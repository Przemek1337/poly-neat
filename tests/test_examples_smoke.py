"""End-to-end smoke tests: every example's ``run_experiment`` runs to completion.

These tests verify wiring, not quality. Each example is shrunk to a population
of 8 evolved for 2 generations, its artifacts are redirected into a temporary
directory, and its dataset loaders are replaced with same-shape synthetic
tensors so that the suite never reaches the network.

No assertion is made about fitness or accuracy. A smoke test that asserts on a
score becomes a flaky test, and the examples' scores are the thing most likely
to move legitimately.

The examples themselves know nothing about any of this: there is no test-only
flag, environment variable or hook in the example files. An example run by a
person always loads the real dataset.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
import torch

import polyneat as pn
from examples._datasets import ClassificationDataset, split_features_and_labels
from examples._experiment import EXAMPLE_REGISTRY
from polyneat.config.algorithm_config import AlgorithmConfig

_SMOKE_POPULATION_SIZE = 8
_SMOKE_GENERATION_LIMIT = 2

_IRIS_SAMPLE_COUNT = 150
_IRIS_FEATURE_COUNT = 4
_IRIS_CLASS_COUNT = 3

_MNIST_SMOKE_TRAIN_SIZE = 64
_MNIST_SMOKE_TEST_SIZE = 32
_MNIST_CLASS_COUNT = 10


def _synthetic_iris_dataset(
    *, train_fraction: float = 0.66, random_seed: int
) -> ClassificationDataset:
    """Stand in for ``load_iris`` with the same shapes and return type.

    Builds a synthetic 150-sample Iris and runs it through the real
    ``split_features_and_labels``, so the example's call site and the split seam
    are exercised without touching the UCI file.
    """
    rng = np.random.default_rng(0)
    features = rng.random((_IRIS_SAMPLE_COUNT, _IRIS_FEATURE_COUNT), dtype=np.float32)
    labels = np.arange(_IRIS_SAMPLE_COUNT, dtype=np.int64) % _IRIS_CLASS_COUNT
    return split_features_and_labels(
        torch.from_numpy(features),
        torch.from_numpy(labels),
        train_fraction=train_fraction,
        random_seed=random_seed,
        number_of_classes=_IRIS_CLASS_COUNT,
    )


def _synthetic_mnist_dataset(
    *,
    train_fraction: float = 6 / 7,
    random_seed: int,
    grid_side: int = 7,
    max_train_samples: int | None = 3000,
    max_test_samples: int | None = 2000,
) -> ClassificationDataset:
    """Stand in for ``load_mnist`` with the same signature and return type.

    Subset caps are shrunk well below what the examples ask for, since the point
    is to run the pipeline, not to train. The real ``split_features_and_labels``
    still does the splitting and subsetting.
    """
    rng = np.random.default_rng(random_seed)
    number_of_features = grid_side * grid_side
    total = _MNIST_SMOKE_TRAIN_SIZE + _MNIST_SMOKE_TEST_SIZE
    features = rng.standard_normal((total, number_of_features), dtype=np.float32)
    labels = np.arange(total, dtype=np.int64) % _MNIST_CLASS_COUNT
    return split_features_and_labels(
        torch.from_numpy(features),
        torch.from_numpy(labels),
        train_fraction=train_fraction,
        random_seed=random_seed,
        max_train_samples=min(max_train_samples or total, _MNIST_SMOKE_TRAIN_SIZE),
        max_test_samples=min(max_test_samples or total, _MNIST_SMOKE_TEST_SIZE),
        number_of_classes=_MNIST_CLASS_COUNT,
    )


def _shrink_loaded_config(config: AlgorithmConfig) -> AlgorithmConfig:
    """Cut every knob that drives runtime, leaving the rest of the YAML intact."""
    config.population_size = _SMOKE_POPULATION_SIZE
    # L-NEAT only runs a backpropagation session every `learning_interval_generations`
    # (5 in the YAML), which a 2-generation run would never reach. Forcing the
    # interval to 1 keeps the Lamarckian path inside the smoke test's reach.
    if hasattr(config, "learning_interval_generations"):
        config.learning_interval_generations = 1
    if hasattr(config, "backpropagation_iterations_per_session"):
        config.backpropagation_iterations_per_session = 2
    return config


@pytest.fixture
def shrunk_example_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Patch the shared machinery every example routes through.

    Two seams, each verified to be uniform across all ten examples: the config
    loader and the generation-limit termination criterion. Artifacts go through
    ``run_experiment``'s parameter, dataset loaders are swapped per module.
    """
    original_load = AlgorithmConfig.__dict__["load_from_yaml_file"].__func__

    def load_and_shrink(cls: type[AlgorithmConfig], yaml_file_path: Path):
        return _shrink_loaded_config(original_load(cls, yaml_file_path))

    monkeypatch.setattr(
        AlgorithmConfig, "load_from_yaml_file", classmethod(load_and_shrink)
    )

    original_termination = pn.MaxGenerationsTermination

    def clamped_termination(max_generations: int):
        return original_termination(
            max_generations=min(max_generations, _SMOKE_GENERATION_LIMIT)
        )

    monkeypatch.setattr(pn, "MaxGenerationsTermination", clamped_termination)


def _redirect_dataset_loaders(
    example_module: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Swap the example's dataset loaders for synthetic same-shape builders.

    The examples import their loaders by name, so the names live in the example
    module's own namespace and ``setattr`` on the module reaches them. The real
    loaders are never called.
    """
    if hasattr(example_module, "load_iris"):
        monkeypatch.setattr(example_module, "load_iris", _synthetic_iris_dataset)
    if hasattr(example_module, "load_mnist"):
        monkeypatch.setattr(example_module, "load_mnist", _synthetic_mnist_dataset)


@pytest.mark.parametrize("example_id", sorted(EXAMPLE_REGISTRY))
def test_example_run_experiment_completes_and_writes_artifacts(
    example_id: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    shrunk_example_environment: None,
) -> None:
    example_module = importlib.import_module(EXAMPLE_REGISTRY[example_id])
    _redirect_dataset_loaders(example_module, monkeypatch)

    report = example_module.run_experiment(artifacts_directory=tmp_path)

    assert report.metric_values, f"{example_id} reported no metrics"
    assert report.number_of_generations >= 1
    assert report.runtime_seconds >= 0.0
    written_files = [path for path in tmp_path.rglob("*") if path.is_file()]
    assert written_files, f"{example_id} wrote no artifacts"


def test_every_example_module_is_registered() -> None:
    """Guard against a new example being added without a registry entry."""
    examples_directory = Path(__file__).parent.parent / "examples"
    module_paths = set()
    for path in examples_directory.rglob("*.py"):
        relative_path = path.relative_to(examples_directory)
        if any(part.startswith("_") for part in relative_path.parts):
            continue
        if relative_path.stem == "dataset":
            continue
        module_paths.add("examples." + ".".join(relative_path.with_suffix("").parts))
    assert module_paths == set(EXAMPLE_REGISTRY.values())
