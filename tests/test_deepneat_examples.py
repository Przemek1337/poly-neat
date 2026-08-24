"""End-to-end tests for the two DeepNEAT examples (mnist and fashion_mnist).

Mirrors the shrinking approach of ``tests/test_examples_smoke.py`` (which is
frozen and picks up these two examples automatically once they are
registered), but does not import anything from it: its own small synthetic
dataset builder and its own config/termination shrinking are defined here.

Each example is checked against the ``ExampleModule`` contract, checked for
its ``EXAMPLE_REGISTRY`` entry, has its yaml validated through
``DeepNEATConfig``, and is run once, shrunk, end to end.
"""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path

import numpy as np
import pytest
import torch

import polyneat as pn
from examples._datasets import ClassificationDataset, split_features_and_labels
from examples._experiment import EXAMPLE_REGISTRY
from polyneat.configs.algorithm_config import AlgorithmConfig

_SHRUNK_POPULATION_SIZE = 8
_SHRUNK_GENERATION_LIMIT = 2
_SHRUNK_TRAINING_EPOCHS = 1

_SYNTHETIC_TRAIN_SIZE = 64
_SYNTHETIC_TEST_SIZE = 32
_SYNTHETIC_NUMBER_OF_CLASSES = 10
_SYNTHETIC_GRID_SIDE = 28

_EXPECTED_METRIC_NAMES = frozenset(
    {"validation_accuracy", "test_accuracy", "number_of_layers", "total_parameter_count"}
)

_DEEPNEAT_EXAMPLE_MODULES = (
    ("examples.mnist.deepneat", "mnist/deepneat"),
    ("examples.fashion_mnist.deepneat", "fashion_mnist/deepneat"),
)


def _synthetic_deepneat_dataset(
    *,
    train_fraction: float = 6 / 7,
    random_seed: int,
    grid_side: int = _SYNTHETIC_GRID_SIDE,
    max_train_samples: int | None = None,
    max_test_samples: int | None = None,
) -> ClassificationDataset:
    """Stand in for ``load_mnist``/``load_fashion_mnist`` with the same shape.

    Same signature and return type as the real loaders (see
    ``examples/mnist/dataset.py`` and ``examples/fashion_mnist/dataset.py``),
    so the example's call site is exercised without touching either dataset.
    """
    rng = np.random.default_rng(random_seed)
    number_of_features = grid_side * grid_side
    total = _SYNTHETIC_TRAIN_SIZE + _SYNTHETIC_TEST_SIZE
    features = rng.standard_normal((total, number_of_features), dtype=np.float32)
    labels = np.arange(total, dtype=np.int64) % _SYNTHETIC_NUMBER_OF_CLASSES
    return split_features_and_labels(
        torch.from_numpy(features),
        torch.from_numpy(labels),
        train_fraction=train_fraction,
        random_seed=random_seed,
        max_train_samples=min(max_train_samples or total, _SYNTHETIC_TRAIN_SIZE),
        max_test_samples=min(max_test_samples or total, _SYNTHETIC_TEST_SIZE),
        number_of_classes=_SYNTHETIC_NUMBER_OF_CLASSES,
    )


@pytest.fixture
def shrunk_deepneat_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cut population size, training epochs and the generation limit.

    Two seams the DeepNEAT examples route through: the config loader and the
    ``pn.MaxGenerationsTermination`` attribute looked up at call time.
    """
    original_load = AlgorithmConfig.__dict__["load_from_yaml_file"].__func__

    def load_and_shrink(cls: type[AlgorithmConfig], yaml_file_path: Path):
        config = original_load(cls, yaml_file_path)
        config.population_size = _SHRUNK_POPULATION_SIZE
        config.training_epochs_per_evaluation = _SHRUNK_TRAINING_EPOCHS
        return config

    monkeypatch.setattr(AlgorithmConfig, "load_from_yaml_file", classmethod(load_and_shrink))

    original_termination = pn.MaxGenerationsTermination

    def clamped_termination(max_generations: int):
        return original_termination(
            max_generations=min(max_generations, _SHRUNK_GENERATION_LIMIT)
        )

    monkeypatch.setattr(pn, "MaxGenerationsTermination", clamped_termination)


@pytest.mark.parametrize(
    "module_path,registry_key",
    _DEEPNEAT_EXAMPLE_MODULES,
    ids=[key for _, key in _DEEPNEAT_EXAMPLE_MODULES],
)
def test_module_is_registered(module_path: str, registry_key: str) -> None:
    assert registry_key in EXAMPLE_REGISTRY
    assert EXAMPLE_REGISTRY[registry_key] == module_path


@pytest.mark.parametrize(
    "module_path,_registry_key",
    _DEEPNEAT_EXAMPLE_MODULES,
    ids=[key for _, key in _DEEPNEAT_EXAMPLE_MODULES],
)
def test_module_satisfies_example_contract(module_path: str, _registry_key: str) -> None:
    example_module = importlib.import_module(module_path)

    assert isinstance(example_module.CONFIG_FILE_PATH, Path)
    assert example_module.CONFIG_FILE_PATH.is_file()

    assert callable(example_module.run_experiment)
    signature = inspect.signature(example_module.run_experiment)
    assert set(signature.parameters) == {"device", "random_seed", "artifacts_directory"}


@pytest.mark.parametrize(
    "module_path,_registry_key",
    _DEEPNEAT_EXAMPLE_MODULES,
    ids=[key for _, key in _DEEPNEAT_EXAMPLE_MODULES],
)
def test_yaml_loads_and_validates_as_deepneat_config(
    module_path: str, _registry_key: str
) -> None:
    example_module = importlib.import_module(module_path)
    config = pn.DeepNEATConfig.load_from_yaml_file(example_module.CONFIG_FILE_PATH)
    config.validate()  # must not raise
    assert isinstance(config, pn.DeepNEATConfig)
    assert config.input_image_height == 28
    assert config.input_image_width == 28
    # Every example ships cpu and takes the GPU from --gpu via resolve_device;
    # a yaml saying cuda would fail the suite on any CPU-only machine, and
    # asserting it here catches that on a CUDA machine too.
    assert config.device_for_phenotype_evaluation == "cpu"


@pytest.mark.parametrize(
    "module_path,_registry_key",
    _DEEPNEAT_EXAMPLE_MODULES,
    ids=[key for _, key in _DEEPNEAT_EXAMPLE_MODULES],
)
def test_shrunk_run_reports_all_four_metrics(
    module_path: str,
    _registry_key: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    shrunk_deepneat_environment: None,
) -> None:
    example_module = importlib.import_module(module_path)
    monkeypatch.setattr(example_module, "load_mnist", _synthetic_deepneat_dataset)

    report = example_module.run_experiment(artifacts_directory=tmp_path)

    assert set(report.metric_values) == _EXPECTED_METRIC_NAMES
    for metric_name, metric_value in report.metric_values.items():
        assert isinstance(metric_value, float), f"{metric_name} is not a float"
    # Pins the termination monkeypatch: binding MaxGenerationsTermination at
    # import time instead of looking it up on `pn` leaves every assertion in
    # this file green while silently running the yaml's full 25 generations.
    # The history includes generation 0, hence the +1.
    assert report.number_of_generations == _SHRUNK_GENERATION_LIMIT + 1
    assert report.runtime_seconds >= 0.0

    written_files = [path for path in tmp_path.rglob("*") if path.is_file()]
    assert written_files, f"{module_path} wrote no artifacts"


class _RecordingEvaluator:
    """Wraps ``TrainedNetworkAccuracyEvaluator``, recording what it was given.

    Row sets are captured as hashable tuples so the test can reason about
    which rows each evaluator trained and scored on without knowing anything
    about the dataset the example loaded.
    """

    constructions: list[_RecordingEvaluator] = []
    # Every phenotype any evaluator has already scored. References are kept
    # deliberately: identity is the check, and a garbage-collected object's
    # id can be handed out again.
    already_evaluated: list = []

    def __init__(self, **keyword_arguments: object) -> None:
        self._wrapped = _ORIGINAL_EVALUATOR(**keyword_arguments)
        self.train_rows = _rows_as_hashables(keyword_arguments["train_features"])
        self.validation_rows = _rows_as_hashables(keyword_arguments["validation_features"])
        self.returned_scores: list[float] = []
        self.any_parameter_changed = False
        self.previously_trained_phenotypes: list = []
        _RecordingEvaluator.constructions.append(self)

    def evaluate_batch_of_phenotypes(self, phenotypes: list) -> list[float]:
        self.previously_trained_phenotypes.extend(
            phenotype
            for phenotype in phenotypes
            if any(phenotype is seen for seen in _RecordingEvaluator.already_evaluated)
        )
        _RecordingEvaluator.already_evaluated.extend(phenotypes)
        before = [
            [parameter.detach().clone() for parameter in phenotype.parameters()]
            for phenotype in phenotypes
        ]
        scores = self._wrapped.evaluate_batch_of_phenotypes(phenotypes)
        for phenotype, snapshot in zip(phenotypes, before, strict=True):
            for parameter, original in zip(phenotype.parameters(), snapshot, strict=True):
                if not torch.equal(parameter.detach(), original):
                    self.any_parameter_changed = True
        self.returned_scores.extend(scores)
        return scores


_ORIGINAL_EVALUATOR = pn.TrainedNetworkAccuracyEvaluator


def _rows_as_hashables(features: torch.Tensor) -> set[tuple[float, ...]]:
    """Represent each row of ``features`` as a hashable tuple of its values."""
    return {tuple(row.flatten().tolist()) for row in features}


@pytest.mark.parametrize(
    "module_path,_registry_key",
    _DEEPNEAT_EXAMPLE_MODULES,
    ids=[key for _, key in _DEEPNEAT_EXAMPLE_MODULES],
)
def test_reported_test_accuracy_comes_from_a_retrain_on_unseen_rows(
    module_path: str,
    _registry_key: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    shrunk_deepneat_environment: None,
) -> None:
    """Pin the provenance of the headline number, not just its name and type.

    Without this, three separate corruptions of the final reporting step pass
    the whole suite: scoring on the validation slice evolution already
    optimised against, skipping the retrain and returning a constant, and
    reusing a phenotype trained elsewhere. Each is a plausible edit and each
    would put a wrong accuracy in a thesis, so the shape of the final step is
    asserted directly: two evaluators, the second training on exactly what
    evolution saw and scoring on rows it never did.
    """
    example_module = importlib.import_module(module_path)
    monkeypatch.setattr(example_module, "load_mnist", _synthetic_deepneat_dataset)
    monkeypatch.setattr(_RecordingEvaluator, "constructions", [])
    monkeypatch.setattr(_RecordingEvaluator, "already_evaluated", [])
    monkeypatch.setattr(pn, "TrainedNetworkAccuracyEvaluator", _RecordingEvaluator)

    report = example_module.run_experiment(artifacts_directory=tmp_path)

    fitness_evaluator, final_evaluator = _RecordingEvaluator.constructions

    evolution_saw = fitness_evaluator.train_rows | fitness_evaluator.validation_rows
    assert final_evaluator.validation_rows.isdisjoint(evolution_saw)
    assert final_evaluator.validation_rows.isdisjoint(final_evaluator.train_rows)
    assert final_evaluator.train_rows == evolution_saw
    assert fitness_evaluator.train_rows.isdisjoint(fitness_evaluator.validation_rows)

    assert final_evaluator.any_parameter_changed, "the final genome was never retrained"
    # "Retrained from scratch" means the weights start fresh. A phenotype that
    # some earlier evaluator already trained would still show changed
    # parameters here, so parameter movement alone does not say it. Object
    # identity does: the final network must be one nothing has scored before.
    assert final_evaluator.previously_trained_phenotypes == []
    assert report.metric_values["test_accuracy"] == final_evaluator.returned_scores[-1]
