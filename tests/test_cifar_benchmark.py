"""The CIFAR-10 experiment benchmark: DeepNEAT runs, the pilot hides the test, resume reproduces.

CIFAR runs DeepNEAT only - EXACT is single-channel by construction, so the
two-method comparison lives on MNIST. These tests replace the CIFAR loader with a
small synthetic same-shape set so the plumbing is exercised offline: the search,
both tracks, the checkpoint round-trip, the pilot's refusal to score the test,
the pinned preprocessing genes, and an interrupted search resuming to the same
selected model.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from examples._benchmark.execution import ExecutionOptions
from examples._datasets import ClassificationDataset
from examples.cifar10._methods import make_deepneat_search
from examples.cifar10._profiles import CONFIGS_DIRECTORY
from examples.cifar10._protocol import Cifar10BenchmarkSettings, run_cifar10_protocol
from polyneat.configs.deepneat.deepneat_config import DeepNEATConfig
from polyneat.evaluators.multiclass_accuracy_evaluator import TrainedMulticlassAccuracyEvaluator
from polyneat.training.model_checkpoint import ModelCheckpoint
from polyneat.training.training_recipe import LearningRateSchedule, TrainingRecipe

_SMALL_RECIPE = TrainingRecipe(
    learning_rate=0.01,
    momentum=0.9,
    weight_decay=0.0005,
    batch_size=32,
    number_of_epochs=1,
    schedule=LearningRateSchedule.CONSTANT,
)


def _synthetic_cifar10(
    *,
    random_seed: int,
    max_train_samples: int | None = None,
    max_test_samples: int | None = None,
    standardize: bool = True,
) -> ClassificationDataset:
    """A small CIFAR-shaped set in ``[0, 1]``, so the suite never downloads."""
    del max_train_samples, max_test_samples, standardize
    rng = np.random.default_rng(random_seed)
    train = rng.random((120, 3 * 32 * 32), dtype=np.float32)
    test = rng.random((60, 3 * 32 * 32), dtype=np.float32)
    return ClassificationDataset(
        train_features=torch.from_numpy(train),
        train_labels=torch.arange(120, dtype=torch.long) % 10,
        test_features=torch.from_numpy(test),
        test_labels=torch.arange(60, dtype=torch.long) % 10,
        number_of_classes=10,
    )


@pytest.fixture(autouse=True)
def _offline_cifar(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reach the loader through the module the protocol calls at run time."""
    monkeypatch.setattr("examples.cifar10._deepneat.load_cifar10", _synthetic_cifar10)


def _settings(**overrides) -> Cifar10BenchmarkSettings:
    defaults = dict(
        protocol_id="cifar10-test",
        dataset_release="cifar10/official",
        dataset_license="fixture",
        image_side=32,
        channels=3,
        number_of_classes=10,
        split_seed=11,
        validation_fraction=0.2,
        search_seed=101,
        retraining_seeds=(),
        candidate_recipe=_SMALL_RECIPE,
        track_b_recipe=_SMALL_RECIPE,
        inference_batch_size=64,
        maximum_phenotype_parameters=5_000_000,
        maximum_training_samples=120,
        maximum_test_samples=60,
        uses_augmentation=True,
        execution=ExecutionOptions(mode="pilot"),
    )
    return Cifar10BenchmarkSettings(**{**defaults, **overrides})


def _deepneat_config() -> DeepNEATConfig:
    return DeepNEATConfig(
        population_size=2,
        number_of_input_nodes=1,
        number_of_output_nodes=10,
        number_of_classes=10,
        input_image_channels=3,
        input_image_height=32,
        input_image_width=32,
        available_filter_counts=(4,),
        available_kernel_sizes=(3,),
        available_dense_unit_counts=(8,),
        maximum_total_parameter_count=5_000_000,
    )


def _search():
    return make_deepneat_search(_deepneat_config(), number_of_generations=2)


def test_smoke_completes_and_reports_both_tracks(tmp_path: Path) -> None:
    report = run_cifar10_protocol(
        _settings(execution=ExecutionOptions(mode="smoke")),
        _search(),
        method_name="deepneat",
        artifacts_directory=tmp_path,
    )

    assert 0.0 <= report.metric_values["validation_accuracy"] <= 1.0
    assert report.metric_values["selected_parameter_count"] > 0
    assert "track_a_seed101_test_accuracy" in report.metric_values
    assert (tmp_path / "run_report.json").is_file()


def test_pilot_never_scores_the_official_test(tmp_path: Path) -> None:
    run_cifar10_protocol(
        _settings(execution=ExecutionOptions(mode="pilot")),
        _search(),
        method_name="deepneat",
        artifacts_directory=tmp_path,
    )
    report = json.loads((tmp_path / "run_report.json").read_text(encoding="utf-8"))

    assert report["mode"] == "pilot"
    assert report["official_test"] == {}


def test_the_profile_pins_every_extra_preprocessing_gene() -> None:
    payload = yaml.safe_load(
        (CONFIGS_DIRECTORY / "deepneat_smoke.yaml").read_text(encoding="utf-8")
    )
    config = DeepNEATConfig.from_dict(payload["algorithm"])
    assert isinstance(config, DeepNEATConfig)

    assert config.global_cropped_image_size_max == 0
    assert config.available_horizontal_flip_options == (False,)
    assert config.available_variance_normalization_options == (False,)


def test_an_interrupted_search_resumes_to_the_same_selected_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings()
    continuous = run_cifar10_protocol(
        settings, _search(), method_name="deepneat",
        artifacts_directory=tmp_path / "continuous",
    )

    original = TrainedMulticlassAccuracyEvaluator._evaluate_one

    def interrupt_second(self, phenotype, evaluation_id):
        if len(self.evaluation_records) == 1:
            raise KeyboardInterrupt
        return original(self, phenotype, evaluation_id)

    with monkeypatch.context() as context:
        context.setattr(TrainedMulticlassAccuracyEvaluator, "_evaluate_one", interrupt_second)
        with pytest.raises(KeyboardInterrupt):
            run_cifar10_protocol(
                settings, _search(), method_name="deepneat",
                artifacts_directory=tmp_path / "resumed",
            )

    resumed = run_cifar10_protocol(
        replace(settings, execution=ExecutionOptions(mode="pilot", resume=True)),
        _search(),
        method_name="deepneat",
        artifacts_directory=tmp_path / "resumed",
    )

    assert resumed.metric_values["validation_accuracy"] == (
        continuous.metric_values["validation_accuracy"]
    )
    model_id = f"deepneat_track_a_seed{settings.search_seed}"
    expected, _ = ModelCheckpoint.read_file(
        tmp_path / "continuous" / "checkpoints" / f"{model_id}.pt"
    )
    actual, _ = ModelCheckpoint.read_file(tmp_path / "resumed" / "checkpoints" / f"{model_id}.pt")
    assert expected.compute_sha256() == actual.compute_sha256()
