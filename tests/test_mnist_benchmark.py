"""The MNIST comparison benchmark: both methods run, the pilot hides the test, resume reproduces.

These tests run on a small subset of the real MNIST set (present locally, no
network) so the plumbing is exercised end to end without paying for a full run.
The scientific numbers are not asserted - a subset of a few hundred digits
measures nothing - only that the protocol behaves: both algorithms complete, a
pilot never scores the official test, a full run refuses to start without its
lock, the extra preprocessing genes stay pinned, and an interrupted search
resumes to exactly the model it would have selected uninterrupted.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from examples._benchmark.execution import ExecutionLockError, ExecutionOptions
from examples.mnist import benchmark_deepneat, benchmark_exact
from examples.mnist._methods import make_deepneat_search, make_exact_search
from examples.mnist._profiles import CONFIGS_DIRECTORY
from examples.mnist._protocol import MnistBenchmarkSettings, run_mnist_protocol
from polyneat.configs.deepneat.deepneat_config import DeepNEATConfig
from polyneat.configs.exact.exact_config import EXACTConfig
from polyneat.evaluators.multiclass_accuracy_evaluator import (
    PretrainedMulticlassAccuracyEvaluator,
    TrainedMulticlassAccuracyEvaluator,
)
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


def _settings(**overrides) -> MnistBenchmarkSettings:
    defaults = dict(
        protocol_id="mnist-test",
        dataset_release="mnist/official",
        dataset_license="fixture",
        image_side=28,
        channels=1,
        number_of_classes=10,
        split_seed=11,
        validation_fraction=0.2,
        search_seed=101,
        retraining_seeds=(),
        candidate_recipe=_SMALL_RECIPE,
        track_b_recipe=_SMALL_RECIPE,
        inference_batch_size=64,
        maximum_phenotype_parameters=5_000_000,
        maximum_training_samples=150,
        maximum_test_samples=80,
        uses_augmentation=True,
        execution=ExecutionOptions(mode="pilot"),
    )
    return MnistBenchmarkSettings(**{**defaults, **overrides})


def _deepneat_config() -> DeepNEATConfig:
    return DeepNEATConfig(
        population_size=2,
        number_of_input_nodes=1,
        number_of_output_nodes=10,
        number_of_classes=10,
        input_image_channels=1,
        input_image_height=28,
        input_image_width=28,
        available_filter_counts=(4,),
        available_kernel_sizes=(3,),
        available_dense_unit_counts=(8,),
        maximum_total_parameter_count=5_000_000,
    )


def _exact_config() -> EXACTConfig:
    return EXACTConfig(
        population_size=2,
        number_of_input_nodes=1,
        number_of_output_nodes=10,
        input_image_height=28,
        input_image_width=28,
        training_batch_size=32,
        number_of_training_epochs_per_genome=1,
        use_simplex_hyperparameter_optimization=False,
    )


def _search_for(method: str):
    if method == "deepneat":
        return make_deepneat_search(_deepneat_config(), number_of_generations=2)
    return make_exact_search(_exact_config(), number_of_generations=2)


@pytest.mark.parametrize(
    ("module", "method"),
    [(benchmark_deepneat, "deepneat"), (benchmark_exact, "exact")],
)
def test_smoke_profile_completes_and_reports_both_tracks(module, method, tmp_path: Path) -> None:
    report = module.run_experiment(artifacts_directory=tmp_path)

    assert 0.0 <= report.metric_values["validation_accuracy"] <= 1.0
    assert report.metric_values["selected_parameter_count"] > 0
    # smoke evaluates the test, so both tracks report an accuracy on it
    assert "track_a_seed101_test_accuracy" in report.metric_values
    assert any(
        key.startswith("track_b_") and key.endswith("_test_accuracy")
        for key in report.metric_values
    )
    assert 0.0 <= report.metric_values["failed_evaluation_fraction"] <= 1.0
    assert (tmp_path / "run_report.json").is_file()


@pytest.mark.parametrize("method", ["deepneat", "exact"])
def test_pilot_never_scores_the_official_test(method: str, tmp_path: Path) -> None:
    run_mnist_protocol(
        _settings(execution=ExecutionOptions(mode="pilot")),
        _search_for(method),
        method_name=method,
        artifacts_directory=tmp_path,
    )
    report = json.loads((tmp_path / "run_report.json").read_text(encoding="utf-8"))

    assert report["mode"] == "pilot"
    assert report["official_test"] == {}


def test_full_mode_requires_a_protocol_lock() -> None:
    with pytest.raises(ExecutionLockError, match="requires --protocol-lock"):
        ExecutionOptions(mode="full")


def test_the_deepneat_profile_pins_every_extra_preprocessing_gene() -> None:
    payload = yaml.safe_load(
        (CONFIGS_DIRECTORY / "deepneat_smoke.yaml").read_text(encoding="utf-8")
    )
    config = DeepNEATConfig.from_dict(payload["algorithm"])
    assert isinstance(config, DeepNEATConfig)

    assert config.global_cropped_image_size_max == 0
    assert config.global_spatial_scaling_max == 0.0
    assert config.available_horizontal_flip_options == (False,)
    assert config.available_variance_normalization_options == (False,)


@pytest.mark.parametrize(
    ("method", "evaluator_class"),
    [
        ("deepneat", TrainedMulticlassAccuracyEvaluator),
        ("exact", PretrainedMulticlassAccuracyEvaluator),
    ],
)
def test_an_interrupted_search_resumes_to_the_same_selected_model(
    method: str, evaluator_class, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings()
    continuous = run_mnist_protocol(
        settings, _search_for(method), method_name=method,
        artifacts_directory=tmp_path / "continuous",
    )

    original = evaluator_class._evaluate_one

    def interrupt_second(self, phenotype, evaluation_id):
        if len(self.evaluation_records) == 1:
            raise KeyboardInterrupt
        return original(self, phenotype, evaluation_id)

    with monkeypatch.context() as context:
        context.setattr(evaluator_class, "_evaluate_one", interrupt_second)
        with pytest.raises(KeyboardInterrupt):
            run_mnist_protocol(
                settings, _search_for(method), method_name=method,
                artifacts_directory=tmp_path / "resumed",
            )

    resumed = run_mnist_protocol(
        replace(settings, execution=ExecutionOptions(mode="pilot", resume=True)),
        _search_for(method),
        method_name=method,
        artifacts_directory=tmp_path / "resumed",
    )

    assert resumed.metric_values["validation_accuracy"] == (
        continuous.metric_values["validation_accuracy"]
    )
    model_id = f"{method}_track_a_seed{settings.search_seed}"
    expected, _ = ModelCheckpoint.read_file(
        tmp_path / "continuous" / "checkpoints" / f"{model_id}.pt"
    )
    actual, _ = ModelCheckpoint.read_file(tmp_path / "resumed" / "checkpoints" / f"{model_id}.pt")
    assert expected.compute_sha256() == actual.compute_sha256()
