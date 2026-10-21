"""A diverged track B retraining is recorded as failed and does not discard the run."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

import examples.pediatric_pneumonia._protocol as protocol
from examples.pediatric_pneumonia._methods import make_fixed_cnn_baseline
from examples.pediatric_pneumonia._protocol import BenchmarkSettings, run_pneumonia_protocol
from examples.pediatric_pneumonia._synthetic_archive import (
    SyntheticArchiveSpec,
    write_synthetic_archive,
)
from polyneat.evaluators.bootstrap_confidence_intervals import BootstrapConfig
from polyneat.nn.fixed_convolutional_network import FixedConvolutionalNetworkConfig
from polyneat.training.training_recipe import LearningRateSchedule, TrainingRecipe

_IMAGE_SIDE = 32
_DIVERGING_SEED = 202


@pytest.fixture(scope="module")
def synthetic_archive(tmp_path_factory: pytest.TempPathFactory) -> Path:
    directory = tmp_path_factory.mktemp("pneumonia_divergence") / "data"
    return write_synthetic_archive(
        directory,
        random_seed=31337,
        spec=SyntheticArchiveSpec(
            train_normal_patients=10,
            train_pneumonia_patients=14,
            validation_normal_patients=3,
            validation_pneumonia_patients=3,
            test_normal_patients=4,
            test_pneumonia_patients=6,
            images_per_patient=2,
        ),
    )


def _settings(archive: Path) -> BenchmarkSettings:
    recipe = TrainingRecipe(
        learning_rate=0.02,
        momentum=0.9,
        weight_decay=0.0005,
        batch_size=8,
        number_of_epochs=2,
        schedule=LearningRateSchedule.CONSTANT,
    )
    return BenchmarkSettings(
        data_directory=archive,
        protocol_id="protocol-test-v1",
        dataset_release="synthetic/fixture",
        dataset_license="not-applicable-synthetic",
        image_side=_IMAGE_SIDE,
        split_seed=11,
        search_seed=101,
        retraining_seeds=(201, _DIVERGING_SEED),
        bootstrap_seed=13,
        candidate_recipe=recipe,
        track_b_recipe=recipe,
        inference_batch_size=8,
        bootstrap=BootstrapConfig(number_of_replicates=20, maximum_attempts=200),
        maximum_phenotype_parameters=2_000_000,
    )


@pytest.fixture
def one_retraining_diverges(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make one retraining end with NaN weights, as a diverged SGD run does."""
    original = protocol.retrain_topology_for_track_b

    def retrain(settings, data, selected, retraining_seed):
        model, preprocessor, summary = original(settings, data, selected, retraining_seed)
        if retraining_seed == _DIVERGING_SEED:
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.fill_(float("nan"))
        return model, preprocessor, summary

    monkeypatch.setattr(protocol, "retrain_topology_for_track_b", retrain)


@pytest.mark.usefixtures("one_retraining_diverges")
def test_a_diverged_retraining_is_recorded_and_the_rest_is_scored(
    synthetic_archive: Path, tmp_path: Path
) -> None:
    artifacts = tmp_path / "run"
    report = run_pneumonia_protocol(
        _settings(synthetic_archive),
        make_fixed_cnn_baseline(
            FixedConvolutionalNetworkConfig(convolution_channels=(4, 8), dropout_probability=0.0)
        ),
        method_name="fixed_cnn",
        artifacts_directory=artifacts,
    )

    assert "track_a_seed101_test_auroc" in report.metric_values
    assert "track_b_seed201_test_auroc" in report.metric_values
    assert f"track_b_seed{_DIVERGING_SEED}_test_auroc" not in report.metric_values
    assert report.metric_values["track_b_failed_retraining_count"] == 1.0

    run_report = json.loads((artifacts / "run_report.json").read_text(encoding="utf-8"))
    failed = run_report["track_b_failed_retrainings"]
    assert [entry["retraining_seed"] for entry in failed] == [_DIVERGING_SEED]
    assert "non-finite" in failed[0]["reason"]
    assert f"fixed_cnn_track_b_seed{_DIVERGING_SEED}" not in run_report["official_test"]


def test_a_run_without_divergence_reports_zero_failed_retrainings(
    synthetic_archive: Path, tmp_path: Path
) -> None:
    artifacts = tmp_path / "run"
    report = run_pneumonia_protocol(
        _settings(synthetic_archive),
        make_fixed_cnn_baseline(
            FixedConvolutionalNetworkConfig(convolution_channels=(4, 8), dropout_probability=0.0)
        ),
        method_name="fixed_cnn",
        artifacts_directory=artifacts,
    )

    assert report.metric_values["track_b_failed_retraining_count"] == 0.0
    run_report = json.loads((artifacts / "run_report.json").read_text(encoding="utf-8"))
    assert run_report["track_b_failed_retrainings"] == []
