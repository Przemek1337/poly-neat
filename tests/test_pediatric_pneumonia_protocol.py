"""End-to-end tests of the pediatric pneumonia protocol on a synthetic archive."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from examples.pediatric_pneumonia._methods import (
    make_deepneat_search,
    make_fixed_cnn_baseline,
    make_random_search,
)
from examples.pediatric_pneumonia._protocol import (
    BenchmarkSettings,
    prepare_data,
    run_pneumonia_protocol,
)
from examples.pediatric_pneumonia._synthetic_archive import (
    SyntheticArchiveSpec,
    write_synthetic_archive,
)
from examples.pediatric_pneumonia.dataset import load_split_tensors
from polyneat.configs.deepneat.deepneat_config import DeepNEATConfig
from polyneat.evaluators.binary_inference import predict_binary
from polyneat.evaluators.bootstrap_confidence_intervals import BootstrapConfig
from polyneat.nn.fixed_convolutional_network import FixedConvolutionalNetworkConfig
from polyneat.training.image_preprocessing import (
    ImagePreprocessingConfig,
    ImagePreprocessor,
)
from polyneat.training.model_checkpoint import ModelCheckpoint
from polyneat.training.training_recipe import LearningRateSchedule, TrainingRecipe

_IMAGE_SIDE = 32


@pytest.fixture(scope="module")
def synthetic_archive(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A small archive shared by every test in this module."""
    directory = tmp_path_factory.mktemp("pneumonia_protocol") / "data"
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


def _settings(archive: Path, **overrides) -> BenchmarkSettings:
    recipe = TrainingRecipe(
        learning_rate=0.02,
        momentum=0.9,
        weight_decay=0.0005,
        batch_size=8,
        number_of_epochs=2,
        schedule=LearningRateSchedule.CONSTANT,
    )
    defaults = dict(
        data_directory=archive,
        protocol_id="protocol-test-v1",
        dataset_release="synthetic/fixture",
        dataset_license="not-applicable-synthetic",
        image_side=_IMAGE_SIDE,
        split_seed=11,
        search_seed=101,
        retraining_seeds=(201,),
        bootstrap_seed=13,
        candidate_recipe=recipe,
        track_b_recipe=recipe,
        inference_batch_size=8,
        bootstrap=BootstrapConfig(number_of_replicates=20, maximum_attempts=200),
        maximum_phenotype_parameters=2_000_000,
    )
    return BenchmarkSettings(**{**defaults, **overrides})


def _fixed_cnn_search():
    return make_fixed_cnn_baseline(
        FixedConvolutionalNetworkConfig(
            convolution_channels=(4, 8), dropout_probability=0.0
        )
    )


def _deepneat_config() -> DeepNEATConfig:
    return DeepNEATConfig(
        population_size=3,
        number_of_input_nodes=1,
        number_of_output_nodes=2,
        number_of_classes=2,
        random_seed=101,
        input_image_channels=1,
        input_image_height=_IMAGE_SIDE,
        input_image_width=_IMAGE_SIDE,
        available_filter_counts=(4, 8),
        available_kernel_sizes=(3,),
        available_dense_unit_counts=(8, 16),
        maximum_total_parameter_count=2_000_000,
    )


class TestProtocolEndToEnd:
    def test_the_fixed_cnn_baseline_completes_both_tracks(
        self, synthetic_archive: Path, tmp_path: Path
    ) -> None:
        report = run_pneumonia_protocol(
            _settings(synthetic_archive),
            _fixed_cnn_search(),
            method_name="fixed_cnn",
            artifacts_directory=tmp_path / "run",
        )
        assert report.succeeded
        assert "track_a_seed101_test_auroc" in report.metric_values
        assert "track_b_seed201_test_auroc" in report.metric_values
        assert 0.0 <= report.metric_values["search_validation_auroc"] <= 1.0

    def test_deepneat_and_random_search_share_the_protocol(
        self, synthetic_archive: Path, tmp_path: Path
    ) -> None:
        for method_name, search in (
            ("deepneat", make_deepneat_search(_deepneat_config(), number_of_generations=2)),
            (
                "random_search_deepneat",
                make_random_search(
                    _deepneat_config(),
                    number_of_candidates=3,
                    minimum_structural_mutations=1,
                    maximum_structural_mutations=2,
                ),
            ),
        ):
            report = run_pneumonia_protocol(
                _settings(synthetic_archive),
                search,
                method_name=method_name,
                artifacts_directory=tmp_path / method_name,
            )
            assert report.succeeded
            assert (tmp_path / method_name / "run_report.json").is_file()
            assert (tmp_path / method_name / "manifest.json").is_file()

    def test_every_run_writes_the_artifacts_a_rerun_would_need(
        self, synthetic_archive: Path, tmp_path: Path
    ) -> None:
        artifacts = tmp_path / "run"
        report = run_pneumonia_protocol(
            _settings(synthetic_archive),
            _fixed_cnn_search(),
            method_name="fixed_cnn",
            artifacts_directory=artifacts,
        )
        assert set(report.artifact_paths) == {
            "manifest",
            "run_report",
            "checkpoints",
            "predictions",
        }
        assert len(list((artifacts / "checkpoints").glob("*.pt"))) == 2
        assert len(list((artifacts / "predictions").glob("*_official_test.json"))) == 2

        run_report = json.loads((artifacts / "run_report.json").read_text(encoding="utf-8"))
        assert run_report["effective_configuration"]["split_seed"] == 11
        assert run_report["manifest_sha256"]
        assert run_report["patient_independence_established"] is False

    def test_two_seeds_write_into_separate_directories(
        self, synthetic_archive: Path, tmp_path: Path
    ) -> None:
        first = run_pneumonia_protocol(
            _settings(synthetic_archive, search_seed=101),
            _fixed_cnn_search(),
            method_name="fixed_cnn",
            artifacts_directory=tmp_path / "seed_101",
        )
        second = run_pneumonia_protocol(
            _settings(synthetic_archive, search_seed=202),
            _fixed_cnn_search(),
            method_name="fixed_cnn",
            artifacts_directory=tmp_path / "seed_202",
        )
        assert first.artifact_paths["run_report"] != second.artifact_paths["run_report"]
        assert (tmp_path / "seed_101" / "run_report.json").is_file()
        assert (tmp_path / "seed_202" / "run_report.json").is_file()


class TestCheckpointsAndThresholds:
    @staticmethod
    def _run(archive: Path, artifacts: Path):
        report = run_pneumonia_protocol(
            _settings(archive),
            _fixed_cnn_search(),
            method_name="fixed_cnn",
            artifacts_directory=artifacts,
        )
        run_report = json.loads((artifacts / "run_report.json").read_text(encoding="utf-8"))
        return report, run_report

    def test_each_model_gets_its_own_threshold_bound_to_its_own_checkpoint(
        self, synthetic_archive: Path, tmp_path: Path
    ) -> None:
        _, run_report = self._run(synthetic_archive, tmp_path / "run")
        frozen = run_report["frozen_models"]
        assert len(frozen) == 2
        bindings = {
            model["details"]["threshold"]["binding_sha256"] for model in frozen.values()
        }
        assert len(bindings) == 2, "track A and track B must not share a threshold binding"
        digests = {model["checkpoint_sha256"] for model in frozen.values()}
        assert len(digests) == 2

    def test_thresholds_are_selected_only_on_the_threshold_split(
        self, synthetic_archive: Path, tmp_path: Path
    ) -> None:
        _, run_report = self._run(synthetic_archive, tmp_path / "run")
        for model in run_report["frozen_models"].values():
            assert model["details"]["threshold"]["split_name"] == "threshold_validation"

    def test_the_saved_checkpoint_reproduces_its_own_test_predictions(
        self, synthetic_archive: Path, tmp_path: Path
    ) -> None:
        artifacts = tmp_path / "run"
        self._run(synthetic_archive, artifacts)

        model_id = "fixed_cnn_track_b_seed201"
        checkpoint, _ = ModelCheckpoint.read_file(artifacts / "checkpoints" / f"{model_id}.pt")
        exported = json.loads(
            (artifacts / "predictions" / f"{model_id}_official_test.json").read_text(
                encoding="utf-8"
            )
        )

        from polyneat.nn.fixed_convolutional_network import FixedConvolutionalNetwork

        restored_model = FixedConvolutionalNetwork(
            FixedConvolutionalNetworkConfig(
                convolution_channels=(4, 8), dropout_probability=0.0
            )
        )
        checkpoint.restore_into(restored_model)
        restored_preprocessor = ImagePreprocessor(
            ImagePreprocessingConfig(target_side=_IMAGE_SIDE)
        )
        restored_preprocessor.load_state_dict(checkpoint.preprocessing_state)

        data = prepare_data(_settings(synthetic_archive))
        test_split = load_split_tensors(
            data.manifest,
            data_directory=synthetic_archive,
            split_name="official_test",
            target_side=_IMAGE_SIDE,
            manifest_sha256=data.manifest_sha256,
        )
        recomputed = predict_binary(
            restored_model,
            images=test_split.images,
            labels=test_split.labels,
            example_ids=test_split.example_ids,
            group_ids=test_split.group_ids,
            preprocessor=restored_preprocessor,
            batch_size=8,
            device_for_computation=torch.device("cpu"),
            model_id=model_id,
            stage="track_b",
            split_name="official_test",
        )
        exported_probabilities = torch.tensor(
            [record["positive_class_probability"] for record in exported["records"]],
            dtype=torch.float64,
        )
        assert torch.allclose(
            recomputed.positive_class_probabilities, exported_probabilities, atol=1e-9
        )

    def test_track_b_refits_its_own_preprocessing_on_the_wider_split(
        self, synthetic_archive: Path, tmp_path: Path
    ) -> None:
        artifacts = tmp_path / "run"
        self._run(synthetic_archive, artifacts)
        track_a, _ = ModelCheckpoint.read_file(
            artifacts / "checkpoints" / "fixed_cnn_track_a_seed101.pt"
        )
        track_b, _ = ModelCheckpoint.read_file(
            artifacts / "checkpoints" / "fixed_cnn_track_b_seed201.pt"
        )
        assert (
            track_a.preprocessing_state["fitted_mean"]
            != track_b.preprocessing_state["fitted_mean"]
        ), "track B trains on train+search_validation and must fit its own statistics"

    def test_a_prediction_file_carries_the_ids_needed_to_trace_it(
        self, synthetic_archive: Path, tmp_path: Path
    ) -> None:
        artifacts = tmp_path / "run"
        self._run(synthetic_archive, artifacts)
        exported = json.loads(
            (
                artifacts / "predictions" / "fixed_cnn_track_a_seed101_official_test.json"
            ).read_text(encoding="utf-8")
        )
        assert exported["split_name"] == "official_test"
        first_record = exported["records"][0]
        assert set(first_record) == {
            "example_id",
            "group_id",
            "label",
            "positive_class_probability",
            "logit_negative",
            "logit_positive",
        }


class TestDeepNeatWinnerIsTheModelThatWon:
    def test_the_kept_weights_reproduce_the_selected_fitness(
        self, synthetic_archive: Path
    ) -> None:
        """A rebuilt DeepNEAT genome would score differently; the snapshot must not."""
        from examples.pediatric_pneumonia._protocol import prepare_data, run_search_stage

        settings = _settings(synthetic_archive)
        data = prepare_data(settings)
        selected, preprocessor = run_search_stage(
            settings, data, make_deepneat_search(_deepneat_config(), number_of_generations=2)
        )

        from polyneat.evaluators.binary_classification_metrics import compute_auroc

        predictions = predict_binary(
            selected.track_a_model,
            images=data.search_validation.images,
            labels=data.search_validation.labels,
            example_ids=data.search_validation.example_ids,
            group_ids=data.search_validation.group_ids,
            preprocessor=preprocessor,
            batch_size=8,
            device_for_computation=torch.device("cpu"),
            model_id="winner",
            stage="track_a",
            split_name="search_validation",
        )
        assert compute_auroc(
            predictions.labels, predictions.positive_class_probabilities
        ) == pytest.approx(selected.selection_fitness, abs=1e-6)

    def test_rebuilding_the_topology_gives_a_different_untrained_model(
        self, synthetic_archive: Path
    ) -> None:
        from examples.pediatric_pneumonia._protocol import prepare_data, run_search_stage

        settings = _settings(synthetic_archive)
        data = prepare_data(settings)
        selected, _ = run_search_stage(
            settings, data, make_deepneat_search(_deepneat_config(), number_of_generations=2)
        )
        rebuilt = selected.rebuild_model()
        trained_parameters = list(selected.track_a_model.parameters())
        rebuilt_parameters = list(rebuilt.parameters())
        assert len(trained_parameters) == len(rebuilt_parameters)
        assert any(
            not torch.allclose(trained, fresh)
            for trained, fresh in zip(trained_parameters, rebuilt_parameters, strict=True)
        ), "a rebuilt genome must not already carry the winner's trained weights"


class TestSearchStagePermissions:
    def test_the_search_never_receives_the_threshold_or_test_splits(
        self, synthetic_archive: Path
    ) -> None:
        settings = _settings(synthetic_archive)
        data = prepare_data(settings)
        seen_splits: list[str] = []

        def recording_search(context):
            seen_splits.append(context.train.split_name)
            seen_splits.append(context.search_validation.split_name)
            return _fixed_cnn_search()(context)

        from examples.pediatric_pneumonia._protocol import run_search_stage

        run_search_stage(settings, data, recording_search)
        assert seen_splits == ["train", "search_validation"]
        assert "threshold_validation" not in seen_splits
        assert "official_test" not in seen_splits

    def test_track_a_statistics_are_fitted_on_train_alone(
        self, synthetic_archive: Path
    ) -> None:
        from examples.pediatric_pneumonia._protocol import build_preprocessor

        settings = _settings(synthetic_archive)
        data = prepare_data(settings)
        train_only = build_preprocessor(settings, data.train.images)
        train_and_validation = build_preprocessor(
            settings,
            torch.cat([data.train.images, data.search_validation.images]),
        )
        assert (
            train_only.state_dict()["fitted_mean"]
            != train_and_validation.state_dict()["fitted_mean"]
        )
