"""Tests for the wall-clock budget, failure-aware selection and run checkpoints."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from polyneat.evaluators.binary_auroc_evaluator import (
    TrainedBinaryAurocEvaluator,
    ValidationSplit,
)
from polyneat.nn.fixed_convolutional_network import (
    FixedConvolutionalNetwork,
    FixedConvolutionalNetworkConfig,
)
from polyneat.runner.evaluation_record import EvaluationStatus
from polyneat.runner.run_checkpoint import (
    RUN_CHECKPOINT_SCHEMA_VERSION,
    CheckpointResumeError,
    RunCheckpoint,
    RunCheckpointStore,
    build_run_binding,
    verify_checkpoint_is_resumable,
)
from polyneat.runner.wall_clock_budget import (
    WallClockBudget,
    WallClockBudgetTermination,
)
from polyneat.training.class_weights import compute_balanced_class_weights
from polyneat.training.image_preprocessing import (
    ImagePreprocessingConfig,
    ImagePreprocessor,
)
from polyneat.training.supervised_trainer import DEADLINE_REACHED, SupervisedTrainer
from polyneat.training.training_recipe import TrainingRecipe

_CPU = torch.device("cpu")
_SIDE = 16


class FakeClock:
    """A clock the test moves by hand, so no test ever waits."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class TestWallClockBudget:
    def test_time_is_only_charged_while_running(self) -> None:
        clock = FakeClock()
        budget = WallClockBudget(100.0, clock=clock)
        clock.advance(30.0)
        assert budget.consumed_seconds == 0.0, "a budget that never started spends nothing"

        budget.start()
        clock.advance(40.0)
        assert budget.consumed_seconds == pytest.approx(40.0)
        assert budget.remaining_seconds == pytest.approx(60.0)
        assert not budget.is_exhausted

    def test_downtime_between_pause_and_resume_is_not_charged(self) -> None:
        clock = FakeClock()
        budget = WallClockBudget(100.0, clock=clock)
        budget.start()
        clock.advance(10.0)
        budget.pause()
        clock.advance(500.0)
        budget.start()
        clock.advance(5.0)
        assert budget.consumed_seconds == pytest.approx(15.0)

    def test_exhaustion_is_reported_through_should_stop(self) -> None:
        clock = FakeClock()
        budget = WallClockBudget(10.0, clock=clock)
        budget.start()
        assert budget.should_stop() is False
        clock.advance(10.0)
        assert budget.should_stop() is True
        assert budget.remaining_seconds == 0.0

    def test_termination_criterion_fires_on_exhaustion(self) -> None:
        clock = FakeClock()
        budget = WallClockBudget(10.0, clock=clock)
        criterion = WallClockBudgetTermination(budget)
        budget.start()
        assert criterion.should_terminate_evolution(None) is False
        clock.advance(11.0)
        assert criterion.should_terminate_evolution(None) is True
        assert criterion.termination_reason_label == "budget_exhausted"

    def test_resuming_charges_work_already_done(self) -> None:
        clock = FakeClock()
        first_segment = WallClockBudget(100.0, clock=clock)
        first_segment.start()
        clock.advance(60.0)
        state = first_segment.state_dict()

        second_clock = FakeClock()
        second_segment = WallClockBudget(100.0, clock=second_clock)
        second_segment.load_state_dict(state)
        second_segment.start()
        second_clock.advance(30.0)
        assert second_segment.consumed_seconds == pytest.approx(90.0)
        assert not second_segment.is_exhausted
        second_clock.advance(20.0)
        assert second_segment.is_exhausted

    def test_a_state_for_a_different_allowance_is_refused(self) -> None:
        budget = WallClockBudget(100.0)
        with pytest.raises(ValueError, match="allowance"):
            budget.load_state_dict(
                {"schema_version": "1.0", "total_seconds": 200.0, "consumed_seconds": 1.0}
            )

    def test_a_non_positive_allowance_is_refused(self) -> None:
        with pytest.raises(ValueError, match="total_seconds"):
            WallClockBudget(0.0)


def _labelled_images(number_per_class: int = 16) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(11)
    negatives = torch.rand(number_per_class, 1, _SIDE, _SIDE, generator=generator) * 0.3
    positives = torch.rand(number_per_class, 1, _SIDE, _SIDE, generator=generator) * 0.3
    positives[:, :, 4:10, 4:10] += 0.6
    images = torch.cat([negatives, positives]).clamp(0.0, 1.0)
    labels = torch.cat([torch.zeros(number_per_class), torch.ones(number_per_class)]).long()
    return images, labels


class TestBudgetInsideEvaluation:
    @staticmethod
    def _evaluator(budget: WallClockBudget):
        images, labels = _labelled_images()
        preprocessor = ImagePreprocessor(ImagePreprocessingConfig(target_side=_SIDE))
        preprocessor.fit_standardization(images)
        trainer = SupervisedTrainer(
            recipe=TrainingRecipe(number_of_epochs=20, batch_size=8, learning_rate=0.01),
            preprocessor=preprocessor,
            device_for_computation=_CPU,
            class_weights=compute_balanced_class_weights(labels, 2),
            should_stop=budget.should_stop,
        )
        return TrainedBinaryAurocEvaluator(
            train_images=images,
            train_labels=labels,
            trainer=trainer,
            root_seed=2,
            validation=ValidationSplit(
                images=images,
                labels=labels,
                example_ids=tuple(f"v{i}" for i in range(len(labels))),
                group_ids=tuple(f"g{i}" for i in range(len(labels))),
                split_name="search_validation",
            ),
            preprocessor=preprocessor,
            device_for_computation=_CPU,
            inference_batch_size=16,
            should_stop=budget.should_stop,
        )

    @staticmethod
    def _model() -> FixedConvolutionalNetwork:
        return FixedConvolutionalNetwork(
            FixedConvolutionalNetworkConfig(convolution_channels=(4,), dropout_probability=0.0)
        )

    def test_a_candidate_cut_off_mid_training_does_not_compete(self) -> None:
        clock = FakeClock()
        budget = WallClockBudget(5.0, clock=clock)
        budget.start()
        images, labels = _labelled_images()
        preprocessor = ImagePreprocessor(ImagePreprocessingConfig(target_side=_SIDE))
        preprocessor.fit_standardization(images)

        batches_seen = {"count": 0}

        def ticking_should_stop() -> bool:
            batches_seen["count"] += 1
            if batches_seen["count"] > 3:
                clock.advance(10.0)
            return budget.should_stop()

        trainer = SupervisedTrainer(
            recipe=TrainingRecipe(number_of_epochs=20, batch_size=8),
            preprocessor=preprocessor,
            device_for_computation=_CPU,
            should_stop=ticking_should_stop,
        )
        result = trainer.train(
            self._model(),
            images=images,
            labels=labels,
            batch_order_generator=torch.Generator().manual_seed(1),
            augmentation_generator=torch.Generator().manual_seed(2),
        )
        assert result.status == DEADLINE_REACHED
        assert result.completed_epochs < 20

    def test_an_exhausted_budget_marks_remaining_candidates_failed(self) -> None:
        clock = FakeClock()
        budget = WallClockBudget(1.0, clock=clock)
        budget.start()
        evaluator = self._evaluator(budget)
        clock.advance(5.0)

        evaluator.evaluate_batch_of_phenotypes([self._model(), self._model()])
        assert all(
            record.status is EvaluationStatus.FAILED_DEADLINE
            for record in evaluator.evaluation_records
        )
        assert evaluator.best_fitness is None, (
            "a run where nothing completed must not present a best model"
        )

    def test_the_best_completed_evaluation_survives_a_later_deadline(self) -> None:
        clock = FakeClock()
        budget = WallClockBudget(60.0, clock=clock)
        budget.start()
        evaluator = self._evaluator(budget)

        evaluator.evaluate_batch_of_phenotypes([self._model()])
        assert evaluator.evaluation_records[0].status is EvaluationStatus.SUCCEEDED
        completed_fitness = evaluator.best_fitness

        clock.advance(120.0)
        evaluator.evaluate_batch_of_phenotypes([self._model()])
        assert evaluator.evaluation_records[1].status is EvaluationStatus.FAILED_DEADLINE
        assert evaluator.best_fitness == completed_fitness


def _checkpoint(
    *, generation: int = 3, algorithm_state: dict | None = None, binding: dict | None = None
) -> RunCheckpoint:
    return RunCheckpoint(
        schema_version=RUN_CHECKPOINT_SCHEMA_VERSION,
        run_id="run-1",
        stage="search",
        generation_number=generation,
        genome_kind="DeepNEATGenome",
        population_payload=[{"node_genes": [], "connection_genes": [], "index": index}
                            for index in range(4)],
        algorithm_state=algorithm_state,
        random_generator_state=np.random.default_rng(7).bit_generator.state,
        torch_stream_states={},
        best_genome_payload={"node_genes": [], "connection_genes": []},
        best_fitness=0.81,
        best_model_reference="gen2/cand5",
        evaluation_records=[
            {"evaluation_id": "gen0/cand0", "status": "succeeded", "fitness": 0.7}
        ],
        budget_state={"schema_version": "1.0", "total_seconds": 100.0, "consumed_seconds": 42.0},
        binding=binding or {"manifest_sha256": "a" * 64},
    )


class TestRunCheckpoints:
    def test_saving_only_the_best_genome_is_not_resumable(self) -> None:
        checkpoint = _checkpoint(algorithm_state=None)
        assert checkpoint.is_resumable is False
        with pytest.raises(CheckpointResumeError, match="no exported algorithm state"):
            verify_checkpoint_is_resumable(
                checkpoint, expected_binding={"manifest_sha256": "a" * 64}
            )

    def test_a_complete_checkpoint_resumes(self) -> None:
        checkpoint = _checkpoint(
            algorithm_state={"species": [[0, 1], [2, 3]], "innovation_high_water_mark": 42}
        )
        assert checkpoint.is_resumable
        verify_checkpoint_is_resumable(
            checkpoint, expected_binding={"manifest_sha256": "a" * 64}
        )

    def test_a_changed_manifest_or_lock_refuses_the_resume(self) -> None:
        checkpoint = _checkpoint(
            algorithm_state={"species": []},
            binding={"manifest_sha256": "a" * 64, "protocol_lock_sha256": "b" * 64},
        )
        with pytest.raises(CheckpointResumeError, match="manifest_sha256 changed"):
            verify_checkpoint_is_resumable(
                checkpoint,
                expected_binding={
                    "manifest_sha256": "c" * 64,
                    "protocol_lock_sha256": "b" * 64,
                },
            )

    def test_the_binding_covers_the_effective_configuration(self) -> None:
        first = build_run_binding(
            manifest_sha256="a" * 64,
            protocol_lock_sha256="b" * 64,
            effective_configuration={"population_size": 20},
        )
        second = build_run_binding(
            manifest_sha256="a" * 64,
            protocol_lock_sha256="b" * 64,
            effective_configuration={"population_size": 21},
        )
        assert first["effective_configuration_sha256"] != second["effective_configuration_sha256"]

    def test_store_roundtrips_and_finds_the_latest_generation(self, tmp_path: Path) -> None:
        store = RunCheckpointStore(tmp_path / "checkpoints")
        store.write(_checkpoint(generation=1, algorithm_state={"species": []}))
        store.write(_checkpoint(generation=12, algorithm_state={"species": []}))
        store.write(_checkpoint(generation=7, algorithm_state={"species": []}))

        latest = store.read_latest()
        assert latest is not None
        assert latest.generation_number == 12
        assert latest.best_fitness == 0.81
        assert latest.budget_state["consumed_seconds"] == 42.0

    def test_an_edited_checkpoint_is_refused(self, tmp_path: Path) -> None:
        store = RunCheckpointStore(tmp_path / "checkpoints")
        path = store.write(_checkpoint(algorithm_state={"species": []}))
        payload = path.read_text(encoding="utf-8").replace(
            '"best_fitness": 0.81', '"best_fitness": 0.99'
        )
        path.write_text(payload, encoding="utf-8")
        with pytest.raises(CheckpointResumeError, match="modified after it was written"):
            store.read_latest()

    def test_an_empty_store_has_no_latest_checkpoint(self, tmp_path: Path) -> None:
        assert RunCheckpointStore(tmp_path / "empty").read_latest() is None

    def test_the_random_stream_position_survives_the_roundtrip(self, tmp_path: Path) -> None:
        generator = np.random.default_rng(3)
        generator.random(5)
        checkpoint = _checkpoint(algorithm_state={"species": []})
        object.__setattr__(checkpoint, "random_generator_state", generator.bit_generator.state)
        expected_next = generator.random(3)

        store = RunCheckpointStore(tmp_path / "checkpoints")
        store.write(checkpoint)
        restored_state = store.read_latest().random_generator_state

        restored_generator = np.random.default_rng()
        restored_generator.bit_generator.state = restored_state
        assert np.allclose(restored_generator.random(3), expected_next)


class TestBenchmarkArtifactDirectories:
    def test_each_seed_gets_its_own_directory(self, tmp_path: Path) -> None:
        from benchmarks.run_benchmark import _artifacts_directory_for_seed

        first = _artifacts_directory_for_seed(tmp_path, "pediatric_pneumonia/deepneat", 101)
        second = _artifacts_directory_for_seed(tmp_path, "pediatric_pneumonia/deepneat", 102)
        assert first != second
        assert first.is_dir() and second.is_dir()
        assert first.name == "seed_101"

    def test_omitting_the_root_keeps_the_historical_no_artifacts_run(self, tmp_path: Path) -> None:
        from benchmarks.run_benchmark import _artifacts_directory_for_seed

        assert _artifacts_directory_for_seed(None, "xor/neat", 0) is None


class TestExperimentReportExtensions:
    def test_the_original_three_field_contract_still_constructs(self) -> None:
        from examples._experiment import ExperimentReport

        report = ExperimentReport(
            metric_values={"auroc": 0.9}, number_of_generations=3, runtime_seconds=1.5
        )
        assert report.succeeded
        assert report.effective_configuration == {}
        assert report.artifact_paths == {}

    def test_a_failed_run_says_so_instead_of_reporting_metrics(self) -> None:
        from examples._experiment import ExperimentReport

        report = ExperimentReport(
            metric_values={},
            number_of_generations=2,
            runtime_seconds=9.0,
            status="failed",
            failure_reason="no candidate could be evaluated within the budget",
        )
        assert not report.succeeded
        assert report.metric_values == {}
        assert "budget" in report.failure_reason

    def test_undefined_metrics_carry_their_reason(self) -> None:
        from examples._experiment import ExperimentReport

        report = ExperimentReport(
            metric_values={"auroc": 0.8},
            number_of_generations=1,
            runtime_seconds=1.0,
            undefined_metrics={"precision": "the model predicted no positives"},
        )
        assert "precision" not in report.metric_values
        assert report.undefined_metrics["precision"]
