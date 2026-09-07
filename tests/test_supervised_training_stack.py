"""Tests for the shared trainer, initialization, checkpoints and AUROC evaluators."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from polyneat.evaluators.binary_auroc_evaluator import (
    PretrainedBinaryAurocEvaluator,
    TrainedBinaryAurocEvaluator,
    ValidationSplit,
)
from polyneat.evaluators.binary_inference import predict_binary
from polyneat.nn.fixed_convolutional_network import (
    FixedConvolutionalNetwork,
    FixedConvolutionalNetworkConfig,
)
from polyneat.runner.evaluation_record import (
    UNSELECTABLE_FITNESS,
    EvaluationRecord,
    EvaluationStatus,
    count_by_status,
    fitness_values_for_selection,
)
from polyneat.training.class_weights import compute_balanced_class_weights
from polyneat.training.image_preprocessing import (
    ImageAugmentationConfig,
    ImagePreprocessingConfig,
    ImagePreprocessor,
)
from polyneat.training.model_checkpoint import (
    CheckpointError,
    ModelCheckpoint,
    capture_model_checkpoint,
)
from polyneat.training.parameter_initialization import initialize_module_parameters
from polyneat.training.random_streams import TrainingRandomStreams
from polyneat.training.supervised_trainer import (
    COMPLETED,
    DEADLINE_REACHED,
    SupervisedTrainer,
    TrainingDataError,
)
from polyneat.training.training_recipe import (
    LearningRateSchedule,
    OptimizerName,
    TrainingRecipe,
)

_CPU = torch.device("cpu")
_SIDE = 16


def _labelled_images(
    number_per_class: int = 24, *, seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Images whose positive class carries a bright patch, so learning is possible."""
    generator = torch.Generator().manual_seed(seed)
    negatives = torch.rand(number_per_class, 1, _SIDE, _SIDE, generator=generator) * 0.3
    positives = torch.rand(number_per_class, 1, _SIDE, _SIDE, generator=generator) * 0.3
    positives[:, :, 4:10, 4:10] += 0.6
    images = torch.cat([negatives, positives]).clamp(0.0, 1.0)
    labels = torch.cat(
        [torch.zeros(number_per_class), torch.ones(number_per_class)]
    ).to(torch.long)
    return images, labels


def _fitted_preprocessor(images: torch.Tensor, *, augment: bool = True) -> ImagePreprocessor:
    preprocessor = ImagePreprocessor(
        ImagePreprocessingConfig(target_side=_SIDE),
        ImageAugmentationConfig() if augment else None,
    )
    preprocessor.fit_standardization(images)
    return preprocessor


def _small_model() -> FixedConvolutionalNetwork:
    return FixedConvolutionalNetwork(
        FixedConvolutionalNetworkConfig(convolution_channels=(4, 8), dropout_probability=0.0)
    )


def _trainer(
    preprocessor: ImagePreprocessor,
    *,
    recipe: TrainingRecipe | None = None,
    class_weights: torch.Tensor | None = None,
    should_stop=None,
) -> SupervisedTrainer:
    return SupervisedTrainer(
        recipe=recipe or TrainingRecipe(number_of_epochs=3, batch_size=16, learning_rate=0.02),
        preprocessor=preprocessor,
        device_for_computation=_CPU,
        class_weights=class_weights,
        should_stop=should_stop,
    )


class TestTrainingRecipe:
    def test_constant_schedule_never_moves(self) -> None:
        recipe = TrainingRecipe(schedule=LearningRateSchedule.CONSTANT, learning_rate=0.05)
        assert recipe.learning_rate_for_epoch(0) == 0.05
        assert recipe.learning_rate_for_epoch(29) == 0.05

    def test_cosine_schedule_decays_to_the_floor(self) -> None:
        recipe = TrainingRecipe(
            schedule=LearningRateSchedule.COSINE,
            learning_rate=0.1,
            minimum_learning_rate=0.001,
            number_of_epochs=10,
        )
        assert recipe.learning_rate_for_epoch(0) == pytest.approx(0.1)
        assert recipe.learning_rate_for_epoch(9) == pytest.approx(0.001)

    def test_step_schedule_multiplies_at_each_boundary(self) -> None:
        recipe = TrainingRecipe(
            schedule=LearningRateSchedule.STEP,
            learning_rate=0.1,
            step_schedule_gamma=0.1,
            step_schedule_epoch_interval=5,
        )
        assert recipe.learning_rate_for_epoch(4) == pytest.approx(0.1)
        assert recipe.learning_rate_for_epoch(5) == pytest.approx(0.01)

    def test_optimizer_is_built_fresh_with_no_carried_state(self) -> None:
        model = _small_model()
        recipe = TrainingRecipe(optimizer=OptimizerName.SGD, momentum=0.9)
        optimizer = recipe.build_optimizer(list(model.parameters()))
        assert optimizer.state == {}
        assert optimizer.param_groups[0]["momentum"] == 0.9

    def test_invalid_recipe_values_are_refused(self) -> None:
        with pytest.raises(ValueError, match="number_of_epochs"):
            TrainingRecipe(number_of_epochs=0)
        with pytest.raises(ValueError, match="batch_size"):
            TrainingRecipe(batch_size=0)


class TestSupervisedTrainer:
    def test_a_completed_session_reports_its_work(self) -> None:
        images, labels = _labelled_images()
        trainer = _trainer(_fitted_preprocessor(images))
        streams = TrainingRandomStreams.derive(root_seed=1, evaluation_id="e0", track="A")
        result = trainer.train(
            _small_model(),
            images=images,
            labels=labels,
            batch_order_generator=streams.batch_order,
            augmentation_generator=streams.augmentation,
        )
        assert result.status == COMPLETED
        assert result.completed_epochs == 3
        assert result.optimizer_steps == 3 * 3  # 48 samples, batch 16
        assert result.examples_processed == 3 * 48
        assert result.final_batch_loss is not None
        assert not result.was_interrupted

    def test_training_actually_changes_the_parameters(self) -> None:
        images, labels = _labelled_images()
        model = _small_model()
        before = [parameter.detach().clone() for parameter in model.parameters()]
        streams = TrainingRandomStreams.derive(root_seed=1, evaluation_id="e0", track="B")
        _trainer(_fitted_preprocessor(images)).train(
            model,
            images=images,
            labels=labels,
            batch_order_generator=streams.batch_order,
            augmentation_generator=streams.augmentation,
        )
        after = list(model.parameters())
        assert any(
            not torch.allclose(old, new) for old, new in zip(before, after, strict=True)
        )

    def test_the_same_streams_reproduce_the_same_model(self) -> None:
        images, labels = _labelled_images()
        preprocessor = _fitted_preprocessor(images)

        def train_once() -> list[torch.Tensor]:
            torch.manual_seed(123)
            model = _small_model()
            streams = TrainingRandomStreams.derive(root_seed=9, evaluation_id="e1", track="A")
            _trainer(preprocessor).train(
                model,
                images=images,
                labels=labels,
                batch_order_generator=streams.batch_order,
                augmentation_generator=streams.augmentation,
            )
            return [parameter.detach().clone() for parameter in model.parameters()]

        first = train_once()
        second = train_once()
        assert all(torch.allclose(a, b) for a, b in zip(first, second, strict=True))

    def test_class_weights_reach_the_loss(self) -> None:
        images, labels = _labelled_images()
        preprocessor = _fitted_preprocessor(images, augment=False)

        def train_with(weights: torch.Tensor | None) -> torch.Tensor:
            torch.manual_seed(7)
            model = _small_model()
            streams = TrainingRandomStreams.derive(root_seed=4, evaluation_id="e2", track="A")
            _trainer(preprocessor, class_weights=weights).train(
                model,
                images=images,
                labels=labels,
                batch_order_generator=streams.batch_order,
                augmentation_generator=streams.augmentation,
            )
            return next(iter(model.parameters())).detach().clone()

        unweighted = train_with(None)
        weighted = train_with(torch.tensor([1.0, 20.0]))
        assert not torch.allclose(unweighted, weighted)

    def test_balanced_weights_come_from_the_split_being_trained_on(self) -> None:
        images, labels = _labelled_images()
        imbalanced_labels = labels.clone()
        imbalanced_labels[:12] = 1
        assert not torch.allclose(
            compute_balanced_class_weights(labels, 2),
            compute_balanced_class_weights(imbalanced_labels, 2),
        )

    def test_a_deadline_stops_the_session_and_says_so(self) -> None:
        images, labels = _labelled_images()
        remaining_batches = {"count": 4}

        def should_stop() -> bool:
            remaining_batches["count"] -= 1
            return remaining_batches["count"] <= 0

        result = _trainer(
            _fitted_preprocessor(images),
            recipe=TrainingRecipe(number_of_epochs=10, batch_size=16),
            should_stop=should_stop,
        ).train(
            _small_model(),
            images=images,
            labels=labels,
            batch_order_generator=torch.Generator().manual_seed(1),
            augmentation_generator=torch.Generator().manual_seed(2),
        )
        assert result.status == DEADLINE_REACHED
        assert result.was_interrupted
        assert result.completed_epochs < 10

    def test_a_split_with_no_usable_minibatch_is_refused(self) -> None:
        images, labels = _labelled_images(number_per_class=1)
        with pytest.raises(TrainingDataError, match="no minibatch larger than one"):
            _trainer(
                _fitted_preprocessor(images),
                recipe=TrainingRecipe(number_of_epochs=1, batch_size=1),
            ).train(
                _small_model(),
                images=images,
                labels=labels,
                batch_order_generator=torch.Generator().manual_seed(1),
                augmentation_generator=torch.Generator().manual_seed(2),
            )

    def test_misaligned_or_empty_data_is_refused(self) -> None:
        images, labels = _labelled_images()
        trainer = _trainer(_fitted_preprocessor(images))
        with pytest.raises(TrainingDataError, match="rows but labels"):
            trainer.train(
                _small_model(),
                images=images,
                labels=labels[:-1],
                batch_order_generator=torch.Generator().manual_seed(1),
                augmentation_generator=torch.Generator().manual_seed(2),
            )


class TestSharedInitialization:
    def test_every_layer_is_touched(self) -> None:
        model = _small_model()
        touched = initialize_module_parameters(model, torch.Generator().manual_seed(1))
        assert touched >= 4

    def test_the_same_generator_reproduces_the_same_parameters(self) -> None:
        first, second = _small_model(), _small_model()
        initialize_module_parameters(first, torch.Generator().manual_seed(5))
        initialize_module_parameters(second, torch.Generator().manual_seed(5))
        assert all(
            torch.allclose(a, b)
            for a, b in zip(first.parameters(), second.parameters(), strict=True)
        )

    def test_normalization_layers_are_reset_to_identity(self) -> None:
        normalization = nn.BatchNorm2d(3)
        normalization.weight.data.fill_(3.0)
        normalization.bias.data.fill_(2.0)
        normalization.running_mean.data.fill_(5.0)
        normalization.running_var.data.fill_(9.0)
        normalization.num_batches_tracked.data.fill_(17)

        initialize_module_parameters(normalization, torch.Generator().manual_seed(1))
        assert torch.allclose(normalization.weight, torch.ones(3))
        assert torch.allclose(normalization.bias, torch.zeros(3))
        assert torch.allclose(normalization.running_mean, torch.zeros(3))
        assert torch.allclose(normalization.running_var, torch.ones(3))
        assert int(normalization.num_batches_tracked) == 0

    def test_initialization_does_not_depend_on_global_torch_state(self) -> None:
        torch.manual_seed(0)
        first = _small_model()
        initialize_module_parameters(first, torch.Generator().manual_seed(11))
        first_weights = [p.detach().clone() for p in first.parameters()]

        torch.manual_seed(999)
        torch.rand(10_000)
        second = _small_model()
        initialize_module_parameters(second, torch.Generator().manual_seed(11))
        assert all(
            torch.allclose(a, b)
            for a, b in zip(first_weights, second.parameters(), strict=True)
        )


class TestModelCheckpoint:
    @staticmethod
    def _checkpoint(model, preprocessor) -> ModelCheckpoint:
        return capture_model_checkpoint(
            model,
            model_id="winner",
            stage="track_a",
            genome_kind="FixedConvolutionalNetworkConfig",
            genome_payload={"architecture": model.config.describe()},
            preprocessing_state=preprocessor.state_dict(),
            metadata={"selection_fitness": 0.87},
        )

    def test_a_snapshot_does_not_follow_later_training(self) -> None:
        images, labels = _labelled_images()
        preprocessor = _fitted_preprocessor(images)
        model = _small_model()
        checkpoint = self._checkpoint(model, preprocessor)
        digest_before = checkpoint.compute_sha256()

        streams = TrainingRandomStreams.derive(root_seed=1, evaluation_id="e", track="A")
        _trainer(preprocessor).train(
            model,
            images=images,
            labels=labels,
            batch_order_generator=streams.batch_order,
            augmentation_generator=streams.augmentation,
        )
        assert checkpoint.compute_sha256() == digest_before
        assert not torch.allclose(
            checkpoint.model_state["_classifier.weight"],
            model.state_dict()["_classifier.weight"],
        )

    def test_restoring_reproduces_the_predictions_of_the_frozen_model(self) -> None:
        images, labels = _labelled_images()
        preprocessor = _fitted_preprocessor(images, augment=False)
        model = _small_model()
        streams = TrainingRandomStreams.derive(root_seed=2, evaluation_id="e", track="A")
        _trainer(preprocessor).train(
            model,
            images=images,
            labels=labels,
            batch_order_generator=streams.batch_order,
            augmentation_generator=streams.augmentation,
        )
        checkpoint = self._checkpoint(model, preprocessor)
        original = predict_binary(
            model,
            images=images,
            labels=labels,
            example_ids=tuple(f"e{i}" for i in range(len(labels))),
            group_ids=tuple(f"g{i}" for i in range(len(labels))),
            preprocessor=preprocessor,
            batch_size=16,
            device_for_computation=_CPU,
            model_id="winner",
            stage="track_a",
            split_name="official_test",
        )

        restored_model = _small_model()
        checkpoint.restore_into(restored_model)
        restored_preprocessor = ImagePreprocessor(ImagePreprocessingConfig(target_side=_SIDE))
        restored_preprocessor.load_state_dict(checkpoint.preprocessing_state)
        restored = predict_binary(
            restored_model,
            images=images,
            labels=labels,
            example_ids=original.example_ids,
            group_ids=original.group_ids,
            preprocessor=restored_preprocessor,
            batch_size=16,
            device_for_computation=_CPU,
            model_id="winner",
            stage="track_a",
            split_name="official_test",
        )
        assert torch.allclose(
            original.positive_class_probabilities, restored.positive_class_probabilities
        )

    def test_file_roundtrip_and_tamper_detection(self, tmp_path: Path) -> None:
        images, _ = _labelled_images()
        preprocessor = _fitted_preprocessor(images)
        checkpoint = self._checkpoint(_small_model(), preprocessor)
        checkpoint_path = tmp_path / "winner.pt"
        written_digest = checkpoint.write_file(checkpoint_path)

        reloaded, read_digest = ModelCheckpoint.read_file(checkpoint_path)
        assert read_digest == written_digest
        assert reloaded.model_id == "winner"
        assert reloaded.metadata["selection_fitness"] == 0.87

        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        payload["model_state"]["_classifier.bias"] += 1.0
        torch.save(payload, checkpoint_path)
        with pytest.raises(CheckpointError, match="modified after it was written"):
            ModelCheckpoint.read_file(checkpoint_path)

    def test_digest_changes_when_a_single_weight_changes(self) -> None:
        images, _ = _labelled_images()
        preprocessor = _fitted_preprocessor(images)
        model = _small_model()
        first = self._checkpoint(model, preprocessor).compute_sha256()
        with torch.no_grad():
            model._classifier.bias += 1e-3
        assert self._checkpoint(model, preprocessor).compute_sha256() != first

    def test_reading_a_missing_checkpoint_raises(self, tmp_path: Path) -> None:
        with pytest.raises(CheckpointError, match="does not exist"):
            ModelCheckpoint.read_file(tmp_path / "absent.pt")


class _DegenerateModel(FixedConvolutionalNetwork):
    """A phenotype that reports itself unusable, the way a dead graph does."""

    is_degenerate = True


class _NonFiniteModel(FixedConvolutionalNetwork):
    """A phenotype that emits NaN, which the protocol calls a failed evaluation."""

    def forward_pass(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return super().forward_pass(input_tensor) * float("nan")


class TestEvaluationRecords:
    def test_a_success_needs_a_finite_fitness(self) -> None:
        with pytest.raises(ValueError, match="needs a finite fitness"):
            EvaluationRecord(evaluation_id="e", status=EvaluationStatus.SUCCEEDED)

    def test_a_failure_needs_a_reason_and_carries_no_score(self) -> None:
        with pytest.raises(ValueError, match="carries no failure reason"):
            EvaluationRecord(evaluation_id="e", status=EvaluationStatus.FAILED_DEADLINE)
        with pytest.raises(ValueError, match="must not present a score"):
            EvaluationRecord(
                evaluation_id="e",
                status=EvaluationStatus.FAILED_DEADLINE,
                failure_reason="out of time",
                fitness=0.9,
            )

    def test_failures_project_to_an_unselectable_sentinel(self) -> None:
        records = [
            EvaluationRecord(
                evaluation_id="ok", status=EvaluationStatus.SUCCEEDED, fitness=0.6
            ),
            EvaluationRecord(
                evaluation_id="oom",
                status=EvaluationStatus.FAILED_OUT_OF_MEMORY,
                failure_reason="CUDA out of memory",
            ),
        ]
        assert fitness_values_for_selection(records) == [0.6, UNSELECTABLE_FITNESS]
        assert max(fitness_values_for_selection(records)) == 0.6
        assert count_by_status(records) == {"succeeded": 1, "failed_out_of_memory": 1}


class TestBinaryAurocEvaluators:
    @staticmethod
    def _setup(should_stop=None, maximum_parameters: int | None = 1_000_000):
        images, labels = _labelled_images(seed=3)
        preprocessor = _fitted_preprocessor(images)
        validation = ValidationSplit(
            images=images,
            labels=labels,
            example_ids=tuple(f"v{i}" for i in range(len(labels))),
            group_ids=tuple(f"g{i}" for i in range(len(labels))),
            split_name="search_validation",
        )
        evaluator = TrainedBinaryAurocEvaluator(
            train_images=images,
            train_labels=labels,
            trainer=_trainer(
                preprocessor, class_weights=compute_balanced_class_weights(labels, 2)
            ),
            root_seed=5,
            validation=validation,
            preprocessor=preprocessor,
            device_for_computation=_CPU,
            inference_batch_size=16,
            maximum_phenotype_parameters=maximum_parameters,
            should_stop=should_stop,
        )
        return evaluator, images, labels, validation, preprocessor

    def test_training_evaluator_scores_and_records_each_candidate(self) -> None:
        evaluator, *_ = self._setup()
        fitnesses = evaluator.evaluate_batch_of_phenotypes([_small_model(), _small_model()])
        assert len(fitnesses) == 2
        assert all(0.0 <= fitness <= 1.0 for fitness in fitnesses)
        assert all(
            record.status is EvaluationStatus.SUCCEEDED
            for record in evaluator.evaluation_records
        )
        assert all(record.optimizer_steps > 0 for record in evaluator.evaluation_records)

    def test_the_best_candidate_is_snapshotted_as_it_was_scored(self) -> None:
        evaluator, *_ = self._setup()
        evaluator.evaluate_batch_of_phenotypes([_small_model(), _small_model()])
        assert evaluator.best_model_state is not None
        assert evaluator.best_evaluation_id is not None
        assert evaluator.best_fitness == max(
            record.fitness for record in evaluator.evaluation_records
        )

    def test_a_degenerate_phenotype_fails_without_being_trained(self) -> None:
        evaluator, *_ = self._setup()
        fitnesses = evaluator.evaluate_batch_of_phenotypes([_DegenerateModel()])
        record = evaluator.evaluation_records[0]
        assert record.status is EvaluationStatus.FAILED_INVALID_PHENOTYPE
        assert record.optimizer_steps == 0
        assert fitnesses == [UNSELECTABLE_FITNESS]

    def test_a_candidate_over_the_parameter_budget_is_rejected(self) -> None:
        evaluator, *_ = self._setup(maximum_parameters=10)
        evaluator.evaluate_batch_of_phenotypes([_small_model()])
        record = evaluator.evaluation_records[0]
        assert record.status is EvaluationStatus.FAILED_INVALID_PHENOTYPE
        assert "exceeds the frozen budget" in record.failure_reason

    def test_an_exhausted_budget_fails_the_candidate_instead_of_scoring_it(self) -> None:
        evaluator, *_ = self._setup(should_stop=lambda: True)
        evaluator.evaluate_batch_of_phenotypes([_small_model()])
        record = evaluator.evaluation_records[0]
        assert record.status is EvaluationStatus.FAILED_DEADLINE
        assert record.fitness is None

    def test_a_non_finite_model_is_a_failed_evaluation_not_a_zero(self) -> None:
        evaluator, *_ = self._setup()
        fitnesses = evaluator.evaluate_batch_of_phenotypes([_NonFiniteModel()])
        record = evaluator.evaluation_records[0]
        assert record.status is EvaluationStatus.FAILED_NON_FINITE
        assert record.fitness is None
        assert fitnesses[0] == UNSELECTABLE_FITNESS

    def test_pretrained_evaluator_does_not_train(self) -> None:
        _, images, labels, validation, preprocessor = self._setup()
        model = _small_model()
        before = [parameter.detach().clone() for parameter in model.parameters()]
        evaluator = PretrainedBinaryAurocEvaluator(
            validation=validation,
            preprocessor=preprocessor,
            device_for_computation=_CPU,
            inference_batch_size=16,
        )
        evaluator.evaluate_batch_of_phenotypes([model])
        assert evaluator.evaluation_records[0].status is EvaluationStatus.SUCCEEDED
        assert evaluator.evaluation_records[0].optimizer_steps == 0
        assert all(
            torch.allclose(old, new)
            for old, new in zip(before, model.parameters(), strict=True)
        )
