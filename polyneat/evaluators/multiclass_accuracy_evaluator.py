"""Fitness by top-1 accuracy on a validation split, for the digit and object tasks.

The counterpart of :mod:`polyneat.evaluators.binary_auroc_evaluator` for the
MNIST and CIFAR benchmarks, where the label is one of many classes and AUROC
does not apply. Everything except the metric - budgets, the failure taxonomy,
the best-model snapshot, the resumable state, the split between a method that
trains before scoring and one that only scores - is inherited unchanged from
:class:`~polyneat.evaluators.scored_candidate_evaluator.ScoredCandidateEvaluatorBase`.
Only the scoring here is different: a batched forward pass, an argmax, and the
fraction that matches the label.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import torch

from polyneat.core.component_protocols import Phenotype
from polyneat.evaluators.binary_classification_metrics import MetricInputError
from polyneat.evaluators.binary_predictions import NonFinitePredictionError
from polyneat.evaluators.scored_candidate_evaluator import (
    ScoredCandidateEvaluatorBase,
)
from polyneat.evaluators.scored_candidate_evaluator import (
    as_trainable_model as _as_trainable_model,
)
from polyneat.evaluators.scored_candidate_evaluator import (
    count_parameters as _count_parameters,
)
from polyneat.evaluators.scored_candidate_evaluator import (
    is_out_of_memory_error as _is_out_of_memory_error,
)
from polyneat.logging_utils.custom_logger import get_logger
from polyneat.runner.evaluation_record import EvaluationRecord, EvaluationStatus
from polyneat.training.image_preprocessing import ImagePreprocessor
from polyneat.training.random_streams import TrainingRandomStreams
from polyneat.training.supervised_trainer import COMPLETED, SupervisedTrainer
from polyneat.training.trainable_model import TrainableModel, move_model_to_device

logger = get_logger(__name__)


@dataclass(frozen=True)
class AccuracyValidationSplit:
    """The split a candidate's accuracy is measured on.

    Unlike the binary split it carries no group ids: the multi-class benchmarks
    report a point accuracy, not a grouped bootstrap, so there is nothing to
    resample by.

    Attributes:
        images: ``NCHW`` batch, unpreprocessed.
        labels: Long tensor of class indices.
        split_name: Name of the split, recorded for provenance.
    """

    images: torch.Tensor
    labels: torch.Tensor
    split_name: str


def validation_top1_accuracy(
    model: TrainableModel,
    *,
    images: torch.Tensor,
    labels: torch.Tensor,
    preprocessor: ImagePreprocessor,
    batch_size: int,
    device_for_computation: torch.device,
) -> float:
    """Top-1 accuracy of one runnable model on a labelled split.

    Args:
        model: A model satisfying the trainable-model contract.
        images: ``NCHW`` batch, unpreprocessed.
        labels: Long tensor of class indices, aligned with ``images``.
        preprocessor: The model's fitted preprocessing, applied with
            ``training=False`` so no augmentation runs and nothing is fitted.
        batch_size: Rows per forward pass.
        device_for_computation: Device to run on.

    Returns:
        The fraction of rows whose argmax logit equals the label.

    Raises:
        MetricInputError: If the split is empty.
        NonFinitePredictionError: If the model emits a non-finite logit.
    """
    total = int(labels.shape[0])
    if total == 0:
        raise MetricInputError("cannot score accuracy on an empty validation split")
    model = move_model_to_device(model, device_for_computation)
    model.eval()
    correct = 0
    with torch.no_grad():
        for batch_start in range(0, total, batch_size):
            batch_images = images[batch_start : batch_start + batch_size].to(device_for_computation)
            preprocessed = preprocessor.apply(batch_images, training=False)
            logits = model.forward_pass(preprocessed)
            if not bool(torch.isfinite(logits).all()):
                raise NonFinitePredictionError(
                    "the model emitted a non-finite logit during accuracy scoring"
                )
            predicted = logits.argmax(dim=1).detach().cpu()
            batch_labels = labels[batch_start : batch_start + batch_size].detach().cpu()
            correct += int((predicted == batch_labels).sum())
    return correct / total


class _MulticlassAccuracyEvaluatorBase(ScoredCandidateEvaluatorBase):
    """Top-1 accuracy scoring on top of the shared candidate bookkeeping."""

    def __init__(
        self,
        *,
        validation: AccuracyValidationSplit,
        preprocessor: ImagePreprocessor,
        **base_arguments,
    ) -> None:
        """Fix the split and the preprocessing every candidate is scored under.

        Args:
            validation: Split the fitness is measured on. For the search stage
                this is the fixed validation split and nothing else.
            preprocessor: Already-fitted image path. Inference never fits it.
            **base_arguments: Device, batch size, budget and stage, passed to
                the shared base.
        """
        super().__init__(**base_arguments)
        self._validation = validation
        self._preprocessor = preprocessor

    def _score_phenotype(self, phenotype: Phenotype, evaluation_id: str) -> float:
        return validation_top1_accuracy(
            _as_trainable_model(phenotype),
            images=self._validation.images,
            labels=self._validation.labels,
            preprocessor=self._preprocessor,
            batch_size=self._inference_batch_size,
            device_for_computation=self._device_for_computation,
        )


class PretrainedMulticlassAccuracyEvaluator(_MulticlassAccuracyEvaluatorBase):
    """Scores a phenotype that its algorithm already trained.

    EXACT's evaluator on the multi-class tasks: it writes trained kernels back
    into the genotype between generations, so its phenotype arrives trained and
    is only scored here.
    """

    def _evaluate_one(self, phenotype: Phenotype, evaluation_id: str) -> EvaluationRecord:
        rejection = self._rejection_record(phenotype, evaluation_id)
        if rejection is not None:
            return rejection

        evaluation_started_at = time.perf_counter()
        try:
            fitness = self._score_phenotype(phenotype, evaluation_id)
        except (NonFinitePredictionError, MetricInputError) as scoring_error:
            return EvaluationRecord(
                evaluation_id=evaluation_id,
                status=EvaluationStatus.FAILED_NON_FINITE,
                failure_reason=str(scoring_error),
                wall_clock_seconds=time.perf_counter() - evaluation_started_at,
                parameter_count=_count_parameters(phenotype),
            )
        except Exception as runtime_error:
            if not _is_out_of_memory_error(runtime_error):
                raise
            return EvaluationRecord(
                evaluation_id=evaluation_id,
                status=EvaluationStatus.FAILED_OUT_OF_MEMORY,
                failure_reason=str(runtime_error),
                wall_clock_seconds=time.perf_counter() - evaluation_started_at,
                parameter_count=_count_parameters(phenotype),
            )

        self._remember_if_best(phenotype, evaluation_id, fitness)
        return EvaluationRecord(
            evaluation_id=evaluation_id,
            status=EvaluationStatus.SUCCEEDED,
            fitness=fitness,
            wall_clock_seconds=time.perf_counter() - evaluation_started_at,
            parameter_count=_count_parameters(phenotype),
            model_reference=evaluation_id,
        )


class TrainedMulticlassAccuracyEvaluator(_MulticlassAccuracyEvaluatorBase):
    """Trains each phenotype from scratch, then scores it by validation accuracy.

    The evaluator for the weightless-genome methods on the multi-class tasks:
    DeepNEAT and the random search over its space. Every candidate gets fresh
    parameters from its own initialization stream and the shared training
    recipe, so no candidate gains an advantage from its population position.
    """

    def __init__(
        self,
        *,
        train_images: torch.Tensor,
        train_labels: torch.Tensor,
        trainer: SupervisedTrainer,
        root_seed: int,
        reinitializer: Callable[[Phenotype, torch.Generator], None] | None = None,
        **base_arguments,
    ) -> None:
        """Add the training split and the trainer to the shared scoring setup.

        Args:
            train_images: Training images for candidate training.
            train_labels: Their class indices.
            trainer: The shared trainer, already carrying the recipe, the fitted
                preprocessing and the class weights.
            root_seed: Root of the per-candidate random streams.
            reinitializer: How to give a candidate fresh parameters. Defaults to
                the phenotype's own ``reinitialize_parameters``.
            **base_arguments: Passed through to the shared base.
        """
        super().__init__(**base_arguments)
        self._train_images = train_images
        self._train_labels = train_labels
        self._trainer = trainer
        self._root_seed = root_seed
        self._reinitializer = reinitializer or _reinitialize_with_phenotype_method

    def _evaluate_one(self, phenotype: Phenotype, evaluation_id: str) -> EvaluationRecord:
        rejection = self._rejection_record(phenotype, evaluation_id)
        if rejection is not None:
            return rejection

        parameter_count = _count_parameters(phenotype)
        evaluation_started_at = time.perf_counter()
        streams = TrainingRandomStreams.derive(
            root_seed=self._root_seed, evaluation_id=evaluation_id, track="A"
        )
        try:
            self._reinitializer(phenotype, streams.parameter_initialization)
            training_result = self._trainer.train(
                _as_trainable_model(phenotype),
                images=self._train_images,
                labels=self._train_labels,
                batch_order_generator=streams.batch_order,
                augmentation_generator=streams.augmentation,
            )
            if training_result.status != COMPLETED:
                return EvaluationRecord(
                    evaluation_id=evaluation_id,
                    status=EvaluationStatus.FAILED_DEADLINE,
                    failure_reason=(
                        "the budget ran out after "
                        f"{training_result.completed_epochs} epochs; an unfinished candidate "
                        "does not compete with fully evaluated ones"
                    ),
                    wall_clock_seconds=time.perf_counter() - evaluation_started_at,
                    parameter_count=parameter_count,
                    optimizer_steps=training_result.optimizer_steps,
                    examples_processed=training_result.examples_processed,
                )
            fitness = self._score_phenotype(phenotype, evaluation_id)
        except (NonFinitePredictionError, MetricInputError) as scoring_error:
            return EvaluationRecord(
                evaluation_id=evaluation_id,
                status=EvaluationStatus.FAILED_NON_FINITE,
                failure_reason=str(scoring_error),
                wall_clock_seconds=time.perf_counter() - evaluation_started_at,
                parameter_count=parameter_count,
            )
        except Exception as runtime_error:
            if not _is_out_of_memory_error(runtime_error):
                raise
            return EvaluationRecord(
                evaluation_id=evaluation_id,
                status=EvaluationStatus.FAILED_OUT_OF_MEMORY,
                failure_reason=str(runtime_error),
                wall_clock_seconds=time.perf_counter() - evaluation_started_at,
                parameter_count=parameter_count,
            )

        self._remember_if_best(phenotype, evaluation_id, fitness)
        return EvaluationRecord(
            evaluation_id=evaluation_id,
            status=EvaluationStatus.SUCCEEDED,
            fitness=fitness,
            wall_clock_seconds=time.perf_counter() - evaluation_started_at,
            parameter_count=parameter_count,
            optimizer_steps=training_result.optimizer_steps,
            examples_processed=training_result.examples_processed,
            model_reference=evaluation_id,
            details={"training": training_result.to_serializable_dict()},
        )


def _reinitialize_with_phenotype_method(phenotype: Phenotype, generator: torch.Generator) -> None:
    """Reinitialize through the phenotype's own method, seeding it explicitly.

    Raises:
        TypeError: If the phenotype cannot reinitialize itself, which would let
            a candidate silently keep a previous candidate's weights.
    """
    reinitialize = getattr(phenotype, "reinitialize_parameters", None)
    if not callable(reinitialize):
        raise TypeError(
            "TrainedMulticlassAccuracyEvaluator needs phenotypes that implement "
            "reinitialize_parameters() so each candidate really starts from its own stream"
        )
    seed = int(torch.randint(0, 2**62, (1,), generator=generator).item())
    previous_state = torch.random.get_rng_state()
    try:
        torch.manual_seed(seed)
        reinitialize()
    finally:
        torch.random.set_rng_state(previous_state)
