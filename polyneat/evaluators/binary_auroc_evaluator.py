"""Fitness by AUROC on a validation split, for trained and pretrained phenotypes.

Two evaluators, because the two algorithms this benchmark compares learn in
genuinely different places. DeepNEAT carries no weights in its genome, so every
evaluation trains a fresh network and the training belongs in the evaluator.
EXACT is Lamarckian and trains between generations, so by the time its
phenotype reaches an evaluator it is already trained and must only be scored.
Forcing one class to do both would mean a flag deciding whether training
happens, which is exactly the kind of hidden branch the protocol asks to avoid.

The metric-neutral bookkeeping - budgets, failure taxonomy, best-model
snapshots, resumable state - lives in
:mod:`polyneat.evaluators.scored_candidate_evaluator` and is shared with the
multi-class accuracy evaluator. This module adds only the binary AUROC scoring
on a validation split, and the two ways of reaching it.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import torch

from polyneat.core.component_protocols import Phenotype
from polyneat.evaluators.binary_classification_metrics import MetricInputError, compute_auroc
from polyneat.evaluators.binary_inference import predict_binary
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

logger = get_logger(__name__)


@dataclass(frozen=True)
class ValidationSplit:
    """The split a candidate's fitness is measured on.

    Attributes:
        images: ``NCHW`` batch, unpreprocessed.
        labels: Long tensor of class indices.
        example_ids: Manifest ids, aligned with the rows.
        group_ids: Split groups, aligned with the rows.
        split_name: Name of the split, recorded in every prediction set so a
            fitness computed on the wrong split is visible in the artifact.
    """

    images: torch.Tensor
    labels: torch.Tensor
    example_ids: tuple[str, ...]
    group_ids: tuple[str, ...]
    split_name: str


class _BinaryAurocEvaluatorBase(ScoredCandidateEvaluatorBase):
    """Binary AUROC scoring on top of the shared candidate bookkeeping."""

    def __init__(
        self,
        *,
        validation: ValidationSplit,
        preprocessor: ImagePreprocessor,
        **base_arguments,
    ) -> None:
        """Fix the split and the preprocessing every candidate is scored under.

        Args:
            validation: Split the fitness is measured on. For the search stage
                this is ``search_validation`` and nothing else.
            preprocessor: Already-fitted image path. Inference never fits it.
            **base_arguments: Device, batch size, budget and stage, passed to
                :class:`~polyneat.evaluators.scored_candidate_evaluator
                .ScoredCandidateEvaluatorBase`.
        """
        super().__init__(**base_arguments)
        self._validation = validation
        self._preprocessor = preprocessor

    def _score_phenotype(self, phenotype: Phenotype, evaluation_id: str) -> float:
        """AUROC of one phenotype on the validation split.

        Raises:
            NonFinitePredictionError: If the model emitted NaN or Inf.
            MetricInputError: If the split cannot support a ranking metric.
        """
        predictions = predict_binary(
            _as_trainable_model(phenotype),
            images=self._validation.images,
            labels=self._validation.labels,
            example_ids=self._validation.example_ids,
            group_ids=self._validation.group_ids,
            preprocessor=self._preprocessor,
            batch_size=self._inference_batch_size,
            device_for_computation=self._device_for_computation,
            model_id=evaluation_id,
            stage=self._stage,
            split_name=self._validation.split_name,
        )
        return compute_auroc(predictions.labels, predictions.positive_class_probabilities)


class PretrainedBinaryAurocEvaluator(_BinaryAurocEvaluatorBase):
    """Scores a phenotype that its algorithm already trained.

    This is EXACT's evaluator. EXACT writes trained kernels back into the
    genotype between generations, so by the time a phenotype reaches fitness
    evaluation it carries the weights it learned. Training it again here would
    add an unbudgeted training pass that the published algorithm does not
    perform.
    """

    def _evaluate_one(self, phenotype: Phenotype, evaluation_id: str) -> EvaluationRecord:
        rejection = self._rejection_record(phenotype, evaluation_id)
        if rejection is not None:
            return rejection

        evaluation_started_at = time.perf_counter()
        try:
            fitness = self._score_phenotype(phenotype, evaluation_id)
        except NonFinitePredictionError as non_finite_error:
            return EvaluationRecord(
                evaluation_id=evaluation_id,
                status=EvaluationStatus.FAILED_NON_FINITE,
                failure_reason=str(non_finite_error),
                wall_clock_seconds=time.perf_counter() - evaluation_started_at,
                parameter_count=_count_parameters(phenotype),
            )
        except MetricInputError as metric_error:
            return EvaluationRecord(
                evaluation_id=evaluation_id,
                status=EvaluationStatus.FAILED_NON_FINITE,
                failure_reason=str(metric_error),
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


class TrainedBinaryAurocEvaluator(_BinaryAurocEvaluatorBase):
    """Trains each phenotype from scratch, then scores it by validation AUROC.

    This is the evaluator for the algorithms whose genome carries no weights:
    DeepNEAT, the random search over DeepNEAT's space, and the fixed CNN
    baseline. Every candidate gets fresh parameters drawn from its own
    initialization stream, a weighted cross-entropy against the class balance
    of the training split, and the shared augmentation - so no candidate can
    gain an advantage from where it happened to sit in the population.

    Trained weights are never written back into a genome. A snapshot of the
    best candidate is kept, because the checkpoint that earned the selected
    fitness is the model track A reports on, and decoding the genome again
    would not reproduce it.
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
            trainer: The shared trainer, already carrying the recipe, the
                fitted preprocessing and the class weights.
            root_seed: Root of the per-candidate random streams. Every candidate
                derives its own initialization, batch order and augmentation
                from it and its evaluation id, so the streams do not depend on
                population position or on model size.
            reinitializer: How to give a candidate fresh parameters. Defaults to
                calling the phenotype's own ``reinitialize_parameters``, which is
                what DeepNEAT phenotypes provide.
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
        except (NonFinitePredictionError, MetricInputError) as non_finite_error:
            return EvaluationRecord(
                evaluation_id=evaluation_id,
                status=EvaluationStatus.FAILED_NON_FINITE,
                failure_reason=str(non_finite_error),
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

    A phenotype that owns an initialization scheme keeps it - DeepNEAT evolves a
    weight-scaling gene, and track A must respect that gene. The generator is
    still the source of the draws, so the stream stays independent of anything
    else the process did.

    Raises:
        TypeError: If the phenotype cannot reinitialize itself. Training a
            candidate that silently kept a previous candidate's weights would
            make every fitness after the first meaningless.
    """
    reinitialize = getattr(phenotype, "reinitialize_parameters", None)
    if not callable(reinitialize):
        raise TypeError(
            "TrainedBinaryAurocEvaluator needs phenotypes that implement "
            "reinitialize_parameters() so each candidate really starts from its own stream"
        )
    seed = int(torch.randint(0, 2**62, (1,), generator=generator).item())
    previous_state = torch.random.get_rng_state()
    try:
        torch.manual_seed(seed)
        reinitialize()
    finally:
        torch.random.set_rng_state(previous_state)
