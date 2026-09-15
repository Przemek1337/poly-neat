"""Fitness by AUROC on a validation split, for trained and pretrained phenotypes.

Two evaluators, because the two algorithms this benchmark compares learn in
genuinely different places. DeepNEAT carries no weights in its genome, so every
evaluation trains a fresh network and the training belongs in the evaluator.
EXACT is Lamarckian and trains between generations, so by the time its
phenotype reaches an evaluator it is already trained and must only be scored.
Forcing one class to do both would mean a flag deciding whether training
happens, which is exactly the kind of hidden branch the protocol asks to avoid.

What both share is everything that is not algorithm-specific: the same
preprocessing, the same batched inference, the same AUROC on the same split,
the same failure taxonomy, and the same rule that a failed evaluation carries a
reason instead of a score.

The existing :class:`~polyneat.evaluators.trained_network_accuracy_evaluator
.TrainedNetworkAccuracyEvaluator` keeps its accuracy fitness and its API
untouched; this is an addition beside it, not a replacement.
"""

from __future__ import annotations

import copy
import time
from collections.abc import Callable
from dataclasses import dataclass

import torch

from polyneat.core.component_protocols import Phenotype
from polyneat.core.type_aliases import FitnessValue
from polyneat.evaluators.binary_classification_metrics import MetricInputError, compute_auroc
from polyneat.evaluators.binary_inference import predict_binary
from polyneat.evaluators.binary_predictions import NonFinitePredictionError
from polyneat.logging_utils.custom_logger import get_logger
from polyneat.runner.evaluation_record import (
    EvaluationRecord,
    EvaluationStatus,
    fitness_values_for_selection,
)
from polyneat.training.image_preprocessing import ImagePreprocessor
from polyneat.training.random_streams import TrainingRandomStreams
from polyneat.training.supervised_trainer import COMPLETED, SupervisedTrainer
from polyneat.training.trainable_model import TrainableModel

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


def _is_out_of_memory_error(error: BaseException) -> bool:
    """Whether an exception is a device out-of-memory condition."""
    if isinstance(error, torch.cuda.OutOfMemoryError):
        return True
    return isinstance(error, RuntimeError) and "out of memory" in str(error).lower()


def _count_parameters(phenotype: Phenotype) -> int:
    """Trainable parameter count of a phenotype, or zero when it has none."""
    parameters = getattr(phenotype, "parameters", None)
    if not callable(parameters):
        return 0
    return sum(int(parameter.numel()) for parameter in parameters())


def _as_trainable_model(phenotype: Phenotype) -> TrainableModel:
    """Check that a phenotype really supports the operations about to be used.

    The runner hands out phenotypes, whose contract is only a forward pass.
    These evaluators additionally train them, snapshot them and run batched
    inference on them, which is the narrower TrainableModel contract. Checking
    it here turns a phenotype that cannot do those things into a clear error
    instead of an attribute error somewhere inside a training loop.

    Raises:
        TypeError: If the phenotype does not satisfy the trainable contract.
    """
    if not isinstance(phenotype, TrainableModel):
        raise TypeError(
            f"{type(phenotype).__name__} does not satisfy the trainable-model contract "
            "(forward_pass, parameters, train, eval, state_dict, load_state_dict); these "
            "evaluators train and snapshot the phenotypes they are given"
        )
    return phenotype


class _BinaryAurocEvaluatorBase:
    """Shared scoring, failure handling and best-model bookkeeping."""

    def __init__(
        self,
        *,
        validation: ValidationSplit,
        preprocessor: ImagePreprocessor,
        device_for_computation: torch.device,
        inference_batch_size: int,
        maximum_phenotype_parameters: int | None = None,
        should_stop: Callable[[], bool] | None = None,
        stage: str = "track_a",
    ) -> None:
        """Fix the split, the preprocessing and the limits every candidate faces.

        Args:
            validation: Split the fitness is measured on. For the search stage
                this is ``search_validation`` and nothing else.
            preprocessor: Already-fitted image path. Inference never fits it.
            device_for_computation: Device to run on.
            inference_batch_size: Rows per forward pass during scoring.
            maximum_phenotype_parameters: Candidates above this budget are
                recorded as invalid rather than trained. ``None`` disables the
                limit, which is only appropriate outside a budgeted series.
            should_stop: Checked before each candidate and inside training.
                Once it returns ``True`` the remaining candidates are recorded
                as deadline failures instead of being started.
            stage: Stage label carried into prediction sets and checkpoints.
        """
        self._validation = validation
        self._preprocessor = preprocessor
        self._device_for_computation = device_for_computation
        self._inference_batch_size = inference_batch_size
        self._maximum_phenotype_parameters = maximum_phenotype_parameters
        self._should_stop = should_stop
        self._stage = stage

        self._generation_counter = 0
        self._evaluation_records: list[EvaluationRecord] = []
        self._best_fitness: float | None = None
        self._best_evaluation_id: str | None = None
        self._best_model_state: dict | None = None
        self._best_parameter_count: int | None = None

    @property
    def evaluation_records(self) -> tuple[EvaluationRecord, ...]:
        """Every evaluation this evaluator has performed, in order."""
        return tuple(self._evaluation_records)

    def state_dict(self) -> dict:
        """Independent state of scoring; never includes dataset tensors."""
        return copy.deepcopy(
            {
                "generation_counter": self._generation_counter,
                "records": [record.to_serializable_dict() for record in self._evaluation_records],
                "best_fitness": self._best_fitness,
                "best_id": self._best_evaluation_id,
                "best_model_state": self._best_model_state,
                "best_parameter_count": self._best_parameter_count,
            }
        )

    def load_state_dict(self, state: dict) -> None:
        self._generation_counter = int(state["generation_counter"])
        self._evaluation_records = [
            EvaluationRecord(**{**row, "status": EvaluationStatus(row["status"])})
            for row in state["records"]
        ]
        self._best_fitness = state["best_fitness"]
        self._best_evaluation_id = state["best_id"]
        self._best_model_state = copy.deepcopy(state["best_model_state"])
        self._best_parameter_count = state["best_parameter_count"]

    def evaluate_candidate(self, phenotype: Phenotype, evaluation_id: str) -> EvaluationRecord:
        """Evaluate one stable ID for a checkpoint-aware, sequential runner."""
        previous_best = (
            self._best_fitness,
            self._best_evaluation_id,
            self._best_model_state,
            self._best_parameter_count,
        )
        record = self._evaluate_one(phenotype, evaluation_id)
        if record.is_selectable and self._should_stop is not None and self._should_stop():
            (
                self._best_fitness,
                self._best_evaluation_id,
                self._best_model_state,
                self._best_parameter_count,
            ) = previous_best
            record = EvaluationRecord(
                evaluation_id=evaluation_id,
                status=EvaluationStatus.FAILED_DEADLINE,
                failure_reason="evaluation did not complete before the deadline",
                wall_clock_seconds=record.wall_clock_seconds,
                optimizer_steps=record.optimizer_steps,
                examples_processed=record.examples_processed,
                parameter_count=record.parameter_count,
            )
        self._evaluation_records.append(record)
        return record

    @property
    def best_fitness(self) -> float | None:
        """Highest fitness seen, or ``None`` when nothing has succeeded."""
        return self._best_fitness

    @property
    def best_evaluation_id(self) -> str | None:
        """Evaluation id of the best candidate seen so far."""
        return self._best_evaluation_id

    @property
    def best_model_state(self) -> dict | None:
        """Snapshot of the best candidate's parameters, taken when it was scored.

        This is the point of keeping it: re-decoding the winning genome would
        build a different network with fresh weights, and for DeepNEAT that
        network never earned the selected fitness at all.
        """
        return self._best_model_state

    def evaluate_batch_of_phenotypes(self, phenotypes: list[Phenotype]) -> list[FitnessValue]:
        """Evaluate one generation and return fitnesses in population order.

        Failed candidates come back as an unselectable sentinel; the reason,
        the cost and the status live in :attr:`evaluation_records`, which is
        what a report and a budget summary are built from.
        """
        generation_records: list[EvaluationRecord] = []
        for position_in_batch, phenotype in enumerate(phenotypes):
            evaluation_id = f"gen{self._generation_counter}/cand{position_in_batch}"
            generation_records.append(self._evaluate_one(phenotype, evaluation_id))
        self._evaluation_records.extend(generation_records)
        self._generation_counter += 1
        return fitness_values_for_selection(generation_records)

    def _evaluate_one(self, phenotype: Phenotype, evaluation_id: str) -> EvaluationRecord:
        """Evaluate one candidate; implemented by the concrete evaluators."""
        raise NotImplementedError

    def _rejection_record(
        self, phenotype: Phenotype, evaluation_id: str
    ) -> EvaluationRecord | None:
        """Return a failure record when a candidate must not be trained at all."""
        if self._should_stop is not None and self._should_stop():
            return EvaluationRecord(
                evaluation_id=evaluation_id,
                status=EvaluationStatus.FAILED_DEADLINE,
                failure_reason="the budget was exhausted before this candidate started",
                parameter_count=_count_parameters(phenotype),
            )
        if getattr(phenotype, "is_degenerate", False):
            return EvaluationRecord(
                evaluation_id=evaluation_id,
                status=EvaluationStatus.FAILED_INVALID_PHENOTYPE,
                failure_reason="the genome expresses no usable input-to-output path",
            )
        parameter_count = _count_parameters(phenotype)
        if (
            self._maximum_phenotype_parameters is not None
            and parameter_count > self._maximum_phenotype_parameters
        ):
            return EvaluationRecord(
                evaluation_id=evaluation_id,
                status=EvaluationStatus.FAILED_INVALID_PHENOTYPE,
                failure_reason=(
                    f"{parameter_count} parameters exceeds the frozen budget of "
                    f"{self._maximum_phenotype_parameters}"
                ),
                parameter_count=parameter_count,
            )
        return None

    def score_phenotype(self, phenotype: Phenotype, evaluation_id: str) -> float:
        """AUROC of one phenotype on the validation split, without training it.

        Public because a caller that restored a winner's weights has to be able
        to check that the restored model reproduces the fitness the search
        selected on, rather than trusting that it does.

        Raises:
            NonFinitePredictionError: If the model emitted NaN or Inf.
            MetricInputError: If the split cannot support a ranking metric.
        """
        return self._score_phenotype(phenotype, evaluation_id)

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

    def _remember_if_best(self, phenotype: Phenotype, evaluation_id: str, fitness: float) -> None:
        """Snapshot the candidate when it is the best scored so far.

        Ties keep the earlier candidate, matching the protocol's tie-break on a
        stable evaluation order.
        """
        if self._should_stop is not None and self._should_stop():
            return
        parameter_count = _count_parameters(phenotype)
        if self._best_fitness is not None:
            if fitness < self._best_fitness or (
                fitness == self._best_fitness
                and self._best_parameter_count is not None
                and parameter_count >= self._best_parameter_count
            ):
                return
        state_dict_method = getattr(phenotype, "state_dict", None)
        self._best_fitness = fitness
        self._best_parameter_count = parameter_count
        self._best_evaluation_id = evaluation_id
        self._best_model_state = (
            None
            if not callable(state_dict_method)
            else {
                parameter_name: value.detach().cpu().clone()
                if isinstance(value, torch.Tensor)
                else value
                for parameter_name, value in state_dict_method().items()
            }
        )


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
            root_seed: Root of the per-candidate random streams. Every
                candidate derives its own initialization, batch order and
                augmentation from it and its evaluation id, so the streams do
                not depend on population position or on model size.
            reinitializer: How to give a candidate fresh parameters. Defaults
                to calling the phenotype's own ``reinitialize_parameters``,
                which is what DeepNEAT phenotypes provide.
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

    A phenotype that owns an initialization scheme keeps it - DeepNEAT evolves
    a weight-scaling gene, and track A must respect that gene. The generator is
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
