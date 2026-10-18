"""Metric-neutral bookkeeping shared by the fitness evaluators.

Two benchmarks score candidates on two different metrics - AUROC for the binary
pneumonia task, top-1 accuracy for the multi-class digit and object tasks - but
everything around the metric is the same: rejecting a candidate that is
degenerate, over budget or past the deadline; snapshotting the best model so the
checkpoint is the network that earned the fitness rather than one rebuilt from
its genome; recording a reason for every failure; and exporting an independent
state so a sequential search can stop and resume.

That shared machinery lives here. A concrete evaluator supplies only two things:
how to score one already-runnable model (``_score_phenotype``) and how to reach
that score for one candidate (``_evaluate_one``), the latter differing because a
weightless-genome method trains before scoring while a Lamarckian one only
scores. This module knows nothing about which metric is used.
"""

from __future__ import annotations

import copy
from collections.abc import Callable

import torch

from polyneat.core.component_protocols import Phenotype
from polyneat.core.type_aliases import FitnessValue
from polyneat.runner.evaluation_record import (
    EvaluationRecord,
    EvaluationStatus,
    fitness_values_for_selection,
)
from polyneat.training.trainable_model import TrainableModel


def count_parameters(phenotype: Phenotype) -> int:
    """Trainable parameter count of a phenotype, or zero when it has none."""
    parameters = getattr(phenotype, "parameters", None)
    if not callable(parameters):
        return 0
    return sum(int(parameter.numel()) for parameter in parameters())


def is_out_of_memory_error(error: BaseException) -> bool:
    """Whether an exception is a device out-of-memory condition."""
    if isinstance(error, torch.cuda.OutOfMemoryError):
        return True
    return isinstance(error, RuntimeError) and "out of memory" in str(error).lower()


def as_trainable_model(phenotype: Phenotype) -> TrainableModel:
    """Check that a phenotype supports training, snapshotting and inference.

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


class ScoredCandidateEvaluatorBase:
    """Shared scoring bookkeeping: budgets, failures and best-model snapshots."""

    def __init__(
        self,
        *,
        device_for_computation: torch.device,
        inference_batch_size: int,
        maximum_phenotype_parameters: int | None = None,
        should_stop: Callable[[], bool] | None = None,
        stage: str = "track_a",
    ) -> None:
        """Fix the limits and device every candidate faces.

        Args:
            device_for_computation: Device to run on.
            inference_batch_size: Rows per forward pass during scoring.
            maximum_phenotype_parameters: Candidates above this budget are
                recorded as invalid rather than trained. ``None`` disables the
                limit, which is only appropriate outside a budgeted series.
            should_stop: Checked before each candidate and inside training. Once
                it returns ``True`` the remaining candidates are recorded as
                deadline failures instead of being started.
            stage: Stage label carried into prediction sets and checkpoints.
        """
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
        """Restore the scoring state saved by :meth:`state_dict`."""
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
        build a different network with fresh weights, and for a weightless
        genome that network never earned the selected fitness at all.
        """
        return self._best_model_state

    def evaluate_batch_of_phenotypes(self, phenotypes: list[Phenotype]) -> list[FitnessValue]:
        """Evaluate one generation and return fitnesses in population order.

        Failed candidates come back as an unselectable sentinel; the reason, the
        cost and the status live in :attr:`evaluation_records`, which is what a
        report and a budget summary are built from.
        """
        generation_records: list[EvaluationRecord] = []
        for position_in_batch, phenotype in enumerate(phenotypes):
            evaluation_id = f"gen{self._generation_counter}/cand{position_in_batch}"
            generation_records.append(self._evaluate_one(phenotype, evaluation_id))
        self._evaluation_records.extend(generation_records)
        self._generation_counter += 1
        return fitness_values_for_selection(generation_records)

    def score_phenotype(self, phenotype: Phenotype, evaluation_id: str) -> float:
        """Fitness of one phenotype on the validation split, without training it.

        Public because a caller that restored a winner's weights has to be able
        to check that the restored model reproduces the fitness the search
        selected on, rather than trusting that it does.
        """
        return self._score_phenotype(phenotype, evaluation_id)

    def _evaluate_one(self, phenotype: Phenotype, evaluation_id: str) -> EvaluationRecord:
        """Evaluate one candidate; implemented by the concrete evaluators."""
        raise NotImplementedError

    def _score_phenotype(self, phenotype: Phenotype, evaluation_id: str) -> float:
        """Score one runnable phenotype; implemented by the metric evaluators."""
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
                parameter_count=count_parameters(phenotype),
            )
        if getattr(phenotype, "is_degenerate", False):
            return EvaluationRecord(
                evaluation_id=evaluation_id,
                status=EvaluationStatus.FAILED_INVALID_PHENOTYPE,
                failure_reason="the genome expresses no usable input-to-output path",
            )
        parameter_count = count_parameters(phenotype)
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

    def _remember_if_best(self, phenotype: Phenotype, evaluation_id: str, fitness: float) -> None:
        """Snapshot the candidate when it is the best scored so far.

        Ties keep the earlier candidate, matching the protocol's tie-break on a
        stable evaluation order.
        """
        if self._should_stop is not None and self._should_stop():
            return
        parameter_count = count_parameters(phenotype)
        if self._best_fitness is not None and (
            fitness < self._best_fitness
            or (
                fitness == self._best_fitness
                and self._best_parameter_count is not None
                and parameter_count >= self._best_parameter_count
            )
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
