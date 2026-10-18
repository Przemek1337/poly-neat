"""What one candidate evaluation produced, including when it produced nothing.

A bare ``list[float]`` of fitnesses cannot express the three things the
protocol needs from a budgeted search: that an evaluation failed and why, what
it cost, and which model it left behind. Encoding failure as a very small
number is worse than useless here, because a numeric sentinel is still a
number: it can win a tournament, become a parent, or turn out to be the best
candidate of a generation where everything else also failed.

So a record carries an explicit status. A successful one must have a finite
fitness; a failed one must have a reason. Both are enforced in
``__post_init__`` rather than left to the caller, and
:func:`fitness_values_for_selection` is the only place that turns records back
into the ``list[FitnessValue]`` the existing generational loop expects.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum

from polyneat.core.type_aliases import FitnessValue


class EvaluationStatus(StrEnum):
    """How one candidate evaluation ended.

    Attributes:
        SUCCEEDED: Trained and scored under the full protocol. Eligible for
            selection.
        FAILED_OUT_OF_MEMORY: Ran out of device memory. Recorded with its cost,
            never retried at a smaller batch size or resolution, because that
            would evaluate a different configuration than the one declared.
        FAILED_NON_FINITE: Produced NaN or Inf loss or predictions.
        FAILED_INVALID_PHENOTYPE: Could not be expressed as a usable network,
            for example a degenerate graph or one over the parameter budget.
        FAILED_DEADLINE: The budget ran out before the evaluation finished. An
            unfinished candidate does not compete with fully evaluated ones.
    """

    SUCCEEDED = "succeeded"
    FAILED_OUT_OF_MEMORY = "failed_out_of_memory"
    FAILED_NON_FINITE = "failed_non_finite"
    FAILED_INVALID_PHENOTYPE = "failed_invalid_phenotype"
    FAILED_DEADLINE = "failed_deadline"


# Selection must never see a failed candidate as a competitor. Negative infinity
# is used only at the boundary where the record list is projected onto the
# existing ``list[FitnessValue]`` contract, and only for candidates the caller
# has already been told to exclude.
UNSELECTABLE_FITNESS: FitnessValue = -math.inf


@dataclass(frozen=True)
class EvaluationRecord:
    """One candidate evaluation, its outcome and its cost.

    Attributes:
        evaluation_id: Stable id, unique within a run, used to derive random
            streams and to name artifacts.
        status: How the evaluation ended.
        fitness: The selection fitness. Present exactly when the evaluation
            succeeded.
        failure_reason: Why it failed. Present exactly when it did.
        wall_clock_seconds: Cost of the evaluation, charged to the budget
            whether it succeeded or not.
        parameter_count: Trainable parameters of the expressed phenotype.
        optimizer_steps: Parameter updates actually applied.
        examples_processed: Training examples fed forward.
        model_reference: Identifier of the checkpoint this evaluation left
            behind, when one was kept.
        details: Free-form extras carried into the artifact directory.
    """

    evaluation_id: str
    status: EvaluationStatus
    fitness: FitnessValue | None = None
    failure_reason: str | None = None
    wall_clock_seconds: float = 0.0
    parameter_count: int = 0
    optimizer_steps: int = 0
    examples_processed: int = 0
    model_reference: str | None = None
    details: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status is EvaluationStatus.SUCCEEDED:
            if self.fitness is None or not math.isfinite(self.fitness):
                raise ValueError(
                    f"evaluation {self.evaluation_id!r} is marked succeeded but its fitness is "
                    f"{self.fitness!r}; a success needs a finite fitness"
                )
        else:
            if not self.failure_reason:
                raise ValueError(
                    f"evaluation {self.evaluation_id!r} is marked {self.status} but carries no "
                    "failure reason"
                )
            if self.fitness is not None:
                raise ValueError(
                    f"evaluation {self.evaluation_id!r} failed but still carries fitness "
                    f"{self.fitness!r}; a failed candidate must not present a score"
                )

    @property
    def is_selectable(self) -> bool:
        """Whether this candidate may be selected, become a parent, or win."""
        return self.status is EvaluationStatus.SUCCEEDED

    def to_serializable_dict(self) -> dict:
        """Return the record as JSON-compatible data."""
        return {
            "evaluation_id": self.evaluation_id,
            "status": str(self.status),
            "fitness": self.fitness,
            "failure_reason": self.failure_reason,
            "wall_clock_seconds": self.wall_clock_seconds,
            "parameter_count": self.parameter_count,
            "optimizer_steps": self.optimizer_steps,
            "examples_processed": self.examples_processed,
            "model_reference": self.model_reference,
            "details": self.details,
        }


def fitness_values_for_selection(records: list[EvaluationRecord]) -> list[FitnessValue]:
    """Project records onto the ``list[FitnessValue]`` the generational loop takes.

    Failed evaluations map to negative infinity. That is a deliberate boundary
    conversion and not a score: every caller of this function has the records
    themselves and is expected to exclude the failures from selection rather
    than rely on the sentinel losing every comparison.

    Args:
        records: Evaluation records in population order.

    Returns:
        One fitness per record, in the same order.
    """
    selection_values: list[FitnessValue] = []
    for record in records:
        # is_selectable already implies a finite fitness (__post_init__ enforces
        # it); the explicit None check keeps that guarantee visible to a reader
        # and to a type checker instead of relying on the invariant holding.
        if record.is_selectable and record.fitness is not None:
            selection_values.append(record.fitness)
        else:
            selection_values.append(UNSELECTABLE_FITNESS)
    return selection_values


def count_by_status(records: list[EvaluationRecord]) -> dict[str, int]:
    """Count records per status, so a report can state the failure rate."""
    counts: dict[str, int] = {}
    for record in records:
        counts[str(record.status)] = counts.get(str(record.status), 0) + 1
    return counts
