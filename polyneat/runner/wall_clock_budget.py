"""The wall-clock budget a search runs under, and the clock it reads.

The protocol's primary budget is active wall-clock time on one exclusive
device. Two consequences shape this class.

First, the clock is injected. A budget that could only be tested by actually
waiting would be tested badly or not at all, so every test drives a fake clock
and the production path passes ``time.perf_counter``.

Second, the budget distinguishes *active* time from elapsed time. Infrastructure
downtime does not consume a search budget - a run paused because a node was
rebooted has not spent its allowance - so pausing and resuming are explicit
operations rather than something inferred from a gap in timestamps.

The budget exposes :meth:`should_stop` as a plain callable so the trainer and
the evaluator can check it at batch boundaries without importing anything from
the runner package.
"""

from __future__ import annotations

import time
from collections.abc import Callable

from polyneat.logging_utils.custom_logger import get_logger

logger = get_logger(__name__)

BUDGET_STATE_SCHEMA_VERSION = "1.0"


class WallClockBudget:
    """Tracks how much active wall-clock time a stage has left."""

    def __init__(
        self,
        total_seconds: float,
        *,
        clock: Callable[[], float] = time.perf_counter,
        already_consumed_seconds: float = 0.0,
    ) -> None:
        """Create a budget that has not started running yet.

        Args:
            total_seconds: Allowance for this stage.
            clock: Monotonic source of seconds. Injected so tests can drive it.
            already_consumed_seconds: Time spent by an earlier segment of the
                same run, for example before a resume. Work already done is
                charged to the budget rather than forgiven.

        Raises:
            ValueError: If the allowance is not positive or the consumed time
                is negative.
        """
        if total_seconds <= 0.0:
            raise ValueError(f"total_seconds must be > 0, got {total_seconds}")
        if already_consumed_seconds < 0.0:
            raise ValueError(
                f"already_consumed_seconds must be >= 0, got {already_consumed_seconds}"
            )
        self._total_seconds = float(total_seconds)
        self._clock = clock
        self._consumed_seconds = float(already_consumed_seconds)
        self._segment_started_at: float | None = None

    def start(self) -> None:
        """Begin, or resume, charging time to this budget."""
        if self._segment_started_at is None:
            self._segment_started_at = self._clock()

    def pause(self) -> None:
        """Stop charging time, banking whatever the current segment used.

        Called around infrastructure downtime, so a stall does not silently
        eat the search allowance.
        """
        if self._segment_started_at is not None:
            self._consumed_seconds += self._clock() - self._segment_started_at
            self._segment_started_at = None

    @property
    def total_seconds(self) -> float:
        """The full allowance of this stage."""
        return self._total_seconds

    @property
    def consumed_seconds(self) -> float:
        """Active seconds charged so far, including the running segment."""
        if self._segment_started_at is None:
            return self._consumed_seconds
        return self._consumed_seconds + (self._clock() - self._segment_started_at)

    @property
    def remaining_seconds(self) -> float:
        """Seconds left, never negative."""
        return max(0.0, self._total_seconds - self.consumed_seconds)

    @property
    def is_exhausted(self) -> bool:
        """Whether no allowance remains."""
        return self.remaining_seconds <= 0.0

    def should_stop(self) -> bool:
        """Callable form, for trainers and evaluators to poll at boundaries.

        Returning ``True`` means "do not start more work". Work already in
        flight finishes; the overhead of that one uninterruptible operation is
        reported rather than hidden.
        """
        return self.is_exhausted

    def state_dict(self) -> dict:
        """Capture consumed time so a resumed run does not restart its budget."""
        return {
            "schema_version": BUDGET_STATE_SCHEMA_VERSION,
            "total_seconds": self._total_seconds,
            "consumed_seconds": self.consumed_seconds,
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore consumed time captured by :meth:`state_dict`.

        Raises:
            ValueError: If the state was written by another schema version, or
                describes a different allowance - resuming a two-hour budget
                from a four-hour record would silently change the experiment.
        """
        if state.get("schema_version") != BUDGET_STATE_SCHEMA_VERSION:
            raise ValueError(
                f"budget state has schema version {state.get('schema_version')!r}, this build "
                f"reads {BUDGET_STATE_SCHEMA_VERSION!r}"
            )
        if float(state["total_seconds"]) != self._total_seconds:
            raise ValueError(
                f"budget state was written for a {state['total_seconds']}s allowance but this "
                f"budget is {self._total_seconds}s"
            )
        self._consumed_seconds = float(state["consumed_seconds"])
        self._segment_started_at = None
        logger.info(
            "Budget resumed with %.1fs of %.1fs already consumed",
            self._consumed_seconds,
            self._total_seconds,
        )


class WallClockBudgetTermination:
    """Ends the generational loop once the search budget is spent.

    Composed with the budget rather than owning it, because the same budget
    object is also polled inside training and candidate evaluation. The
    criterion is what stops the *loop*; :meth:`WallClockBudget.should_stop` is
    what stops work inside one generation.
    """

    def __init__(self, budget: WallClockBudget) -> None:
        """Wrap ``budget`` as a termination criterion."""
        self._budget = budget

    def should_terminate_evolution(self, context) -> bool:  # noqa: ANN001 - protocol shape
        """Whether the budget has run out."""
        return self._budget.is_exhausted

    @property
    def termination_reason_label(self) -> str:
        """Label recorded in the evolution result."""
        return "budget_exhausted"
