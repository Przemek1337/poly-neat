from __future__ import annotations

from typing import Protocol, runtime_checkable

from polyneat.runner.context import RunContext


@runtime_checkable
class TerminationCriterion(Protocol):
    def should_terminate_evolution(self, context: RunContext) -> bool: ...

    @property
    def termination_reason_label(self) -> str: ...


class MaxGenerationsTermination:
    def __init__(self, max_generations: int):
        self._max_generations = max_generations

    def should_terminate_evolution(self, context: RunContext) -> bool:
        return context.current_generation_number >= self._max_generations

    @property
    def termination_reason_label(self) -> str:
        return "max_generations_reached"


class TargetFitnessTermination:
    def __init__(self, target_fitness: float):
        self._target_fitness = target_fitness

    def should_terminate_evolution(self, context: RunContext) -> bool:
        return (
            context.current_best_fitness is not None
            and context.current_best_fitness >= self._target_fitness
        )

    @property
    def termination_reason_label(self) -> str:
        return "target_fitness_reached"


class FitnessStagnationTermination:
    def __init__(self, stagnation_generations: int):
        self._stagnation_generations = stagnation_generations

    def should_terminate_evolution(self, context: RunContext) -> bool:
        history = context.history_of_generation_statistics
        if len(history) < self._stagnation_generations:
            return False
        recent = history[-self._stagnation_generations :]
        return recent[0].best_fitness >= recent[-1].best_fitness

    @property
    def termination_reason_label(self) -> str:
        return "stagnation_limit_reached"


class CompositeTermination:
    """Terminates when any one criterion fires (OR logic)."""

    def __init__(self, criteria: list[TerminationCriterion]):
        self._criteria = criteria
        self._fired_criterion: TerminationCriterion | None = None

    def should_terminate_evolution(self, context: RunContext) -> bool:
        for criterion in self._criteria:
            if criterion.should_terminate_evolution(context):
                self._fired_criterion = criterion
                return True
        return False

    @property
    def termination_reason_label(self) -> str:
        if self._fired_criterion is not None:
            return self._fired_criterion.termination_reason_label
        return "manually_stopped"
