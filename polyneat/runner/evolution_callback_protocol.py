from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from polyneat.core.component_protocols import Genome
from polyneat.core.type_aliases import FitnessValue

if TYPE_CHECKING:
    from polyneat.core.generation_statistics import GenerationStatistics
    from polyneat.core.population import Population
    from polyneat.runner.run_context import RunContext


@runtime_checkable
class EvolutionCallback(Protocol):
    """Observer of the evolution loop.

    Hooks fire in this order per run: ``on_run_started`` once, then per
    generation ``on_generation_started`` → ``on_population_evaluated`` →
    (``on_new_best_genome_found`` when applicable) →
    ``on_generation_completed``, and finally ``on_run_completed`` once.
    """

    def on_run_started(self, context: RunContext) -> None: ...
    def on_generation_started(self, context: RunContext, population: Population) -> None: ...
    def on_population_evaluated(
        self,
        context: RunContext,
        population: Population,
        fitnesses: list[FitnessValue],
    ) -> None: ...
    def on_generation_completed(
        self,
        context: RunContext,
        new_population: Population,
        statistics: GenerationStatistics,
    ) -> None: ...
    def on_new_best_genome_found(
        self,
        context: RunContext,
        best_genome: Genome,
        best_fitness: FitnessValue,
    ) -> None: ...
    def on_run_completed(self, context: RunContext, final_result: object) -> None: ...


class BaseEvolutionCallback:
    """Convenience base with no-op defaults; subclasses override only the hooks they need."""

    def on_run_started(self, context: RunContext) -> None:
        pass

    def on_generation_started(self, context: RunContext, population: Population) -> None:
        pass

    def on_population_evaluated(
        self,
        context: RunContext,
        population: Population,
        fitnesses: list[FitnessValue],
    ) -> None:
        pass

    def on_generation_completed(
        self,
        context: RunContext,
        new_population: Population,
        statistics: GenerationStatistics,
    ) -> None:
        pass

    def on_new_best_genome_found(
        self,
        context: RunContext,
        best_genome: Genome,
        best_fitness: FitnessValue,
    ) -> None:
        pass

    def on_run_completed(self, context: RunContext, final_result: object) -> None:
        pass
