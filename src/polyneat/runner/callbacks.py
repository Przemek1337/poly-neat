from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from polyneat.core.protocols import Genome
from polyneat.core.type_aliases import FitnessValue

if TYPE_CHECKING:
    from polyneat.core.population import Population
    from polyneat.core.statistics import GenerationStatistics
    from polyneat.runner.context import RunContext


@runtime_checkable
class EvolutionCallback(Protocol):
    def on_run_started(self, context: "RunContext") -> None: ...
    def on_generation_started(self, context: "RunContext", population: "Population") -> None: ...
    def on_population_evaluated(
        self,
        context: "RunContext",
        population: "Population",
        fitnesses: list[FitnessValue],
    ) -> None: ...
    def on_generation_completed(
        self,
        context: "RunContext",
        new_population: "Population",
        statistics: "GenerationStatistics",
    ) -> None: ...
    def on_new_best_genome_found(
        self,
        context: "RunContext",
        best_genome: Genome,
        best_fitness: FitnessValue,
    ) -> None: ...
    def on_run_completed(self, context: "RunContext", final_result: object) -> None: ...


class BaseEvolutionCallback:
    """Convenience base with no-op defaults; subclass and override only what you need."""

    def on_run_started(self, context: "RunContext") -> None:
        pass

    def on_generation_started(self, context: "RunContext", population: "Population") -> None:
        pass

    def on_population_evaluated(
        self,
        context: "RunContext",
        population: "Population",
        fitnesses: list[FitnessValue],
    ) -> None:
        pass

    def on_generation_completed(
        self,
        context: "RunContext",
        new_population: "Population",
        statistics: "GenerationStatistics",
    ) -> None:
        pass

    def on_new_best_genome_found(
        self,
        context: "RunContext",
        best_genome: Genome,
        best_fitness: FitnessValue,
    ) -> None:
        pass

    def on_run_completed(self, context: "RunContext", final_result: object) -> None:
        pass
