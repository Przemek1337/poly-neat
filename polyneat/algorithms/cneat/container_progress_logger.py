from __future__ import annotations

from typing import TYPE_CHECKING

from rich.console import Console
from rich.table import Table

from polyneat.algorithms.cneat.class_genome_container import ClassGenomeContainer
from polyneat.runner.evolution_callback_protocol import BaseEvolutionCallback

if TYPE_CHECKING:
    from polyneat.core.generation_statistics import GenerationStatistics
    from polyneat.core.population import Population
    from polyneat.runner.run_context import RunContext


class ContainerProgressLogger(BaseEvolutionCallback):
    """Rich-formatted per-generation console output prepared for runs with container."""

    def __init__(self, container: ClassGenomeContainer) -> None:
        self._rich_console = Console()
        self._container = container

    def on_generation_completed(
        self,
        context: RunContext,
        new_population: Population,
        statistics: GenerationStatistics,
    ) -> None:
        best_fitness_per_class = [
            self._container.best_fitness_for_class(class_label_index)
            for class_label_index in range(self._container.number_of_class_labels)
        ]
        rows = [
            f"c{class_label_index} best={best_fitness:.4f}"
            if best_fitness is not None
            else f"c{class_label_index} best=--"
            for class_label_index, best_fitness in enumerate(best_fitness_per_class)
        ]

        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_row(
            f"[cyan]Gen {statistics.generation_number:>4}[/cyan]",
            f"overall best=[green]{statistics.best_fitness:.4f}[/green]",
            *rows,
            f"mean={statistics.mean_fitness:.4f}",
            f"species={statistics.number_of_species}",
            f"elapsed={statistics.elapsed_seconds:.2f}s",
        )
        self._rich_console.print(table)
