from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import TYPE_CHECKING

from rich.console import Console
from rich.table import Table

from polyneat.core.protocols import Genome
from polyneat.core.type_aliases import FitnessValue
from polyneat.runner.callbacks import BaseEvolutionCallback
from polyneat.utils.serialization import save_as_json, save_as_pickle

if TYPE_CHECKING:
    from polyneat.core.population import Population
    from polyneat.core.statistics import GenerationStatistics
    from polyneat.runner.context import RunContext

logger = logging.getLogger(__name__)


class ConsoleStatisticsLogger(BaseEvolutionCallback):
    def __init__(self) -> None:
        self._console = Console()

    def on_generation_completed(
        self,
        context: "RunContext",
        new_population: "Population",
        statistics: "GenerationStatistics",
    ) -> None:
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_row(
            f"[cyan]Gen {statistics.generation_number:>4}[/cyan]",
            f"best=[green]{statistics.best_fitness:.4f}[/green]",
            f"mean={statistics.mean_fitness:.4f}",
            f"species={statistics.number_of_species}",
            f"elapsed={statistics.elapsed_seconds:.2f}s",
        )
        self._console.print(table)

    def on_new_best_genome_found(
        self,
        context: "RunContext",
        best_genome: Genome,
        best_fitness: FitnessValue,
    ) -> None:
        self._console.print(
            f"  [bold green]★ New best fitness: {best_fitness:.6f}[/bold green]"
        )


class TensorBoardLogger(BaseEvolutionCallback):
    def __init__(self, log_directory: Path, run_name: str | None = None) -> None:
        self._log_directory = log_directory
        self._run_name = run_name
        self._writer = None

    def on_run_started(self, context: "RunContext") -> None:
        from torch.utils.tensorboard import SummaryWriter  # type: ignore[import]

        log_dir = self._log_directory / (self._run_name or context.run_id)
        self._writer = SummaryWriter(log_dir=str(log_dir))

    def on_generation_completed(
        self,
        context: "RunContext",
        new_population: "Population",
        statistics: "GenerationStatistics",
    ) -> None:
        if self._writer is None:
            return
        step = statistics.generation_number
        self._writer.add_scalar("fitness/best", statistics.best_fitness, step)
        self._writer.add_scalar("fitness/mean", statistics.mean_fitness, step)
        self._writer.add_scalar("fitness/median", statistics.median_fitness, step)
        if statistics.number_of_species is not None:
            self._writer.add_scalar("population/num_species", statistics.number_of_species, step)
        for metric_name, metric_value in statistics.extra_metrics.items():
            self._writer.add_scalar(f"extra/{metric_name}", metric_value, step)

    def on_run_completed(self, context: "RunContext", final_result: object) -> None:
        if self._writer is not None:
            self._writer.close()


class BestGenomePersister(BaseEvolutionCallback):
    def __init__(
        self,
        output_directory: Path,
        save_every_new_best: bool = True,
        save_on_run_completed: bool = True,
    ) -> None:
        self._output_directory = output_directory
        self._save_every_new_best = save_every_new_best
        self._save_on_run_completed = save_on_run_completed

    def on_run_started(self, context: "RunContext") -> None:
        self._output_directory.mkdir(parents=True, exist_ok=True)

    def _persist(self, genome: Genome, label: str) -> None:
        json_path = self._output_directory / f"{label}.json"
        pkl_path = self._output_directory / f"{label}.pkl"
        save_as_json(genome.to_serializable_dict(), json_path)
        save_as_pickle(genome, pkl_path)
        logger.debug("Saved best genome to %s and %s", json_path, pkl_path)

    def on_new_best_genome_found(
        self,
        context: "RunContext",
        best_genome: Genome,
        best_fitness: FitnessValue,
    ) -> None:
        if self._save_every_new_best:
            self._persist(best_genome, "best_genome")

    def on_run_completed(self, context: "RunContext", final_result: object) -> None:
        if self._save_on_run_completed and context.current_best_genome is not None:
            self._persist(context.current_best_genome, "best_genome")


class NetworkTopologyVisualizer(BaseEvolutionCallback):
    """Renders network topology as PNG/SVG using networkx + matplotlib."""

    def __init__(
        self,
        output_directory: Path,
        render_every_n_generations: int | None = None,
        render_on_run_completed: bool = True,
    ) -> None:
        self._output_directory = output_directory
        self._render_every_n_generations = render_every_n_generations
        self._render_on_run_completed = render_on_run_completed

    def on_run_started(self, context: "RunContext") -> None:
        topology_dir = self._output_directory / "topology"
        topology_dir.mkdir(parents=True, exist_ok=True)

    def on_generation_completed(
        self,
        context: "RunContext",
        new_population: "Population",
        statistics: "GenerationStatistics",
    ) -> None:
        if (
            self._render_every_n_generations is not None
            and statistics.generation_number % self._render_every_n_generations == 0
            and context.current_best_genome is not None
        ):
            self._render(
                context.current_best_genome,
                self._output_directory / "topology"
                / f"gen_{statistics.generation_number:04d}_best.png",
            )

    def on_run_completed(self, context: "RunContext", final_result: object) -> None:
        if self._render_on_run_completed and context.current_best_genome is not None:
            self._render(
                context.current_best_genome,
                self._output_directory / "topology" / "final_best.svg",
            )

    def _render(self, genome: Genome, output_path: Path) -> None:
        # Import here to avoid hard dependency at module load time
        from polyneat.viz.network_topology import render_genome_topology

        render_genome_topology(genome=genome, output_path=output_path)
        logger.debug("Rendered topology to %s", output_path)
