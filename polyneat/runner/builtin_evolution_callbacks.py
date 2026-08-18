from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from rich.console import Console
from rich.table import Table
from torch.utils.tensorboard.writer import SummaryWriter

from polyneat.core.component_protocols import Genome
from polyneat.core.type_aliases import FitnessValue
from polyneat.logging_utils.custom_logger import get_logger
from polyneat.runner.evolution_callback_protocol import BaseEvolutionCallback
from polyneat.utils.artifact_serialization import save_as_json, save_as_pickle
from polyneat.viz.network_topology_renderer import render_genome_topology

if TYPE_CHECKING:
    from polyneat.core.generation_statistics import GenerationStatistics
    from polyneat.core.population import Population
    from polyneat.runner.run_context import RunContext

logger = get_logger(__name__)


class ConsoleStatisticsLogger(BaseEvolutionCallback):
    """Rich-formatted per-generation console output."""

    def __init__(self) -> None:
        self._rich_console = Console()

    def on_generation_completed(
        self,
        context: RunContext,
        new_population: Population,
        statistics: GenerationStatistics,
    ) -> None:
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_row(
            f"[cyan]Gen {statistics.generation_number:>4}[/cyan]",
            f"best=[green]{statistics.best_fitness:.4f}[/green]",
            f"mean={statistics.mean_fitness:.4f}",
            f"species={statistics.number_of_species}",
            f"elapsed={statistics.elapsed_seconds:.2f}s",
        )
        self._rich_console.print(table)

    def on_new_best_genome_found(
        self,
        context: RunContext,
        best_genome: Genome,
        best_fitness: FitnessValue,
    ) -> None:
        self._rich_console.print(f"  [bold green]NEW BEST fitness: {best_fitness:.6f}[/bold green]")


class TensorBoardLogger(BaseEvolutionCallback):
    """Writes per-generation scalar metrics to a TensorBoard event file."""

    def __init__(
        self,
        log_directory: Path,
        run_name: str | None = None,
        run_label: str | None = None,
    ) -> None:
        """Configure where and under what name the run's events are written.

        Args:
            log_directory: Directory the run's event subdirectory is created in.
            run_name: Exact subdirectory name, overriding everything else. Rarely
                needed; prefer ``run_label``.
            run_label: Human-readable prefix (e.g. ``retina-leo``) prepended to
                the auto-generated run id, so a run reads as
                ``retina-leo_<run_id>`` instead of a bare timestamp yet stays
                unique across re-runs.
        """
        self._log_directory = log_directory
        self._run_name = run_name
        self._run_label = run_label
        self._writer: SummaryWriter | None = None

    def on_run_started(self, context: RunContext) -> None:
        if self._run_name is not None:
            run_directory_name = self._run_name
        elif self._run_label is not None:
            run_directory_name = f"{self._run_label}_{context.run_id}"
        else:
            run_directory_name = context.run_id
        resolved_log_directory = self._log_directory / run_directory_name
        self._writer = SummaryWriter(log_dir=str(resolved_log_directory))

    def on_generation_completed(
        self,
        context: RunContext,
        new_population: Population,
        statistics: GenerationStatistics,
    ) -> None:
        if self._writer is None:
            return
        current_step = statistics.generation_number
        self._writer.add_scalar("fitness/best", statistics.best_fitness, current_step)
        self._writer.add_scalar("fitness/mean", statistics.mean_fitness, current_step)
        self._writer.add_scalar("fitness/median", statistics.median_fitness, current_step)
        if statistics.number_of_species is not None:
            self._writer.add_scalar(
                "population/num_species", statistics.number_of_species, current_step
            )
        for metric_name, metric_value in statistics.extra_metrics.items():
            self._writer.add_scalar(f"extra/{metric_name}", metric_value, current_step)
        # Flush each generation: the default SummaryWriter buffers for up to
        # flush_secs (120s), which makes a live run look frozen in TensorBoard.
        self._writer.flush()

    def on_run_completed(self, context: RunContext, final_result: object) -> None:
        if self._writer is not None:
            self._writer.close()


class BestGenomePersister(BaseEvolutionCallback):
    """Saves the best genome as JSON (canonical) + pickle (convenience)."""

    def __init__(
        self,
        output_directory: Path,
        save_every_new_best: bool = True,
        save_on_run_completed: bool = True,
    ) -> None:
        self._output_directory = output_directory
        self._save_every_new_best = save_every_new_best
        self._save_on_run_completed = save_on_run_completed

    def on_run_started(self, context: RunContext) -> None:
        self._output_directory.mkdir(parents=True, exist_ok=True)

    def on_new_best_genome_found(
        self,
        context: RunContext,
        best_genome: Genome,
        best_fitness: FitnessValue,
    ) -> None:
        if self._save_every_new_best:
            self._persist_best_genome_to_disk(best_genome, label="best_genome")

    def on_run_completed(self, context: RunContext, final_result: object) -> None:
        if self._save_on_run_completed and context.current_best_genome is not None:
            self._persist_best_genome_to_disk(context.current_best_genome, label="best_genome")

    def _persist_best_genome_to_disk(self, genome: Genome, label: str) -> None:
        json_file_path = self._output_directory / f"{label}.json"
        pickle_file_path = self._output_directory / f"{label}.pkl"
        save_as_json(genome.to_serializable_dict(), json_file_path)
        save_as_pickle(genome, pickle_file_path)
        logger.debug("Saved best genome to %s and %s", json_file_path, pickle_file_path)


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

    def on_run_started(self, context: RunContext) -> None:
        topology_output_directory = self._output_directory / "topology"
        topology_output_directory.mkdir(parents=True, exist_ok=True)

    def on_generation_completed(
        self,
        context: RunContext,
        new_population: Population,
        statistics: GenerationStatistics,
    ) -> None:
        if (
            self._render_every_n_generations is not None
            and statistics.generation_number % self._render_every_n_generations == 0
            and context.current_best_genome is not None
        ):
            self._render_best_genome_topology(
                context.current_best_genome,
                self._output_directory
                / "topology"
                / f"gen_{statistics.generation_number:04d}_best.png",
            )

    def on_run_completed(self, context: RunContext, final_result: object) -> None:
        if self._render_on_run_completed and context.current_best_genome is not None:
            self._render_best_genome_topology(
                context.current_best_genome,
                self._output_directory / "topology" / "final_best.svg",
            )

    def _render_best_genome_topology(self, genome: Genome, output_path: Path) -> None:
        render_genome_topology(genome=genome, output_path=output_path)
        logger.debug("Rendered topology to %s", output_path)
