from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from polyneat.core.component_protocols import Genome
from polyneat.core.generation_statistics import GenerationStatistics
from polyneat.core.type_aliases import FitnessValue


@dataclass
class RunContext:
    """Mutable state of an in-progress run, passed to every callback.

    Attributes:
        run_id: Unique id of the run (timestamp + random suffix).
        run_started_at: Wall-clock start time.
        current_generation_number: Generation currently being processed.
        total_generations_planned: Planned generation count; ``-1`` if open-ended.
        history_of_generation_statistics: Statistics of completed generations.
        current_best_genome: Best genome seen so far, if any.
        current_best_fitness: Its raw fitness, if any.
        configuration_snapshot: Free-form config dump for reproducibility.
    """

    run_id: str
    run_started_at: datetime
    current_generation_number: int
    total_generations_planned: int
    history_of_generation_statistics: list[GenerationStatistics]
    current_best_genome: Genome | None
    current_best_fitness: FitnessValue | None
    configuration_snapshot: dict[str, Any]
