from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from polyneat.core.protocols import Genome
from polyneat.core.statistics import GenerationStatistics
from polyneat.core.type_aliases import FitnessValue


@dataclass
class RunContext:
    run_id: str
    run_started_at: datetime
    current_generation_number: int
    total_generations_planned: int
    history_of_generation_statistics: list[GenerationStatistics]
    current_best_genome: Genome | None
    current_best_fitness: FitnessValue | None
    configuration_snapshot: dict[str, Any]
