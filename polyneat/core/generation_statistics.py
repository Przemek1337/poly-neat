from __future__ import annotations

from dataclasses import dataclass, field

from polyneat.core.type_aliases import FitnessValue


@dataclass(frozen=True)
class GenerationStatistics:
    generation_number: int
    best_fitness: FitnessValue
    mean_fitness: FitnessValue
    median_fitness: FitnessValue
    number_of_species: int | None
    number_of_genomes_evaluated: int
    elapsed_seconds: float
    extra_metrics: dict[str, float] = field(default_factory=dict)
