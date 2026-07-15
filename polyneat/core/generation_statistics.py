from __future__ import annotations

from dataclasses import dataclass, field

from polyneat.core.type_aliases import FitnessValue


@dataclass(frozen=True)
class GenerationStatistics:
    """Summary metrics of one evaluated generation.

    Attributes:
        generation_number: Generation the statistics describe.
        best_fitness: Highest raw fitness in the generation.
        mean_fitness: Mean raw fitness.
        median_fitness: Median raw fitness.
        number_of_species: Species count after stagnation removal; ``None``
            for algorithms without speciation.
        number_of_genomes_evaluated: Population size that was evaluated.
        elapsed_seconds: Wall-clock time of the reproduction step.
        extra_metrics: Algorithm-specific scalars (logged by TensorBoard
            under ``extra/<name>``).
    """

    generation_number: int
    best_fitness: FitnessValue
    mean_fitness: FitnessValue
    median_fitness: FitnessValue
    number_of_species: int | None
    number_of_genomes_evaluated: int
    elapsed_seconds: float
    extra_metrics: dict[str, float] = field(default_factory=dict)
