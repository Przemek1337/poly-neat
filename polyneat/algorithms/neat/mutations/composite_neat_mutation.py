from __future__ import annotations

from numpy.random import Generator

from polyneat.algorithms.neat.global_innovation_tracker import GlobalInnovationTracker
from polyneat.algorithms.neat.neat_genome import NEATGenome
from polyneat.core.component_protocols import MutationOperator


class CompositeNEATMutation:
    """Applies a fixed ordered list of NEAT mutation operators to a genome.

    Order matters — the canonical NEAT pipeline is:

        1. WeightModificationMutation
        2. AddConnectionMutation
        3. AddNodeMutation
        4. ToggleConnectionEnabledMutation

    Each operator decides internally whether it actually fires.
    """

    def __init__(self, ordered_individual_mutations: list[MutationOperator]) -> None:
        self._ordered_individual_mutations = ordered_individual_mutations

    def apply_to_genome(
        self,
        genome: NEATGenome,
        rng: Generator,
        innovation_tracker: GlobalInnovationTracker,
    ) -> NEATGenome:
        current_genome = genome
        for individual_mutation_operator in self._ordered_individual_mutations:
            current_genome = individual_mutation_operator.apply_to_genome(  # type: ignore[assignment]
                current_genome,
                rng,
                innovation_tracker,
            )
        return current_genome
