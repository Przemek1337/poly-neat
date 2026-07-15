from __future__ import annotations

from numpy.random import Generator

from polyneat.core.component_protocols import InnovationTracker, MutationOperator
from polyneat.core.neat.neat_genome import NEATGenome


class CompositeNEATMutation:
    """Applies a fixed ordered list of NEAT mutation operators to a genome.

    Order matters — the canonical NEAT pipeline is:

        1. WeightModificationMutation
        2. AddConnectionMutation
        3. AddNodeMutation
        4. ToggleConnectionEnabledMutation

    Each operator decides internally whether it actually fires.
    """

    def __init__(self, ordered_individual_mutations: list[MutationOperator[NEATGenome]]) -> None:
        self._ordered_individual_mutations = ordered_individual_mutations

    def apply_to_genome(
        self,
        genome: NEATGenome,
        rng: Generator,
        innovation_tracker: InnovationTracker,
    ) -> NEATGenome:
        """Apply each configured mutation in order and return the final genome.

        Args:
            genome: Genome to mutate; never modified in place.
            rng: Source of randomness shared by all component mutations.
            innovation_tracker: Tracker consulted by structural mutations.

        Returns:
            The genome after all mutations that chose to fire.
        """
        current_genome = genome
        for individual_mutation_operator in self._ordered_individual_mutations:
            current_genome = individual_mutation_operator.apply_to_genome(
                current_genome,
                rng,
                innovation_tracker,
            )
        return current_genome
