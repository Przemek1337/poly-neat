from __future__ import annotations

from numpy.random import Generator

from polyneat.core.component_protocols import InnovationTracker, MutationOperator
from polyneat.core.neat.neat_genome import NEATGenome


class CompositeNEATMutation:
    """Applies a fixed ordered list of NEAT mutation operators to a genome.

    Order matters. The pipeline wired up by ``NEATAlgorithm._build_mutation`` is:

        1. WeightModificationMutation
        2. AddConnectionMutation
        3. AddNodeMutation
        4. ToggleConnectionEnabledMutation

    Steps 1-3 are the three mutation types of the paper (section 3.1): weight
    mutation plus the two structural mutations, add-connection and add-node.
    Step 4 is an addition of this implementation, not an operator of the paper -
    see :class:`ToggleConnectionEnabledMutation` for why.
    The order is a choice of this implementation; the paper does not prescribe
    one. Structural mutations run after weight mutation so a freshly added
    connection keeps the weight its own operator drew for it.

    Each operator decides internally whether it actually fires.

    References:
        Stanley, K. O., & Miikkulainen, R. (2002). Evolving Neural Networks
            through Augmenting Topologies. *Evolutionary Computation*, 10(2), 99-127.
        (Mutation types: section 3.1, Figure 3.)
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
