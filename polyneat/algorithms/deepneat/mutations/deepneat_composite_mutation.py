"""Apply DeepNEAT's mutation operators in a fixed order.

References:
    Miikkulainen, R., et al. (2017). Evolving Deep Neural Networks. *arXiv:1703.00548*.
        DOI: 10.1016/B978-0-12-815480-9.00015-3
"""

from __future__ import annotations

from numpy.random import Generator

from polyneat.algorithms.deepneat.deepneat_genome import DeepNEATGenome
from polyneat.core.component_protocols import InnovationTracker, MutationOperator


class DeepNEATCompositeMutation:
    """Runs a fixed ordered list of DeepNEAT mutation operators.

    The pipeline wired by ``DeepNEATAlgorithm._build_mutation`` is:

        1. LayerHyperparameterMutation
        2. AddTensorEdgeMutation
        3. AddLayerNodeMutation
        4. ToggleTensorEdgeMutation

    It mirrors the order of ``CompositeNEATMutation``: the parameter-level
    operator runs before the structural ones, so a freshly inserted layer keeps
    the hyperparameters its own sampler drew for it. Each operator decides
    internally whether it fires.
    """

    def __init__(
        self, ordered_individual_mutations: list[MutationOperator[DeepNEATGenome]]
    ) -> None:
        """Store the operators in application order.

        Args:
            ordered_individual_mutations: Operators to run, in order.
        """
        self._ordered_individual_mutations = ordered_individual_mutations

    def apply_to_genome(
        self,
        genome: DeepNEATGenome,
        rng: Generator,
        innovation_tracker: InnovationTracker,
    ) -> DeepNEATGenome:
        """Apply every operator in order and return the final genome.

        Args:
            genome: Genome to mutate; never modified in place.
            rng: Source of randomness shared by all operators.
            innovation_tracker: Tracker consulted by the structural operators.

        Returns:
            The genome after all operators that chose to fire.
        """
        current_genome = genome
        for individual_mutation_operator in self._ordered_individual_mutations:
            current_genome = individual_mutation_operator.apply_to_genome(
                current_genome, rng, innovation_tracker
            )
        return current_genome
