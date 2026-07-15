from __future__ import annotations

from numpy.random import Generator

from polyneat.core.component_protocols import InnovationTracker
from polyneat.core.neat.neat_genome import ConnectionGene, NEATGenome


class WeightModificationMutation:
    """Per-connection weight perturbation or full replacement.

    For each connection:
      * with ``probability_of_perturbation`` the weight is nudged by a Gaussian
        with std ``perturbation_strength_sigma``;
      * else with ``probability_of_replacement`` the weight is drawn uniformly
        from ``[initial_weight_range_min, initial_weight_range_max]``;
      * otherwise the weight is left unchanged.

    The two probabilities are checked in that order per connection (they are
    *not* independent), which matches Stanley's convention. Weight mutation
    is the non-structural mutation of the paper (section 3.1): "connection
    weights mutate as in any NE system, with each connection either perturbed
    or not".
    """

    def __init__(
        self,
        probability_of_perturbation: float,
        probability_of_replacement: float,
        perturbation_strength_sigma: float,
        initial_weight_range_min: float,
        initial_weight_range_max: float,
    ) -> None:
        self._probability_of_perturbation = probability_of_perturbation
        self._probability_of_replacement = probability_of_replacement
        self._perturbation_strength_sigma = perturbation_strength_sigma
        self._initial_weight_range_min = initial_weight_range_min
        self._initial_weight_range_max = initial_weight_range_max

    def apply_to_genome(
        self,
        genome: NEATGenome,
        rng: Generator,
        innovation_tracker: InnovationTracker,
    ) -> NEATGenome:
        new_connection_genes: list[ConnectionGene] = []
        for existing_connection_gene in genome.connection_genes:
            sampled_uniform_value = rng.random()
            if sampled_uniform_value < self._probability_of_perturbation:
                perturbation_delta = float(rng.normal(0.0, self._perturbation_strength_sigma))
                perturbed_weight_value = existing_connection_gene.weight + perturbation_delta
                new_connection_genes.append(
                    ConnectionGene(
                        innovation_id=existing_connection_gene.innovation_id,
                        source_node_id=existing_connection_gene.source_node_id,
                        target_node_id=existing_connection_gene.target_node_id,
                        weight=perturbed_weight_value,
                        is_enabled=existing_connection_gene.is_enabled,
                    )
                )
                continue

            if sampled_uniform_value < (
                self._probability_of_perturbation + self._probability_of_replacement
            ):
                replaced_weight_value = float(
                    rng.uniform(self._initial_weight_range_min, self._initial_weight_range_max)
                )
                new_connection_genes.append(
                    ConnectionGene(
                        innovation_id=existing_connection_gene.innovation_id,
                        source_node_id=existing_connection_gene.source_node_id,
                        target_node_id=existing_connection_gene.target_node_id,
                        weight=replaced_weight_value,
                        is_enabled=existing_connection_gene.is_enabled,
                    )
                )
                continue

            new_connection_genes.append(existing_connection_gene)

        return NEATGenome(
            node_genes=genome.node_genes,
            connection_genes=tuple(new_connection_genes),
        )
