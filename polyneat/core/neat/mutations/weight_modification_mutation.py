from __future__ import annotations

from numpy.random import Generator

from polyneat.core.component_protocols import InnovationTracker
from polyneat.core.neat.neat_genome import ConnectionGene, NEATGenome


class WeightModificationMutation:
    """Weight perturbation or full replacement, gated per genome then per connection.

    Two-level scheme of Stanley & Miikkulainen (2002), section 4.1: "There was
    an 80% chance of a genome having its connection weights mutated, in which
    case each weight had a 90% chance of being uniformly perturbed and a 10%
    chance of being assigned a new random value."

    So first a single draw decides whether *this genome* mutates its weights at
    all; only then is each connection considered:

      * with ``probability_of_perturbation`` the weight is nudged by a value
        drawn uniformly from ``[-magnitude, +magnitude]``;
      * else with ``probability_of_replacement`` the weight is drawn uniformly
        from ``[initial_weight_range_min, initial_weight_range_max]``;
      * otherwise the weight is left unchanged.

    The genome-level gate matters beyond the marginal rates: it keeps a whole
    genome's weights intact together, so roughly one genome in five carries its
    weight vector into the next generation untouched. Drawing independently per
    connection would make that outcome vanish for anything but a tiny genome.

    The per-connection probabilities are checked in the order above and are
    therefore *not* independent, which matches Stanley's convention.

    The perturbation is *uniform*, not Gaussian: section 4.1 says each weight is
    "uniformly perturbed", and the reference implementation draws
    ``randposneg() * randfloat() * power`` in ``Genome::mutate_link_weights``,
    i.e. uniformly from ``[-power, power]`` - despite its ``GAUSSIAN`` enum name.
    A bounded perturbation cannot produce the rare large jumps of a normal
    distribution, which is what the separate replacement branch is there for.

    References:
        Stanley, K. O., & Miikkulainen, R. (2002). Evolving Neural Networks
            through Augmenting Topologies. *Evolutionary Computation*, 10(2), 99-127.
        (Non-structural mutation: section 3.1; the rates above: section 4.1.)
    """

    def __init__(
        self,
        probability_of_genome_weight_mutation: float,
        probability_of_perturbation: float,
        probability_of_replacement: float,
        weight_perturbation_magnitude: float,
        initial_weight_range_min: float,
        initial_weight_range_max: float,
    ) -> None:
        self._probability_of_genome_weight_mutation = probability_of_genome_weight_mutation
        self._probability_of_perturbation = probability_of_perturbation
        self._probability_of_replacement = probability_of_replacement
        self._weight_perturbation_magnitude = weight_perturbation_magnitude
        self._initial_weight_range_min = initial_weight_range_min
        self._initial_weight_range_max = initial_weight_range_max

    def apply_to_genome(
        self,
        genome: NEATGenome,
        rng: Generator,
        innovation_tracker: InnovationTracker,
    ) -> NEATGenome:
        """Return the genome with its weights mutated, or unchanged.

        Args:
            genome: Genome to mutate; left untouched, a new one is returned.
            rng: Random generator threaded through the run.
            innovation_tracker: Unused - weight mutation adds no structure.

        Returns:
            A new genome with mutated weights, or the input genome itself when
            the genome-level draw does not fire.
        """
        if rng.random() >= self._probability_of_genome_weight_mutation:
            return genome

        new_connection_genes: list[ConnectionGene] = []
        for existing_connection_gene in genome.connection_genes:
            sampled_uniform_value = rng.random()
            if sampled_uniform_value < self._probability_of_perturbation:
                perturbation_delta = float(
                    rng.uniform(
                        -self._weight_perturbation_magnitude,
                        self._weight_perturbation_magnitude,
                    )
                )
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
