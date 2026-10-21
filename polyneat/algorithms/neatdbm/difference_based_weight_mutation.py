"""Difference-based weight mutation of NEAT-DBM.

References:
    Stanovov, V., Akhmedova, Sh., & Semenkin, E. (2021). Neuroevolution of augmented topologies
        with difference-based mutation. *IOP Conference Series: Materials Science and
        Engineering*, 1047, 012075.
        DOI: 10.1088/1757-899X/1047/1/012075
"""

from __future__ import annotations

import dataclasses

from polyneat.core.neat.neat_genome import ConnectionGene, NEATGenome
from polyneat.core.type_aliases import InnovationId


class DifferenceBasedWeightMutation:
    """Differential-evolution-style weight recombination across four genomes.

    Implements section 3 of the paper: given a target genome and three
    distinct donor genomes ``r1``, ``r2``, ``r3``, the connection genes whose
    innovation numbers are present in all four genomes are located, and at
    those shared positions the target's weight is replaced by
    ``w_r1 + F * (w_r2 - w_r3)``. Positions where all four genomes carry an
    identical weight are skipped ("the genes having the same weight values
    are not considered"), and when no eligible position exists the target
    genome is returned unchanged ("if there is at least one innovation
    number for all 4 individuals, the following procedure is used").

    The operator changes weights only; topology, gene order, and
    ``is_enabled`` flags of the target genome are preserved. It is
    deterministic — donor selection carries all randomness and happens in
    :class:`~polyneat.algorithms.neatdbm.neatdbm_algorithm.NEATDBMAlgorithm`.
    """

    def __init__(self, scaling_factor: float) -> None:
        self._scaling_factor = scaling_factor

    def apply_to_genome_with_donors(
        self,
        genome: NEATGenome,
        donor_base_genome: NEATGenome,
        donor_difference_first_genome: NEATGenome,
        donor_difference_second_genome: NEATGenome,
    ) -> NEATGenome:
        """Recombine the shared connection weights of four genomes.

        Args:
            genome: Target genome whose weights are modified. Corresponds to
                the offspring ``OFS`` of the paper; its own weights do not
                enter the formula, they only help determine which genes are
                shared.
            donor_base_genome: Donor ``r1`` supplying the base weights.
            donor_difference_first_genome: Donor ``r2``, the minuend of the
                difference term.
            donor_difference_second_genome: Donor ``r3``, the subtrahend of
                the difference term.

        Returns:
            A new genome with recombined weights at the eligible shared
            positions, or ``genome`` itself when no position is eligible.
        """
        base_weight_by_innovation_id = {
            connection_gene.innovation_id: connection_gene.weight
            for connection_gene in donor_base_genome.connection_genes
        }
        first_weight_by_innovation_id = {
            connection_gene.innovation_id: connection_gene.weight
            for connection_gene in donor_difference_first_genome.connection_genes
        }
        second_weight_by_innovation_id = {
            connection_gene.innovation_id: connection_gene.weight
            for connection_gene in donor_difference_second_genome.connection_genes
        }

        eligible_innovation_ids: set[InnovationId] = set()
        for connection_gene in genome.connection_genes:
            innovation_id = connection_gene.innovation_id
            if (
                innovation_id not in base_weight_by_innovation_id
                or innovation_id not in first_weight_by_innovation_id
                or innovation_id not in second_weight_by_innovation_id
            ):
                continue
            all_four_weights_are_identical = (
                connection_gene.weight
                == base_weight_by_innovation_id[innovation_id]
                == first_weight_by_innovation_id[innovation_id]
                == second_weight_by_innovation_id[innovation_id]
            )
            if all_four_weights_are_identical:
                continue
            eligible_innovation_ids.add(innovation_id)

        if not eligible_innovation_ids:
            return genome

        recombined_connection_genes: list[ConnectionGene] = []
        for connection_gene in genome.connection_genes:
            if connection_gene.innovation_id not in eligible_innovation_ids:
                recombined_connection_genes.append(connection_gene)
                continue
            recombined_weight = base_weight_by_innovation_id[
                connection_gene.innovation_id
            ] + self._scaling_factor * (
                first_weight_by_innovation_id[connection_gene.innovation_id]
                - second_weight_by_innovation_id[connection_gene.innovation_id]
            )
            recombined_connection_genes.append(
                dataclasses.replace(connection_gene, weight=recombined_weight)
            )

        return NEATGenome(
            node_genes=genome.node_genes,
            connection_genes=tuple(recombined_connection_genes),
        )
