"""NEAT crossover: gene alignment by innovation number.

Implements the mating scheme of Stanley & Miikkulainen (2002), section 3.2
and Figure 4: matching genes are inherited randomly, disjoint and excess
genes come from the fitter parent, and a gene disabled in either parent has a
preset chance of staying disabled in the child.
"""

from __future__ import annotations

from numpy.random import Generator

from polyneat.core.neat.neat_genome import ConnectionGene, NEATGenome, NodeGene
from polyneat.nn.topology_utilities import would_directed_edge_create_cycle


def _resolve_enabled_connection_cycles(
    connection_genes: list[ConnectionGene],
) -> list[ConnectionGene]:
    """Disable connections that would create directed cycles, greedy in innovation_id order.

    Processes connections in ascending innovation_id order. Each enabled connection
    is accepted only if it does not form a cycle with the already-accepted enabled
    edges. Disabled-in-input connections are passed through unchanged.
    """
    sorted_indices = sorted(
        range(len(connection_genes)), key=lambda i: connection_genes[i].innovation_id
    )
    accepted_enabled_edges: list[tuple[int, int]] = []
    resolved: list[ConnectionGene] = list(connection_genes)

    for index in sorted_indices:
        conn = connection_genes[index]
        if not conn.is_enabled:
            continue
        if would_directed_edge_create_cycle(
            candidate_source_node_id=conn.source_node_id,
            candidate_target_node_id=conn.target_node_id,
            existing_enabled_edges=accepted_enabled_edges,
        ):
            resolved[index] = ConnectionGene(
                innovation_id=conn.innovation_id,
                source_node_id=conn.source_node_id,
                target_node_id=conn.target_node_id,
                weight=conn.weight,
                is_enabled=False,
            )
        else:
            accepted_enabled_edges.append((conn.source_node_id, conn.target_node_id))

    return resolved


class NEATCrossover:
    """Stanley's NEAT crossover aligned by ``innovation_id`` (paper, section 3.2).

    Matching genes (same innovation_id in both parents) are inherited from
    either parent, with ``probability_of_inheriting_from_fitter_parent_for_matching_genes``
    weighting the choice. Disjoint and excess genes are inherited from the
    fitter parent only. When a gene is disabled in either parent, the offspring
    has a chance to inherit it disabled too — this matches the reference
    implementation and prevents accidentally re-enabling structure the parents
    have collectively pruned.

    Nodes referenced by inherited connections are collected from the fitter
    parent first; any nodes present only in the less-fit parent are added last
    so that the offspring can still express connections whose endpoints came
    from the less-fit parent (this is a rare edge case).
    """

    def __init__(
        self,
        probability_of_inheriting_from_fitter_parent_for_matching_genes: float,
        probability_of_child_gene_remaining_disabled_when_either_parent_disabled: float = 0.75,
    ) -> None:
        self._probability_of_inheriting_from_fitter_parent_for_matching_genes = (
            probability_of_inheriting_from_fitter_parent_for_matching_genes
        )
        self._probability_of_child_gene_remaining_disabled = (
            probability_of_child_gene_remaining_disabled_when_either_parent_disabled
        )

    def apply_to_parents(
        self,
        fitter_parent: NEATGenome,
        less_fit_parent: NEATGenome,
        rng: Generator,
    ) -> NEATGenome:
        """Mate two parents into one child genome.

        Args:
            fitter_parent: Parent with the higher fitness; disjoint and
                excess genes are inherited from it.
            less_fit_parent: The other parent; contributes only matching
                genes (and, rarely, nodes referenced by them).
            rng: Source of randomness for gene choice and enable flips.

        Returns:
            The child genome, with any accidental directed cycles resolved
            by disabling the offending connections.
        """
        fitter_parent_connections_by_innovation_id: dict[int, ConnectionGene] = {
            connection_gene.innovation_id: connection_gene
            for connection_gene in fitter_parent.connection_genes
        }
        less_fit_parent_connections_by_innovation_id: dict[int, ConnectionGene] = {
            connection_gene.innovation_id: connection_gene
            for connection_gene in less_fit_parent.connection_genes
        }

        child_connection_genes: list[ConnectionGene] = []
        for (
            innovation_id,
            fitter_connection_gene,
        ) in fitter_parent_connections_by_innovation_id.items():
            matched_less_fit_connection_gene = less_fit_parent_connections_by_innovation_id.get(
                innovation_id
            )

            if matched_less_fit_connection_gene is None:
                # Disjoint or excess relative to less-fit parent → inherit from fitter.
                child_connection_genes.append(fitter_connection_gene)
                continue

            inherit_from_fitter = (
                rng.random() < self._probability_of_inheriting_from_fitter_parent_for_matching_genes
            )
            chosen_connection_gene = (
                fitter_connection_gene if inherit_from_fitter else matched_less_fit_connection_gene
            )

            either_parent_had_disabled_gene = (not fitter_connection_gene.is_enabled) or (
                not matched_less_fit_connection_gene.is_enabled
            )
            if either_parent_had_disabled_gene:
                child_is_enabled = not (
                    rng.random() < self._probability_of_child_gene_remaining_disabled
                )
                child_connection_genes.append(
                    ConnectionGene(
                        innovation_id=chosen_connection_gene.innovation_id,
                        source_node_id=chosen_connection_gene.source_node_id,
                        target_node_id=chosen_connection_gene.target_node_id,
                        weight=chosen_connection_gene.weight,
                        is_enabled=child_is_enabled,
                    )
                )
            else:
                child_connection_genes.append(chosen_connection_gene)

        node_ids_referenced_by_child_connections: set[int] = set()
        for child_connection_gene in child_connection_genes:
            node_ids_referenced_by_child_connections.add(child_connection_gene.source_node_id)
            node_ids_referenced_by_child_connections.add(child_connection_gene.target_node_id)

        fitter_parent_node_gene_by_id: dict[int, NodeGene] = {
            node_gene.node_id: node_gene for node_gene in fitter_parent.node_genes
        }
        less_fit_parent_node_gene_by_id: dict[int, NodeGene] = {
            node_gene.node_id: node_gene for node_gene in less_fit_parent.node_genes
        }

        child_node_genes: list[NodeGene] = []
        node_ids_already_added: set[int] = set()

        for node_gene in fitter_parent.node_genes:
            child_node_genes.append(node_gene)
            node_ids_already_added.add(node_gene.node_id)

        for required_node_id in node_ids_referenced_by_child_connections:
            if required_node_id in node_ids_already_added:
                continue
            fallback_node_gene = fitter_parent_node_gene_by_id.get(
                required_node_id
            ) or less_fit_parent_node_gene_by_id.get(required_node_id)
            if fallback_node_gene is not None:
                child_node_genes.append(fallback_node_gene)
                node_ids_already_added.add(required_node_id)

        validated_child_connection_genes = _resolve_enabled_connection_cycles(
            child_connection_genes
        )

        return NEATGenome(
            node_genes=tuple(child_node_genes),
            connection_genes=tuple(validated_child_connection_genes),
        )
