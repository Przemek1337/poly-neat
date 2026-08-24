"""Crossover for DeepNEAT genomes, aligned by innovation id.

References:
    Miikkulainen, R., et al. (2017). Evolving Deep Neural Networks. *arXiv:1703.00548*.
        DOI: 10.1016/B978-0-12-815480-9.00015-3
    Stanley, K. O., & Miikkulainen, R. (2002). Evolving Neural Networks through Augmenting
        Topologies. *Evolutionary Computation*, 10(2), 99-127. (Gene alignment: section 3.2.)
"""

from __future__ import annotations

from numpy.random import Generator

from polyneat.algorithms.deepneat.deepneat_genome import (
    DeepNEATGenome,
    LayerNodeGene,
    TensorEdgeGene,
)
from polyneat.logging_utils.custom_logger import get_logger
from polyneat.nn.topology_utilities import would_directed_edge_create_cycle

logger = get_logger(__name__)


class DeepNEATCrossover:
    """Combines two architectures by aligning their edges on innovation id.

    Edges follow NEAT's rule: matching genes are drawn from either parent,
    disjoint and excess genes come from the fitter parent only. Layer nodes are
    inherited as a consequence - the child holds every node its inherited edges
    reference, plus the input and output layers.

    A node present in both parents takes its hyperparameters from one randomly
    chosen parent **as a whole**, never field by field. Mixing fields could
    produce values outside the configured search space, and averaging them would
    invent values the sampler can never draw.

    References:
        Miikkulainen, R., et al. (2017). Evolving Deep Neural Networks.
            *arXiv:1703.00548*. (Gene alignment and mating: Figure 4.)
        Stanley, K. O., & Miikkulainen, R. (2002). Evolving Neural Networks through
            Augmenting Topologies. *Evolutionary Computation*, 10(2), 99-127.
    """

    def __init__(
        self,
        probability_of_inheriting_from_fitter_parent_for_matching_genes: float,
    ) -> None:
        """Store the inheritance bias for matching genes.

        Args:
            probability_of_inheriting_from_fitter_parent_for_matching_genes:
                Chance a matching gene is taken from the fitter parent.
        """
        self._probability_of_inheriting_from_fitter_parent = (
            probability_of_inheriting_from_fitter_parent_for_matching_genes
        )

    def apply_to_parents(
        self,
        fitter_parent: DeepNEATGenome,
        less_fit_parent: DeepNEATGenome,
        rng: Generator,
    ) -> DeepNEATGenome:
        """Return the offspring of two architectures.

        Args:
            fitter_parent: The parent with the higher fitness; supplies every
                disjoint and excess gene.
            less_fit_parent: The other parent; contributes only matching genes.
            rng: Source of randomness for the per-gene inheritance choices.

        Returns:
            A valid, acyclic child genome.

        References:
            Miikkulainen, R., et al. (2017). Evolving Deep Neural Networks.
                *arXiv:1703.00548*. (Mating scheme: Figure 4.)
        """
        less_fit_edges_by_innovation_id = {
            edge.innovation_id: edge for edge in less_fit_parent.edge_genes
        }

        inherited_edges: list[TensorEdgeGene] = []
        for fitter_edge in fitter_parent.edge_genes:
            matching_edge = less_fit_edges_by_innovation_id.get(fitter_edge.innovation_id)
            if matching_edge is None:
                inherited_edges.append(fitter_edge)
                continue
            take_from_fitter = (
                rng.random() < self._probability_of_inheriting_from_fitter_parent
            )
            inherited_edges.append(fitter_edge if take_from_fitter else matching_edge)

        inherited_edges = self._drop_edges_closing_a_cycle(inherited_edges)

        less_fit_nodes_by_id = {node.node_id: node for node in less_fit_parent.node_genes}
        required_node_ids = {fitter_parent.input_node_id, fitter_parent.output_node_id}
        for edge in inherited_edges:
            required_node_ids.add(edge.source_node_id)
            required_node_ids.add(edge.target_node_id)

        inherited_nodes: list[LayerNodeGene] = []
        for fitter_node in fitter_parent.node_genes:
            if fitter_node.node_id not in required_node_ids:
                continue
            matching_node = less_fit_nodes_by_id.get(fitter_node.node_id)
            if matching_node is None or matching_node.layer_type != fitter_node.layer_type:
                inherited_nodes.append(fitter_node)
                continue
            take_from_fitter = (
                rng.random() < self._probability_of_inheriting_from_fitter_parent
            )
            inherited_nodes.append(fitter_node if take_from_fitter else matching_node)

        return DeepNEATGenome(
            node_genes=tuple(inherited_nodes),
            edge_genes=tuple(inherited_edges),
        )

    @staticmethod
    def _drop_edges_closing_a_cycle(
        candidate_edges: list[TensorEdgeGene],
    ) -> list[TensorEdgeGene]:
        """Disable enabled edges that would make the child cyclic.

        Both parents are acyclic, but the child's *combination* of enable bits
        need not be: an edge disabled in the fitter parent may arrive enabled
        from the other one and close a loop. Edges are considered in innovation
        order, so the resolution is deterministic given the inheritance draws.

        Args:
            candidate_edges: The edges inherited so far.

        Returns:
            The same edges, with cycle-closing ones switched off.
        """
        accepted_enabled_edges: list[tuple[int, int]] = []
        resolved_edges: list[TensorEdgeGene] = []
        for edge in sorted(candidate_edges, key=lambda gene: gene.innovation_id):
            if not edge.is_enabled:
                resolved_edges.append(edge)
                continue
            if would_directed_edge_create_cycle(
                candidate_source_node_id=edge.source_node_id,
                candidate_target_node_id=edge.target_node_id,
                existing_enabled_edges=accepted_enabled_edges,
            ):
                logger.debug(
                    "DeepNEATCrossover disabled innov=%d to keep the child acyclic",
                    edge.innovation_id,
                )
                resolved_edges.append(
                    TensorEdgeGene(
                        innovation_id=edge.innovation_id,
                        source_node_id=edge.source_node_id,
                        target_node_id=edge.target_node_id,
                        is_enabled=False,
                    )
                )
                continue
            accepted_enabled_edges.append((edge.source_node_id, edge.target_node_id))
            resolved_edges.append(edge)
        return resolved_edges
