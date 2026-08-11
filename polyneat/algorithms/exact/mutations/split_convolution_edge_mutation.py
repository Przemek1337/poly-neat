"""Split-edge mutation of EXACT (Desell, 2017, section III-B)."""

from __future__ import annotations

from dataclasses import dataclass, replace

from numpy.random import Generator

from polyneat.algorithms.exact.exact_genome import (
    ConvolutionEdgeGene,
    EXACTGenome,
    FilterNodeGene,
)
from polyneat.core.component_protocols import InnovationTracker


@dataclass(frozen=True)
class SplitConvolutionEdgeMutation:
    """Splits one enabled edge with a new filter node halfway between its endpoints.

    The selected edge is disabled (not deleted); a new node of size
    ``round((i_d + o_d) / 2)`` per dimension (minimum 1) and depth
    ``(depth_in + depth_out) / 2`` is inserted, connected by two new edges
    (section III-B). Node id and both edge markings come from
    ``innovation_tracker.get_or_assign_node_split``, so two genomes splitting
    the same edge receive identical markings. When the tracker's markings
    persist across generations, an edge that was already split in this
    genome's ancestry (and later re-enabled) would be handed back the node
    it already contains; that draw is a no-op instead.
    """

    def apply_to_genome(
        self,
        genome: EXACTGenome,
        rng: Generator,
        innovation_tracker: InnovationTracker,
    ) -> EXACTGenome:
        """Return a genome with one edge split, or ``genome`` when no edge is enabled.

        Args:
            genome: Genome to mutate.
            rng: Source of randomness for the edge draw.
            innovation_tracker: Issues the node id and the two edge markings.

        Returns:
            The mutated genome, marked untrained, or ``genome`` unchanged
            when no edge is enabled or the drawn split already exists here.
        """
        enabled_edge_indices = [
            edge_index
            for edge_index, edge_gene in enumerate(genome.edge_genes)
            if edge_gene.is_enabled
        ]
        if not enabled_edge_indices:
            return genome
        selected_edge_index = enabled_edge_indices[
            int(rng.integers(len(enabled_edge_indices)))
        ]
        split_edge_gene = genome.edge_genes[selected_edge_index]
        source_node = genome.get_node_gene_by_id(split_edge_gene.source_node_id)
        target_node = genome.get_node_gene_by_id(split_edge_gene.target_node_id)

        highest_existing_node_id = max(node_gene.node_id for node_gene in genome.node_genes)
        node_split_record = innovation_tracker.get_or_assign_node_split(
            split_connection_innovation_id=split_edge_gene.innovation_id,
            minimum_new_node_id=highest_existing_node_id + 1,
        )
        if genome.get_node_gene_by_id(node_split_record.new_node_id) is not None:
            return genome

        new_node_gene = FilterNodeGene(
            node_id=node_split_record.new_node_id,
            node_type="hidden",
            filter_height=max(
                1, round((source_node.filter_height + target_node.filter_height) / 2)
            ),
            filter_width=max(
                1, round((source_node.filter_width + target_node.filter_width) / 2)
            ),
            depth=(source_node.depth + target_node.depth) / 2.0,
        )
        edge_into_new_node = ConvolutionEdgeGene(
            innovation_id=node_split_record.innovation_id_into_new_node,
            source_node_id=source_node.node_id,
            target_node_id=new_node_gene.node_id,
            is_enabled=True,
            kernel_weights=None,
        )
        edge_out_of_new_node = ConvolutionEdgeGene(
            innovation_id=node_split_record.innovation_id_out_of_new_node,
            source_node_id=new_node_gene.node_id,
            target_node_id=target_node.node_id,
            is_enabled=True,
            kernel_weights=None,
        )
        mutated_edge_genes = tuple(
            replace(edge_gene, is_enabled=False)
            if edge_index == selected_edge_index
            else edge_gene
            for edge_index, edge_gene in enumerate(genome.edge_genes)
        ) + (edge_into_new_node, edge_out_of_new_node)
        return EXACTGenome(
            node_genes=(*genome.node_genes, new_node_gene),
            edge_genes=mutated_edge_genes,
            is_trained=False,
        )
