"""Add-edge mutation of EXACT (Desell, 2017, section III-B)."""

from __future__ import annotations

from dataclasses import dataclass, replace

from numpy.random import Generator

from polyneat.algorithms.exact.exact_genome import ConvolutionEdgeGene, EXACTGenome
from polyneat.core.component_protocols import InnovationTracker


@dataclass(frozen=True)
class AddConvolutionEdgeMutation:
    """Adds an edge between two nodes ``n1``, ``n2`` with ``depth_n1 < depth_n2``.

    The strict depth ordering keeps every edge feed-forward (section III-B).
    The innovation tracker deduplicates the marking: if any genome added the
    same ``(source, target)`` edge this generation, the same innovation id is
    reused.
    """

    def apply_to_genome(
        self,
        genome: EXACTGenome,
        rng: Generator,
        innovation_tracker: InnovationTracker,
    ) -> EXACTGenome:
        """Return a genome with one new enabled edge, or ``genome`` when no pair is free.

        Args:
            genome: Genome to mutate.
            rng: Source of randomness for the endpoint draw.
            innovation_tracker: Issues the new edge's historical marking.

        Returns:
            The mutated genome, marked untrained.
        """
        existing_endpoint_pairs = {
            (edge_gene.source_node_id, edge_gene.target_node_id)
            for edge_gene in genome.edge_genes
        }
        candidate_endpoint_pairs = [
            (source_node.node_id, target_node.node_id)
            for source_node in genome.node_genes
            for target_node in genome.node_genes
            if source_node.depth < target_node.depth
            and (source_node.node_id, target_node.node_id) not in existing_endpoint_pairs
        ]
        if not candidate_endpoint_pairs:
            return genome
        source_node_id, target_node_id = candidate_endpoint_pairs[
            int(rng.integers(len(candidate_endpoint_pairs)))
        ]
        new_edge_gene = ConvolutionEdgeGene(
            innovation_id=innovation_tracker.get_or_assign_innovation_id_for_connection(
                source_node_id=source_node_id,
                target_node_id=target_node_id,
            ),
            source_node_id=source_node_id,
            target_node_id=target_node_id,
            is_enabled=True,
            kernel_weights=None,
        )
        return replace(
            genome,
            edge_genes=(*genome.edge_genes, new_edge_gene),
            is_trained=False,
        )
