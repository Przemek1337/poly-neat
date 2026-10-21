"""Add-node mutation of EXACT (Desell, 2017, section III-B).

References:
    Desell, T. (2017). Developing a Volunteer Computing Project to Evolve
        Convolutional Neural Networks and Their Hyperparameters. 2017 IEEE
        13th International Conference on e-Science, pp. 19-28.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import cast

from numpy.random import Generator

from polyneat.algorithms.exact.exact_genome import (
    ConvolutionEdgeGene,
    EXACTGenome,
    FilterNodeGene,
)
from polyneat.algorithms.exact.exact_innovation_tracker import EXACTInnovationTracker
from polyneat.core.component_protocols import InnovationTracker

_MAXIMUM_EDGES_PER_SIDE = 5


@dataclass(frozen=True)
class AddConvolutionNodeMutation:
    """Inserts a hidden node at a random depth, wired into the existing graph.

    Per the paper (section III-B): a depth is drawn uniformly from (0, 1),
    which splits the CNN in two since the input sits at depth 0 and the
    outputs at depth 1. A new node is created at that depth, 1-5 edges are
    generated from randomly chosen lesser-depth nodes and 1-5 edges to
    randomly chosen greater-depth nodes (capped by how many candidates
    exist), and the node size is set per dimension to the average of the
    maximum selected input node size and the minimum selected output node
    size.

    Attributes:
        minimum_hidden_filter_size: Floor for the new node's filter size.

    References:
        Desell, T. (2017). Developing a Volunteer Computing Project to Evolve
            Convolutional Neural Networks and Their Hyperparameters. 2017 IEEE
            13th International Conference on e-Science, pp. 19-28.
    """

    minimum_hidden_filter_size: int

    def apply_to_genome(
        self,
        genome: EXACTGenome,
        rng: Generator,
        innovation_tracker: InnovationTracker,
    ) -> EXACTGenome:
        """Return a genome with one new wired hidden node.

        Args:
            genome: Genome to mutate.
            rng: Source of randomness for the depth and the neighbor draws.
            innovation_tracker: Issues the node id and the edge markings;
                must be an :class:`EXACTInnovationTracker`.

        Returns:
            The mutated genome, marked untrained.
        """
        exact_tracker = cast("EXACTInnovationTracker", innovation_tracker)
        existing_depths = {node_gene.depth for node_gene in genome.node_genes}
        new_node_depth = float(rng.uniform(0.0, 1.0))
        while new_node_depth <= 0.0 or new_node_depth in existing_depths:
            new_node_depth = float(rng.uniform(0.0, 1.0))

        lesser_depth_nodes = [
            node_gene
            for node_gene in genome.node_genes
            if node_gene.depth < new_node_depth
        ]
        greater_depth_nodes = [
            node_gene
            for node_gene in genome.node_genes
            if node_gene.depth > new_node_depth
        ]

        number_of_incoming_edges = min(
            int(rng.integers(1, _MAXIMUM_EDGES_PER_SIDE + 1)), len(lesser_depth_nodes)
        )
        number_of_outgoing_edges = min(
            int(rng.integers(1, _MAXIMUM_EDGES_PER_SIDE + 1)), len(greater_depth_nodes)
        )
        selected_source_nodes = [
            lesser_depth_nodes[index]
            for index in rng.choice(
                len(lesser_depth_nodes), size=number_of_incoming_edges, replace=False
            )
        ]
        selected_target_nodes = [
            greater_depth_nodes[index]
            for index in rng.choice(
                len(greater_depth_nodes), size=number_of_outgoing_edges, replace=False
            )
        ]

        maximum_source_height = max(
            node_gene.filter_height for node_gene in selected_source_nodes
        )
        maximum_source_width = max(
            node_gene.filter_width for node_gene in selected_source_nodes
        )
        minimum_target_height = min(
            node_gene.filter_height for node_gene in selected_target_nodes
        )
        minimum_target_width = min(
            node_gene.filter_width for node_gene in selected_target_nodes
        )

        highest_existing_node_id = max(
            node_gene.node_id for node_gene in genome.node_genes
        )
        new_node_gene = FilterNodeGene(
            node_id=exact_tracker.assign_fresh_node_id(
                minimum_new_node_id=highest_existing_node_id + 1
            ),
            node_type="hidden",
            filter_height=max(
                self.minimum_hidden_filter_size,
                round((maximum_source_height + minimum_target_height) / 2),
            ),
            filter_width=max(
                self.minimum_hidden_filter_size,
                round((maximum_source_width + minimum_target_width) / 2),
            ),
            depth=new_node_depth,
        )

        new_edge_genes = tuple(
            ConvolutionEdgeGene(
                innovation_id=exact_tracker.get_or_assign_innovation_id_for_connection(
                    source_node_id=source_node.node_id,
                    target_node_id=new_node_gene.node_id,
                ),
                source_node_id=source_node.node_id,
                target_node_id=new_node_gene.node_id,
                is_enabled=True,
                kernel_weights=None,
            )
            for source_node in selected_source_nodes
        ) + tuple(
            ConvolutionEdgeGene(
                innovation_id=exact_tracker.get_or_assign_innovation_id_for_connection(
                    source_node_id=new_node_gene.node_id,
                    target_node_id=target_node.node_id,
                ),
                source_node_id=new_node_gene.node_id,
                target_node_id=target_node.node_id,
                is_enabled=True,
                kernel_weights=None,
            )
            for target_node in selected_target_nodes
        )
        return replace(
            genome,
            node_genes=(*genome.node_genes, new_node_gene),
            edge_genes=(*genome.edge_genes, *new_edge_genes),
            is_trained=False,
        )
