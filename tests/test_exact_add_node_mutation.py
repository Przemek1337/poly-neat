"""Tests for the add-node mutation (Desell, 2017, section III-B)."""

from __future__ import annotations

import numpy as np

from polyneat.algorithms.exact.exact_genome import (
    ConvolutionEdgeGene,
    EXACTGenome,
    FilterNodeGene,
)
from polyneat.algorithms.exact.exact_innovation_tracker import EXACTInnovationTracker
from polyneat.algorithms.exact.mutations.add_convolution_node_mutation import (
    AddConvolutionNodeMutation,
)


def _build_minimal_genome() -> EXACTGenome:
    return EXACTGenome(
        node_genes=(
            FilterNodeGene(
                node_id=0, node_type="input", filter_height=8, filter_width=8, depth=0.0
            ),
            FilterNodeGene(
                node_id=1, node_type="output", filter_height=1, filter_width=1, depth=1.0
            ),
            FilterNodeGene(
                node_id=2, node_type="output", filter_height=1, filter_width=1, depth=1.0
            ),
        ),
        edge_genes=(
            ConvolutionEdgeGene(
                innovation_id=0, source_node_id=0, target_node_id=1, is_enabled=True
            ),
            ConvolutionEdgeGene(
                innovation_id=1, source_node_id=0, target_node_id=2, is_enabled=True
            ),
        ),
        is_trained=True,
    )


def _seeded_tracker() -> EXACTInnovationTracker:
    tracker = EXACTInnovationTracker()
    tracker.get_or_assign_innovation_id_for_connection(0, 1)
    tracker.get_or_assign_innovation_id_for_connection(0, 2)
    tracker.assign_fresh_node_id(minimum_new_node_id=3)
    return tracker


def test_add_node_inserts_hidden_node_with_wired_edges() -> None:
    genome = _build_minimal_genome()
    mutated = AddConvolutionNodeMutation(minimum_hidden_filter_size=1).apply_to_genome(
        genome=genome, rng=np.random.default_rng(7), innovation_tracker=_seeded_tracker()
    )
    new_nodes = [
        node_gene for node_gene in mutated.node_genes if node_gene.node_type == "hidden"
    ]
    assert len(new_nodes) == 1
    new_node = new_nodes[0]
    assert 0.0 < new_node.depth < 1.0
    incoming = [
        edge_gene
        for edge_gene in mutated.edge_genes
        if edge_gene.target_node_id == new_node.node_id
    ]
    outgoing = [
        edge_gene
        for edge_gene in mutated.edge_genes
        if edge_gene.source_node_id == new_node.node_id
    ]
    assert 1 <= len(incoming) <= 5
    assert 1 <= len(outgoing) <= 5
    assert all(
        edge_gene.is_enabled and edge_gene.kernel_weights is None
        for edge_gene in incoming + outgoing
    )
    # Only the input (8x8) can feed it, only 1x1 outputs can follow:
    # size = round((8 + 1) / 2) = 4 in both dimensions.
    assert (new_node.filter_height, new_node.filter_width) == (4, 4)
    assert mutated.is_trained is False
    assert mutated.are_all_output_nodes_reachable()


def test_add_node_leaves_original_genes_untouched() -> None:
    genome = _build_minimal_genome()
    mutated = AddConvolutionNodeMutation(minimum_hidden_filter_size=1).apply_to_genome(
        genome=genome, rng=np.random.default_rng(3), innovation_tracker=_seeded_tracker()
    )
    assert genome.edge_genes == mutated.edge_genes[: len(genome.edge_genes)]
    assert genome.node_genes == mutated.node_genes[: len(genome.node_genes)]
