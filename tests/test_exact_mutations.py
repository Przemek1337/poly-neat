from __future__ import annotations

import numpy as np

from polyneat.algorithms.exact.exact_genome import (
    ConvolutionEdgeGene,
    EXACTGenome,
    FilterNodeGene,
)
from polyneat.algorithms.exact.mutations.add_convolution_edge_mutation import (
    AddConvolutionEdgeMutation,
)
from polyneat.algorithms.exact.mutations.change_filter_size_mutation import (
    ChangeFilterSizeMutation,
)
from polyneat.algorithms.exact.mutations.disable_convolution_edge_mutation import (
    DisableConvolutionEdgeMutation,
)
from polyneat.algorithms.exact.mutations.enable_convolution_edge_mutation import (
    EnableConvolutionEdgeMutation,
)
from polyneat.core.neat.global_innovation_tracker import GlobalInnovationTracker


def _three_node_genome() -> EXACTGenome:
    """input(28x28, d=0) -> hidden(10x10, d=0.5) -> output(1x1, d=1), edge 0-2 disabled."""
    kernel_19 = tuple(tuple(0.5 for _ in range(19)) for _ in range(19))
    return EXACTGenome(
        node_genes=(
            FilterNodeGene(
                node_id=0, node_type="input", filter_height=28, filter_width=28, depth=0.0
            ),
            FilterNodeGene(
                node_id=1, node_type="output", filter_height=1, filter_width=1, depth=1.0
            ),
            FilterNodeGene(
                node_id=2, node_type="hidden", filter_height=10, filter_width=10, depth=0.5
            ),
        ),
        edge_genes=(
            ConvolutionEdgeGene(
                innovation_id=0,
                source_node_id=0,
                target_node_id=2,
                is_enabled=True,
                kernel_weights=kernel_19,
            ),
            ConvolutionEdgeGene(
                innovation_id=1, source_node_id=2, target_node_id=1, is_enabled=True
            ),
            ConvolutionEdgeGene(
                innovation_id=2, source_node_id=0, target_node_id=1, is_enabled=False
            ),
        ),
        is_trained=True,
    )


def test_disable_edge_disables_one_enabled_edge_and_marks_untrained() -> None:
    genome = _three_node_genome()
    mutated = DisableConvolutionEdgeMutation().apply_to_genome(
        genome, np.random.default_rng(0), GlobalInnovationTracker()
    )
    enabled_before = sum(edge.is_enabled for edge in genome.edge_genes)
    enabled_after = sum(edge.is_enabled for edge in mutated.edge_genes)
    assert enabled_after == enabled_before - 1
    assert len(mutated.edge_genes) == len(genome.edge_genes)  # gene stays in the genome
    assert mutated.is_trained is False


def test_enable_edge_enables_the_disabled_edge() -> None:
    genome = _three_node_genome()
    mutated = EnableConvolutionEdgeMutation().apply_to_genome(
        genome, np.random.default_rng(0), GlobalInnovationTracker()
    )
    assert mutated.get_edge_gene_by_innovation_id(2).is_enabled is True
    assert mutated.is_trained is False


def test_enable_edge_without_disabled_edges_returns_genome_unchanged() -> None:
    genome = _three_node_genome()
    all_enabled = EXACTGenome(
        node_genes=genome.node_genes,
        edge_genes=tuple(
            ConvolutionEdgeGene(
                innovation_id=edge.innovation_id,
                source_node_id=edge.source_node_id,
                target_node_id=edge.target_node_id,
                is_enabled=True,
                kernel_weights=edge.kernel_weights,
            )
            for edge in genome.edge_genes
        ),
        is_trained=True,
    )
    mutated = EnableConvolutionEdgeMutation().apply_to_genome(
        all_enabled, np.random.default_rng(0), GlobalInnovationTracker()
    )
    assert mutated is all_enabled


def test_add_edge_with_no_free_pair_returns_genome_unchanged() -> None:
    genome = _three_node_genome()  # all three depth-ordered pairs already have an edge
    mutated = AddConvolutionEdgeMutation().apply_to_genome(
        genome, np.random.default_rng(0), GlobalInnovationTracker()
    )
    assert mutated is genome


def test_add_edge_connects_a_lower_depth_node_to_a_higher_depth_node() -> None:
    genome = _three_node_genome()
    without_skip_edge = EXACTGenome(
        node_genes=genome.node_genes,
        edge_genes=genome.edge_genes[:2],  # drop the disabled 0 -> 1 edge
        is_trained=True,
    )
    # Seed the tracker with the genome's existing edges, as the initial-population
    # strategy does in a real run, so the new marking cannot collide with them.
    tracker = GlobalInnovationTracker()
    for existing_edge in without_skip_edge.edge_genes:
        tracker.get_or_assign_innovation_id_for_connection(
            source_node_id=existing_edge.source_node_id,
            target_node_id=existing_edge.target_node_id,
        )
    mutated = AddConvolutionEdgeMutation().apply_to_genome(
        without_skip_edge, np.random.default_rng(0), tracker
    )
    assert len(mutated.edge_genes) == 3
    new_edge = mutated.edge_genes[-1]
    assert (new_edge.source_node_id, new_edge.target_node_id) == (0, 1)
    assert new_edge.is_enabled is True
    assert new_edge.kernel_weights is None
    assert mutated.is_trained is False


def test_change_filter_size_resizes_only_hidden_nodes_and_invalidates_kernels() -> None:
    genome = _three_node_genome()
    mutation = ChangeFilterSizeMutation(
        change_height=True,
        change_width=True,
        filter_size_change_options=(2,),
        minimum_filter_size=1,
    )
    mutated = mutation.apply_to_genome(
        genome, np.random.default_rng(0), GlobalInnovationTracker()
    )
    hidden_node = mutated.get_node_gene_by_id(2)
    assert (hidden_node.filter_height, hidden_node.filter_width) == (12, 12)
    assert mutated.get_node_gene_by_id(0).filter_height == 28  # input untouched
    assert mutated.get_node_gene_by_id(1).filter_height == 1  # output untouched
    # Both edges incident to the resized node lose their kernels.
    assert mutated.get_edge_gene_by_innovation_id(0).kernel_weights is None
    assert mutated.get_edge_gene_by_innovation_id(1).kernel_weights is None
    assert mutated.is_trained is False


def test_change_filter_size_single_dimension_and_clamp() -> None:
    genome = _three_node_genome()
    mutation = ChangeFilterSizeMutation(
        change_height=False,
        change_width=True,
        filter_size_change_options=(-2,),
        minimum_filter_size=9,
    )
    mutated = mutation.apply_to_genome(
        genome, np.random.default_rng(0), GlobalInnovationTracker()
    )
    hidden_node = mutated.get_node_gene_by_id(2)
    assert hidden_node.filter_height == 10  # y untouched
    assert hidden_node.filter_width == 9  # 10 - 2 clamped to the minimum


def test_change_filter_size_without_hidden_nodes_returns_genome_unchanged() -> None:
    minimal = EXACTGenome(
        node_genes=(
            FilterNodeGene(
                node_id=0, node_type="input", filter_height=28, filter_width=28, depth=0.0
            ),
            FilterNodeGene(
                node_id=1, node_type="output", filter_height=1, filter_width=1, depth=1.0
            ),
        ),
        edge_genes=(
            ConvolutionEdgeGene(
                innovation_id=0, source_node_id=0, target_node_id=1, is_enabled=True
            ),
        ),
    )
    mutation = ChangeFilterSizeMutation(
        change_height=True,
        change_width=True,
        filter_size_change_options=(-2, -1, 1, 2),
        minimum_filter_size=1,
    )
    mutated = mutation.apply_to_genome(
        minimal, np.random.default_rng(0), GlobalInnovationTracker()
    )
    assert mutated is minimal
