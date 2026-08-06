from __future__ import annotations

import numpy as np

from polyneat.algorithms.exact.exact_genome import (
    ConvolutionEdgeGene,
    EXACTGenome,
    FilterNodeGene,
)
from polyneat.algorithms.exact.mutations.disable_convolution_edge_mutation import (
    DisableConvolutionEdgeMutation,
)
from polyneat.algorithms.exact.mutations.exact_composite_mutation import (
    EXACTCompositeMutation,
)
from polyneat.algorithms.exact.mutations.split_convolution_edge_mutation import (
    SplitConvolutionEdgeMutation,
)
from polyneat.core.neat.global_innovation_tracker import GlobalInnovationTracker


def _minimal_genome() -> EXACTGenome:
    return EXACTGenome(
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
        is_trained=True,
    )


def _tracker_seeded_for(genome: EXACTGenome) -> GlobalInnovationTracker:
    """Tracker whose counters are past the genome's hand-written markings.

    In a real run the initial-population strategy issues every starting edge
    id through the tracker; these tests build genomes literally, so the
    tracker has to be advanced the same way or it would reissue id 0.
    """
    tracker = GlobalInnovationTracker()
    for edge_gene in genome.edge_genes:
        tracker.get_or_assign_innovation_id_for_connection(
            source_node_id=edge_gene.source_node_id,
            target_node_id=edge_gene.target_node_id,
        )
    return tracker


def test_split_edge_creates_midpoint_node_and_two_new_edges() -> None:
    genome = _minimal_genome()
    tracker = _tracker_seeded_for(genome)
    mutated = SplitConvolutionEdgeMutation().apply_to_genome(
        genome, np.random.default_rng(0), tracker
    )
    assert len(mutated.node_genes) == 3
    assert len(mutated.edge_genes) == 3
    # The split edge is disabled but kept (section III-B).
    assert mutated.get_edge_gene_by_innovation_id(0).is_enabled is False
    new_node = next(node for node in mutated.node_genes if node.node_type == "hidden")
    # Size is the midpoint (28+1)/2 -> 14.5 -> round() -> 14; depth is (0+1)/2.
    assert (new_node.filter_height, new_node.filter_width) == (14, 14)
    assert new_node.depth == 0.5
    new_edges = [edge for edge in mutated.edge_genes if edge.innovation_id != 0]
    assert {(edge.source_node_id, edge.target_node_id) for edge in new_edges} == {
        (0, new_node.node_id),
        (new_node.node_id, 1),
    }
    assert all(edge.kernel_weights is None and edge.is_enabled for edge in new_edges)
    assert mutated.is_trained is False


def test_split_edge_reuses_markings_for_the_same_split_within_a_generation() -> None:
    tracker = _tracker_seeded_for(_minimal_genome())
    rng = np.random.default_rng(0)
    first = SplitConvolutionEdgeMutation().apply_to_genome(_minimal_genome(), rng, tracker)
    second = SplitConvolutionEdgeMutation().apply_to_genome(_minimal_genome(), rng, tracker)
    first_hidden = next(node for node in first.node_genes if node.node_type == "hidden")
    second_hidden = next(node for node in second.node_genes if node.node_type == "hidden")
    assert first_hidden.node_id == second_hidden.node_id
    assert {edge.innovation_id for edge in first.edge_genes} == {
        edge.innovation_id for edge in second.edge_genes
    }


def test_split_edge_without_enabled_edges_returns_genome_unchanged() -> None:
    genome = _minimal_genome()
    disabled_only = EXACTGenome(
        node_genes=genome.node_genes,
        edge_genes=(
            ConvolutionEdgeGene(
                innovation_id=0, source_node_id=0, target_node_id=1, is_enabled=False
            ),
        ),
    )
    mutated = SplitConvolutionEdgeMutation().apply_to_genome(
        disabled_only, np.random.default_rng(0), GlobalInnovationTracker()
    )
    assert mutated is disabled_only


def test_composite_applies_the_configured_number_of_mutations() -> None:
    composite = EXACTCompositeMutation(
        mutation_operators=[SplitConvolutionEdgeMutation()],
        operator_selection_probabilities=[1.0],
        number_of_mutations_per_genome=2,
        maximum_attempts_for_reachable_child=5,
    )
    genome = _minimal_genome()
    mutated = composite.apply_to_genome(
        genome, np.random.default_rng(0), _tracker_seeded_for(genome)
    )
    # Two splits: 2 new hidden nodes, 4 new edges alongside the disabled originals.
    hidden_nodes = [node for node in mutated.node_genes if node.node_type == "hidden"]
    assert len(hidden_nodes) == 2
    assert mutated.are_all_output_nodes_reachable() is True
    assert mutated.is_trained is False


def test_composite_discards_children_with_unreachable_outputs() -> None:
    # Disabling the only edge always strands the output, so every attempt is
    # discarded and the parent comes back unchanged (interpretation note 10).
    composite = EXACTCompositeMutation(
        mutation_operators=[DisableConvolutionEdgeMutation()],
        operator_selection_probabilities=[1.0],
        number_of_mutations_per_genome=1,
        maximum_attempts_for_reachable_child=3,
    )
    genome = _minimal_genome()
    mutated = composite.apply_to_genome(
        genome, np.random.default_rng(0), GlobalInnovationTracker()
    )
    assert mutated is genome
