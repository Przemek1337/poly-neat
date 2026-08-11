from __future__ import annotations

import numpy as np

from polyneat.algorithms.exact.exact_crossover import EXACTCrossover
from polyneat.algorithms.exact.exact_genome import (
    ConvolutionEdgeGene,
    EXACTGenome,
    FilterNodeGene,
)

_INPUT = FilterNodeGene(
    node_id=0, node_type="input", filter_height=28, filter_width=28, depth=0.0
)
_OUTPUT = FilterNodeGene(
    node_id=1, node_type="output", filter_height=1, filter_width=1, depth=1.0
)


def _kernel(size: int, value: float) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(value for _ in range(size)) for _ in range(size))


def _fitter_parent() -> EXACTGenome:
    # Hidden node 2 is 10x10 here; edges 0 (0->2), 1 (2->1), 3 (0->1, fitter only).
    return EXACTGenome(
        node_genes=(
            _INPUT,
            _OUTPUT,
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
                kernel_weights=_kernel(19, 1.0),
            ),
            ConvolutionEdgeGene(
                innovation_id=1,
                source_node_id=2,
                target_node_id=1,
                is_enabled=True,
                kernel_weights=_kernel(10, 1.0),
            ),
            ConvolutionEdgeGene(
                innovation_id=3, source_node_id=0, target_node_id=1, is_enabled=True
            ),
        ),
        is_trained=True,
    )


def _less_fit_parent() -> EXACTGenome:
    # Same structure ids, but hidden node 2 is 12x12 and edge 4 (2->1 twin path
    # innovation) exists only here. Edges 0 and 1 match the fitter parent, so
    # the 0 -> 2 -> 1 path survives even when every non-matching edge is
    # inherited disabled. Kernels here fit *this* parent's 12x12 node.
    return EXACTGenome(
        node_genes=(
            _INPUT,
            _OUTPUT,
            FilterNodeGene(
                node_id=2, node_type="hidden", filter_height=12, filter_width=12, depth=0.5
            ),
        ),
        edge_genes=(
            ConvolutionEdgeGene(
                innovation_id=0,
                source_node_id=0,
                target_node_id=2,
                is_enabled=True,
                kernel_weights=_kernel(17, 2.0),
            ),
            ConvolutionEdgeGene(
                innovation_id=1,
                source_node_id=2,
                target_node_id=1,
                is_enabled=True,
                kernel_weights=_kernel(12, 2.0),
            ),
            ConvolutionEdgeGene(
                innovation_id=4,
                source_node_id=2,
                target_node_id=1,
                is_enabled=True,
                kernel_weights=_kernel(12, 2.0),
            ),
        ),
        is_trained=True,
    )


def test_matching_edges_and_nodes_come_from_the_fitter_parent() -> None:
    crossover = EXACTCrossover(
        more_fit_parent_edge_inclusion_rate=1.0,
        less_fit_parent_edge_inclusion_rate=1.0,
    )
    child = crossover.apply_to_parents(
        _fitter_parent(), _less_fit_parent(), np.random.default_rng(0)
    )
    # Node 2 has the fitter parent's 10x10 size.
    assert child.get_node_gene_by_id(2).filter_height == 10
    # Matching edge 0 keeps the fitter parent's 19x19 kernel.
    assert child.get_edge_gene_by_innovation_id(0).kernel_weights == _kernel(19, 1.0)
    assert child.is_trained is False


def test_rate_zero_carries_non_matching_edges_disabled() -> None:
    crossover = EXACTCrossover(
        more_fit_parent_edge_inclusion_rate=0.0,
        less_fit_parent_edge_inclusion_rate=0.0,
    )
    child = crossover.apply_to_parents(
        _fitter_parent(), _less_fit_parent(), np.random.default_rng(0)
    )
    # Section 2.3: edges not added by either parent are carried over disabled.
    assert child.get_edge_gene_by_innovation_id(3).is_enabled is False
    assert child.get_edge_gene_by_innovation_id(4).is_enabled is False
    # Matching edges are unconditional.
    assert child.get_edge_gene_by_innovation_id(0).is_enabled is True


def test_rate_one_includes_non_matching_edges_enabled() -> None:
    crossover = EXACTCrossover(
        more_fit_parent_edge_inclusion_rate=1.0,
        less_fit_parent_edge_inclusion_rate=1.0,
    )
    child = crossover.apply_to_parents(
        _fitter_parent(), _less_fit_parent(), np.random.default_rng(0)
    )
    assert child.get_edge_gene_by_innovation_id(3).is_enabled is True
    assert child.get_edge_gene_by_innovation_id(4).is_enabled is True


def test_kernels_incompatible_with_inherited_node_sizes_are_reset() -> None:
    crossover = EXACTCrossover(
        more_fit_parent_edge_inclusion_rate=1.0,
        less_fit_parent_edge_inclusion_rate=1.0,
    )
    child = crossover.apply_to_parents(
        _fitter_parent(), _less_fit_parent(), np.random.default_rng(0)
    )
    # Edge 4 comes from the less fit parent with a 12x12 kernel, but node 2 is
    # inherited from the fitter parent as 10x10 -> expected kernel 10x10, so
    # the stored kernel is invalidated (interpretation note 11).
    assert child.get_edge_gene_by_innovation_id(4).kernel_weights is None
    # Edge 0's fitter kernel matches the fitter node sizes and survives.
    assert child.get_edge_gene_by_innovation_id(0).kernel_weights is not None


def test_crossover_never_returns_unreachable_child() -> None:
    """With inclusion rates 0.0 every parent-only edge is inherited disabled.
    If the only input->output path is a fitter-parent-only edge, the naive
    child is unreachable; the operator must fall back to a clone of the
    fitter parent, marked untrained."""
    input_node = FilterNodeGene(
        node_id=0, node_type="input", filter_height=4, filter_width=4, depth=0.0
    )
    output_node = FilterNodeGene(
        node_id=1, node_type="output", filter_height=1, filter_width=1, depth=1.0
    )
    only_path_edge = ConvolutionEdgeGene(
        innovation_id=0, source_node_id=0, target_node_id=1, is_enabled=True
    )
    fitter_parent = EXACTGenome(
        node_genes=(input_node, output_node),
        edge_genes=(only_path_edge,),
        is_trained=True,
    )
    less_fit_parent = EXACTGenome(
        node_genes=(input_node, output_node), edge_genes=(), is_trained=True
    )
    crossover = EXACTCrossover(
        more_fit_parent_edge_inclusion_rate=0.0,
        less_fit_parent_edge_inclusion_rate=0.0,
        maximum_attempts_for_reachable_child=3,
    )
    child = crossover.apply_to_parents(
        fitter_parent=fitter_parent,
        less_fit_parent=less_fit_parent,
        rng=np.random.default_rng(0),
    )
    assert child.are_all_output_nodes_reachable()
    assert child.is_trained is False
    assert child.edge_genes == fitter_parent.edge_genes
