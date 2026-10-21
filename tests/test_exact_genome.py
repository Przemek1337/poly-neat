from __future__ import annotations

import pytest

from polyneat.algorithms.exact.exact_genome import (
    ConvolutionEdgeGene,
    EXACTGenome,
    FilterNodeGene,
    InvalidEXACTGenomeError,
    compute_convolution_kernel_size,
)
from polyneat.algorithms.exact.exact_training_hyperparameters import (
    EXACTTrainingHyperparameters,
)


def _input_node() -> FilterNodeGene:
    return FilterNodeGene(
        node_id=0, node_type="input", filter_height=28, filter_width=28, depth=0.0
    )


def _output_node(node_id: int = 1) -> FilterNodeGene:
    return FilterNodeGene(
        node_id=node_id, node_type="output", filter_height=1, filter_width=1, depth=1.0
    )


def _hidden_node(node_id: int = 2, size: int = 10, depth: float = 0.5) -> FilterNodeGene:
    return FilterNodeGene(
        node_id=node_id, node_type="hidden", filter_height=size, filter_width=size, depth=depth
    )


def test_kernel_size_follows_the_paper_formula() -> None:
    # Section 2: conv_d = |out_d - in_d| + 1, in each dimension independently.
    assert compute_convolution_kernel_size(_input_node(), _hidden_node()) == (19, 19)
    assert compute_convolution_kernel_size(_hidden_node(), _output_node()) == (10, 10)
    assert compute_convolution_kernel_size(_hidden_node(), _hidden_node(node_id=3)) == (1, 1)


def test_minimal_genome_is_valid_and_reachable() -> None:
    genome = EXACTGenome(
        node_genes=(_input_node(), _output_node()),
        edge_genes=(
            ConvolutionEdgeGene(
                innovation_id=0, source_node_id=0, target_node_id=1, is_enabled=True
            ),
        ),
    )
    assert genome.is_trained is False
    assert genome.are_all_output_nodes_reachable() is True
    assert genome.get_node_gene_by_id(1) == _output_node()
    assert genome.get_edge_gene_by_innovation_id(0).kernel_weights is None


def test_disabled_only_path_makes_output_unreachable() -> None:
    genome = EXACTGenome(
        node_genes=(_input_node(), _output_node()),
        edge_genes=(
            ConvolutionEdgeGene(
                innovation_id=0, source_node_id=0, target_node_id=1, is_enabled=False
            ),
        ),
    )
    assert genome.are_all_output_nodes_reachable() is False


def test_rejects_duplicate_node_ids_and_innovation_ids() -> None:
    with pytest.raises(InvalidEXACTGenomeError):
        EXACTGenome(node_genes=(_input_node(), _input_node()), edge_genes=())
    with pytest.raises(InvalidEXACTGenomeError):
        EXACTGenome(
            node_genes=(_input_node(), _output_node()),
            edge_genes=(
                ConvolutionEdgeGene(
                    innovation_id=0, source_node_id=0, target_node_id=1, is_enabled=True
                ),
                ConvolutionEdgeGene(
                    innovation_id=0, source_node_id=0, target_node_id=1, is_enabled=False
                ),
            ),
        )


def test_rejects_dangling_edge_endpoints() -> None:
    with pytest.raises(InvalidEXACTGenomeError):
        EXACTGenome(
            node_genes=(_input_node(), _output_node()),
            edge_genes=(
                ConvolutionEdgeGene(
                    innovation_id=0, source_node_id=0, target_node_id=99, is_enabled=True
                ),
            ),
        )


def test_rejects_edges_that_do_not_feed_forward_in_depth() -> None:
    with pytest.raises(InvalidEXACTGenomeError):
        EXACTGenome(
            node_genes=(_input_node(), _output_node()),
            edge_genes=(
                ConvolutionEdgeGene(
                    innovation_id=0, source_node_id=1, target_node_id=0, is_enabled=True
                ),
            ),
        )


def test_rejects_kernel_with_wrong_shape() -> None:
    wrong_kernel = ((1.0, 2.0), (3.0, 4.0))  # 2x2, but 28x28 -> 1x1 needs 28x28
    with pytest.raises(InvalidEXACTGenomeError):
        EXACTGenome(
            node_genes=(_input_node(), _output_node()),
            edge_genes=(
                ConvolutionEdgeGene(
                    innovation_id=0,
                    source_node_id=0,
                    target_node_id=1,
                    is_enabled=True,
                    kernel_weights=wrong_kernel,
                ),
            ),
        )


def test_rejects_non_1x1_output_node_and_non_positive_filter_size() -> None:
    with pytest.raises(InvalidEXACTGenomeError):
        FilterNodeGene(
            node_id=1, node_type="output", filter_height=2, filter_width=1, depth=1.0
        )
    with pytest.raises(InvalidEXACTGenomeError):
        FilterNodeGene(
            node_id=2, node_type="hidden", filter_height=0, filter_width=3, depth=0.5
        )


def test_serialization_round_trip_preserves_everything() -> None:
    kernel = tuple(tuple(float(r * 10 + c) for c in range(10)) for r in range(10))
    genome = EXACTGenome(
        node_genes=(_input_node(), _output_node(), _hidden_node()),
        edge_genes=(
            ConvolutionEdgeGene(
                innovation_id=0, source_node_id=0, target_node_id=2, is_enabled=True
            ),
            ConvolutionEdgeGene(
                innovation_id=1,
                source_node_id=2,
                target_node_id=1,
                is_enabled=True,
                kernel_weights=kernel,
            ),
        ),
        is_trained=True,
    )
    rebuilt = EXACTGenome.from_serializable_dict(genome.to_serializable_dict())
    assert rebuilt == genome


def _example_hyperparameters() -> EXACTTrainingHyperparameters:
    return EXACTTrainingHyperparameters(
        learning_rate=0.0025,
        learning_rate_decay_factor=0.95,
        momentum=0.5,
        momentum_decay_factor=0.95,
        weight_decay=0.0005,
        weight_decay_decay_factor=0.95,
        velocity_reset_interval=1000,
        input_dropout_probability=0.001,
        hidden_dropout_probability=0.1,
        batch_size=50,
        batch_normalization_alpha=0.1,
    )


def test_genome_round_trips_batch_norm_state_and_hyperparameters() -> None:
    genome = EXACTGenome(
        node_genes=(
            FilterNodeGene(
                node_id=0, node_type="input", filter_height=2, filter_width=2, depth=0.0
            ),
            FilterNodeGene(
                node_id=1,
                node_type="output",
                filter_height=1,
                filter_width=1,
                depth=1.0,
            ),
            FilterNodeGene(
                node_id=2,
                node_type="hidden",
                filter_height=2,
                filter_width=2,
                depth=0.5,
                batch_normalization_state=(1.5, -0.5, 0.25, 0.9),
            ),
        ),
        edge_genes=(),
        is_trained=False,
        training_hyperparameters=_example_hyperparameters(),
    )
    restored = EXACTGenome.from_serializable_dict(genome.to_serializable_dict())
    assert restored == genome


def test_from_serializable_dict_tolerates_missing_new_keys() -> None:
    """Artifacts written before these fields existed must still load."""
    payload = {
        "node_genes": [
            {
                "node_id": 0,
                "node_type": "input",
                "filter_height": 2,
                "filter_width": 2,
                "depth": 0.0,
            }
        ],
        "edge_genes": [],
        "is_trained": False,
    }
    genome = EXACTGenome.from_serializable_dict(payload)
    assert genome.node_genes[0].batch_normalization_state is None
    assert genome.training_hyperparameters is None
