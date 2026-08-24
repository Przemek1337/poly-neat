from __future__ import annotations

import pytest

from polyneat.algorithms.deepneat.deepneat_genome import (
    DeepNEATGenome,
    InvalidDeepNEATGenomeError,
    LayerNodeGene,
    TensorEdgeGene,
)

_INPUT = LayerNodeGene(node_id=0, layer_type="input")
_OUTPUT = LayerNodeGene(node_id=1, layer_type="output")
_CONV = LayerNodeGene(
    node_id=2,
    layer_type="conv",
    number_of_filters=32,
    kernel_size=3,
    dropout_rate=0.25,
    uses_batch_normalization=True,
    is_followed_by_max_pooling=True,
)
_DENSE = LayerNodeGene(node_id=3, layer_type="dense", number_of_units=128, dropout_rate=0.1)


def _minimal_genome() -> DeepNEATGenome:
    return DeepNEATGenome(
        node_genes=(_INPUT, _OUTPUT),
        edge_genes=(
            TensorEdgeGene(innovation_id=0, source_node_id=0, target_node_id=1,
                           is_enabled=True),
        ),
    )


def test_minimal_genome_is_valid() -> None:
    genome = _minimal_genome()
    assert len(genome.node_genes) == 2
    assert genome.is_output_reachable_from_input()


def test_conv_node_requires_filters_and_kernel_size() -> None:
    with pytest.raises(InvalidDeepNEATGenomeError):
        LayerNodeGene(node_id=5, layer_type="conv", kernel_size=3)
    with pytest.raises(InvalidDeepNEATGenomeError):
        LayerNodeGene(node_id=5, layer_type="conv", number_of_filters=16)


def test_conv_node_must_not_carry_dense_units() -> None:
    with pytest.raises(InvalidDeepNEATGenomeError):
        LayerNodeGene(
            node_id=5, layer_type="conv", number_of_filters=16, kernel_size=3,
            number_of_units=64,
        )


def test_dense_node_requires_units_and_forbids_conv_fields() -> None:
    with pytest.raises(InvalidDeepNEATGenomeError):
        LayerNodeGene(node_id=5, layer_type="dense")
    with pytest.raises(InvalidDeepNEATGenomeError):
        LayerNodeGene(node_id=5, layer_type="dense", number_of_units=64, kernel_size=3)


def test_input_and_output_nodes_carry_no_hyperparameters() -> None:
    with pytest.raises(InvalidDeepNEATGenomeError):
        LayerNodeGene(node_id=5, layer_type="input", number_of_units=64)
    with pytest.raises(InvalidDeepNEATGenomeError):
        LayerNodeGene(node_id=5, layer_type="output", dropout_rate=0.3)


def test_unknown_layer_type_is_rejected() -> None:
    with pytest.raises(InvalidDeepNEATGenomeError):
        LayerNodeGene(node_id=5, layer_type="lstm")


def test_duplicate_node_ids_are_rejected() -> None:
    with pytest.raises(InvalidDeepNEATGenomeError):
        DeepNEATGenome(
            node_genes=(_INPUT, LayerNodeGene(node_id=0, layer_type="output")),
            edge_genes=(),
        )


def test_duplicate_innovation_ids_are_rejected() -> None:
    with pytest.raises(InvalidDeepNEATGenomeError):
        DeepNEATGenome(
            node_genes=(_INPUT, _OUTPUT, _CONV),
            edge_genes=(
                TensorEdgeGene(innovation_id=0, source_node_id=0, target_node_id=2,
                               is_enabled=True),
                TensorEdgeGene(innovation_id=0, source_node_id=2, target_node_id=1,
                               is_enabled=True),
            ),
        )


def test_dangling_edge_endpoint_is_rejected() -> None:
    with pytest.raises(InvalidDeepNEATGenomeError):
        DeepNEATGenome(
            node_genes=(_INPUT, _OUTPUT),
            edge_genes=(
                TensorEdgeGene(innovation_id=0, source_node_id=0, target_node_id=99,
                               is_enabled=True),
            ),
        )


def test_exactly_one_input_and_one_output_are_required() -> None:
    with pytest.raises(InvalidDeepNEATGenomeError):
        DeepNEATGenome(node_genes=(_INPUT,), edge_genes=())
    with pytest.raises(InvalidDeepNEATGenomeError):
        DeepNEATGenome(
            node_genes=(_INPUT, _OUTPUT, LayerNodeGene(node_id=4, layer_type="output")),
            edge_genes=(),
        )


def test_cycle_among_enabled_edges_is_rejected() -> None:
    with pytest.raises(InvalidDeepNEATGenomeError):
        DeepNEATGenome(
            node_genes=(_INPUT, _OUTPUT, _CONV, _DENSE),
            edge_genes=(
                TensorEdgeGene(innovation_id=0, source_node_id=2, target_node_id=3,
                               is_enabled=True),
                TensorEdgeGene(innovation_id=1, source_node_id=3, target_node_id=2,
                               is_enabled=True),
            ),
        )


def test_cycle_among_disabled_edges_is_allowed() -> None:
    genome = DeepNEATGenome(
        node_genes=(_INPUT, _OUTPUT, _CONV, _DENSE),
        edge_genes=(
            TensorEdgeGene(innovation_id=0, source_node_id=2, target_node_id=3,
                           is_enabled=True),
            TensorEdgeGene(innovation_id=1, source_node_id=3, target_node_id=2,
                           is_enabled=False),
        ),
    )
    assert len(genome.edge_genes) == 2


def test_output_unreachable_when_the_only_path_is_disabled() -> None:
    genome = DeepNEATGenome(
        node_genes=(_INPUT, _OUTPUT),
        edge_genes=(
            TensorEdgeGene(innovation_id=0, source_node_id=0, target_node_id=1,
                           is_enabled=False),
        ),
    )
    assert not genome.is_output_reachable_from_input()


def test_output_reachable_through_a_hidden_layer() -> None:
    genome = DeepNEATGenome(
        node_genes=(_INPUT, _OUTPUT, _CONV),
        edge_genes=(
            TensorEdgeGene(innovation_id=0, source_node_id=0, target_node_id=2,
                           is_enabled=True),
            TensorEdgeGene(innovation_id=1, source_node_id=2, target_node_id=1,
                           is_enabled=True),
        ),
    )
    assert genome.is_output_reachable_from_input()


def test_get_node_gene_by_id_returns_none_for_unknown_ids() -> None:
    genome = _minimal_genome()
    assert genome.get_node_gene_by_id(0) is _INPUT
    assert genome.get_node_gene_by_id(77) is None


def test_clone_returns_an_equal_but_distinct_object() -> None:
    genome = _minimal_genome()
    clone = genome.clone_genome()
    assert clone is not genome
    assert clone.node_genes == genome.node_genes
    assert clone.edge_genes == genome.edge_genes


def test_serialization_round_trip_preserves_everything() -> None:
    genome = DeepNEATGenome(
        node_genes=(_INPUT, _OUTPUT, _CONV, _DENSE),
        edge_genes=(
            TensorEdgeGene(innovation_id=0, source_node_id=0, target_node_id=2,
                           is_enabled=True),
            TensorEdgeGene(innovation_id=1, source_node_id=2, target_node_id=3,
                           is_enabled=True),
            TensorEdgeGene(innovation_id=2, source_node_id=3, target_node_id=1,
                           is_enabled=True),
            TensorEdgeGene(innovation_id=3, source_node_id=0, target_node_id=1,
                           is_enabled=False),
        ),
    )
    restored = DeepNEATGenome.from_serializable_dict(genome.to_serializable_dict())
    assert restored.node_genes == genome.node_genes
    assert restored.edge_genes == genome.edge_genes


def test_serializable_dict_is_json_friendly() -> None:
    import json

    json.dumps(_minimal_genome().to_serializable_dict())
