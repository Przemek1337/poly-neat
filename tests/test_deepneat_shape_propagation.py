from __future__ import annotations

from polyneat.algorithms.deepneat.deepneat_genome import (
    DeepNEATGenome,
    LayerNodeGene,
    TensorEdgeGene,
)
from polyneat.algorithms.deepneat.layer_shape_propagation import (
    TensorShape,
    compute_merge_strategy,
    propagate_tensor_shapes,
    prune_to_nodes_on_an_input_output_path,
)

_INPUT_SHAPE = TensorShape.spatial(channels=1, height=28, width=28)


def _edge(innovation_id: int, source: int, target: int, enabled: bool = True):
    return TensorEdgeGene(
        innovation_id=innovation_id,
        source_node_id=source,
        target_node_id=target,
        is_enabled=enabled,
    )


def test_spatial_and_flat_shapes_report_their_element_counts() -> None:
    assert TensorShape.spatial(channels=3, height=4, width=5).element_count == 60
    assert TensorShape.flat(features=17).element_count == 17


def test_pruning_keeps_only_nodes_on_a_path() -> None:
    genome = DeepNEATGenome(
        node_genes=(
            LayerNodeGene(node_id=0, layer_type="input"),
            LayerNodeGene(node_id=1, layer_type="output"),
            LayerNodeGene(node_id=2, layer_type="conv", number_of_filters=16, kernel_size=3),
            LayerNodeGene(node_id=3, layer_type="dense", number_of_units=64),
        ),
        edge_genes=(
            _edge(0, 0, 2),
            _edge(1, 2, 1),
            _edge(2, 0, 3),  # node 3 is a dead end: it reaches no output
        ),
    )
    assert prune_to_nodes_on_an_input_output_path(genome) == {0, 1, 2}


def test_pruning_returns_empty_when_no_path_exists() -> None:
    genome = DeepNEATGenome(
        node_genes=(
            LayerNodeGene(node_id=0, layer_type="input"),
            LayerNodeGene(node_id=1, layer_type="output"),
        ),
        edge_genes=(_edge(0, 0, 1, enabled=False),),
    )
    assert prune_to_nodes_on_an_input_output_path(genome) == set()


def test_conv_preserves_spatial_size_with_same_padding() -> None:
    genome = DeepNEATGenome(
        node_genes=(
            LayerNodeGene(node_id=0, layer_type="input"),
            LayerNodeGene(node_id=1, layer_type="output"),
            LayerNodeGene(node_id=2, layer_type="conv", number_of_filters=16, kernel_size=5),
        ),
        edge_genes=(_edge(0, 0, 2), _edge(1, 2, 1)),
    )
    shapes = propagate_tensor_shapes(
        genome,
        prune_to_nodes_on_an_input_output_path(genome),
        _INPUT_SHAPE,
        number_of_classes=10,
    )
    assert shapes[2] == TensorShape.spatial(channels=16, height=28, width=28)


def test_max_pooling_halves_the_spatial_size() -> None:
    genome = DeepNEATGenome(
        node_genes=(
            LayerNodeGene(node_id=0, layer_type="input"),
            LayerNodeGene(node_id=1, layer_type="output"),
            LayerNodeGene(
                node_id=2, layer_type="conv", number_of_filters=8, kernel_size=3,
                is_followed_by_max_pooling=True,
            ),
        ),
        edge_genes=(_edge(0, 0, 2), _edge(1, 2, 1)),
    )
    shapes = propagate_tensor_shapes(
        genome,
        prune_to_nodes_on_an_input_output_path(genome),
        _INPUT_SHAPE,
        number_of_classes=10,
    )
    assert shapes[2] == TensorShape.spatial(channels=8, height=14, width=14)


def test_max_pooling_is_skipped_when_the_map_is_too_small() -> None:
    genome = DeepNEATGenome(
        node_genes=(
            LayerNodeGene(node_id=0, layer_type="input"),
            LayerNodeGene(node_id=1, layer_type="output"),
            LayerNodeGene(
                node_id=2, layer_type="conv", number_of_filters=8, kernel_size=3,
                is_followed_by_max_pooling=True,
            ),
        ),
        edge_genes=(_edge(0, 0, 2), _edge(1, 2, 1)),
    )
    shapes = propagate_tensor_shapes(
        genome,
        prune_to_nodes_on_an_input_output_path(genome),
        TensorShape.spatial(channels=1, height=1, width=1),
        number_of_classes=10,
    )
    assert shapes[2] == TensorShape.spatial(channels=8, height=1, width=1)


def test_dense_layer_flattens_its_input() -> None:
    genome = DeepNEATGenome(
        node_genes=(
            LayerNodeGene(node_id=0, layer_type="input"),
            LayerNodeGene(node_id=1, layer_type="output"),
            LayerNodeGene(node_id=2, layer_type="dense", number_of_units=64),
        ),
        edge_genes=(_edge(0, 0, 2), _edge(1, 2, 1)),
    )
    shapes = propagate_tensor_shapes(
        genome,
        prune_to_nodes_on_an_input_output_path(genome),
        _INPUT_SHAPE,
        number_of_classes=10,
    )
    assert shapes[2] == TensorShape.flat(features=64)


def test_conv_after_dense_is_coerced_to_a_dense_layer() -> None:
    # Decision #4 in the spec: this keeps every DAG buildable, so the mutation
    # operators never have to reason about the graph.
    genome = DeepNEATGenome(
        node_genes=(
            LayerNodeGene(node_id=0, layer_type="input"),
            LayerNodeGene(node_id=1, layer_type="output"),
            LayerNodeGene(node_id=2, layer_type="dense", number_of_units=64),
            LayerNodeGene(node_id=3, layer_type="conv", number_of_filters=32, kernel_size=3),
        ),
        edge_genes=(_edge(0, 0, 2), _edge(1, 2, 3), _edge(2, 3, 1)),
    )
    shapes = propagate_tensor_shapes(
        genome,
        prune_to_nodes_on_an_input_output_path(genome),
        _INPUT_SHAPE,
        number_of_classes=10,
    )
    assert shapes[3] == TensorShape.flat(features=32)


def test_two_spatial_inputs_are_pooled_to_the_smaller_and_concatenated() -> None:
    strategy = compute_merge_strategy(
        incoming_shapes=[
            TensorShape.spatial(channels=8, height=28, width=28),
            TensorShape.spatial(channels=16, height=14, width=14),
        ],
        target_layer_type="conv",
    )
    assert not strategy.flatten_inputs
    assert (strategy.pooled_height, strategy.pooled_width) == (14, 14)
    assert strategy.merged_channels == 24


def test_a_flat_input_forces_everything_to_flatten() -> None:
    strategy = compute_merge_strategy(
        incoming_shapes=[
            TensorShape.spatial(channels=2, height=4, width=4),
            TensorShape.flat(features=10),
        ],
        target_layer_type="conv",
    )
    assert strategy.flatten_inputs
    assert strategy.merged_features == 2 * 4 * 4 + 10


def test_a_dense_target_forces_flattening_even_with_spatial_inputs() -> None:
    strategy = compute_merge_strategy(
        incoming_shapes=[TensorShape.spatial(channels=2, height=4, width=4)],
        target_layer_type="dense",
    )
    assert strategy.flatten_inputs
    assert strategy.merged_features == 32


def test_output_target_forces_flattening() -> None:
    strategy = compute_merge_strategy(
        incoming_shapes=[TensorShape.spatial(channels=3, height=2, width=2)],
        target_layer_type="output",
    )
    assert strategy.flatten_inputs
    assert strategy.merged_features == 12


def test_skip_connection_merges_at_the_output() -> None:
    genome = DeepNEATGenome(
        node_genes=(
            LayerNodeGene(node_id=0, layer_type="input"),
            LayerNodeGene(node_id=1, layer_type="output"),
            LayerNodeGene(
                node_id=2, layer_type="conv", number_of_filters=4, kernel_size=3,
                is_followed_by_max_pooling=True,
            ),
        ),
        edge_genes=(_edge(0, 0, 2), _edge(1, 2, 1), _edge(2, 0, 1)),
    )
    shapes = propagate_tensor_shapes(
        genome,
        prune_to_nodes_on_an_input_output_path(genome),
        _INPUT_SHAPE,
        number_of_classes=10,
    )
    # output merges conv (4x14x14 = 784) with the raw input (1x28x28 = 784)
    assert shapes[1].is_spatial is False
