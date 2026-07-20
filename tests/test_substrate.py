from __future__ import annotations

from polyneat.algorithms.hyperneat.substrate import (
    build_grid_sandwich_substrate,
    build_layered_substrate,
)


def test_single_node_layer_is_centered():
    substrate = build_layered_substrate(
        input_layer_size=1,
        hidden_layer_sizes=(),
        output_layer_size=1,
        coordinate_range_min=-1.0,
        coordinate_range_max=1.0,
        include_bias_node=False,
    )
    (only_input_node,) = substrate.input_layer.nodes
    # a single node in a layer sits at the midpoint of the coordinate range
    assert only_input_node.x_coordinate == 0.0


def test_node_ids_are_contiguous_and_input_first():
    substrate = build_layered_substrate(
        input_layer_size=2,
        hidden_layer_sizes=(3,),
        output_layer_size=1,
        coordinate_range_min=-1.0,
        coordinate_range_max=1.0,
        include_bias_node=True,
    )
    all_ids = [node.node_id for node in substrate.all_nodes()]
    assert all_ids == list(range(len(all_ids)))
    # input nodes come first
    assert substrate.input_layer.nodes[0].node_id == 0
    assert substrate.input_layer.nodes[1].node_id == 1
    # bias node is last
    assert substrate.bias_node is not None
    assert substrate.bias_node.node_id == all_ids[-1]
    assert substrate.bias_node.role == "bias"


def test_layers_have_distinct_y_coordinates():
    substrate = build_layered_substrate(
        input_layer_size=2,
        hidden_layer_sizes=(2,),
        output_layer_size=2,
        coordinate_range_min=-1.0,
        coordinate_range_max=1.0,
        include_bias_node=False,
    )
    input_y = substrate.input_layer.nodes[0].y_coordinate
    hidden_y = substrate.hidden_layers[0].nodes[0].y_coordinate
    output_y = substrate.output_layer.nodes[0].y_coordinate
    assert input_y != hidden_y != output_y
    assert len({input_y, hidden_y, output_y}) == 3


def test_adjacent_layer_pairs_are_consecutive_and_feed_forward():
    substrate = build_layered_substrate(
        input_layer_size=2,
        hidden_layer_sizes=(3, 4),
        output_layer_size=1,
        coordinate_range_min=-1.0,
        coordinate_range_max=1.0,
        include_bias_node=False,
    )
    pairs = substrate.feed_forward_layer_adjacent_pairs()
    # input -> h1, h1 -> h2, h2 -> output == 3 adjacent pairs for 4 layers
    assert len(pairs) == 3
    source_first, target_first = pairs[0]
    assert source_first is substrate.input_layer
    assert target_first is substrate.hidden_layers[0]


def test_roles_are_assigned_per_layer():
    substrate = build_layered_substrate(
        input_layer_size=1,
        hidden_layer_sizes=(1,),
        output_layer_size=1,
        coordinate_range_min=-1.0,
        coordinate_range_max=1.0,
        include_bias_node=False,
    )
    assert substrate.input_layer.nodes[0].role == "input"
    assert substrate.hidden_layers[0].nodes[0].role == "hidden"
    assert substrate.output_layer.nodes[0].role == "output"


def test_grid_sandwich_has_two_grids_no_hidden_and_contiguous_ids():
    substrate = build_grid_sandwich_substrate(
        input_grid_shape=(2, 3),
        output_grid_shape=(2, 2),
        coordinate_range_min=-1.0,
        coordinate_range_max=1.0,
        include_bias_node=True,
        output_sheet_shares_input_plane=True,
    )
    assert len(substrate.input_layer.nodes) == 6
    assert len(substrate.output_layer.nodes) == 4
    assert substrate.hidden_layers == ()
    # input sheet ids row-major first, then output sheet, then bias last
    assert [node.node_id for node in substrate.input_layer.nodes] == [0, 1, 2, 3, 4, 5]
    assert [node.node_id for node in substrate.output_layer.nodes] == [6, 7, 8, 9]
    assert substrate.bias_node is not None
    assert substrate.bias_node.node_id == 10
    assert substrate.bias_node.role == "bias"


def test_grid_sandwich_shared_plane_puts_output_on_the_input_square():
    substrate = build_grid_sandwich_substrate(
        input_grid_shape=(3, 3),
        output_grid_shape=(3, 3),
        coordinate_range_min=-1.0,
        coordinate_range_max=1.0,
        include_bias_node=False,
        output_sheet_shares_input_plane=True,
    )
    input_y_coordinates = {node.y_coordinate for node in substrate.input_layer.nodes}
    output_y_coordinates = {node.y_coordinate for node in substrate.output_layer.nodes}
    assert input_y_coordinates == output_y_coordinates


def test_layered_bias_does_not_collide_with_any_node():
    # Regression: the bias used to be placed at (coord_min, coord_min), which is
    # exactly the first input node's coordinate for >= 2 inputs, so the CPPN
    # could not distinguish the bias source from that input.
    substrate = build_layered_substrate(
        input_layer_size=2,
        hidden_layer_sizes=(3,),
        output_layer_size=1,
        coordinate_range_min=-1.0,
        coordinate_range_max=1.0,
        include_bias_node=True,
    )
    assert substrate.bias_node is not None
    bias_coordinate = (substrate.bias_node.x_coordinate, substrate.bias_node.y_coordinate)
    non_bias_coordinates = [
        (node.x_coordinate, node.y_coordinate)
        for node in substrate.all_nodes()
        if node.role != "bias"
    ]
    assert bias_coordinate not in non_bias_coordinates


def test_grid_sandwich_bias_does_not_collide_with_any_node():
    substrate = build_grid_sandwich_substrate(
        input_grid_shape=(3, 3),
        output_grid_shape=(3, 3),
        coordinate_range_min=-1.0,
        coordinate_range_max=1.0,
        include_bias_node=True,
        output_sheet_shares_input_plane=True,
    )
    assert substrate.bias_node is not None
    bias_coordinate = (substrate.bias_node.x_coordinate, substrate.bias_node.y_coordinate)
    non_bias_coordinates = [
        (node.x_coordinate, node.y_coordinate)
        for node in substrate.all_nodes()
        if node.role != "bias"
    ]
    assert bias_coordinate not in non_bias_coordinates


def test_grid_sandwich_separated_lifts_output_above_the_input_plane():
    substrate = build_grid_sandwich_substrate(
        input_grid_shape=(4, 4),
        output_grid_shape=(1, 10),
        coordinate_range_min=-1.0,
        coordinate_range_max=1.0,
        include_bias_node=True,
        output_sheet_shares_input_plane=False,
    )
    assert len(substrate.input_layer.nodes) == 16
    assert len(substrate.output_layer.nodes) == 10
    highest_input_y = max(node.y_coordinate for node in substrate.input_layer.nodes)
    lowest_output_y = min(node.y_coordinate for node in substrate.output_layer.nodes)
    assert lowest_output_y > highest_input_y
