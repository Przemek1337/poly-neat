from __future__ import annotations

import pytest

from polyneat.algorithms.hyperneat.substrate import (
    build_substrate_from_explicit_layer_coordinates,
)

# Two clusters of four, separated by a gap. See the design doc: an even spread
# cannot separate the hemispheres, because within-hemisphere distances (up to
# 0.857) overlap cross-hemisphere ones (from 0.286).
_RETINA_LAYERS = (
    (-1.00, -0.85, -0.70, -0.55, 0.55, 0.70, 0.85, 1.00),
    (-1.00, -0.85, -0.70, -0.55, 0.55, 0.70, 0.85, 1.00),
    (-0.775, 0.775),
)


def _retina_substrate():
    return build_substrate_from_explicit_layer_coordinates(
        layer_x_coordinates=_RETINA_LAYERS,
        coordinate_range_min=-1.0,
        coordinate_range_max=1.0,
        bias_x_coordinate=0.0,
    )


def test_layer_sizes_follow_the_given_coordinates() -> None:
    substrate = _retina_substrate()
    assert len(substrate.input_layer.nodes) == 8
    assert len(substrate.hidden_layers) == 1
    assert len(substrate.hidden_layers[0].nodes) == 8
    assert len(substrate.output_layer.nodes) == 2


def test_x_coordinates_are_taken_verbatim() -> None:
    substrate = _retina_substrate()
    assert tuple(n.x_coordinate for n in substrate.input_layer.nodes) == _RETINA_LAYERS[0]
    assert tuple(n.x_coordinate for n in substrate.output_layer.nodes) == _RETINA_LAYERS[2]


def test_node_ids_are_contiguous_from_zero_in_registration_order() -> None:
    substrate = _retina_substrate()
    assert [node.node_id for node in substrate.all_nodes()] == list(range(19))


def test_roles_are_assigned_by_layer_position() -> None:
    substrate = _retina_substrate()
    assert all(n.role == "input" for n in substrate.input_layer.nodes)
    assert all(n.role == "hidden" for n in substrate.hidden_layers[0].nodes)
    assert all(n.role == "output" for n in substrate.output_layer.nodes)
    assert substrate.bias_node is not None
    assert substrate.bias_node.role == "bias"


def test_bias_sits_at_the_requested_x_and_below_the_input_layer() -> None:
    substrate = _retina_substrate()
    assert substrate.bias_node is not None
    assert substrate.bias_node.x_coordinate == 0.0
    assert substrate.bias_node.y_coordinate < substrate.input_layer.nodes[0].y_coordinate


def test_bias_can_be_omitted() -> None:
    substrate = build_substrate_from_explicit_layer_coordinates(
        layer_x_coordinates=((-1.0, 1.0), (-1.0, 1.0)),
        coordinate_range_min=-1.0,
        coordinate_range_max=1.0,
        bias_x_coordinate=None,
    )
    assert substrate.bias_node is None


def test_layers_get_distinct_y_coordinates_in_order() -> None:
    substrate = _retina_substrate()
    layer_y_values = [layer.nodes[0].y_coordinate for layer in substrate.ordered_layers()]
    assert layer_y_values == sorted(layer_y_values)
    assert len(set(layer_y_values)) == 3


def test_two_layer_substrate_has_no_hidden_layers() -> None:
    substrate = build_substrate_from_explicit_layer_coordinates(
        layer_x_coordinates=((-1.0, 1.0), (0.0,)),
        coordinate_range_min=-1.0,
        coordinate_range_max=1.0,
        bias_x_coordinate=None,
    )
    assert substrate.hidden_layers == ()
    assert len(substrate.output_layer.nodes) == 1


def test_fewer_than_two_layers_is_rejected() -> None:
    with pytest.raises(ValueError):
        build_substrate_from_explicit_layer_coordinates(
            layer_x_coordinates=((-1.0, 1.0),),
            coordinate_range_min=-1.0,
            coordinate_range_max=1.0,
            bias_x_coordinate=None,
        )


def test_empty_layer_is_rejected() -> None:
    with pytest.raises(ValueError):
        build_substrate_from_explicit_layer_coordinates(
            layer_x_coordinates=((-1.0, 1.0), ()),
            coordinate_range_min=-1.0,
            coordinate_range_max=1.0,
            bias_x_coordinate=None,
        )


def test_hemispheres_are_separated_by_a_gap_wider_than_within_hemisphere_spread() -> None:
    # The property the locality seed depends on: every cross-hemisphere distance
    # must exceed every within-hemisphere distance, or no threshold separates
    # them. This is exactly what an even spread fails to provide.
    substrate = _retina_substrate()
    xs = [node.x_coordinate for node in substrate.input_layer.nodes]
    within = [abs(a - b) for a in xs for b in xs if a * b > 0]
    across = [abs(a - b) for a in xs for b in xs if a * b < 0]
    assert max(within) < min(across)


def test_the_seeded_gaussian_threshold_falls_inside_that_gap() -> None:
    # LEO = exp(-(w*dx)^2) + b with w=1, b=-0.5 expresses while |dx| < 0.8326.
    import math

    cutoff = math.sqrt(math.log(2.0))
    substrate = _retina_substrate()
    xs = [node.x_coordinate for node in substrate.input_layer.nodes]
    within = [abs(a - b) for a in xs for b in xs if a * b > 0]
    across = [abs(a - b) for a in xs for b in xs if a * b < 0]
    assert max(within) < cutoff < min(across)
