"""Substrate value types and builders for HyperNEAT.

A *substrate* is the fixed neural network whose connection weights a CPPN paints
as a function of node geometry. This module holds the immutable substrate value
types (`SubstrateNode`, `SubstrateLayer`, `Substrate`) and two ready-made
builders: a 1-D layered stack (`build_layered_substrate`) and a 2-D two-sheet
"sandwich" (`build_grid_sandwich_substrate`).

References:
    Stanley, K. O., D'Ambrosio, D. B., & Gauci, J. (2009). A Hypercube-Based Encoding for
        Evolving Large-Scale Neural Networks. *Artificial Life*, 15(2), 185-212.
        DOI: 10.1162/artl.2009.15.2.15202
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

SubstrateNodeRole = Literal["input", "hidden", "output", "bias"]


@dataclass(frozen=True)
class SubstrateNode:
    """A node in the substrate with fixed coordinates and a fixed role.

    Attributes:
        node_id: Unique id of the node within its substrate.
        x_coordinate: Horizontal position, fed to the CPPN when querying any
            connection touching this node.
        y_coordinate: Vertical position, fed to the CPPN alongside
            ``x_coordinate``.
        role: Whether the node is an ``input``, ``hidden``, ``output``, or
            ``bias`` node.
    """

    node_id: int
    x_coordinate: float
    y_coordinate: float
    role: SubstrateNodeRole


@dataclass(frozen=True)
class SubstrateLayer:
    """A single sheet of substrate nodes.

    Attributes:
        nodes: The nodes that make up this layer, in registration order.
    """

    nodes: tuple[SubstrateNode, ...]


@dataclass(frozen=True)
class Substrate:
    """A layered feed-forward substrate: input sheet -> hidden sheets -> output sheet.

    Restricting to a layered feed-forward layout (rather than the paper's fully
    general 4-D hypercube, which permits recurrent substrate connections) lets
    the substrate ANN run on PolyNEAT's existing feed-forward executor. The
    optional bias node has a fixed coordinate and projects to every non-input
    node, supplying per-node bias in the substrate.

    Attributes:
        input_layer: The sheet that receives the task inputs.
        hidden_layers: Zero or more sheets between input and output, in order.
        output_layer: The sheet whose activations are the task outputs.
        bias_node: An optional constant-source node, or ``None`` when the
            substrate has no bias.
    """

    input_layer: SubstrateLayer
    hidden_layers: tuple[SubstrateLayer, ...]
    output_layer: SubstrateLayer
    bias_node: SubstrateNode | None

    def ordered_layers(self) -> tuple[SubstrateLayer, ...]:
        """Return the layers from input to output.

        Returns:
            The input layer, then each hidden layer in order, then the output
            layer. The bias node is not included.
        """
        return (self.input_layer, *self.hidden_layers, self.output_layer)

    def all_nodes(self) -> tuple[SubstrateNode, ...]:
        """Return every node in the substrate in registration order.

        Returns:
            All layer nodes (input first, then hidden, then output), followed by
            the bias node if one is present.
        """
        nodes: list[SubstrateNode] = []
        for layer in self.ordered_layers():
            nodes.extend(layer.nodes)
        if self.bias_node is not None:
            nodes.append(self.bias_node)
        return tuple(nodes)

    def feed_forward_layer_adjacent_pairs(
        self,
    ) -> list[tuple[SubstrateLayer, SubstrateLayer]]:
        """Return each adjacent ``(source_layer, target_layer)`` pair.

        Returns:
            Consecutive layer pairs from input to output, e.g. ``[(input, h1),
            (h1, output)]`` for one hidden layer. These are the only pairs a
            feed-forward decoder queries the CPPN for.
        """
        layers = self.ordered_layers()
        return [
            (layers[layer_index], layers[layer_index + 1]) for layer_index in range(len(layers) - 1)
        ]


def _evenly_spaced_coordinates(node_count: int, range_min: float, range_max: float) -> list[float]:
    """Return ``node_count`` coordinates spread across ``[range_min, range_max]``.

    Args:
        node_count: How many coordinates to produce.
        range_min: Lower bound of the closed interval.
        range_max: Upper bound of the closed interval.

    Returns:
        A list of ``node_count`` coordinates. A single node is placed at the
        midpoint of the interval; two or more are spaced evenly with the
        endpoints included.
    """
    if node_count == 1:
        return [(range_min + range_max) / 2.0]
    step = (range_max - range_min) / (node_count - 1)
    return [range_min + step * index for index in range(node_count)]


def build_layered_substrate(
    input_layer_size: int,
    hidden_layer_sizes: tuple[int, ...],
    output_layer_size: int,
    coordinate_range_min: float,
    coordinate_range_max: float,
    include_bias_node: bool,
) -> Substrate:
    """Build a 1-D layered substrate, assigning ids input-first then bias-last.

    Each layer gets a distinct y coordinate (layers spread evenly across the
    coordinate range); nodes within a layer spread evenly across x. Node ids are
    contiguous starting at 0 in registration order (all input nodes, then each
    hidden layer, then output, then the bias node) so that the substrate ANN's
    input registration order matches the task input column order.

    Args:
        input_layer_size: Number of input nodes.
        hidden_layer_sizes: Size of each hidden layer, in order; empty for none.
        output_layer_size: Number of output nodes.
        coordinate_range_min: Lower bound of the coordinate range on both axes.
        coordinate_range_max: Upper bound of the coordinate range on both axes.
        include_bias_node: Whether to add a bias node.

    Returns:
        The assembled `Substrate`. When ``include_bias_node`` is True the bias
        sits in a band just below the input layer (at ``y = coordinate_range_min
        - 0.25 * span``) so it never coincides with an input node's coordinate.
    """
    layer_sizes = [input_layer_size, *hidden_layer_sizes, output_layer_size]
    layer_y_coordinates = _evenly_spaced_coordinates(
        len(layer_sizes), coordinate_range_min, coordinate_range_max
    )

    next_node_id = 0
    built_layers: list[SubstrateLayer] = []
    for layer_index, layer_size in enumerate(layer_sizes):
        if layer_index == 0:
            role = "input"
        elif layer_index == len(layer_sizes) - 1:
            role = "output"
        else:
            role = "hidden"
        x_coordinates = _evenly_spaced_coordinates(
            layer_size, coordinate_range_min, coordinate_range_max
        )
        layer_nodes: list[SubstrateNode] = []
        for x_coordinate in x_coordinates:
            layer_nodes.append(
                SubstrateNode(
                    node_id=next_node_id,
                    x_coordinate=x_coordinate,
                    y_coordinate=layer_y_coordinates[layer_index],
                    role=role,
                )
            )
            next_node_id += 1
        built_layers.append(SubstrateLayer(nodes=tuple(layer_nodes)))

    bias_node: SubstrateNode | None = None
    if include_bias_node:
        bias_band_offset = 0.25 * (coordinate_range_max - coordinate_range_min)
        bias_node = SubstrateNode(
            node_id=next_node_id,
            x_coordinate=coordinate_range_min,
            y_coordinate=coordinate_range_min - bias_band_offset,
            role="bias",
        )

    return Substrate(
        input_layer=built_layers[0],
        hidden_layers=tuple(built_layers[1:-1]),
        output_layer=built_layers[-1],
        bias_node=bias_node,
    )


def build_grid_sandwich_substrate(
    input_grid_shape: tuple[int, int],
    output_grid_shape: tuple[int, int],
    coordinate_range_min: float,
    coordinate_range_max: float,
    include_bias_node: bool,
    output_sheet_shares_input_plane: bool = True,
) -> Substrate:
    """Build a two-sheet "sandwich" substrate: a 2-D input grid wired to a 2-D output grid.

    There is no hidden layer, matching the paper's state-space sandwich
    (Stanley et al. 2009, figure 6c). Each ``grid_shape`` is ``(rows,
    columns)``; nodes are laid out row-major so their ids match a row-major
    flatten of the corresponding field. Node ids are contiguous starting at 0:
    all input-sheet nodes, then all output-sheet nodes, then the bias node.

    Args:
        input_grid_shape: ``(rows, columns)`` of the input sheet.
        output_grid_shape: ``(rows, columns)`` of the output sheet.
        coordinate_range_min: Lower bound of the coordinate range on both axes.
        coordinate_range_max: Upper bound of the coordinate range on both axes.
        include_bias_node: Whether to add a bias node (placed below the input
            plane so it never coincides with a sheet node).
        output_sheet_shares_input_plane: When True the output grid occupies the
            same ``[coordinate_range_min, coordinate_range_max]^2`` square as the
            input, so nearby source and target coordinates are spatially local.
            When False the output sheet is lifted into a coordinate band above
            the input plane, letting the CPPN distinguish input positions from
            output positions.

    Returns:
        The assembled two-sheet `Substrate`.

    Raises:
        ValueError: If either grid shape has a non-positive number of rows or
            columns.
    """
    for grid_shape_name, grid_shape in (
        ("input_grid_shape", input_grid_shape),
        ("output_grid_shape", output_grid_shape),
    ):
        if grid_shape[0] < 1 or grid_shape[1] < 1:
            raise ValueError(
                f"{grid_shape_name} must have positive rows and columns, got {grid_shape}"
            )

    coordinate_span = coordinate_range_max - coordinate_range_min
    separated_band_gap = 0.25 * coordinate_span

    next_node_id = 0

    def _build_grid_layer(
        grid_shape: tuple[int, int],
        role: SubstrateNodeRole,
        row_y_coordinates: list[float],
    ) -> SubstrateLayer:
        """Build one row-major grid layer, advancing the shared id counter.

        Args:
            grid_shape: ``(rows, columns)`` of this sheet.
            role: Role assigned to every node in the sheet.
            row_y_coordinates: Y coordinate for each row of the grid.

        Returns:
            The assembled `SubstrateLayer`.
        """
        nonlocal next_node_id
        number_of_rows, number_of_columns = grid_shape
        column_x_coordinates = _evenly_spaced_coordinates(
            number_of_columns, coordinate_range_min, coordinate_range_max
        )
        grid_nodes: list[SubstrateNode] = []
        for row_index in range(number_of_rows):
            for column_index in range(number_of_columns):
                grid_nodes.append(
                    SubstrateNode(
                        node_id=next_node_id,
                        x_coordinate=column_x_coordinates[column_index],
                        y_coordinate=row_y_coordinates[row_index],
                        role=role,
                    )
                )
                next_node_id += 1
        return SubstrateLayer(nodes=tuple(grid_nodes))

    input_row_y_coordinates = _evenly_spaced_coordinates(
        input_grid_shape[0], coordinate_range_min, coordinate_range_max
    )
    input_layer = _build_grid_layer(input_grid_shape, "input", input_row_y_coordinates)

    output_row_y_coordinates = _evenly_spaced_coordinates(
        output_grid_shape[0], coordinate_range_min, coordinate_range_max
    )
    if not output_sheet_shares_input_plane:
        vertical_lift = coordinate_span + separated_band_gap
        output_row_y_coordinates = [
            y_coordinate + vertical_lift for y_coordinate in output_row_y_coordinates
        ]
    output_layer = _build_grid_layer(output_grid_shape, "output", output_row_y_coordinates)

    bias_node: SubstrateNode | None = None
    if include_bias_node:
        bias_node = SubstrateNode(
            node_id=next_node_id,
            x_coordinate=coordinate_range_min,
            y_coordinate=coordinate_range_min - separated_band_gap,
            role="bias",
        )

    return Substrate(
        input_layer=input_layer,
        hidden_layers=(),
        output_layer=output_layer,
        bias_node=bias_node,
    )
