"""Describe the architecture every finished run selected: layers, connections, parameters, metrics.

Invoke as a module from the repository root::

    uv run python -m benchmarks.describe_winners [--results-root benchmarks/results]
    uv run python -m benchmarks.describe_winners --checkpoint path/to/model.pt [...]

The first form walks every series in :data:`benchmarks.summarize_series.SERIES`,
reads each run's ``run_report.json`` and, when the run's ``checkpoints``
directory is present, decodes the track A checkpoint's genome. Track B retrains
the same topology from scratch, so one diagram per run covers both tracks. The
second form describes standalone checkpoint files without a report.

For every model it writes, under the output directory:

- ``<series>/<method>_seed<N>.json`` - layer table, connections, parameter
  counts and the metrics from the report;
- ``<series>/<method>_seed<N>.pdf`` / ``.png`` - a layered drawing of the
  network, rendered with matplotlib;
- ``<series>/<method>_seed<N>.dot`` - the same graph in Graphviz form, for
  ``dot -Tpdf`` when a typeset figure is wanted;

and one ``winners.md`` with a summary table per series and a layer table per
model. The description is derived the same way the phenotypes are built:
DeepNEAT genomes are pruned to the nodes on an enabled input-to-output path and
shapes are propagated with the library's own rules, so a max-pooling flag the
phenotype skips (map smaller than 2x2) is reported as skipped, not as applied.
Parameter counts come from the checkpoint's tensors (buffers excluded) and are
checked against the count the report recorded.

Nothing here reads or writes a lock, and this module sits outside the digested
implementation, so running it cannot invalidate a frozen series.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

from benchmarks.summarize_series import SERIES, SeriesSpec
from polyneat.algorithms.deepneat.deepneat_genome import DeepNEATGenome
from polyneat.algorithms.deepneat.layer_shape_propagation import (
    TensorShape,
    compute_merge_strategy,
    propagate_tensor_shapes,
    prune_to_nodes_on_an_input_output_path,
)
from polyneat.algorithms.exact.exact_genome import EXACTGenome, compute_convolution_kernel_size
from polyneat.logging_utils.custom_logger import get_logger
from polyneat.nn.topology_utilities import compute_topological_order_of_node_ids
from polyneat.training.model_checkpoint import ModelCheckpoint

logger = get_logger(__name__)

_DEFAULT_RESULTS_ROOT = Path(__file__).parent / "results"
_BUFFER_SUFFIXES = ("running_mean", "running_var", "num_batches_tracked")
_DEEPNEAT_MODULE_PREFIX = "_layer_modules_by_node_id."
_EXACT_KERNEL_PREFIX = "_phenotype._kernel_parameters."
_EXACT_BATCH_NORMALIZATION_PREFIX = "_phenotype._batch_normalization_by_node_id."


@dataclass
class LayerRow:
    """One node of the decoded network, as it executes.

    Attributes:
        node_id: Node id in the genome (a running index for the fixed CNN).
        layer: ``input``, ``conv``, ``dense``, ``output``, ``hidden`` (EXACT
            filter) or a fixed-CNN block name.
        filters: Output channels of a convolution, when one applies.
        kernel: Kernel size as ``HxW``, or a comma-separated list for an EXACT
            node with several incoming kernels.
        units: Width of a dense layer.
        uses_batch_normalization: Whether batch normalization follows the layer.
        dropout_rate: Dropout probability after the activation, 0 for none.
        max_pooling: ``applied``, ``skipped`` (flagged but the map is already
            smaller than 2x2) or ``none``.
        inputs: Ids of the nodes feeding this one.
        merge: How several inputs are combined, e.g. ``concat`` or
            ``flatten``; empty for a single spatial input.
        output_shape: Shape the node produces, batch excluded.
        parameters: Trainable scalars this node owns.
    """

    node_id: int
    layer: str
    filters: int | None = None
    kernel: str | None = None
    units: int | None = None
    uses_batch_normalization: bool = False
    dropout_rate: float = 0.0
    max_pooling: str = "none"
    inputs: list[int] = field(default_factory=list)
    merge: str = ""
    output_shape: str = ""
    parameters: int = 0


@dataclass
class ArchitectureDescription:
    """Everything known about one frozen model's architecture.

    Attributes:
        genome_kind: Class name of the genome the checkpoint was decoded from.
        input_shape: Input tensor shape, batch excluded.
        layers: Executed nodes in topological (or depth) order.
        edges: Executed connections as ``(source, target)`` node ids.
        genome_node_count: Nodes in the genome, including pruned ones.
        genome_enabled_edge_count: Enabled edges in the genome.
        genome_disabled_edge_count: Disabled edges in the genome.
        total_parameters: Trainable scalars, counted from the checkpoint.
        notes: Method-specific remarks, e.g. the activation EXACT uses.
    """

    genome_kind: str
    input_shape: str
    layers: list[LayerRow]
    edges: list[tuple[int, int]]
    genome_node_count: int
    genome_enabled_edge_count: int
    genome_disabled_edge_count: int
    total_parameters: int
    notes: list[str] = field(default_factory=list)

    def count_layers(self, layer: str) -> int:
        """Number of executed nodes of one layer type."""
        return sum(1 for row in self.layers if row.layer == layer)

    @property
    def max_pooling_count(self) -> int:
        """Max-pooling operations that actually run."""
        return sum(1 for row in self.layers if row.max_pooling == "applied")

    @property
    def dropout_count(self) -> int:
        """Layers followed by a non-zero dropout."""
        return sum(1 for row in self.layers if row.dropout_rate > 0.0)

    @property
    def batch_normalization_count(self) -> int:
        """Layers followed by batch normalization."""
        return sum(1 for row in self.layers if row.uses_batch_normalization)


def _is_parameter(tensor_name: str) -> bool:
    return not tensor_name.endswith(_BUFFER_SUFFIXES)


def _format_shape(shape: TensorShape) -> str:
    if shape.is_spatial:
        return f"{shape.channels}x{shape.height}x{shape.width}"
    return str(shape.features)


def _parameters_by_key(model_state: dict, prefix: str) -> dict[str, int]:
    """Sum trainable scalars per first key segment after ``prefix``."""
    counts: dict[str, int] = defaultdict(int)
    for tensor_name, tensor in model_state.items():
        if tensor_name.startswith(prefix) and _is_parameter(tensor_name):
            counts[tensor_name[len(prefix) :].split(".", 1)[0]] += int(tensor.numel())
    return counts


def count_trainable_parameters(model_state: dict) -> int:
    """Trainable scalars in a state dict; normalization statistics are buffers."""
    return sum(int(tensor.numel()) for name, tensor in model_state.items() if _is_parameter(name))


def _infer_deepneat_input_channels(
    genome: DeepNEATGenome, model_state: dict, input_side: int
) -> int:
    """Read the input channel count off a layer fed by the input node alone.

    A layer with several inputs concatenates them, so its weight reflects the
    sum of their channels and says nothing about the input's own.
    """
    for edge in genome.edge_genes:
        if not edge.is_enabled or edge.source_node_id != genome.input_node_id:
            continue
        target = genome.get_node_gene_by_id(edge.target_node_id)
        incoming = sum(
            1
            for other in genome.edge_genes
            if other.is_enabled and other.target_node_id == target.node_id
        )
        if incoming != 1:
            continue
        weight_name = (
            f"{_DEEPNEAT_MODULE_PREFIX}{target.node_id}.0.weight"
            if target.layer_type in ("conv", "dense")
            else f"{_DEEPNEAT_MODULE_PREFIX}{target.node_id}.weight"
        )
        weight = model_state.get(weight_name)
        if weight is None:
            continue
        if target.layer_type == "conv":
            return int(weight.shape[1])
        return int(weight.shape[1]) // (input_side * input_side)
    return 1


def describe_deepneat(
    genome_payload: dict,
    model_state: dict,
    *,
    input_side: int,
    input_channels: int | None = None,
    number_of_classes: int | None = None,
) -> ArchitectureDescription:
    """Decode a DeepNEAT genome into the network its phenotype executes.

    Args:
        genome_payload: ``DeepNEATGenome.to_serializable_dict()`` output.
        model_state: The checkpoint's tensors.
        input_side: Side of the square input image.
        input_channels: Input channels; inferred from the weights when ``None``.
        number_of_classes: Output width; inferred from the weights when ``None``.

    Returns:
        The executed layers, connections and parameter counts.
    """
    genome = DeepNEATGenome.from_serializable_dict(genome_payload)
    if input_channels is None:
        input_channels = _infer_deepneat_input_channels(genome, model_state, input_side)
    if number_of_classes is None:
        output_bias = model_state.get(f"{_DEEPNEAT_MODULE_PREFIX}{genome.output_node_id}.bias")
        number_of_classes = int(output_bias.shape[0]) if output_bias is not None else 2
    cropped_image_size = genome.global_hyperparameters.cropped_image_size
    side = cropped_image_size if cropped_image_size > 0 else input_side
    input_shape = TensorShape.spatial(channels=input_channels, height=side, width=side)

    retained_node_ids = prune_to_nodes_on_an_input_output_path(genome)
    shapes_by_node_id = propagate_tensor_shapes(
        genome=genome,
        retained_node_ids=retained_node_ids,
        input_shape=input_shape,
        number_of_classes=number_of_classes,
    )
    edges = [
        (edge.source_node_id, edge.target_node_id)
        for edge in genome.edge_genes
        if edge.is_enabled
        and edge.source_node_id in retained_node_ids
        and edge.target_node_id in retained_node_ids
    ]
    incoming_by_target: dict[int, list[int]] = defaultdict(list)
    for source_node_id, target_node_id in edges:
        incoming_by_target[target_node_id].append(source_node_id)
    parameters_by_node = _parameters_by_key(model_state, _DEEPNEAT_MODULE_PREFIX)

    layers: list[LayerRow] = []
    for node_id in compute_topological_order_of_node_ids(
        all_node_ids=retained_node_ids, enabled_directed_edges=edges
    ):
        node = genome.get_node_gene_by_id(node_id)
        row = LayerRow(
            node_id=node_id,
            layer=node.layer_type,
            output_shape=_format_shape(shapes_by_node_id[node_id]),
            parameters=parameters_by_node.get(str(node_id), 0),
        )
        if node.layer_type != "input":
            sources = sorted(incoming_by_target[node_id])
            incoming_shapes = [shapes_by_node_id[source] for source in sources]
            merge = compute_merge_strategy(incoming_shapes, node.layer_type)
            row.inputs = sources
            merge_steps = []
            if len(sources) > 1:
                merge_steps.append(f"concat {len(sources)}")
                if any(
                    shape.is_spatial
                    and (shape.height, shape.width) != (merge.pooled_height, merge.pooled_width)
                    for shape in incoming_shapes
                ):
                    merge_steps.append(f"max-pool to {merge.pooled_height}x{merge.pooled_width}")
            if merge.flatten_inputs and any(shape.is_spatial for shape in incoming_shapes):
                merge_steps.append(f"flatten {merge.merged_features}")
            row.merge = ", ".join(merge_steps)
        if node.layer_type in ("conv", "dense"):
            row.uses_batch_normalization = node.uses_batch_normalization
            row.dropout_rate = node.dropout_rate
        if node.layer_type == "conv":
            row.filters = node.number_of_filters
            row.kernel = f"{node.kernel_size}x{node.kernel_size}"
            if node.is_followed_by_max_pooling:
                row.max_pooling = (
                    "applied" if merge.pooled_height >= 2 and merge.pooled_width >= 2 else "skipped"
                )
        elif node.layer_type == "dense":
            row.units = node.number_of_units
        elif node.layer_type == "output":
            row.units = number_of_classes
        layers.append(row)

    return ArchitectureDescription(
        genome_kind="DeepNEATGenome",
        input_shape=_format_shape(input_shape),
        layers=layers,
        edges=edges,
        genome_node_count=len(genome.node_genes),
        genome_enabled_edge_count=sum(1 for edge in genome.edge_genes if edge.is_enabled),
        genome_disabled_edge_count=sum(1 for edge in genome.edge_genes if not edge.is_enabled),
        total_parameters=count_trainable_parameters(model_state),
        notes=[
            "conv: Conv2d (padding 'same') -> [BatchNorm2d] -> ReLU -> [Dropout] "
            "-> [MaxPool2d 2x2]",
            "dense: Linear -> [BatchNorm1d] -> ReLU -> [Dropout]; output: Linear to logits",
            "several inputs are max-pooled to the smallest map and concatenated on channels",
        ],
    )


def describe_exact(genome_payload: dict, model_state: dict) -> ArchitectureDescription:
    """Decode an EXACT genome: filters as nodes, one convolution kernel per edge.

    Args:
        genome_payload: ``EXACTGenome.to_serializable_dict()`` output.
        model_state: The checkpoint's tensors.

    Returns:
        The filters in depth order, their connections and parameter counts.
    """
    genome = EXACTGenome.from_serializable_dict(genome_payload)
    node_by_id = {node.node_id: node for node in genome.node_genes}
    enabled_edges = [edge for edge in genome.edge_genes if edge.is_enabled]
    incoming_by_target: dict[int, list] = defaultdict(list)
    for edge in enabled_edges:
        incoming_by_target[edge.target_node_id].append(edge)
    kernel_parameters = _parameters_by_key(model_state, _EXACT_KERNEL_PREFIX)
    normalization_parameters = _parameters_by_key(model_state, _EXACT_BATCH_NORMALIZATION_PREFIX)

    layers: list[LayerRow] = []
    for node in sorted(genome.node_genes, key=lambda gene: (gene.depth, gene.node_id)):
        incoming = sorted(incoming_by_target[node.node_id], key=lambda edge: edge.source_node_id)
        kernels = []
        for edge in incoming:
            height, width = compute_convolution_kernel_size(node_by_id[edge.source_node_id], node)
            kernels.append(f"{height}x{width}")
        layers.append(
            LayerRow(
                node_id=node.node_id,
                layer=node.node_type,
                filters=1,
                kernel=", ".join(kernels) or None,
                uses_batch_normalization=str(node.node_id) in normalization_parameters,
                inputs=[edge.source_node_id for edge in incoming],
                merge=f"sum {len(incoming)}" if len(incoming) > 1 else "",
                output_shape=f"1x{node.filter_height}x{node.filter_width}",
                parameters=sum(
                    kernel_parameters.get(str(edge.innovation_id), 0) for edge in incoming
                )
                + normalization_parameters.get(str(node.node_id), 0),
            )
        )
    input_node = next(node for node in genome.node_genes if node.node_type == "input")
    return ArchitectureDescription(
        genome_kind="EXACTGenome",
        input_shape=f"1x{input_node.filter_height}x{input_node.filter_width}",
        layers=layers,
        edges=[(edge.source_node_id, edge.target_node_id) for edge in enabled_edges],
        genome_node_count=len(genome.node_genes),
        genome_enabled_edge_count=len(enabled_edges),
        genome_disabled_edge_count=len(genome.edge_genes) - len(enabled_edges),
        total_parameters=count_trainable_parameters(model_state),
        notes=[
            "every node is one single-channel feature map; every edge is one convolution "
            "with kernel |out - in| + 1, so maps shrink by the kernel size, not by pooling",
            "a node sums its incoming convolutions; hidden nodes: [BatchNorm2d] -> leaky ReLU "
            "clamped at the configured maximum; output nodes are 1x1 class logits",
        ],
    )


def describe_fixed_cnn(
    genome_payload: dict, model_state: dict, *, input_side: int
) -> ArchitectureDescription:
    """Describe the hand-designed baseline from its recorded configuration.

    Args:
        genome_payload: The configuration the baseline recorded.
        model_state: The checkpoint's tensors.
        input_side: Side of the square input image.

    Returns:
        Its blocks in order, as a chain.
    """
    parameters_by_module = _parameters_by_key(model_state, "_feature_extractor.")
    kernel_size = genome_payload["kernel_size"]
    uses_batch_normalization = genome_payload["uses_batch_normalization"]
    first_weight = model_state["_feature_extractor.0.weight"]
    channels, side = int(first_weight.shape[1]), input_side
    layers = [LayerRow(node_id=0, layer="input", output_shape=f"{channels}x{side}x{side}")]
    module_index = 0
    for filters in genome_payload["convolution_channels"]:
        block_modules = 4 if uses_batch_normalization else 3
        side = -(-side // 2)
        layers.append(
            LayerRow(
                node_id=len(layers),
                layer="conv",
                filters=filters,
                kernel=f"{kernel_size}x{kernel_size}",
                uses_batch_normalization=uses_batch_normalization,
                max_pooling="applied",
                inputs=[len(layers) - 1],
                output_shape=f"{filters}x{side}x{side}",
                parameters=sum(
                    parameters_by_module.get(str(index), 0)
                    for index in range(module_index, module_index + block_modules)
                ),
            )
        )
        module_index += block_modules
    classifier_parameters = sum(
        int(tensor.numel())
        for name, tensor in model_state.items()
        if name.startswith("_classifier.")
    )
    number_of_classes = int(model_state["_classifier.bias"].shape[0])
    layers.append(
        LayerRow(
            node_id=len(layers),
            layer="output",
            units=number_of_classes,
            dropout_rate=genome_payload["dropout_probability"],
            inputs=[len(layers) - 1],
            merge=f"global average pool {genome_payload['convolution_channels'][-1]}",
            output_shape=str(number_of_classes),
            parameters=classifier_parameters,
        )
    )
    return ArchitectureDescription(
        genome_kind="FixedConvolutionalNetworkConfig",
        input_shape=f"{channels}x{input_side}x{input_side}",
        layers=layers,
        edges=[(index, index + 1) for index in range(len(layers) - 1)],
        genome_node_count=len(layers),
        genome_enabled_edge_count=len(layers) - 1,
        genome_disabled_edge_count=0,
        total_parameters=count_trainable_parameters(model_state),
        notes=[
            "conv block: Conv2d -> [BatchNorm2d] -> ReLU -> MaxPool2d 2x2 (ceil)",
            "head: global average pooling -> Dropout -> Linear (dropout listed on the output row)",
        ],
    )


def describe_checkpoint(
    checkpoint: ModelCheckpoint,
    *,
    input_side: int | None = None,
    input_channels: int | None = None,
    number_of_classes: int | None = None,
) -> ArchitectureDescription | None:
    """Pick the decoder for a checkpoint's genome kind.

    Returns:
        The description, or ``None`` for a genome kind this module cannot decode.
    """
    side = input_side or checkpoint.preprocessing_state.get("target_side")
    if checkpoint.genome_kind == "DeepNEATGenome":
        return describe_deepneat(
            checkpoint.genome_payload,
            checkpoint.model_state,
            input_side=side,
            input_channels=input_channels,
            number_of_classes=number_of_classes,
        )
    if checkpoint.genome_kind == "EXACTGenome":
        return describe_exact(checkpoint.genome_payload, checkpoint.model_state)
    if checkpoint.genome_kind == "FixedConvolutionalNetworkConfig":
        return describe_fixed_cnn(
            checkpoint.genome_payload, checkpoint.model_state, input_side=side
        )
    logger.warning("cannot decode genome kind %s", checkpoint.genome_kind)
    return None


def _has_no_input(row: LayerRow) -> bool:
    """An EXACT filter left without enabled incoming edges; it feeds a constant."""
    return row.layer == "hidden" and not row.inputs


def _node_label(row: LayerRow) -> str:
    """Multi-line label for one node in the drawings."""
    if row.layer == "input":
        return f"Input\n{row.output_shape}"
    if row.layer in ("hidden", "output") and row.filters == 1:
        kind = "Output" if row.layer == "output" else "Filter"
        lines = [f"{kind} {row.output_shape.split('x', 1)[1]}"]
        if row.uses_batch_normalization:
            lines.append("BN")
        if _has_no_input(row):
            lines.append("(no input)")
        return "\n".join(lines)
    lines = []
    if row.merge:
        lines.append(f"[{row.merge}]")
    if row.layer == "conv":
        lines.append(f"Conv {row.kernel}, {row.filters}")
    elif row.layer == "dense":
        lines.append(f"Dense {row.units}")
    else:
        lines.append(f"Output (Linear {row.units})")
    extras = []
    if row.uses_batch_normalization:
        extras.append("BN")
    if row.layer in ("conv", "dense"):
        extras.append("ReLU")
    if row.dropout_rate > 0.0:
        extras.append(f"Dropout {row.dropout_rate:.2f}")
    if row.max_pooling == "applied":
        extras.append("MaxPool 2x2")
    if extras:
        lines.append(" · ".join(extras))
    lines.append(f"-> {row.output_shape}  ({row.parameters:,} p)")
    return "\n".join(lines)


def _compute_layout(description: ArchitectureDescription) -> dict[int, tuple[float, float]]:
    """Grid position of each node: its row is the longest path from the input.

    Rows are in topological order already, so one pass suffices. Output nodes
    share the last row, which keeps EXACT's per-class outputs side by side.
    """
    rank: dict[int, int] = {}
    for row in description.layers:
        rank[row.node_id] = max((rank.get(source, 0) + 1 for source in row.inputs), default=0)
    last_row = max(rank.values())
    for row in description.layers:
        if row.layer == "output":
            rank[row.node_id] = last_row
    columns: dict[int, list[int]] = defaultdict(list)
    for row in description.layers:
        columns[rank[row.node_id]].append(row.node_id)
    positions = {}
    for level, node_ids in columns.items():
        for index, node_id in enumerate(node_ids):
            positions[node_id] = (index - (len(node_ids) - 1) / 2, -float(level))
    return positions


def render_matplotlib(description: ArchitectureDescription, title: str, stem: Path) -> None:
    """Draw the network top to bottom and save ``stem.pdf`` and ``stem.png``."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as pyplot

    positions = _compute_layout(description)
    is_exact = description.genome_kind == "EXACTGenome"
    horizontal_spacing = 1.6 if is_exact else 3.6
    vertical_spacing = 1.0 if is_exact else 1.5
    widest = max(
        sum(1 for position in positions.values() if position[1] == level)
        for level in {position[1] for position in positions.values()}
    )
    levels = len({position[1] for position in positions.values()})
    # A connection that skips rows between two nodes in the same column would be
    # drawn over the chain between them, so it is routed as an arc beside the
    # column instead, alternating sides; shorter skips sit inside longer ones.
    side_routed = sorted(
        (
            (source, target)
            for source, target in description.edges
            if source in positions
            and target in positions
            and abs(positions[source][0] - positions[target][0]) < 1e-9
            and positions[source][1] - positions[target][1] > 1.5
        ),
        key=lambda edge: positions[edge[0]][1] - positions[edge[1]][1],
    )
    side_of_edge = {edge: 1 if index % 2 == 0 else -1 for index, edge in enumerate(side_routed)}
    nesting_of_edge = {edge: index // 2 for index, edge in enumerate(side_routed)}
    arc_gap = horizontal_spacing * (0.2 if is_exact else 0.12)
    arc_margin = arc_gap * (len(side_routed) + 1) / 2 + arc_gap
    figure, axes = pyplot.subplots(
        figsize=(
            max(6.0, widest * horizontal_spacing * 0.9 + 2 * arc_margin),
            max(3.0, levels * vertical_spacing),
        )
    )
    scaled = {
        node_id: (x * horizontal_spacing, y * vertical_spacing)
        for node_id, (x, y) in positions.items()
    }
    colors = {
        "input": "#9ecae1",
        "conv": "#fdd0a2",
        "dense": "#c7e9c0",
        "output": "#d9d9d9",
        "hidden": "#fdd0a2",
    }
    labels = {}
    for row in description.layers:
        x, y = scaled[row.node_id]
        labels[row.node_id] = axes.text(
            x,
            y,
            _node_label(row),
            ha="center",
            va="center",
            fontsize=6 if is_exact else 7,
            bbox={
                "boxstyle": "round,pad=0.35",
                "facecolor": colors.get(row.layer, "#eeeeee"),
                "edgecolor": "#333333",
                "lw": 0.6,
                "linestyle": "--" if _has_no_input(row) else "-",
            },
            zorder=2,
        )
    xs = [position[0] for position in scaled.values()]
    ys = [position[1] for position in scaled.values()]
    axes.set_xlim(
        min(xs) - horizontal_spacing * 0.7 - arc_margin,
        max(xs) + horizontal_spacing * 0.7 + arc_margin,
    )
    axes.set_ylim(min(ys) - vertical_spacing * 0.7, max(ys) + vertical_spacing * 0.7)
    axes.axis("off")
    axes.set_title(title, fontsize=9)
    figure.tight_layout()
    # Arrows run from the bottom edge of the source box to the top edge of the
    # target box; the boxes' extents are only known once the figure is drawn.
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    to_data = axes.transData.inverted()
    box_extents = {}
    for node_id, label in labels.items():
        (left, bottom), (right, top) = to_data.transform(
            label.get_bbox_patch().get_window_extent(renderer).get_points()
        )
        box_extents[node_id] = (left, bottom, right, top)
    to_display = axes.transData.transform
    for source, target in description.edges:
        if source not in scaled or target not in scaled:
            continue
        connection_style = "arc3,rad=0.0"
        start = (scaled[source][0], box_extents[source][1])
        end = (scaled[target][0], box_extents[target][3])
        side = side_of_edge.get((source, target))
        if side is not None:
            # Leave from and arrive at the box sides, bulging outwards by a
            # fixed distance; arc3 bulges by rad * length / 2 in display space.
            start = (box_extents[source][2 if side > 0 else 0], scaled[source][1])
            end = (box_extents[target][2 if side > 0 else 0], scaled[target][1])
            bulge = arc_gap * (nesting_of_edge[(source, target)] + 1)
            bulge_in_display = to_display((bulge, 0.0))[0] - to_display((0.0, 0.0))[0]
            length_in_display = abs(to_display(start)[1] - to_display(end)[1])
            connection_style = f"arc3,rad={-side * 2 * bulge_in_display / length_in_display:.4f}"
        axes.annotate(
            "",
            xy=end,
            xytext=start,
            arrowprops={
                "arrowstyle": "-|>",
                "color": "#555555",
                "lw": 0.6 if is_exact else 0.8,
                "shrinkA": 0,
                "shrinkB": 0,
                "connectionstyle": connection_style,
            },
            zorder=1,
        )
    stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(stem.with_suffix(".pdf"))
    figure.savefig(stem.with_suffix(".png"), dpi=200)
    pyplot.close(figure)


def render_dot(description: ArchitectureDescription, title: str) -> str:
    """The same graph as Graphviz source, for ``dot -Tpdf model.dot -o model.pdf``."""
    lines = [
        "digraph network {",
        f'  label="{title}"; labelloc=t; rankdir=TB;',
        '  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10];',
    ]
    colors = {"input": "#9ecae1", "conv": "#fdd0a2", "dense": "#c7e9c0", "hidden": "#fdd0a2"}
    for row in description.layers:
        label = _node_label(row).replace('"', "'").replace("\n", "\\n")
        style = ', style="rounded,filled,dashed"' if _has_no_input(row) else ""
        lines.append(
            f'  n{row.node_id} [label="{label}", '
            f'fillcolor="{colors.get(row.layer, "#d9d9d9")}"{style}];'
        )
    for source, target in description.edges:
        lines.append(f"  n{source} -> n{target};")
    lines.append("}")
    return "\n".join(lines) + "\n"


@dataclass
class WinnerRecord:
    """One run's selected model: where it came from, what it is, how it scored."""

    problem: str
    series_directory: str
    method: str
    seed: str
    model_id: str
    checkpoint_path: str | None
    parameter_count_reported: int | None
    selection_fitness: float | None
    metrics: dict
    architecture: ArchitectureDescription | None = None


def _input_settings(report: dict) -> tuple[int | None, int | None, int | None]:
    configuration = report.get("effective_configuration", {})
    algorithm = configuration.get("profile_payload", {}).get("algorithm", {})
    side = algorithm.get("input_image_height") or configuration.get("image_side")
    return side, algorithm.get("input_image_channels"), algorithm.get("number_of_classes")


def extract_metrics(report: dict, track_a_model_id: str, spec: SeriesSpec) -> dict:
    """The test metrics of track A and the mean of track B, plus search statistics."""
    official = report.get("official_test", {})
    track_a = official.get(track_a_model_id, {})
    track_b = [entry for entry in official.values() if entry.get("track") == "track_b"]
    metrics: dict = {}
    if spec.reports_operating_point:
        reference = track_a.get("metrics_at_reference_threshold", {})
        selected = track_a.get("metrics_at_selected_threshold", {})
        interval = track_a.get("confidence_intervals", {}).get("auroc", {})
        metrics["track_a_test_auroc"] = reference.get("auroc")
        metrics["track_a_test_auroc_ci95"] = [
            interval.get("lower_bound"),
            interval.get("upper_bound"),
        ]
        metrics["track_a_test_average_precision"] = reference.get("average_precision")
        for name in (
            "threshold",
            "sensitivity",
            "specificity",
            "accuracy",
            "balanced_accuracy",
            "f1",
        ):
            metrics[f"track_a_selected_threshold_{name}"] = selected.get(name)
        track_b_values = [
            entry.get("metrics_at_reference_threshold", {}).get("auroc") for entry in track_b
        ]
        metric_name = "auroc"
    else:
        metrics["track_a_test_accuracy"] = track_a.get("test_accuracy")
        track_b_values = [entry.get("test_accuracy") for entry in track_b]
        metric_name = "accuracy"
    track_b_values = [value for value in track_b_values if value is not None]
    metrics[f"track_b_test_{metric_name}_values"] = track_b_values
    metrics[f"track_b_test_{metric_name}_mean"] = (
        sum(track_b_values) / len(track_b_values) if track_b_values else None
    )
    summary = report.get("summary", {})
    metrics["number_of_generations"] = summary.get("number_of_generations")
    metrics["runtime_seconds"] = summary.get("runtime_seconds")
    metrics["evaluation_status_counts"] = report.get("evaluation_status_counts")
    metrics["track_b_failed_retrainings"] = len(report.get("track_b_failed_retrainings", []))
    return metrics


def collect_series_winners(results_root: Path, spec: SeriesSpec) -> list[WinnerRecord]:
    """Read every finished run of one series and decode its track A checkpoint."""
    records = []
    for report_path in sorted(
        (results_root / spec.series_directory).glob("runs/*/seed_*/run_report.json")
    ):
        run_directory = report_path.parent
        report = json.loads(report_path.read_text(encoding="utf-8"))
        track_a_ids = [
            model_id
            for model_id, frozen in report.get("frozen_models", {}).items()
            if frozen.get("track") == "track_a"
        ]
        if not track_a_ids:
            logger.warning("%s has no track A model; skipped", report_path)
            continue
        model_id = track_a_ids[0]
        details = report["frozen_models"][model_id].get("details", {})
        checkpoint_path = run_directory / "checkpoints" / f"{model_id}.pt"
        record = WinnerRecord(
            problem=spec.problem,
            series_directory=spec.series_directory,
            method=run_directory.parent.name,
            seed=run_directory.name.removeprefix("seed_"),
            model_id=model_id,
            checkpoint_path=str(checkpoint_path) if checkpoint_path.is_file() else None,
            parameter_count_reported=details.get("parameter_count"),
            selection_fitness=details.get("selection_fitness"),
            metrics=extract_metrics(report, model_id, spec),
        )
        if record.checkpoint_path is None:
            logger.warning("checkpoint %s is missing; architecture not decoded", checkpoint_path)
        else:
            side, channels, classes = _input_settings(report)
            checkpoint, _ = ModelCheckpoint.read_file(checkpoint_path)
            record.architecture = describe_checkpoint(
                checkpoint, input_side=side, input_channels=channels, number_of_classes=classes
            )
            _check_parameter_count(record)
        records.append(record)
    return records


def _check_parameter_count(record: WinnerRecord) -> None:
    if record.architecture is None or record.parameter_count_reported is None:
        return
    if record.architecture.total_parameters != record.parameter_count_reported:
        logger.warning(
            "%s: checkpoint holds %d trainable parameters but the report recorded %d",
            record.model_id,
            record.architecture.total_parameters,
            record.parameter_count_reported,
        )


def _format_number(value: float | int | None, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, int):
        return f"{value:,}"
    return f"{value:.{digits}f}"


def _format_track_b(metrics: dict, metric: str) -> str:
    """Track B mean, flagging retrainings that diverged and so carry no score."""
    mean = _format_number(metrics.get(f"track_b_test_{metric}_mean"))
    failed = metrics.get("track_b_failed_retrainings", 0)
    if not failed:
        return mean
    if metrics.get(f"track_b_test_{metric}_mean") is None:
        return f"all {failed} diverged"
    return f"{mean} ({failed} diverged)"


def _summary_table(records: list[WinnerRecord], spec: SeriesSpec) -> list[str]:
    metric = "auroc" if spec.reports_operating_point else "accuracy"
    header = [
        "method",
        "seed",
        "params [M]",
        "conv",
        "dense",
        "maxpool",
        "dropout",
        "BN",
        "connections",
        f"track A test {metric}",
        f"track B test {metric} (mean)",
    ]
    if spec.reports_operating_point:
        header += ["sens@thr", "spec@thr"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for record in records:
        architecture = record.architecture
        parameters = (
            architecture.total_parameters if architecture else record.parameter_count_reported
        )
        if architecture is None:
            structure = ["?"] * 6
        elif architecture.genome_kind == "EXACTGenome":
            structure = [
                f"{architecture.count_layers('hidden')} filters",
                "-",
                "0",
                "0",
                str(architecture.batch_normalization_count),
                str(len(architecture.edges)),
            ]
        else:
            structure = [
                str(architecture.count_layers("conv")),
                str(architecture.count_layers("dense")),
                str(architecture.max_pooling_count),
                str(architecture.dropout_count),
                str(architecture.batch_normalization_count),
                str(len(architecture.edges)),
            ]
        cells = [
            record.method,
            record.seed,
            f"{parameters / 1e6:.3f}" if parameters is not None else "-",
            *structure,
            _format_number(record.metrics.get(f"track_a_test_{metric}")),
            _format_track_b(record.metrics, metric),
        ]
        if spec.reports_operating_point:
            cells += [
                _format_number(record.metrics.get("track_a_selected_threshold_sensitivity")),
                _format_number(record.metrics.get("track_a_selected_threshold_specificity")),
            ]
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def _layer_table(architecture: ArchitectureDescription) -> list[str]:
    lines = [
        "| node | layer | filters/units | kernel | BN | dropout | maxpool | inputs | merge "
        "| output | params |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in architecture.layers:
        width = row.filters if row.filters is not None else row.units
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.node_id),
                    row.layer,
                    "-" if width is None else str(width),
                    row.kernel or "-",
                    "yes" if row.uses_batch_normalization else "-",
                    f"{row.dropout_rate:.2f}" if row.dropout_rate > 0 else "-",
                    row.max_pooling if row.max_pooling != "none" else "-",
                    ", ".join(map(str, row.inputs)) or "-",
                    row.merge or "-",
                    row.output_shape,
                    f"{row.parameters:,}",
                ]
            )
            + " |"
        )
    return lines


def _record_stem(output_directory: Path, record: WinnerRecord) -> Path:
    series = record.series_directory.replace("/", "-")
    return output_directory / series / f"{record.method}_seed{record.seed}"


def write_outputs(
    records_by_spec: list[tuple[SeriesSpec, list[WinnerRecord]]], output_directory: Path
) -> Path:
    """Write the per-model files and ``winners.md``; return the Markdown path."""
    sections = ["# Selected architectures", ""]
    for spec, records in records_by_spec:
        sections += [f"## {spec.problem} ({spec.series_directory})", ""]
        if not records:
            sections += [f"No finished runs under {spec.series_directory}.", ""]
            continue
        sections += _summary_table(records, spec) + [""]
        missing = [record.model_id for record in records if record.architecture is None]
        if missing:
            sections += [
                f"Checkpoints missing for {len(missing)} model(s), so their structure is unknown: "
                + ", ".join(missing),
                "",
            ]
        for record in records:
            stem = _record_stem(output_directory, record)
            stem.parent.mkdir(parents=True, exist_ok=True)
            payload = asdict(record)
            stem.with_suffix(".json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
            if record.architecture is None:
                continue
            title = f"{spec.problem} · {record.method} · seed {record.seed}"
            render_matplotlib(record.architecture, title, stem)
            stem.with_suffix(".dot").write_text(
                render_dot(record.architecture, title), encoding="utf-8"
            )
            architecture = record.architecture
            relative = stem.relative_to(output_directory).as_posix()
            sections += [
                f"### {record.method}, seed {record.seed} - "
                f"{architecture.total_parameters:,} parameters",
                "",
                f"Input {architecture.input_shape}; {len(architecture.layers)} executed nodes of "
                f"{architecture.genome_node_count} in the genome; "
                f"{len(architecture.edges)} executed "
                f"connections ({architecture.genome_enabled_edge_count} enabled, "
                f"{architecture.genome_disabled_edge_count} disabled in the genome).",
                "",
                *(f"- {note}" for note in architecture.notes),
                "",
                *_layer_table(architecture),
                "",
                f"![{record.model_id}]({relative}.png)",
                "",
            ]
    output_directory.mkdir(parents=True, exist_ok=True)
    markdown_path = output_directory / "winners.md"
    markdown_path.write_text("\n".join(sections) + "\n", encoding="utf-8")
    return markdown_path


def describe_standalone_checkpoints(checkpoint_paths: list[Path], output_directory: Path) -> None:
    """Describe checkpoint files directly, without a run report."""
    for checkpoint_path in checkpoint_paths:
        checkpoint, _ = ModelCheckpoint.read_file(checkpoint_path)
        architecture = describe_checkpoint(checkpoint)
        if architecture is None:
            continue
        # Every run names its checkpoints the same way, so keep the run directory.
        stem = output_directory / f"{checkpoint_path.parent.parent.name}_{checkpoint_path.stem}"
        title = f"{checkpoint.model_id} ({checkpoint.genome_kind})"
        render_matplotlib(architecture, title, stem)
        stem.with_suffix(".dot").write_text(render_dot(architecture, title), encoding="utf-8")
        stem.with_suffix(".json").write_text(
            json.dumps(
                {"metadata": checkpoint.metadata, "architecture": asdict(architecture)}, indent=2
            ),
            encoding="utf-8",
        )
        logger.info(
            "%s: %s, %d parameters, %d executed nodes, %d connections\n%s",
            checkpoint_path,
            architecture.genome_kind,
            architecture.total_parameters,
            len(architecture.layers),
            len(architecture.edges),
            "\n".join(_layer_table(architecture)),
        )


def main(argument_list: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results-root", type=Path, default=_DEFAULT_RESULTS_ROOT)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=None,
        help="where to write (default: <results-root>/winners)",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        nargs="+",
        default=None,
        help="describe these checkpoint files instead of walking the series",
    )
    arguments = parser.parse_args(argument_list)
    output_directory = arguments.output_directory or arguments.results_root / "winners"
    if arguments.checkpoint:
        describe_standalone_checkpoints(arguments.checkpoint, output_directory)
        return
    records_by_spec = [
        (spec, collect_series_winners(arguments.results_root, spec)) for spec in SERIES
    ]
    markdown_path = write_outputs(records_by_spec, output_directory)
    logger.info("%s\nWritten to %s", markdown_path.read_text(encoding="utf-8"), markdown_path)


if __name__ == "__main__":
    main()
