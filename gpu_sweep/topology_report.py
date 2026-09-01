"""Two views of an evolved network: a layered picture and a text description.

``polyneat.viz.network_topology_renderer`` lays nodes out with a force
simulation, which shows clusters but hides layer structure. Here every node is
assigned an explicit layer - its longest path from the inputs - and
``networkx.multipartite_layout`` pins each layer to its own column, so the
depth NEAT has grown is visible at a glance. Every picture is paired with a
``.txt`` description holding the same facts as text, which is easier to skim
across 120 cells than 120 images.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import networkx

NODE_TYPE_TO_FILL_COLOR: dict[str, str] = {
    "input": "#8ecae6",
    "bias": "#ffb703",
    "hidden": "#d3d3d3",
    "output": "#90be6d",
}
UNKNOWN_NODE_TYPE_FILL_COLOR = "#f28482"


def _enabled_edges(genome: object) -> list[tuple[int, int]]:
    """Source/target pairs of every enabled connection gene."""
    return [
        (connection_gene.source_node_id, connection_gene.target_node_id)
        for connection_gene in genome.connection_genes
        if connection_gene.is_enabled
    ]


def compute_layer_index_by_node_id(genome: object) -> dict[int, int]:
    """Assign every node a layer, so the drawing reads inputs-left to outputs-right.

    Inputs and bias nodes are layer 0. Every other node sits one past the
    deepest node feeding it through an enabled connection, which is its longest
    path from the inputs. Output nodes are then all pulled onto one final
    column, one past the deepest non-output node, so they line up even when
    some are reached by shorter paths than others.

    A hidden node with no enabled incoming connection is **not** layer 0. It is
    unreachable - which happens routinely, since ``AddNodeMutation`` disables
    the edge it splits and FS-NEAT starts from a single connection - and
    putting it in layer 0 draws it in the input column, where it reads as an
    input. Those nodes go to their own layer 1 instead, so the input column
    contains inputs and nothing else.

    Args:
        genome: Anything exposing ``node_genes`` and ``connection_genes``.

    Returns:
        ``{node_id: layer_index}`` covering every node in the genome.
    """
    node_type_by_node_id = {
        node_gene.node_id: node_gene.node_type for node_gene in genome.node_genes
    }
    incoming_source_ids: dict[int, list[int]] = {
        node_id: [] for node_id in node_type_by_node_id
    }
    for source_node_id, target_node_id in _enabled_edges(genome):
        if target_node_id in incoming_source_ids:
            incoming_source_ids[target_node_id].append(source_node_id)

    layer_index_by_node_id: dict[int, int] = {}

    def depth_of(node_id: int, visiting: frozenset[int]) -> int:
        if node_id in layer_index_by_node_id:
            return layer_index_by_node_id[node_id]
        if node_type_by_node_id.get(node_id) in ("input", "bias") or node_id in visiting:
            return 0
        source_ids = incoming_source_ids.get(node_id, [])
        if not source_ids:
            # Unreachable, not an input. Layer 1 keeps it out of the input
            # column; it will still be drawn with no incoming edges, which is
            # what makes it recognisable as orphaned.
            return 1
        deepest_source = max(
            depth_of(source_id, visiting | {node_id}) for source_id in source_ids
        )
        return deepest_source + 1

    for node_id in node_type_by_node_id:
        layer_index_by_node_id[node_id] = depth_of(node_id, frozenset())

    non_output_layers = [
        layer_index
        for node_id, layer_index in layer_index_by_node_id.items()
        if node_type_by_node_id[node_id] != "output"
    ]
    output_layer_index = (max(non_output_layers) + 1) if non_output_layers else 1
    for node_id, node_type in node_type_by_node_id.items():
        if node_type == "output":
            layer_index_by_node_id[node_id] = output_layer_index
    return layer_index_by_node_id


def describe_topology(
    genome: object,
    *,
    title: str,
    structure_notes: dict[str, object],
) -> str:
    """Render the same structural facts as the picture, as plain text.

    Args:
        genome: Anything exposing ``node_genes`` and ``connection_genes``.
        title: Heading line, normally ``"<dataset>/<algorithm> <label>"``.
        structure_notes: Extra facts from the runner, printed verbatim at the
            end - HyperNEAT's substrate sizes, ensemble notes, and so on.

    Returns:
        A multi-line description, ending with a trailing newline.
    """
    layer_index_by_node_id = compute_layer_index_by_node_id(genome)
    node_type_by_node_id = {
        node_gene.node_id: node_gene.node_type for node_gene in genome.node_genes
    }
    activation_by_node_id = {
        node_gene.node_id: getattr(node_gene, "activation_function_name", "?")
        for node_gene in genome.node_genes
    }
    type_counts = {
        node_type: sum(1 for value in node_type_by_node_id.values() if value == node_type)
        for node_type in ("input", "bias", "hidden", "output")
    }
    enabled_connection_count = len(_enabled_edges(genome))
    total_connection_count = len(list(genome.connection_genes))

    lines = [
        f"{title}",
        "=" * len(title),
        "",
        (
            f"Nodes: {len(node_type_by_node_id)} "
            f"(input {type_counts['input']}, bias {type_counts['bias']}, "
            f"hidden {type_counts['hidden']}, output {type_counts['output']})"
        ),
        f"Connections: {enabled_connection_count} enabled of {total_connection_count} total",
        f"Depth: {max(layer_index_by_node_id.values(), default=0) + 1} layers",
        "",
        "Layers (inputs first, outputs last):",
    ]
    for layer_index in sorted(set(layer_index_by_node_id.values())):
        node_ids_in_layer = sorted(
            node_id
            for node_id, value in layer_index_by_node_id.items()
            if value == layer_index
        )
        composition = ", ".join(
            f"{node_type_by_node_id[node_id]} {node_id}"
            f"[{activation_by_node_id.get(node_id, '?')}]"
            for node_id in node_ids_in_layer
        )
        lines.append(
            f"  layer {layer_index}: {len(node_ids_in_layer)} nodes  [{composition}]"
        )

    if structure_notes:
        lines.extend(["", "Notes:"])
        for note_name, note_value in structure_notes.items():
            lines.append(f"  {note_name}: {note_value}")
    return "\n".join(lines) + "\n"


def render_layered_topology(genome: object, output_path: Path, *, title: str) -> None:
    """Draw the genome with one column per layer and save it to ``output_path``.

    Args:
        genome: Anything exposing ``node_genes`` and ``connection_genes``.
        output_path: Target ``.png`` file; parents are created.
        title: Figure title.
    """
    matplotlib.use("Agg")
    import matplotlib.pyplot as pyplot

    layer_index_by_node_id = compute_layer_index_by_node_id(genome)
    node_type_by_node_id = {
        node_gene.node_id: node_gene.node_type for node_gene in genome.node_genes
    }

    directed_graph = networkx.DiGraph()
    for node_id, layer_index in layer_index_by_node_id.items():
        directed_graph.add_node(node_id, layer=layer_index)
    for source_node_id, target_node_id in _enabled_edges(genome):
        directed_graph.add_edge(source_node_id, target_node_id)

    positions = networkx.multipartite_layout(directed_graph, subset_key="layer")
    node_fill_colors = [
        NODE_TYPE_TO_FILL_COLOR.get(
            node_type_by_node_id.get(node_id, ""), UNKNOWN_NODE_TYPE_FILL_COLOR
        )
        for node_id in directed_graph.nodes()
    ]

    number_of_layers = max(layer_index_by_node_id.values(), default=0) + 1
    widest_layer = max(
        (
            sum(1 for value in layer_index_by_node_id.values() if value == layer_index)
            for layer_index in set(layer_index_by_node_id.values())
        ),
        default=1,
    )
    figure_width = max(6.0, 2.0 * number_of_layers)
    figure_height = max(4.0, min(20.0, 0.35 * widest_layer))
    show_labels = len(layer_index_by_node_id) <= 60

    pyplot.figure(figsize=(figure_width, figure_height))
    networkx.draw_networkx(
        directed_graph,
        pos=positions,
        node_color=node_fill_colors,
        with_labels=show_labels,
        arrows=True,
        node_size=220 if show_labels else 40,
        font_size=7,
        width=0.5,
        edge_color="#999999",
    )
    legend_handles = [
        pyplot.Line2D(
            [], [], marker="o", linestyle="", color=fill_color, label=node_type
        )
        for node_type, fill_color in NODE_TYPE_TO_FILL_COLOR.items()
    ]
    pyplot.legend(handles=legend_handles, loc="upper right", fontsize=7, framealpha=0.9)
    pyplot.title(f"{title}  —  {number_of_layers} layers, inputs left to outputs right")
    pyplot.axis("off")
    pyplot.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pyplot.savefig(output_path, format="png", dpi=150, bbox_inches="tight")
    pyplot.close()


def write_topology_report(
    genome: object,
    output_directory: Path,
    base_name: str,
    *,
    title: str,
    structure_notes: dict[str, object],
) -> None:
    """Write ``<base_name>.png`` and ``<base_name>.txt`` into ``output_directory``.

    Args:
        genome: Anything exposing ``node_genes`` and ``connection_genes``.
        output_directory: Directory to write into; created if absent.
        base_name: Filename stem shared by both artifacts.
        title: Heading used in both the figure and the text file.
        structure_notes: Extra facts appended to the text description.
    """
    output_directory.mkdir(parents=True, exist_ok=True)
    render_layered_topology(genome, output_directory / f"{base_name}.png", title=title)
    (output_directory / f"{base_name}.txt").write_text(
        describe_topology(genome, title=title, structure_notes=structure_notes),
        encoding="utf-8",
    )


@dataclass(frozen=True)
class RecordedNode:
    """One node, as stored for later drawing."""

    node_id: int
    node_type: str
    activation_function_name: str


@dataclass(frozen=True)
class RecordedConnection:
    """One connection, as stored for later drawing."""

    source_node_id: int
    target_node_id: int
    weight: float
    is_enabled: bool


@dataclass(frozen=True)
class TopologyRecord:
    """A stored network structure that stands in for a genome.

    The attribute names match ``NEATGenome``'s deliberately, so every function
    in this module accepts a live genome or a record read back from disk with
    no branching.
    """

    node_genes: list[RecordedNode]
    connection_genes: list[RecordedConnection]


def genome_to_record_payload(
    genome: object, *, title: str, structure_notes: dict[str, object]
) -> dict[str, object]:
    """Flatten a genome into a JSON-safe payload.

    Only what the drawing needs is kept: node identity, type and activation,
    plus each connection's endpoints, weight and enabled flag. Innovation
    numbers, fitness and every algorithm-specific field are dropped - this is a
    picture's worth of structure, not a genome that can be evolved further.
    """
    return {
        "title": title,
        "structure_notes": structure_notes,
        "nodes": [
            {
                "node_id": node_gene.node_id,
                "node_type": node_gene.node_type,
                "activation_function_name": getattr(
                    node_gene, "activation_function_name", "?"
                ),
            }
            for node_gene in genome.node_genes
        ],
        "connections": [
            {
                "source_node_id": connection_gene.source_node_id,
                "target_node_id": connection_gene.target_node_id,
                "weight": float(connection_gene.weight),
                "is_enabled": bool(connection_gene.is_enabled),
            }
            for connection_gene in genome.connection_genes
        ],
    }


def topology_record_from_dict(
    payload: dict[str, object],
) -> tuple[TopologyRecord, str, dict[str, object]]:
    """Rebuild a record, its title and its notes from a stored payload.

    Returns:
        ``(record, title, structure_notes)``.
    """
    record = TopologyRecord(
        node_genes=[
            RecordedNode(
                node_id=int(node["node_id"]),
                node_type=str(node["node_type"]),
                activation_function_name=str(node["activation_function_name"]),
            )
            for node in payload["nodes"]
        ],
        connection_genes=[
            RecordedConnection(
                source_node_id=int(connection["source_node_id"]),
                target_node_id=int(connection["target_node_id"]),
                weight=float(connection["weight"]),
                is_enabled=bool(connection["is_enabled"]),
            )
            for connection in payload["connections"]
        ],
    )
    return record, str(payload.get("title", "")), dict(payload.get("structure_notes", {}))


def write_topology_record(
    genome: object,
    output_directory: Path,
    base_name: str,
    *,
    title: str,
    structure_notes: dict[str, object],
) -> None:
    """Store ``<base_name>.json`` for later drawing. Draws nothing itself.

    This is what the sweep's child processes call. Rendering is a separate,
    manual step so that no GPU run spends its timeout inside matplotlib.
    """
    output_directory.mkdir(parents=True, exist_ok=True)
    payload = genome_to_record_payload(
        genome, title=title, structure_notes=structure_notes
    )
    (output_directory / f"{base_name}.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )


def render_topology_records(topology_directory: Path) -> int:
    """Draw every stored record in ``topology_directory``.

    Run this by hand after a sweep, as often as you like - it only reads the
    stored JSON, so restyling costs nothing and re-running the sweep is never
    required.

    Args:
        topology_directory: Directory holding ``*.json`` topology records.

    Returns:
        How many records were drawn. Unreadable records are reported and
        skipped rather than aborting the batch.
    """
    number_drawn = 0
    for json_path in sorted(topology_directory.glob("*.json")):
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            record, title, structure_notes = topology_record_from_dict(payload)
        except (json.JSONDecodeError, OSError, KeyError, TypeError) as read_error:
            print(f"skipping unreadable topology record {json_path.name}: {read_error}")
            continue
        write_topology_report(
            record,
            topology_directory,
            json_path.stem,
            title=title or json_path.stem,
            structure_notes=structure_notes,
        )
        number_drawn += 1
    return number_drawn
