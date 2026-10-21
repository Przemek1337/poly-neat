from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, TypeGuard

import matplotlib
import networkx

from polyneat.logging_utils.custom_logger import get_logger

if TYPE_CHECKING:
    from polyneat.core.component_protocols import Genome

logger = get_logger(__name__)


class _NeatShapedGenome(Protocol):
    """Structural type for genomes the renderer can draw.

    Any genome exposing ``node_genes`` and ``connection_genes`` sequences of
    NEAT-style genes can be rendered, regardless of its concrete class.
    """

    node_genes: tuple[Any, ...]
    connection_genes: tuple[Any, ...]


NODE_TYPE_TO_FILL_COLOR: dict[str, str] = {
    "input": "#4fc3f7",
    "output": "#81c784",
    "bias": "#ffb74d",
    "hidden": "#e0e0e0",
}
UNKNOWN_NODE_TYPE_FILL_COLOR = "#e0e0e0"


def render_genome_topology(genome: Genome, output_path: Path) -> None:
    """Render the genome's network topology to ``output_path`` (PNG or SVG).

    The genome must expose ``node_genes`` and ``connection_genes`` attributes
    with fields matching the NEAT gene shape. Genomes that do not conform are
    skipped with a warning rather than raised, so this renderer is safe to
    attach as an unconditional callback.

    Args:
        genome: Genome to draw; only enabled connections become edges.
        output_path: Target file; suffix picks the format (``.svg`` or PNG).
    """
    if not _genome_has_neat_shape(genome):
        logger.warning(
            "render_genome_topology skipped: genome %r lacks node_genes/connection_genes",
            genome,
        )
        return

    matplotlib.use("Agg")
    import matplotlib.pyplot as pyplot

    directed_graph = networkx.DiGraph()
    node_id_to_fill_color: dict[int, str] = {}
    for node_gene in genome.node_genes:
        directed_graph.add_node(node_gene.node_id)
        node_id_to_fill_color[node_gene.node_id] = NODE_TYPE_TO_FILL_COLOR.get(
            node_gene.node_type, UNKNOWN_NODE_TYPE_FILL_COLOR
        )

    for connection_gene in genome.connection_genes:
        if connection_gene.is_enabled:
            directed_graph.add_edge(
                connection_gene.source_node_id,
                connection_gene.target_node_id,
                weight=connection_gene.weight,
            )

    spring_layout_positions = networkx.spring_layout(directed_graph, seed=0)
    node_fill_colors_in_graph_order = [
        node_id_to_fill_color.get(node_id, UNKNOWN_NODE_TYPE_FILL_COLOR)
        for node_id in directed_graph.nodes()
    ]

    pyplot.figure(figsize=(8, 6))
    networkx.draw_networkx(
        directed_graph,
        pos=spring_layout_positions,
        node_color=node_fill_colors_in_graph_order,
        with_labels=True,
        arrows=True,
        node_size=600,
        font_size=8,
    )
    pyplot.axis("off")
    pyplot.tight_layout()

    output_format = "svg" if output_path.suffix.lower() == ".svg" else "png"
    pyplot.savefig(output_path, format=output_format, dpi=150, bbox_inches="tight")
    pyplot.close()


def _genome_has_neat_shape(genome: object) -> TypeGuard[_NeatShapedGenome]:
    return hasattr(genome, "node_genes") and hasattr(genome, "connection_genes")
