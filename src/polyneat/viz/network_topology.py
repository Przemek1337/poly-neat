from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from polyneat.core.protocols import Genome


def render_genome_topology(genome: "Genome", output_path: Path) -> None:
    """Render the genome's network topology to a PNG or SVG file.

    Requires the genome to be a NEATGenome; non-NEAT genomes are silently skipped.
    """
    try:
        from polyneat.algorithms.neat.genome import NEATGenome
    except ImportError:
        return

    if not isinstance(genome, NEATGenome):
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import networkx as nx

    graph = nx.DiGraph()

    node_colors: dict[int, str] = {}
    for node in genome.node_genes:
        graph.add_node(node.node_id)
        if node.node_type == "input":
            node_colors[node.node_id] = "#4fc3f7"
        elif node.node_type == "output":
            node_colors[node.node_id] = "#81c784"
        elif node.node_type == "bias":
            node_colors[node.node_id] = "#ffb74d"
        else:
            node_colors[node.node_id] = "#e0e0e0"

    for conn in genome.connection_genes:
        if conn.is_enabled:
            graph.add_edge(conn.source_node_id, conn.target_node_id, weight=conn.weight)

    pos = nx.spring_layout(graph, seed=0)
    color_list = [node_colors.get(n, "#e0e0e0") for n in graph.nodes()]
    edge_weights = [graph[u][v]["weight"] for u, v in graph.edges()]

    plt.figure(figsize=(8, 6))
    nx.draw_networkx(
        graph,
        pos=pos,
        node_color=color_list,
        with_labels=True,
        arrows=True,
        node_size=600,
        font_size=8,
    )
    plt.axis("off")
    plt.tight_layout()

    suffix = output_path.suffix.lower()
    fmt = "svg" if suffix == ".svg" else "png"
    plt.savefig(output_path, format=fmt, dpi=150, bbox_inches="tight")
    plt.close()
