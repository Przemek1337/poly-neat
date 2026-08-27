"""Modularity measurements on a decoded substrate.

Used to compare plain HyperNEAT with HyperNEAT-LEO on a task whose structure is
modular: LEO's claim is that a locality-seeded link expression output yields
networks with markedly fewer connections crossing between the task's
sub-problems, at comparable fitness.

References:
    Verbancsics, P., & Stanley, K. O. (2011). Constraining Connectivity to Encourage
        Modularity in HyperNEAT. *GECCO '11: Proceedings of the 13th Annual Conference on
        Genetic and Evolutionary Computation*, pp. 1483-1490. DOI: 10.1145/2001576.2001776
"""

from __future__ import annotations

from polyneat.algorithms.hyperneat.substrate import Substrate
from polyneat.core.neat.neat_genome import NEATGenome
from polyneat.nn.topology_utilities import find_node_ids_with_enabled_path_to_any_target


def count_expressed_connections(decoded_substrate_genome: NEATGenome) -> int:
    """Count the enabled connections a decoder actually put into the substrate.

    Args:
        decoded_substrate_genome: The synthetic genome produced by a HyperNEAT or
            HyperNEAT-LEO decoder.

    Returns:
        The number of enabled connection genes.
    """
    return sum(1 for gene in decoded_substrate_genome.connection_genes if gene.is_enabled)


def count_cross_hemisphere_connections(
    substrate: Substrate,
    decoded_substrate_genome: NEATGenome,
) -> int:
    """Count enabled connections joining nodes on opposite sides of ``x = 0``.

    A connection counts when the x coordinates of its endpoints have opposite
    signs. Nodes sitting exactly on ``x = 0`` - the bias, by convention - belong
    to neither hemisphere, so their connections never count.

    Args:
        substrate: The substrate the genome was decoded onto; supplies the node
            coordinates.
        decoded_substrate_genome: The synthetic genome produced by the decoder.

    Returns:
        The number of enabled connections crossing between hemispheres.
    """
    x_coordinate_by_node_id = {node.node_id: node.x_coordinate for node in substrate.all_nodes()}
    crossing_connection_count = 0
    for gene in decoded_substrate_genome.connection_genes:
        if not gene.is_enabled:
            continue
        source_x = x_coordinate_by_node_id.get(gene.source_node_id)
        target_x = x_coordinate_by_node_id.get(gene.target_node_id)
        if source_x is None or target_x is None:
            continue
        if source_x * target_x < 0.0:
            crossing_connection_count += 1
    return crossing_connection_count


def count_cross_hemisphere_input_dependencies(
    substrate: Substrate,
    decoded_substrate_genome: NEATGenome,
) -> int:
    """Count (output, input) pairs where an output depends on the other hemisphere.

    This is the measure the modularity claim is actually about. Counting
    *connections* by coordinate sign - what
    :func:`count_cross_hemisphere_connections` does - conflates wiring locality
    with functional modularity: a hidden node's side is arbitrary, so a network
    routing left inputs through right-side hidden nodes into the left output
    never mixes left and right information, yet every one of its connections
    looks like a crossing.

    What matters is reachability. An output is modular when no input on the
    opposite side has an enabled path to it.

    Args:
        substrate: The substrate the genome was decoded onto; supplies the node
            coordinates and roles.
        decoded_substrate_genome: The synthetic genome produced by the decoder.

    Returns:
        The number of ``(output node, input node)`` pairs on opposite sides of
        ``x = 0`` connected by an enabled path. Zero means perfectly modular; on
        the retina substrate the maximum is 8 (two outputs times four opposite
        inputs).
    """
    x_coordinate_by_node_id = {node.node_id: node.x_coordinate for node in substrate.all_nodes()}
    input_node_ids = [node.node_id for node in substrate.input_layer.nodes]
    enabled_directed_edges = [
        (gene.source_node_id, gene.target_node_id)
        for gene in decoded_substrate_genome.connection_genes
        if gene.is_enabled
    ]

    crossing_dependency_count = 0
    for output_node in substrate.output_layer.nodes:
        output_x = x_coordinate_by_node_id[output_node.node_id]
        if output_x == 0.0:
            continue
        reaching_input_node_ids = find_node_ids_with_enabled_path_to_any_target(
            candidate_source_node_ids=input_node_ids,
            target_node_ids={output_node.node_id},
            enabled_directed_edges=enabled_directed_edges,
        )
        crossing_dependency_count += sum(
            1
            for input_node_id in reaching_input_node_ids
            if x_coordinate_by_node_id[input_node_id] * output_x < 0.0
        )
    return crossing_dependency_count
