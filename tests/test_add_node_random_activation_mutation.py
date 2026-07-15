from __future__ import annotations

import numpy as np

from polyneat.algorithms.hyperneat.add_node_random_activation_mutation import (
    AddNodeWithRandomActivationMutation,
)
from polyneat.core.neat.global_innovation_tracker import GlobalInnovationTracker
from polyneat.core.neat.neat_genome import ConnectionGene, NEATGenome, NodeGene


def _minimal_genome_with_one_connection() -> NEATGenome:
    return NEATGenome(
        node_genes=(
            NodeGene(node_id=0, node_type="input", activation_function_name="identity"),
            NodeGene(node_id=1, node_type="output", activation_function_name="identity"),
        ),
        connection_genes=(
            ConnectionGene(
                innovation_id=0,
                source_node_id=0,
                target_node_id=1,
                weight=0.5,
                is_enabled=True,
            ),
        ),
    )


def _tracker_seeded_for(genome: NEATGenome) -> GlobalInnovationTracker:
    """A tracker whose id counter has advanced past the genome's existing edges.

    Mirrors the real flow where the initial population assigns each connection's
    innovation id through the tracker, so a later add-node mutation issues fresh
    ids that do not collide with existing ones.
    """
    tracker = GlobalInnovationTracker()
    for connection_gene in genome.connection_genes:
        tracker.get_or_assign_innovation_id_for_connection(
            source_node_id=connection_gene.source_node_id,
            target_node_id=connection_gene.target_node_id,
        )
    return tracker


def test_inserted_hidden_node_uses_a_function_from_the_set():
    function_set = ("gaussian", "sine", "absolute_value")
    mutation = AddNodeWithRandomActivationMutation(
        probability_of_application=1.0,
        available_activation_function_names=function_set,
    )
    genome = _minimal_genome_with_one_connection()
    mutated = mutation.apply_to_genome(
        genome=genome,
        rng=np.random.default_rng(0),
        innovation_tracker=_tracker_seeded_for(genome),
    )
    hidden_nodes = [n for n in mutated.node_genes if n.node_type == "hidden"]
    assert len(hidden_nodes) == 1
    assert hidden_nodes[0].activation_function_name in function_set


def test_probability_zero_returns_genome_unchanged():
    mutation = AddNodeWithRandomActivationMutation(
        probability_of_application=0.0,
        available_activation_function_names=("gaussian",),
    )
    genome = _minimal_genome_with_one_connection()
    mutated = mutation.apply_to_genome(
        genome=genome,
        rng=np.random.default_rng(0),
        innovation_tracker=_tracker_seeded_for(genome),
    )
    assert mutated is genome


def test_different_seeds_can_select_different_functions():
    function_set = ("gaussian", "sine", "absolute_value", "sigmoid", "identity")
    mutation = AddNodeWithRandomActivationMutation(
        probability_of_application=1.0,
        available_activation_function_names=function_set,
    )
    genome = _minimal_genome_with_one_connection()
    selected_functions = set()
    for seed in range(30):
        mutated = mutation.apply_to_genome(
            genome=genome,
            rng=np.random.default_rng(seed),
            innovation_tracker=_tracker_seeded_for(genome),
        )
        (hidden_node,) = [n for n in mutated.node_genes if n.node_type == "hidden"]
        selected_functions.add(hidden_node.activation_function_name)
    # random selection should reach more than one function across 30 seeds
    assert len(selected_functions) >= 2
