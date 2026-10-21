from __future__ import annotations

import numpy as np

from polyneat.core.neat.global_innovation_tracker import GlobalInnovationTracker
from polyneat.core.neat.mutations.add_node_mutation import AddNodeMutation
from polyneat.core.neat.neat_genome import ConnectionGene, NEATGenome, NodeGene

_BIAS_TO_OUTPUT_INNOVATION_ID = 2
_INPUT_ZERO_TO_OUTPUT_INNOVATION_ID = 0


def _minimal_start_genome() -> NEATGenome:
    """Two inputs plus bias, all wired to one output - NEAT's minimal start."""
    return NEATGenome(
        node_genes=(
            NodeGene(node_id=0, node_type="input", activation_function_name="identity"),
            NodeGene(node_id=1, node_type="input", activation_function_name="identity"),
            NodeGene(node_id=2, node_type="bias", activation_function_name="identity"),
            NodeGene(node_id=3, node_type="output", activation_function_name="sigmoid"),
        ),
        connection_genes=(
            ConnectionGene(
                innovation_id=_INPUT_ZERO_TO_OUTPUT_INNOVATION_ID,
                source_node_id=0,
                target_node_id=3,
                weight=0.1,
                is_enabled=True,
            ),
            ConnectionGene(
                innovation_id=1, source_node_id=1, target_node_id=3, weight=0.2, is_enabled=True
            ),
            ConnectionGene(
                innovation_id=_BIAS_TO_OUTPUT_INNOVATION_ID,
                source_node_id=2,
                target_node_id=3,
                weight=0.3,
                is_enabled=True,
            ),
        ),
    )


def _tracker_seeded_for(genome: NEATGenome) -> GlobalInnovationTracker:
    """Tracker whose counter has advanced past the genome's existing edges."""
    tracker = GlobalInnovationTracker()
    for connection_gene in genome.connection_genes:
        tracker.get_or_assign_innovation_id_for_connection(
            source_node_id=connection_gene.source_node_id,
            target_node_id=connection_gene.target_node_id,
        )
    return tracker


class _SplitChosenConnectionMutation(AddNodeMutation):
    """AddNodeMutation that always splits one chosen connection, so tests are exact."""

    def __init__(self, innovation_id_to_split: int) -> None:
        super().__init__(
            probability_of_application=1.0,
            activation_function_name_for_new_hidden_node="sigmoid",
        )
        self._innovation_id_to_split = innovation_id_to_split

    def _choose_connection_to_split(self, enabled_connection_genes, rng):
        for connection_gene in enabled_connection_genes:
            if connection_gene.innovation_id == self._innovation_id_to_split:
                return connection_gene
        raise AssertionError("chosen connection is not enabled in this genome")


def _hidden_node_ids(genome: NEATGenome) -> list[int]:
    return [node.node_id for node in genome.node_genes if node.node_type == "hidden"]


def _new_connections(mutated: NEATGenome, original: NEATGenome) -> list[ConnectionGene]:
    original_innovation_ids = {gene.innovation_id for gene in original.connection_genes}
    return [
        gene
        for gene in mutated.connection_genes
        if gene.innovation_id not in original_innovation_ids
    ]


def test_splitting_different_connections_yields_different_node_ids() -> None:
    """The historical-marking guarantee: different structure, different node id."""
    genome = _minimal_start_genome()
    tracker = _tracker_seeded_for(genome)
    rng = np.random.default_rng(0)

    split_of_bias_edge = _SplitChosenConnectionMutation(
        _BIAS_TO_OUTPUT_INNOVATION_ID
    ).apply_to_genome(genome, rng, tracker)
    split_of_input_edge = _SplitChosenConnectionMutation(
        _INPUT_ZERO_TO_OUTPUT_INNOVATION_ID
    ).apply_to_genome(genome, rng, tracker)

    assert _hidden_node_ids(split_of_bias_edge) != _hidden_node_ids(split_of_input_edge)


def test_splitting_different_connections_shares_no_innovation_ids() -> None:
    """Aliased innovation ids would make crossover align unrelated structure."""
    genome = _minimal_start_genome()
    tracker = _tracker_seeded_for(genome)
    rng = np.random.default_rng(0)

    split_of_bias_edge = _SplitChosenConnectionMutation(
        _BIAS_TO_OUTPUT_INNOVATION_ID
    ).apply_to_genome(genome, rng, tracker)
    split_of_input_edge = _SplitChosenConnectionMutation(
        _INPUT_ZERO_TO_OUTPUT_INNOVATION_ID
    ).apply_to_genome(genome, rng, tracker)

    innovations_from_bias_split = {
        gene.innovation_id for gene in _new_connections(split_of_bias_edge, genome)
    }
    innovations_from_input_split = {
        gene.innovation_id for gene in _new_connections(split_of_input_edge, genome)
    }
    assert innovations_from_bias_split.isdisjoint(innovations_from_input_split)


def test_splitting_the_same_connection_twice_reuses_node_and_innovation_ids() -> None:
    """Same structural mutation in one generation must get the same markings."""
    genome = _minimal_start_genome()
    tracker = _tracker_seeded_for(genome)
    rng = np.random.default_rng(0)

    mutation = _SplitChosenConnectionMutation(_BIAS_TO_OUTPUT_INNOVATION_ID)
    first_child = mutation.apply_to_genome(genome, rng, tracker)
    second_child = mutation.apply_to_genome(genome, rng, tracker)

    assert _hidden_node_ids(first_child) == _hidden_node_ids(second_child)
    assert {gene.innovation_id for gene in _new_connections(first_child, genome)} == {
        gene.innovation_id for gene in _new_connections(second_child, genome)
    }


def test_new_node_id_does_not_collide_with_existing_nodes() -> None:
    genome = _minimal_start_genome()
    tracker = _tracker_seeded_for(genome)
    mutated = _SplitChosenConnectionMutation(_BIAS_TO_OUTPUT_INNOVATION_ID).apply_to_genome(
        genome, np.random.default_rng(0), tracker
    )
    existing_node_ids = {node.node_id for node in genome.node_genes}
    assert set(_hidden_node_ids(mutated)).isdisjoint(existing_node_ids)


def test_split_connection_is_disabled_and_replaced_by_two_connections() -> None:
    genome = _minimal_start_genome()
    mutated = _SplitChosenConnectionMutation(_BIAS_TO_OUTPUT_INNOVATION_ID).apply_to_genome(
        genome, np.random.default_rng(0), _tracker_seeded_for(genome)
    )
    split_gene = mutated.get_connection_gene_by_innovation_id(_BIAS_TO_OUTPUT_INNOVATION_ID)
    assert split_gene is not None
    assert not split_gene.is_enabled
    assert len(_new_connections(mutated, genome)) == 2


def test_new_connection_weights_follow_the_paper() -> None:
    """Section 3.1: weight 1.0 into the new node, original weight out of it."""
    genome = _minimal_start_genome()
    original_weight = genome.get_connection_gene_by_innovation_id(
        _BIAS_TO_OUTPUT_INNOVATION_ID
    ).weight
    mutated = _SplitChosenConnectionMutation(_BIAS_TO_OUTPUT_INNOVATION_ID).apply_to_genome(
        genome, np.random.default_rng(0), _tracker_seeded_for(genome)
    )
    (new_hidden_node_id,) = _hidden_node_ids(mutated)
    new_connections = _new_connections(mutated, genome)
    (connection_into_new_node,) = [
        gene for gene in new_connections if gene.target_node_id == new_hidden_node_id
    ]
    (connection_out_of_new_node,) = [
        gene for gene in new_connections if gene.source_node_id == new_hidden_node_id
    ]
    assert connection_into_new_node.weight == 1.0
    assert connection_out_of_new_node.weight == original_weight


def test_mutation_does_not_fire_when_probability_is_zero() -> None:
    genome = _minimal_start_genome()
    mutation = AddNodeMutation(
        probability_of_application=0.0,
        activation_function_name_for_new_hidden_node="sigmoid",
    )
    assert (
        mutation.apply_to_genome(genome, np.random.default_rng(0), _tracker_seeded_for(genome))
        is genome
    )


def test_genome_without_enabled_connections_is_returned_unchanged() -> None:
    genome = NEATGenome(
        node_genes=(
            NodeGene(node_id=0, node_type="input", activation_function_name="identity"),
            NodeGene(node_id=1, node_type="output", activation_function_name="sigmoid"),
        ),
        connection_genes=(
            ConnectionGene(
                innovation_id=0, source_node_id=0, target_node_id=1, weight=0.5, is_enabled=False
            ),
        ),
    )
    mutation = AddNodeMutation(
        probability_of_application=1.0,
        activation_function_name_for_new_hidden_node="sigmoid",
    )
    assert (
        mutation.apply_to_genome(genome, np.random.default_rng(0), GlobalInnovationTracker())
        is genome
    )


def test_same_split_from_genomes_with_different_node_id_ranges_stays_valid() -> None:
    """Two genomes splitting one edge share a node id; it must clash with neither.

    The cached id is reused across genomes, so a genome carrying a *higher*
    hidden node id than the one that first triggered the split must not receive
    an id it already owns. Safe because the tracker's counter always exceeds
    every id it has issued, but that invariant is worth pinning.
    """
    lean_genome = _minimal_start_genome()
    tracker = _tracker_seeded_for(lean_genome)

    # A genome that already carries a high hidden node id from an earlier split.
    established_split = tracker.get_or_assign_node_split(
        split_connection_innovation_id=1, minimum_new_node_id=4
    )
    high_node_id = established_split.new_node_id
    bulky_genome = NEATGenome(
        node_genes=lean_genome.node_genes
        + (NodeGene(node_id=high_node_id, node_type="hidden", activation_function_name="sigmoid"),),
        connection_genes=lean_genome.connection_genes
        + (
            ConnectionGene(
                innovation_id=established_split.innovation_id_into_new_node,
                source_node_id=1,
                target_node_id=high_node_id,
                weight=1.0,
                is_enabled=True,
            ),
        ),
    )
    tracker.reset_for_new_generation()

    mutation = _SplitChosenConnectionMutation(_BIAS_TO_OUTPUT_INNOVATION_ID)
    rng = np.random.default_rng(0)
    from_lean = mutation.apply_to_genome(lean_genome, rng, tracker)
    from_bulky = mutation.apply_to_genome(bulky_genome, rng, tracker)

    shared_new_node_id = (
        set(_hidden_node_ids(from_lean)) & set(_hidden_node_ids(from_bulky))
    )
    assert shared_new_node_id, "the same split must yield the same node id"
    assert shared_new_node_id.isdisjoint({high_node_id})
    assert len({node.node_id for node in from_bulky.node_genes}) == len(from_bulky.node_genes)
