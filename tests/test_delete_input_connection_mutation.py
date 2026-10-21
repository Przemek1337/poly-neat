from __future__ import annotations

import numpy as np
import pytest

from polyneat.algorithms.fdneat.mutations.delete_input_connection_mutation import (
    DeleteInputConnectionMutation,
)
from polyneat.core.neat.global_innovation_tracker import GlobalInnovationTracker
from polyneat.core.neat.neat_genome import ConnectionGene, NEATGenome, NodeGene

# Node layout used by every test below:
#   0, 1 -> input      2 -> bias      3 -> output      4 -> hidden
_NODE_GENES = (
    NodeGene(node_id=0, node_type="input", activation_function_name="identity"),
    NodeGene(node_id=1, node_type="input", activation_function_name="identity"),
    NodeGene(node_id=2, node_type="bias", activation_function_name="identity"),
    NodeGene(node_id=3, node_type="output", activation_function_name="sigmoid"),
    NodeGene(node_id=4, node_type="hidden", activation_function_name="sigmoid"),
)


def _genome_with(connection_genes: tuple[ConnectionGene, ...]) -> NEATGenome:
    return NEATGenome(node_genes=_NODE_GENES, connection_genes=connection_genes)


def _full_genome() -> NEATGenome:
    """Two deletable candidates (innov 0 and 1) plus three protected genes."""
    return _genome_with(
        (
            ConnectionGene(
                innovation_id=0, source_node_id=0, target_node_id=3, weight=0.5, is_enabled=True
            ),
            ConnectionGene(
                innovation_id=1, source_node_id=1, target_node_id=3, weight=0.6, is_enabled=True
            ),
            ConnectionGene(
                innovation_id=2, source_node_id=2, target_node_id=3, weight=0.7, is_enabled=True
            ),
            ConnectionGene(
                innovation_id=3, source_node_id=4, target_node_id=3, weight=0.8, is_enabled=True
            ),
            ConnectionGene(
                innovation_id=4, source_node_id=0, target_node_id=4, weight=0.9, is_enabled=False
            ),
        )
    )


@pytest.fixture
def tracker() -> GlobalInnovationTracker:
    return GlobalInnovationTracker()


def test_does_not_fire_when_draw_is_above_probability(tracker) -> None:
    genome = _full_genome()
    mutation = DeleteInputConnectionMutation(probability_of_application=0.0)
    result = mutation.apply_to_genome(genome, np.random.default_rng(0), tracker)
    assert result.connection_genes == genome.connection_genes


def test_removes_exactly_one_connection_gene(tracker) -> None:
    genome = _full_genome()
    mutation = DeleteInputConnectionMutation(probability_of_application=1.0)
    result = mutation.apply_to_genome(genome, np.random.default_rng(0), tracker)
    assert len(result.connection_genes) == len(genome.connection_genes) - 1


def test_gene_is_removed_not_merely_disabled(tracker) -> None:
    genome = _full_genome()
    mutation = DeleteInputConnectionMutation(probability_of_application=1.0)
    result = mutation.apply_to_genome(genome, np.random.default_rng(0), tracker)

    surviving_innovation_ids = {gene.innovation_id for gene in result.connection_genes}
    original_innovation_ids = {gene.innovation_id for gene in genome.connection_genes}
    removed_innovation_ids = original_innovation_ids - surviving_innovation_ids

    assert len(removed_innovation_ids) == 1
    # Gene 4 was already disabled; every other survivor must still be enabled,
    # proving the operator deleted rather than switched something off.
    assert all(
        gene.is_enabled for gene in result.connection_genes if gene.innovation_id != 4
    )


@pytest.mark.parametrize("seed", range(30))
def test_never_removes_bias_or_hidden_sourced_connections(tracker, seed: int) -> None:
    genome = _full_genome()
    mutation = DeleteInputConnectionMutation(probability_of_application=1.0)
    result = mutation.apply_to_genome(genome, np.random.default_rng(seed), tracker)
    surviving = {gene.innovation_id for gene in result.connection_genes}
    assert 2 in surviving, "bias-sourced connection must never be deleted"
    assert 3 in surviving, "hidden-sourced connection must never be deleted"


@pytest.mark.parametrize("seed", range(30))
def test_only_enabled_input_sourced_connections_are_candidates(tracker, seed: int) -> None:
    genome = _full_genome()
    mutation = DeleteInputConnectionMutation(probability_of_application=1.0)
    result = mutation.apply_to_genome(genome, np.random.default_rng(seed), tracker)
    removed = {gene.innovation_id for gene in genome.connection_genes} - {
        gene.innovation_id for gene in result.connection_genes
    }
    assert removed <= {0, 1}, f"deleted a non-candidate gene: {removed}"


def test_genome_with_single_connection_is_left_alone(tracker) -> None:
    genome = _genome_with(
        (
            ConnectionGene(
                innovation_id=0, source_node_id=0, target_node_id=3, weight=0.5, is_enabled=True
            ),
        )
    )
    mutation = DeleteInputConnectionMutation(probability_of_application=1.0)
    result = mutation.apply_to_genome(genome, np.random.default_rng(0), tracker)
    assert result.connection_genes == genome.connection_genes


def test_genome_without_enabled_input_connections_is_left_alone(tracker) -> None:
    genome = _genome_with(
        (
            ConnectionGene(
                innovation_id=0, source_node_id=0, target_node_id=3, weight=0.5, is_enabled=False
            ),
            ConnectionGene(
                innovation_id=2, source_node_id=2, target_node_id=3, weight=0.7, is_enabled=True
            ),
        )
    )
    mutation = DeleteInputConnectionMutation(probability_of_application=1.0)
    result = mutation.apply_to_genome(genome, np.random.default_rng(0), tracker)
    assert result.connection_genes == genome.connection_genes


def test_original_genome_is_not_mutated_in_place(tracker) -> None:
    genome = _full_genome()
    genes_before = genome.connection_genes
    mutation = DeleteInputConnectionMutation(probability_of_application=1.0)
    mutation.apply_to_genome(genome, np.random.default_rng(0), tracker)
    assert genome.connection_genes is genes_before
    assert len(genome.connection_genes) == 5


def test_result_passes_genome_validation(tracker) -> None:
    genome = _full_genome()
    mutation = DeleteInputConnectionMutation(probability_of_application=1.0)
    result = mutation.apply_to_genome(genome, np.random.default_rng(0), tracker)
    # __post_init__ validates on construction; rebuilding must not raise.
    NEATGenome(node_genes=result.node_genes, connection_genes=result.connection_genes)


def test_node_genes_are_carried_over_untouched(tracker) -> None:
    genome = _full_genome()
    mutation = DeleteInputConnectionMutation(probability_of_application=1.0)
    result = mutation.apply_to_genome(genome, np.random.default_rng(0), tracker)
    assert result.node_genes == _NODE_GENES


def test_tracker_is_left_untouched(tracker) -> None:
    # The operator creates no structure, so it must not consume innovation ids.
    innovation_counter_before = tracker.next_innovation_id_snapshot
    mutation = DeleteInputConnectionMutation(probability_of_application=1.0)
    mutation.apply_to_genome(_full_genome(), np.random.default_rng(0), tracker)
    assert tracker.next_innovation_id_snapshot == innovation_counter_before


def test_repeated_application_eventually_stops_at_one_connection(tracker) -> None:
    genome = _genome_with(
        (
            ConnectionGene(
                innovation_id=0, source_node_id=0, target_node_id=3, weight=0.5, is_enabled=True
            ),
            ConnectionGene(
                innovation_id=1, source_node_id=1, target_node_id=3, weight=0.6, is_enabled=True
            ),
        )
    )
    mutation = DeleteInputConnectionMutation(probability_of_application=1.0)
    rng = np.random.default_rng(0)
    for _application in range(10):
        genome = mutation.apply_to_genome(genome, rng, tracker)
    assert len(genome.connection_genes) == 1
