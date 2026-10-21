from __future__ import annotations

import pytest

from polyneat.algorithms.neatdbm.difference_based_weight_mutation import (
    DifferenceBasedWeightMutation,
)
from polyneat.core.neat.neat_genome import ConnectionGene, NEATGenome, NodeGene


def _build_genome_with_connection_weights(
    weights_by_innovation_id: dict[int, float],
    disabled_innovation_ids: frozenset[int] = frozenset(),
) -> NEATGenome:
    """Build a minimal genome carrying one connection per given innovation id."""
    node_genes = (
        NodeGene(node_id=0, node_type="bias", activation_function_name="sigmoid"),
        NodeGene(node_id=1, node_type="input", activation_function_name="sigmoid"),
        NodeGene(node_id=2, node_type="output", activation_function_name="sigmoid"),
    )
    connection_genes = tuple(
        ConnectionGene(
            innovation_id=innovation_id,
            source_node_id=0 if position % 2 == 0 else 1,
            target_node_id=2,
            weight=weight,
            is_enabled=innovation_id not in disabled_innovation_ids,
        )
        for position, (innovation_id, weight) in enumerate(
            sorted(weights_by_innovation_id.items())
        )
    )
    return NEATGenome(node_genes=node_genes, connection_genes=connection_genes)


def test_shared_gene_weight_follows_difference_formula() -> None:
    # w_new = w_r1 + F * (w_r2 - w_r3) at the shared innovation id (paper, section 3)
    target_genome = _build_genome_with_connection_weights({0: 0.2, 1: 0.9})
    donor_base = _build_genome_with_connection_weights({0: 0.5})
    donor_first = _build_genome_with_connection_weights({0: 0.8})
    donor_second = _build_genome_with_connection_weights({0: 0.1})

    mutated_genome = DifferenceBasedWeightMutation(
        scaling_factor=0.5
    ).apply_to_genome_with_donors(
        genome=target_genome,
        donor_base_genome=donor_base,
        donor_difference_first_genome=donor_first,
        donor_difference_second_genome=donor_second,
    )

    shared_gene = mutated_genome.get_connection_gene_by_innovation_id(0)
    assert shared_gene is not None
    assert shared_gene.weight == pytest.approx(0.5 + 0.5 * (0.8 - 0.1))
    # gene 1 exists only in the target, so it does not participate
    unshared_gene = mutated_genome.get_connection_gene_by_innovation_id(1)
    assert unshared_gene is not None
    assert unshared_gene.weight == 0.9


def test_returns_genome_unchanged_when_no_shared_innovation_id() -> None:
    target_genome = _build_genome_with_connection_weights({0: 0.2})
    donor_base = _build_genome_with_connection_weights({1: 0.5})
    donor_first = _build_genome_with_connection_weights({0: 0.8})
    donor_second = _build_genome_with_connection_weights({0: 0.1})

    mutated_genome = DifferenceBasedWeightMutation(
        scaling_factor=0.5
    ).apply_to_genome_with_donors(
        genome=target_genome,
        donor_base_genome=donor_base,
        donor_difference_first_genome=donor_first,
        donor_difference_second_genome=donor_second,
    )

    assert mutated_genome is target_genome


def test_positions_with_identical_weights_everywhere_are_skipped() -> None:
    # "The genes having the same weight values are not considered" (paper, section 3)
    target_genome = _build_genome_with_connection_weights({0: 0.3})
    donor_base = _build_genome_with_connection_weights({0: 0.3})
    donor_first = _build_genome_with_connection_weights({0: 0.3})
    donor_second = _build_genome_with_connection_weights({0: 0.3})

    mutated_genome = DifferenceBasedWeightMutation(
        scaling_factor=0.5
    ).apply_to_genome_with_donors(
        genome=target_genome,
        donor_base_genome=donor_base,
        donor_difference_first_genome=donor_first,
        donor_difference_second_genome=donor_second,
    )

    assert mutated_genome is target_genome


def test_gene_missing_in_one_donor_does_not_participate() -> None:
    target_genome = _build_genome_with_connection_weights({0: 0.2, 1: 0.4})
    donor_base = _build_genome_with_connection_weights({0: 0.5, 1: 0.6})
    donor_first = _build_genome_with_connection_weights({0: 0.8, 1: 0.7})
    donor_second = _build_genome_with_connection_weights({0: 0.1})  # id 1 missing

    mutated_genome = DifferenceBasedWeightMutation(
        scaling_factor=0.5
    ).apply_to_genome_with_donors(
        genome=target_genome,
        donor_base_genome=donor_base,
        donor_difference_first_genome=donor_first,
        donor_difference_second_genome=donor_second,
    )

    participating_gene = mutated_genome.get_connection_gene_by_innovation_id(0)
    assert participating_gene is not None
    assert participating_gene.weight == pytest.approx(0.5 + 0.5 * (0.8 - 0.1))
    excluded_gene = mutated_genome.get_connection_gene_by_innovation_id(1)
    assert excluded_gene is not None
    assert excluded_gene.weight == 0.4


def test_topology_and_enabled_flags_are_preserved() -> None:
    target_genome = _build_genome_with_connection_weights(
        {0: 0.2, 1: 0.9}, disabled_innovation_ids=frozenset({0})
    )
    donor_base = _build_genome_with_connection_weights({0: 0.5})
    donor_first = _build_genome_with_connection_weights({0: 0.8})
    donor_second = _build_genome_with_connection_weights({0: 0.1})

    mutated_genome = DifferenceBasedWeightMutation(
        scaling_factor=0.5
    ).apply_to_genome_with_donors(
        genome=target_genome,
        donor_base_genome=donor_base,
        donor_difference_first_genome=donor_first,
        donor_difference_second_genome=donor_second,
    )

    assert mutated_genome.node_genes == target_genome.node_genes
    for original_gene, mutated_gene in zip(
        target_genome.connection_genes, mutated_genome.connection_genes, strict=True
    ):
        assert mutated_gene.innovation_id == original_gene.innovation_id
        assert mutated_gene.source_node_id == original_gene.source_node_id
        assert mutated_gene.target_node_id == original_gene.target_node_id
        assert mutated_gene.is_enabled == original_gene.is_enabled
