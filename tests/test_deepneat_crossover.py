from __future__ import annotations

import numpy as np
import pytest

from polyneat.algorithms.deepneat.deepneat_crossover import DeepNEATCrossover
from polyneat.algorithms.deepneat.deepneat_genome import (
    DeepNEATGenome,
    LayerNodeGene,
    TensorEdgeGene,
)

_INPUT = LayerNodeGene(node_id=0, layer_type="input")
_OUTPUT = LayerNodeGene(node_id=1, layer_type="output")


def _conv(node_id: int, filters: int) -> LayerNodeGene:
    return LayerNodeGene(
        node_id=node_id, layer_type="conv", number_of_filters=filters, kernel_size=3
    )


def _fitter_parent() -> DeepNEATGenome:
    return DeepNEATGenome(
        node_genes=(_INPUT, _OUTPUT, _conv(2, 16), _conv(3, 32)),
        edge_genes=(
            TensorEdgeGene(innovation_id=0, source_node_id=0, target_node_id=2,
                           is_enabled=True),
            TensorEdgeGene(innovation_id=1, source_node_id=2, target_node_id=3,
                           is_enabled=True),
            TensorEdgeGene(innovation_id=2, source_node_id=3, target_node_id=1,
                           is_enabled=True),
            TensorEdgeGene(innovation_id=7, source_node_id=0, target_node_id=1,
                           is_enabled=True),
        ),
    )


def _other_parent() -> DeepNEATGenome:
    return DeepNEATGenome(
        node_genes=(_INPUT, _OUTPUT, _conv(2, 128)),
        edge_genes=(
            TensorEdgeGene(innovation_id=0, source_node_id=0, target_node_id=2,
                           is_enabled=True),
            TensorEdgeGene(innovation_id=5, source_node_id=2, target_node_id=1,
                           is_enabled=True),
        ),
    )


@pytest.fixture
def crossover() -> DeepNEATCrossover:
    return DeepNEATCrossover(
        probability_of_inheriting_from_fitter_parent_for_matching_genes=0.5
    )


@pytest.mark.parametrize("seed", range(25))
def test_child_keeps_only_the_fitter_parents_disjoint_and_excess_edges(
    crossover, seed: int
) -> None:
    child = crossover.apply_to_parents(
        _fitter_parent(), _other_parent(), np.random.default_rng(seed)
    )
    child_innovation_ids = {edge.innovation_id for edge in child.edge_genes}
    assert 5 not in child_innovation_ids, "inherited a disjoint gene from the worse parent"
    assert {0, 1, 2, 7} == child_innovation_ids


@pytest.mark.parametrize("seed", range(25))
def test_every_referenced_node_exists_in_the_child(crossover, seed: int) -> None:
    child = crossover.apply_to_parents(
        _fitter_parent(), _other_parent(), np.random.default_rng(seed)
    )
    known_node_ids = {node.node_id for node in child.node_genes}
    for edge in child.edge_genes:
        assert edge.source_node_id in known_node_ids
        assert edge.target_node_id in known_node_ids


@pytest.mark.parametrize("seed", range(25))
def test_child_always_has_exactly_one_input_and_one_output(crossover, seed: int) -> None:
    child = crossover.apply_to_parents(
        _fitter_parent(), _other_parent(), np.random.default_rng(seed)
    )
    assert sum(1 for n in child.node_genes if n.layer_type == "input") == 1
    assert sum(1 for n in child.node_genes if n.layer_type == "output") == 1


def test_matching_node_hyperparameters_can_come_from_either_parent(crossover) -> None:
    observed_filter_counts = set()
    for seed in range(60):
        child = crossover.apply_to_parents(
            _fitter_parent(), _other_parent(), np.random.default_rng(seed)
        )
        node_two = child.get_node_gene_by_id(2)
        assert node_two is not None
        observed_filter_counts.add(node_two.number_of_filters)
    assert observed_filter_counts == {16, 128}


def test_node_present_in_only_the_fitter_parent_is_taken_from_it(crossover) -> None:
    for seed in range(25):
        child = crossover.apply_to_parents(
            _fitter_parent(), _other_parent(), np.random.default_rng(seed)
        )
        node_three = child.get_node_gene_by_id(3)
        if node_three is not None:
            assert node_three.number_of_filters == 32


@pytest.mark.parametrize("seed", range(25))
def test_child_is_acyclic_and_valid(crossover, seed: int) -> None:
    # Construction revalidates; a cycle or a dangling edge would raise.
    crossover.apply_to_parents(_fitter_parent(), _other_parent(), np.random.default_rng(seed))


def test_crossing_a_genome_with_itself_reproduces_it(crossover) -> None:
    parent = _fitter_parent()
    child = crossover.apply_to_parents(parent, parent, np.random.default_rng(0))
    assert {e.innovation_id for e in child.edge_genes} == {
        e.innovation_id for e in parent.edge_genes
    }
    assert {n.node_id for n in child.node_genes} == {n.node_id for n in parent.node_genes}


def test_equal_fitness_allows_unique_genes_from_either_parent(crossover) -> None:
    observed_innovation_ids: set[int] = set()
    for seed in range(100):
        child = crossover.apply_to_parents(
            _fitter_parent(),
            _other_parent(),
            np.random.default_rng(seed),
            parents_have_equal_fitness=True,
        )
        observed_innovation_ids.update(edge.innovation_id for edge in child.edge_genes)
    assert 5 in observed_innovation_ids
    assert 7 in observed_innovation_ids


def test_disabled_matching_gene_uses_neat_disabled_inheritance_rule() -> None:
    fitter = _fitter_parent()
    other = DeepNEATGenome(
        node_genes=_other_parent().node_genes,
        edge_genes=(
            TensorEdgeGene(0, 0, 2, False),
            TensorEdgeGene(5, 2, 1, True),
        ),
    )
    always_disabled = DeepNEATCrossover(
        probability_of_inheriting_from_fitter_parent_for_matching_genes=1.0,
        probability_of_child_gene_remaining_disabled_when_either_parent_disabled=1.0,
    )
    child = always_disabled.apply_to_parents(fitter, other, np.random.default_rng(0))
    matching_edge = next(edge for edge in child.edge_genes if edge.innovation_id == 0)
    assert not matching_edge.is_enabled
