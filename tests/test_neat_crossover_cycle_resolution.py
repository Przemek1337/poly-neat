"""Cycle resolution during crossover: semantics and cost.

``NEATCrossover`` may combine two parents' genes into a child holding a
directed cycle, which this library's feed-forward phenotype cannot execute.
``_resolve_enabled_connection_cycles`` breaks such cycles by disabling the
offending genes, greedily in innovation order. These tests pin that behaviour
and guard the cost of computing it, which is quadratic in the number of genes
unless the reachability adjacency is carried across the scan.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from polyneat.core.neat.neat_crossover import (
    NEATCrossover,
    _resolve_enabled_connection_cycles,
)
from polyneat.core.neat.neat_genome import ConnectionGene, NEATGenome, NodeGene
from polyneat.nn.topology_utilities import (
    build_outgoing_adjacency_from_directed_edges,
    would_directed_edge_create_cycle,
    would_directed_edge_create_cycle_in_adjacency,
)

WIDE_GENOME_INPUT_COUNT = 7070
"""Feature count of the leukemia microarray set, the widest in the sweep."""

WIDE_GENOME_OUTPUT_COUNT = 2

CYCLE_RESOLUTION_TIME_BUDGET_SECONDS = 2.0
"""Generous ceiling: the carried-adjacency scan runs this in under 0.05 s."""


def _connection(
    innovation_id: int,
    source_node_id: int,
    target_node_id: int,
    *,
    weight: float = 0.5,
    is_enabled: bool = True,
) -> ConnectionGene:
    return ConnectionGene(
        innovation_id=innovation_id,
        source_node_id=source_node_id,
        target_node_id=target_node_id,
        weight=weight,
        is_enabled=is_enabled,
    )


def _build_wide_fully_connected_genome(weight_offset: float) -> NEATGenome:
    """A generation-0 genome shaped like leukemia's: every input wired to every output.

    Mirrors ``build_fully_connected_initial_population`` — the paper's minimal
    start (Stanley & Miikkulainen, 2002, section 3.4) — without paying for a
    whole population.
    """
    bias_node_id = WIDE_GENOME_INPUT_COUNT
    first_output_node_id = WIDE_GENOME_INPUT_COUNT + 1

    node_genes = [
        NodeGene(node_id=node_id, node_type="input", activation_function_name="identity")
        for node_id in range(WIDE_GENOME_INPUT_COUNT)
    ]
    node_genes.append(
        NodeGene(node_id=bias_node_id, node_type="bias", activation_function_name="identity")
    )
    node_genes.extend(
        NodeGene(
            node_id=first_output_node_id + output_index,
            node_type="output",
            activation_function_name="steepened_sigmoid",
        )
        for output_index in range(WIDE_GENOME_OUTPUT_COUNT)
    )

    connection_genes: list[ConnectionGene] = []
    next_innovation_id = 0
    for source_node_id in [*range(WIDE_GENOME_INPUT_COUNT), bias_node_id]:
        for output_index in range(WIDE_GENOME_OUTPUT_COUNT):
            connection_genes.append(
                _connection(
                    next_innovation_id,
                    source_node_id,
                    first_output_node_id + output_index,
                    weight=weight_offset + next_innovation_id * 1e-6,
                )
            )
            next_innovation_id += 1

    return NEATGenome(node_genes=tuple(node_genes), connection_genes=tuple(connection_genes))


class TestWouldDirectedEdgeCreateCycleInAdjacency:
    """The adjacency-taking form callers use to avoid rebuilding per query."""

    def test_reports_a_self_loop_as_a_cycle(self) -> None:
        assert would_directed_edge_create_cycle_in_adjacency(
            candidate_source_node_id=3,
            candidate_target_node_id=3,
            outgoing_targets_by_source_node_id={},
        )

    def test_reports_a_cycle_when_the_target_reaches_back_to_the_source(self) -> None:
        assert would_directed_edge_create_cycle_in_adjacency(
            candidate_source_node_id=1,
            candidate_target_node_id=2,
            outgoing_targets_by_source_node_id={2: [3], 3: [1]},
        )

    def test_reports_no_cycle_when_the_target_cannot_reach_the_source(self) -> None:
        assert not would_directed_edge_create_cycle_in_adjacency(
            candidate_source_node_id=1,
            candidate_target_node_id=2,
            outgoing_targets_by_source_node_id={2: [3], 3: [4]},
        )

    def test_leaves_the_supplied_adjacency_untouched(self) -> None:
        """Callers carry one adjacency across many queries; probing must not grow it."""
        adjacency = {2: [3]}
        would_directed_edge_create_cycle_in_adjacency(
            candidate_source_node_id=1,
            candidate_target_node_id=2,
            outgoing_targets_by_source_node_id=adjacency,
        )
        assert adjacency == {2: [3]}

    def test_agrees_with_the_edge_list_form(self) -> None:
        edges = [(2, 3), (3, 1), (5, 6)]
        for source_node_id, target_node_id in [(1, 2), (1, 5), (6, 5), (4, 4)]:
            assert would_directed_edge_create_cycle_in_adjacency(
                candidate_source_node_id=source_node_id,
                candidate_target_node_id=target_node_id,
                outgoing_targets_by_source_node_id=(
                    build_outgoing_adjacency_from_directed_edges(edges)
                ),
            ) == would_directed_edge_create_cycle(
                candidate_source_node_id=source_node_id,
                candidate_target_node_id=target_node_id,
                existing_enabled_edges=edges,
            )


class TestResolveEnabledConnectionCycles:
    """Semantics of the greedy innovation-order pass."""

    def test_disables_the_later_gene_that_closes_a_cycle(self) -> None:
        genes = [
            _connection(1, 1, 2),
            _connection(2, 2, 3),
            _connection(3, 3, 1),
        ]
        resolved = _resolve_enabled_connection_cycles(genes)
        assert [gene.is_enabled for gene in resolved] == [True, True, False]

    def test_keeps_an_acyclic_gene_set_fully_enabled(self) -> None:
        genes = [
            _connection(1, 1, 3),
            _connection(2, 2, 3),
            _connection(3, 3, 4),
        ]
        resolved = _resolve_enabled_connection_cycles(genes)
        assert all(gene.is_enabled for gene in resolved)

    def test_resolves_in_innovation_order_not_list_order(self) -> None:
        """The lower innovation id is accepted even when it appears later in the list."""
        genes = [
            _connection(9, 2, 1),
            _connection(4, 1, 2),
        ]
        resolved = _resolve_enabled_connection_cycles(genes)
        assert [gene.innovation_id for gene in resolved] == [9, 4]
        assert [gene.is_enabled for gene in resolved] == [False, True]

    def test_passes_already_disabled_genes_through_untouched(self) -> None:
        genes = [
            _connection(1, 1, 2),
            _connection(2, 2, 1, is_enabled=False),
        ]
        resolved = _resolve_enabled_connection_cycles(genes)
        assert resolved[1] == genes[1]

    def test_ignores_disabled_genes_when_deciding_later_cycles(self) -> None:
        """A disabled 2->3 must not make 3->1 look like a cycle."""
        genes = [
            _connection(1, 1, 2),
            _connection(2, 2, 3, is_enabled=False),
            _connection(3, 3, 1),
        ]
        resolved = _resolve_enabled_connection_cycles(genes)
        assert [gene.is_enabled for gene in resolved] == [True, False, True]

    def test_resolves_a_wide_generation_zero_gene_set_within_the_time_budget(self) -> None:
        """14,142 input->output genes hold no cycle; proving that must stay cheap.

        Rebuilding the reachability adjacency per gene makes this quadratic —
        roughly 100 million list appends, about 18 s — which dominated every
        generation of a sweep run on the microarray datasets.
        """
        genes = list(_build_wide_fully_connected_genome(0.5).connection_genes)

        start_time = time.perf_counter()
        resolved = _resolve_enabled_connection_cycles(genes)
        elapsed_seconds = time.perf_counter() - start_time

        assert resolved == genes
        assert elapsed_seconds < CYCLE_RESOLUTION_TIME_BUDGET_SECONDS


class TestCrossoverAtGenerationZeroWidth:
    """The user-facing path: mating two leukemia-shaped parents."""

    def test_mating_two_wide_parents_completes_within_the_time_budget(self) -> None:
        fitter_parent = _build_wide_fully_connected_genome(0.5)
        less_fit_parent = _build_wide_fully_connected_genome(-0.5)
        crossover = NEATCrossover(
            probability_of_inheriting_from_fitter_parent_for_matching_genes=0.5
        )
        rng = np.random.default_rng(7)

        start_time = time.perf_counter()
        child = crossover.apply_to_parents(fitter_parent, less_fit_parent, rng)
        elapsed_seconds = time.perf_counter() - start_time

        assert len(child.connection_genes) == len(fitter_parent.connection_genes)
        assert all(gene.is_enabled for gene in child.connection_genes)
        assert elapsed_seconds < CYCLE_RESOLUTION_TIME_BUDGET_SECONDS

    def test_child_stays_acyclic_when_parents_disagree_on_edge_direction(self) -> None:
        node_genes = (
            NodeGene(node_id=0, node_type="input", activation_function_name="identity"),
            NodeGene(node_id=1, node_type="hidden", activation_function_name="steepened_sigmoid"),
            NodeGene(node_id=2, node_type="hidden", activation_function_name="steepened_sigmoid"),
            NodeGene(node_id=3, node_type="output", activation_function_name="steepened_sigmoid"),
        )
        fitter_parent = NEATGenome(
            node_genes=node_genes,
            connection_genes=(
                _connection(1, 0, 1),
                _connection(2, 1, 2),
                _connection(3, 2, 1),
                _connection(4, 2, 3),
            ),
        )
        less_fit_parent = NEATGenome(
            node_genes=node_genes,
            connection_genes=(_connection(1, 0, 1),),
        )
        crossover = NEATCrossover(
            probability_of_inheriting_from_fitter_parent_for_matching_genes=1.0
        )

        child = crossover.apply_to_parents(
            fitter_parent, less_fit_parent, np.random.default_rng(1)
        )

        enabled_by_innovation_id = {
            gene.innovation_id: gene.is_enabled for gene in child.connection_genes
        }
        assert enabled_by_innovation_id[3] is False
        assert enabled_by_innovation_id[2] is True


@pytest.mark.parametrize(
    ("existing_edges", "candidate_edge", "expected_cycle"),
    [
        ([], (1, 2), False),
        ([(2, 1)], (1, 2), True),
        ([(2, 3), (3, 4)], (1, 2), False),
        ([(2, 3), (3, 1)], (1, 2), True),
        ([], (1, 1), True),
    ],
)
def test_edge_list_form_keeps_its_public_behaviour(
    existing_edges: list[tuple[int, int]],
    candidate_edge: tuple[int, int],
    expected_cycle: bool,
) -> None:
    """``AddConnectionMutation`` and the toggle mutation still call this form."""
    candidate_source_node_id, candidate_target_node_id = candidate_edge
    assert (
        would_directed_edge_create_cycle(
            candidate_source_node_id=candidate_source_node_id,
            candidate_target_node_id=candidate_target_node_id,
            existing_enabled_edges=existing_edges,
        )
        is expected_cycle
    )
