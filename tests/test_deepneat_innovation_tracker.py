from __future__ import annotations

from polyneat.algorithms.deepneat.deepneat_innovation_tracker import (
    DeepNEATInnovationTracker,
)


def test_same_edge_gets_the_same_innovation_id_within_a_generation() -> None:
    tracker = DeepNEATInnovationTracker()
    first = tracker.get_or_assign_innovation_id_for_connection(
        source_node_id=0, target_node_id=1
    )
    second = tracker.get_or_assign_innovation_id_for_connection(
        source_node_id=0, target_node_id=1
    )
    assert first == second


def test_different_edges_get_different_innovation_ids() -> None:
    tracker = DeepNEATInnovationTracker()
    first = tracker.get_or_assign_innovation_id_for_connection(
        source_node_id=0, target_node_id=1
    )
    second = tracker.get_or_assign_innovation_id_for_connection(
        source_node_id=0, target_node_id=2
    )
    assert first != second


def test_markings_survive_a_generation_boundary() -> None:
    # DeepNEAT keeps a master innovation list for the whole search, like EXACT:
    # architectures are compared across many generations, so re-issuing an id
    # for an edge seen earlier would misalign crossover.
    tracker = DeepNEATInnovationTracker()
    before = tracker.get_or_assign_innovation_id_for_connection(
        source_node_id=0, target_node_id=1
    )
    tracker.reset_for_new_generation()
    after = tracker.get_or_assign_innovation_id_for_connection(
        source_node_id=0, target_node_id=1
    )
    assert before == after


def test_splitting_the_same_edge_twice_reuses_the_record() -> None:
    tracker = DeepNEATInnovationTracker()
    first = tracker.get_or_assign_node_split(
        split_connection_innovation_id=0, minimum_new_node_id=3
    )
    second = tracker.get_or_assign_node_split(
        split_connection_innovation_id=0, minimum_new_node_id=3
    )
    assert first == second


def test_splitting_different_edges_yields_distinct_node_ids() -> None:
    tracker = DeepNEATInnovationTracker()
    first = tracker.get_or_assign_node_split(
        split_connection_innovation_id=0, minimum_new_node_id=3
    )
    second = tracker.get_or_assign_node_split(
        split_connection_innovation_id=1, minimum_new_node_id=3
    )
    assert first.new_node_id != second.new_node_id
    assert first.innovation_id_into_new_node != second.innovation_id_into_new_node
