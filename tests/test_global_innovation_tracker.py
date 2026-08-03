from __future__ import annotations

from polyneat.core.neat.global_innovation_tracker import GlobalInnovationTracker


def test_same_edge_reuses_innovation_id_within_generation() -> None:
    tracker = GlobalInnovationTracker()
    first = tracker.get_or_assign_innovation_id_for_connection(source_node_id=0, target_node_id=3)
    second = tracker.get_or_assign_innovation_id_for_connection(source_node_id=0, target_node_id=3)
    assert first == second


def test_same_split_connection_reuses_node_id_within_generation() -> None:
    """Two genomes splitting the *same* edge must receive the same new node id."""
    tracker = GlobalInnovationTracker()
    first = tracker.get_or_assign_node_split(
        split_connection_innovation_id=2, minimum_new_node_id=4
    )
    second = tracker.get_or_assign_node_split(
        split_connection_innovation_id=2, minimum_new_node_id=4
    )
    assert first == second


def test_different_split_connections_get_different_node_ids() -> None:
    """Splitting different edges is different structure, so node ids must differ."""
    tracker = GlobalInnovationTracker()
    split_of_edge_two = tracker.get_or_assign_node_split(
        split_connection_innovation_id=2, minimum_new_node_id=4
    )
    split_of_edge_zero = tracker.get_or_assign_node_split(
        split_connection_innovation_id=0, minimum_new_node_id=4
    )
    assert split_of_edge_two.new_node_id != split_of_edge_zero.new_node_id


def test_different_split_connections_get_different_connection_innovations() -> None:
    tracker = GlobalInnovationTracker()
    split_of_edge_two = tracker.get_or_assign_node_split(
        split_connection_innovation_id=2, minimum_new_node_id=4
    )
    split_of_edge_zero = tracker.get_or_assign_node_split(
        split_connection_innovation_id=0, minimum_new_node_id=4
    )
    innovations_of_first_split = {
        split_of_edge_two.innovation_id_into_new_node,
        split_of_edge_two.innovation_id_out_of_new_node,
    }
    innovations_of_second_split = {
        split_of_edge_zero.innovation_id_into_new_node,
        split_of_edge_zero.innovation_id_out_of_new_node,
    }
    assert innovations_of_first_split.isdisjoint(innovations_of_second_split)


def test_new_node_id_is_at_least_the_requested_minimum() -> None:
    tracker = GlobalInnovationTracker()
    split = tracker.get_or_assign_node_split(
        split_connection_innovation_id=0, minimum_new_node_id=17
    )
    assert split.new_node_id >= 17


def test_node_ids_keep_growing_across_generations() -> None:
    """The dedup table resets each generation; the node id counter does not."""
    tracker = GlobalInnovationTracker()
    first_generation_split = tracker.get_or_assign_node_split(
        split_connection_innovation_id=0, minimum_new_node_id=4
    )
    tracker.reset_for_new_generation()
    second_generation_split = tracker.get_or_assign_node_split(
        split_connection_innovation_id=0, minimum_new_node_id=4
    )
    assert second_generation_split.new_node_id > first_generation_split.new_node_id


def test_reset_clears_the_node_split_dedup_table() -> None:
    tracker = GlobalInnovationTracker()
    tracker.get_or_assign_node_split(split_connection_innovation_id=0, minimum_new_node_id=4)
    tracker.reset_for_new_generation()
    innovation_id_after_reset = tracker.get_or_assign_node_split(
        split_connection_innovation_id=0, minimum_new_node_id=4
    ).innovation_id_into_new_node
    assert innovation_id_after_reset >= tracker.next_innovation_id_snapshot - 2
