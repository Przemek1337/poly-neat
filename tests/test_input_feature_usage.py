from __future__ import annotations

from polyneat.nn.topology_utilities import find_node_ids_with_enabled_path_to_any_target


def test_direct_edge_to_target_counts() -> None:
    assert find_node_ids_with_enabled_path_to_any_target(
        candidate_source_node_ids={0, 1},
        target_node_ids={3},
        enabled_directed_edges=[(0, 3)],
    ) == [0]


def test_indirect_path_through_hidden_node_counts() -> None:
    assert find_node_ids_with_enabled_path_to_any_target(
        candidate_source_node_ids={0, 1},
        target_node_ids={3},
        enabled_directed_edges=[(1, 4), (4, 3)],
    ) == [1]


def test_edge_into_a_dead_end_does_not_count() -> None:
    # Input 0 feeds hidden node 4, but 4 goes nowhere: the feature is unused.
    # Counting outgoing edges alone would wrongly report it as used.
    assert find_node_ids_with_enabled_path_to_any_target(
        candidate_source_node_ids={0, 1},
        target_node_ids={3},
        enabled_directed_edges=[(0, 4), (1, 3)],
    ) == [1]


def test_long_chain_is_followed_to_the_end() -> None:
    assert find_node_ids_with_enabled_path_to_any_target(
        candidate_source_node_ids={0},
        target_node_ids={9},
        enabled_directed_edges=[(0, 5), (5, 6), (6, 7), (7, 8), (8, 9)],
    ) == [0]


def test_result_is_sorted_and_deduplicated() -> None:
    assert find_node_ids_with_enabled_path_to_any_target(
        candidate_source_node_ids={2, 0, 1},
        target_node_ids={3},
        enabled_directed_edges=[(2, 3), (0, 3), (1, 3), (0, 3)],
    ) == [0, 1, 2]


def test_no_edges_means_no_used_features() -> None:
    assert (
        find_node_ids_with_enabled_path_to_any_target(
            candidate_source_node_ids={0, 1},
            target_node_ids={3},
            enabled_directed_edges=[],
        )
        == []
    )


def test_target_that_is_also_a_candidate_counts_itself() -> None:
    assert find_node_ids_with_enabled_path_to_any_target(
        candidate_source_node_ids={3},
        target_node_ids={3},
        enabled_directed_edges=[],
    ) == [3]


def test_multiple_targets_any_of_which_counts() -> None:
    assert find_node_ids_with_enabled_path_to_any_target(
        candidate_source_node_ids={0, 1},
        target_node_ids={8, 9},
        enabled_directed_edges=[(0, 8), (1, 7)],
    ) == [0]


def test_candidates_not_in_the_graph_at_all_are_ignored() -> None:
    assert find_node_ids_with_enabled_path_to_any_target(
        candidate_source_node_ids={0, 1, 99},
        target_node_ids={3},
        enabled_directed_edges=[(0, 3), (1, 3)],
    ) == [0, 1]


def test_accepts_ranges_as_well_as_sets() -> None:
    assert find_node_ids_with_enabled_path_to_any_target(
        candidate_source_node_ids=range(3),
        target_node_ids=range(3, 4),
        enabled_directed_edges=[(2, 3)],
    ) == [2]
