from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Iterable


def compute_topological_order_of_node_ids(
    all_node_ids: Iterable[int],
    enabled_directed_edges: Iterable[tuple[int, int]],
) -> list[int]:
    """Return the node ids in a valid topological order using Kahn's algorithm.

    A cycle in ``enabled_directed_edges`` raises ``ValueError`` — feed-forward
    NEAT phenotypes are not allowed to contain cycles, so callers should have
    rejected such structures earlier (via :func:`would_edge_create_cycle`).
    """
    node_id_set: set[int] = set(all_node_ids)
    incoming_edge_count_by_node_id: dict[int, int] = defaultdict(int)
    outgoing_targets_by_source_node_id: dict[int, list[int]] = defaultdict(list)

    for source_node_id, target_node_id in enabled_directed_edges:
        outgoing_targets_by_source_node_id[source_node_id].append(target_node_id)
        incoming_edge_count_by_node_id[target_node_id] += 1
        node_id_set.add(source_node_id)
        node_id_set.add(target_node_id)

    ready_queue: deque[int] = deque(
        node_id for node_id in node_id_set if incoming_edge_count_by_node_id[node_id] == 0
    )
    topologically_sorted_node_ids: list[int] = []

    while ready_queue:
        node_id = ready_queue.popleft()
        topologically_sorted_node_ids.append(node_id)
        for target_node_id in outgoing_targets_by_source_node_id[node_id]:
            incoming_edge_count_by_node_id[target_node_id] -= 1
            if incoming_edge_count_by_node_id[target_node_id] == 0:
                ready_queue.append(target_node_id)

    if len(topologically_sorted_node_ids) != len(node_id_set):
        raise ValueError(
            "compute_topological_order_of_node_ids: cycle detected — enabled edges "
            "form a directed cycle, which is not allowed for feed-forward phenotypes"
        )

    return topologically_sorted_node_ids


def would_directed_edge_create_cycle(
    candidate_source_node_id: int,
    candidate_target_node_id: int,
    existing_enabled_edges: Iterable[tuple[int, int]],
) -> bool:
    """Return True if adding the candidate edge would introduce a directed cycle.

    Runs a BFS from the candidate's target: if the source is reachable, the new
    edge would close a cycle back to itself.
    """
    if candidate_source_node_id == candidate_target_node_id:
        return True

    outgoing_targets_by_source_node_id: dict[int, list[int]] = defaultdict(list)
    for source_node_id, target_node_id in existing_enabled_edges:
        outgoing_targets_by_source_node_id[source_node_id].append(target_node_id)

    bfs_frontier: deque[int] = deque([candidate_target_node_id])
    visited_node_ids: set[int] = {candidate_target_node_id}

    while bfs_frontier:
        current_node_id = bfs_frontier.popleft()
        if current_node_id == candidate_source_node_id:
            return True
        for next_node_id in outgoing_targets_by_source_node_id[current_node_id]:
            if next_node_id not in visited_node_ids:
                visited_node_ids.add(next_node_id)
                bfs_frontier.append(next_node_id)

    return False
