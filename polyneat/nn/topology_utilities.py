from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Iterable, Mapping, Sequence


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


def build_outgoing_adjacency_from_directed_edges(
    directed_edges: Iterable[tuple[int, int]],
) -> dict[int, list[int]]:
    """Group directed edges by their source node.

    Args:
        directed_edges: ``(source, target)`` pairs. Duplicates are preserved, so
            a repeated edge appears once per occurrence in its source's list.

    Returns:
        A ``defaultdict`` from source node id to its target node ids in
        encounter order. Sources with no outgoing edge are absent from the map,
        and reading one through ``[]`` inserts it - use ``get`` to leave the map
        unchanged.
    """
    outgoing_targets_by_source_node_id: dict[int, list[int]] = defaultdict(list)
    for source_node_id, target_node_id in directed_edges:
        outgoing_targets_by_source_node_id[source_node_id].append(target_node_id)
    return outgoing_targets_by_source_node_id


def would_directed_edge_create_cycle_in_adjacency(
    candidate_source_node_id: int,
    candidate_target_node_id: int,
    outgoing_targets_by_source_node_id: Mapping[int, Sequence[int]],
) -> bool:
    """Return True if the candidate edge would close a cycle in the given adjacency.

    Breadth-first search from the candidate's target: reaching the candidate's
    source means the graph already carries a path back, so adding the edge would
    close a loop. A self-loop is reported without searching. The search visits
    only the part of the graph reachable from the target, and reads the adjacency
    without writing to it, so one map can answer any number of candidates.

    Args:
        candidate_source_node_id: Node the prospective edge leaves.
        candidate_target_node_id: Node the prospective edge enters.
        outgoing_targets_by_source_node_id: Adjacency of the edges already
            accepted, as built by :func:`build_outgoing_adjacency_from_directed_edges`.

    Returns:
        Whether accepting the candidate edge would create a directed cycle.
    """
    if candidate_source_node_id == candidate_target_node_id:
        return True

    bfs_frontier: deque[int] = deque([candidate_target_node_id])
    visited_node_ids: set[int] = {candidate_target_node_id}

    while bfs_frontier:
        current_node_id = bfs_frontier.popleft()
        if current_node_id == candidate_source_node_id:
            return True
        for next_node_id in outgoing_targets_by_source_node_id.get(current_node_id, ()):
            if next_node_id not in visited_node_ids:
                visited_node_ids.add(next_node_id)
                bfs_frontier.append(next_node_id)

    return False


def would_directed_edge_create_cycle(
    candidate_source_node_id: int,
    candidate_target_node_id: int,
    existing_enabled_edges: Iterable[tuple[int, int]],
) -> bool:
    """Return True if adding the candidate edge would introduce a directed cycle.

    Edge-list form of :func:`would_directed_edge_create_cycle_in_adjacency`: it
    groups ``existing_enabled_edges`` by source, asks the same question, and
    discards the grouping. That build is O(edges) and is paid whether or not the
    search then takes a single step, so a caller asking about many candidates
    against one slowly growing edge set should keep the adjacency itself and call
    the other form.
    """
    return would_directed_edge_create_cycle_in_adjacency(
        candidate_source_node_id=candidate_source_node_id,
        candidate_target_node_id=candidate_target_node_id,
        outgoing_targets_by_source_node_id=build_outgoing_adjacency_from_directed_edges(
            existing_enabled_edges
        ),
    )


def find_node_ids_with_enabled_path_to_any_target(
    candidate_source_node_ids: Iterable[int],
    target_node_ids: Iterable[int],
    enabled_directed_edges: Iterable[tuple[int, int]],
) -> list[int]:
    """Return the candidates that reach at least one target along enabled edges.

    Used to measure which input features a network actually uses. Counting a
    feature as used because it has an outgoing connection overstates usage: an
    input wired into a hidden node that leads nowhere contributes nothing to any
    output. Requiring a path to an output is the honest measure, and it is what
    makes FS-NEAT and FD-NEAT results comparable, since both report it.

    The search runs backwards from the targets once, which is cheaper than a
    forward search per candidate.

    Args:
        candidate_source_node_ids: Node ids to test, typically the input nodes.
        target_node_ids: Node ids that count as reached, typically the outputs.
            A candidate that is itself a target counts as reaching one.
        enabled_directed_edges: ``(source, target)`` pairs of enabled edges.

    Returns:
        The reaching candidates, sorted ascending, without duplicates.
    """
    incoming_sources_by_target_node_id: dict[int, list[int]] = defaultdict(list)
    for source_node_id, target_node_id in enabled_directed_edges:
        incoming_sources_by_target_node_id[target_node_id].append(source_node_id)

    reaching_node_ids: set[int] = set(target_node_ids)
    bfs_frontier: deque[int] = deque(reaching_node_ids)
    while bfs_frontier:
        current_node_id = bfs_frontier.popleft()
        for source_node_id in incoming_sources_by_target_node_id[current_node_id]:
            if source_node_id not in reaching_node_ids:
                reaching_node_ids.add(source_node_id)
                bfs_frontier.append(source_node_id)

    return sorted(set(candidate_source_node_ids) & reaching_node_ids)
