"""Historical markings: global innovation numbers (paper, section 3.2)."""

from __future__ import annotations

from typing import NamedTuple

from polyneat.core.type_aliases import InnovationId
from polyneat.logging_utils.custom_logger import get_logger

logger = get_logger(__name__)


class NodeSplitRecord(NamedTuple):
    """The three ids a single add-node mutation needs.

    Attributes:
        new_node_id: Id of the hidden node inserted into the split edge.
        innovation_id_into_new_node: Marking of the edge from the split edge's
            source into the new node.
        innovation_id_out_of_new_node: Marking of the edge from the new node
            into the split edge's target.
    """

    new_node_id: int
    innovation_id_into_new_node: InnovationId
    innovation_id_out_of_new_node: InnovationId


class GlobalInnovationTracker:
    """Issues globally-unique InnovationIds for new structural mutations.

    Implements the historical markings of Stanley & Miikkulainen (2002),
    section 3.2. NEAT's crossover aligns genes by their ``innovation_id``. When two genomes
    receive the *same* structural mutation (same source, same target) inside
    the same generation, they must receive the *same* innovation id — otherwise
    matching structures would be treated as disjoint and crossover would waste
    them. The tracker deduplicates within a single generation via
    ``reset_for_new_generation``.

    Add-node mutations are tracked separately by
    :meth:`get_or_assign_node_split`, keyed by the *innovation id of the edge
    being split*. Keying on the endpoint pair alone is not enough: every genome
    of a minimal start carries the same node ids, so two genomes splitting two
    different edges would independently pick the same new node id and their
    outgoing edges would collide on one ``(source, target)`` key — handing
    unrelated structures the same historical marking, which is precisely what
    section 3.2 exists to prevent. Node ids therefore come from a tracker-wide
    counter, as in the reference implementation, not from a per-genome maximum.

    The tracker is an instance (not a static / global), so multiple parallel
    runs do not share state.
    """

    def __init__(self) -> None:
        self._next_innovation_id: InnovationId = 0
        self._next_node_id: int = 0
        self._within_generation_edge_to_innovation_id: dict[tuple[int, int], InnovationId] = {}
        self._within_generation_split_to_node_record: dict[InnovationId, NodeSplitRecord] = {}

    def get_or_assign_innovation_id_for_connection(
        self,
        source_node_id: int,
        target_node_id: int,
    ) -> InnovationId:
        """Return the innovation id for a connection, deduplicated per generation.

        Args:
            source_node_id: Id of the connection's source node.
            target_node_id: Id of the connection's target node.

        Returns:
            The id previously assigned to this ``(source, target)`` pair in
            the current generation, or a freshly incremented one.
        """
        edge_key = (source_node_id, target_node_id)
        cached_innovation_id = self._within_generation_edge_to_innovation_id.get(edge_key)
        if cached_innovation_id is not None:
            return cached_innovation_id

        newly_assigned_innovation_id = self._next_innovation_id
        self._next_innovation_id += 1
        self._within_generation_edge_to_innovation_id[edge_key] = newly_assigned_innovation_id
        return newly_assigned_innovation_id

    def get_or_assign_node_split(
        self,
        split_connection_innovation_id: InnovationId,
        minimum_new_node_id: int,
    ) -> NodeSplitRecord:
        """Return the node id and the two edge markings for splitting one edge.

        Two genomes that split the same edge in the same generation received
        the same structural mutation and must come out with identical markings;
        two genomes that split different edges must not share any of the three
        ids, or crossover would align unrelated structure (paper, section 3.2).

        Args:
            split_connection_innovation_id: Innovation id of the edge being
                split — the identity of the structural mutation.
            minimum_new_node_id: Lower bound the new node id must clear so it
                cannot collide with a node the calling genome already has.

        Returns:
            The ids to use for this split, reused for the rest of the
            generation whenever the same edge is split again.
        """
        cached_record = self._within_generation_split_to_node_record.get(
            split_connection_innovation_id
        )
        if cached_record is not None:
            return cached_record

        new_node_id = max(self._next_node_id, minimum_new_node_id)
        self._next_node_id = new_node_id + 1

        innovation_id_into_new_node = self._assign_fresh_innovation_id()
        innovation_id_out_of_new_node = self._assign_fresh_innovation_id()

        newly_assigned_record = NodeSplitRecord(
            new_node_id=new_node_id,
            innovation_id_into_new_node=innovation_id_into_new_node,
            innovation_id_out_of_new_node=innovation_id_out_of_new_node,
        )
        self._within_generation_split_to_node_record[split_connection_innovation_id] = (
            newly_assigned_record
        )
        return newly_assigned_record

    def _assign_fresh_innovation_id(self) -> InnovationId:
        """Consume and return the next innovation id, bypassing the edge cache.

        Node-split edges are identified by the split they came from, not by
        their endpoint pair, so they must not be deduplicated by ``(source,
        target)``.
        """
        newly_assigned_innovation_id = self._next_innovation_id
        self._next_innovation_id += 1
        return newly_assigned_innovation_id

    def reset_for_new_generation(self) -> None:
        """Clear the within-generation dedup tables. The id counters keep growing."""
        logger.debug(
            "InnovationTracker reset (next_innovation_id=%d, next_node_id=%d, "
            "edge_table_size=%d -> 0, node_split_table_size=%d -> 0)",
            self._next_innovation_id,
            self._next_node_id,
            len(self._within_generation_edge_to_innovation_id),
            len(self._within_generation_split_to_node_record),
        )
        self._within_generation_edge_to_innovation_id.clear()
        self._within_generation_split_to_node_record.clear()

    @property
    def next_innovation_id_snapshot(self) -> InnovationId:
        """Read-only view of the next innovation id — useful for logs and tests."""
        return self._next_innovation_id
