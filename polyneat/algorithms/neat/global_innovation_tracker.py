from __future__ import annotations

from polyneat.core.type_aliases import InnovationId
from polyneat.logging_utils.custom_logger import get_logger

logger = get_logger(__name__)


class GlobalInnovationTracker:
    """Issues globally-unique InnovationIds for new structural mutations.

    NEAT's crossover aligns genes by their ``innovation_id``. When two genomes
    receive the *same* structural mutation (same source, same target) inside
    the same generation, they must receive the *same* innovation id — otherwise
    matching structures would be treated as disjoint and crossover would waste
    them. The tracker deduplicates within a single generation via
    ``reset_for_new_generation``.

    The tracker is an instance (not a static / global), so multiple parallel
    runs do not share state.
    """

    def __init__(self) -> None:
        self._next_innovation_id: InnovationId = 0
        self._within_generation_edge_to_innovation_id: dict[tuple[int, int], InnovationId] = {}

    def get_or_assign_innovation_id_for_connection(
        self,
        source_node_id: int,
        target_node_id: int,
    ) -> InnovationId:
        edge_key = (source_node_id, target_node_id)
        cached_innovation_id = self._within_generation_edge_to_innovation_id.get(edge_key)
        if cached_innovation_id is not None:
            return cached_innovation_id

        newly_assigned_innovation_id = self._next_innovation_id
        self._next_innovation_id += 1
        self._within_generation_edge_to_innovation_id[edge_key] = newly_assigned_innovation_id
        return newly_assigned_innovation_id

    def reset_for_new_generation(self) -> None:
        """Clear the within-generation dedup table. Innovation counter keeps growing."""
        logger.debug(
            "InnovationTracker reset (next_innovation_id=%d, dedup_table_size=%d -> 0)",
            self._next_innovation_id,
            len(self._within_generation_edge_to_innovation_id),
        )
        self._within_generation_edge_to_innovation_id.clear()

    @property
    def next_innovation_id_snapshot(self) -> InnovationId:
        """Read-only view of the next innovation id — useful for logs and tests."""
        return self._next_innovation_id
