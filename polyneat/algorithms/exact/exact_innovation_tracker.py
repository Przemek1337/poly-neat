"""Persistent historical markings for EXACT (Desell, 2017, section III-B).

References:
    Desell, T. (2017). Developing a Volunteer Computing Project to Evolve
        Convolutional Neural Networks and Their Hyperparameters. 2017 IEEE
        13th International Conference on e-Science, pp. 19-28.
"""

from __future__ import annotations

from polyneat.core.neat.global_innovation_tracker import GlobalInnovationTracker
from polyneat.logging_utils.custom_logger import get_logger

logger = get_logger(__name__)


class EXACTInnovationTracker(GlobalInnovationTracker):
    """Master innovation list that never forgets a marking.

    The paper's master process keeps track of all node and edge innovations
    made over the whole search (section III-B): an add-edge mutation reuses
    the marking of any previously seen ``(source, target)`` pair, and
    splitting the same edge always yields the same node id and edge
    markings, no matter how many generations apart. Without persistence the
    same structural edge could carry two different innovation ids, which
    crossover could never align.

    References:
        Desell, T. (2017). Developing a Volunteer Computing Project to Evolve
            Convolutional Neural Networks and Their Hyperparameters. 2017 IEEE
            13th International Conference on e-Science, pp. 19-28.
    """

    def assign_fresh_node_id(self, minimum_new_node_id: int) -> int:
        """Consume and return the next node id for a brand-new hidden node.

        Args:
            minimum_new_node_id: Lower bound the id must clear so it cannot
                collide with a node the calling genome already has.

        Returns:
            A node id never handed out before.
        """
        new_node_id = max(self._next_node_id, minimum_new_node_id)
        self._next_node_id = new_node_id + 1
        return new_node_id

    def reset_for_new_generation(self) -> None:
        """Keep the dedup tables: EXACT markings are search-lifetime."""
        logger.debug(
            "EXACTInnovationTracker generation boundary (markings kept: "
            "%d edges, %d splits)",
            len(self._within_generation_edge_to_innovation_id),
            len(self._within_generation_split_to_node_record),
        )
