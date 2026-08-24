"""Innovation tracker for DeepNEAT: one master list for the whole search.

References:
    Miikkulainen, R., et al. (2017). Evolving Deep Neural Networks. *arXiv:1703.00548*.
        DOI: 10.1016/B978-0-12-815480-9.00015-3
"""

from __future__ import annotations

from polyneat.core.neat.global_innovation_tracker import GlobalInnovationTracker
from polyneat.logging_utils.custom_logger import get_logger

logger = get_logger(__name__)


class DeepNEATInnovationTracker(GlobalInnovationTracker):
    """Keeps historical markings for the entire run, not just one generation.

    Vanilla NEAT clears its deduplication tables at each generation boundary:
    two genomes growing the same edge in *different* generations are treated as
    unrelated innovations, which is what the paper prescribes. DeepNEAT compares
    whole architectures over long runs, and the same layer connection genuinely
    reappearing must keep its identity, or crossover would stop aligning
    structurally identical graphs. This mirrors ``EXACTInnovationTracker``.

    Thread safety: inherited and unchanged - **not** thread-safe. See
    :class:`~polyneat.core.neat.global_innovation_tracker.GlobalInnovationTracker`.
    """

    def reset_for_new_generation(self) -> None:
        """Keep every marking; only log the generation boundary."""
        logger.debug(
            "DeepNEATInnovationTracker generation boundary (markings kept: "
            "next_innovation_id=%d)",
            self.next_innovation_id_snapshot,
        )
