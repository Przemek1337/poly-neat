"""Historical markings for DeepNEAT structural mutations."""

from __future__ import annotations

from polyneat.core.neat.global_innovation_tracker import GlobalInnovationTracker


class DeepNEATInnovationTracker(GlobalInnovationTracker):
    """NEAT historical markings with generation-local mutation deduplication.

    DeepNEAT inherits NEAT's historical-marking mechanism. Consequently the
    cache that recognizes an identical structural mutation is cleared at each
    generation boundary, while the global innovation and node counters remain
    monotonic. The inherited implementation provides exactly those semantics.
    """
