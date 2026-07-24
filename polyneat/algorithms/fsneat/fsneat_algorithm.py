from __future__ import annotations

from dataclasses import dataclass

from numpy.random import Generator

from polyneat.core.neat.initial_population import build_fs_neat_initial_population
from polyneat.core.neat.neat_algorithm import NEATAlgorithm
from polyneat.core.population import Population


@dataclass
class FSNEATAlgorithm(NEATAlgorithm):
    """FS-NEAT (Whiteson et al., 2005): NEAT with automatic feature selection.

    FS-NEAT differs from vanilla NEAT only in its starting point: instead of a
    fully connected minimal topology, every generation-0 genome contains a
    single enabled connection from one randomly chosen input to one randomly
    chosen output. Evolution itself then decides which input features are
    worth wiring in at all, performing feature selection as a side effect of
    the topology search.

    Everything else — configuration (:class:`~polyneat.configs.neat.neat_config.NEATConfig`),
    operators, speciation, and the generational loop — is inherited unchanged
    from :class:`~polyneat.core.neat.neat_algorithm.NEATAlgorithm`.

    References:
        Whiteson, S., Stone, P., Stanley, K. O., Miikkulainen, R., &
        Kohl, N. (2005). Automatic Feature Selection in Neuroevolution.
        *GECCO 2005*.
    """

    def create_initial_population(self, rng: Generator) -> Population:
        """Build the FS-NEAT generation 0 (one random input wired per genome).

        This override has precedence over any ``initial_population_strategy``
        from the config: the sparse start *is* the definition of FS-NEAT, so a
        YAML strategy entry is deliberately ignored.

        Args:
            rng: Source of randomness for connection choice and weights.

        Returns:
            Generation-0 population of single-connection genomes.
        """
        return build_fs_neat_initial_population(self.config, self.innovation_tracker, rng)
