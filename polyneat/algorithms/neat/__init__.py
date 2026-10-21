"""Thin entry point for NEAT.

The full NEAT implementation lives in :mod:`polyneat.core.neat` 
NEAT *is* the core of the library, and derived algorithms subclass
:class:`~polyneat.core.neat.neat_algorithm.NEATAlgorithm`. This
package re-exports the algorithm class NEAT can be imported the same
way it addresses every other algorithm: via ``polyneat.algorithms.<name>``.
"""

from polyneat.core.neat.neat_algorithm import NEATAlgorithm

__all__ = ["NEATAlgorithm"]
