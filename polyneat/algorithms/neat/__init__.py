"""Thin entry point for vanilla NEAT.

The full NEAT implementation lives in :mod:`polyneat.core.neat` — NEAT *is*
the core of the library, and derived algorithms (e.g. FS-NEAT) subclass
:class:`~polyneat.core.neat.neat_algorithm.NEATAlgorithm`. This package
re-exports the algorithm class so user code can keep addressing NEAT the same
way it addresses every other algorithm: via ``polyneat.algorithms.<name>``.
"""

from polyneat.core.neat.neat_algorithm import NEATAlgorithm

__all__ = ["NEATAlgorithm"]
