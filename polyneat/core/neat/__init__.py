"""The NEAT implementation — the core of PolyNEAT.

NEAT (Stanley & Miikkulainen, 2002) *is* the core of this library: every
derived algorithm (FS-NEAT today; DeepNEAT/HyperNEAT-style variants later)
subclasses :class:`~polyneat.core.neat.neat_algorithm.NEATAlgorithm` and
overrides only the factory methods or steps it changes, while the generational
loop stays shared and untouched.

References:
    Stanley, K. O., & Miikkulainen, R. (2002). Evolving Neural Networks
    through Augmenting Topologies. *Evolutionary Computation*, 10(2), 99-127.
"""
