"""HyperNEAT: evolve a CPPN with NEAT and decode it into a substrate ANN.

References:
    Stanley, K. O., D'Ambrosio, D. B., & Gauci, J. (2009). A Hypercube-Based Encoding for
        Evolving Large-Scale Neural Networks. *Artificial Life*, 15(2), 185-212.
        DOI: 10.1162/artl.2009.15.2.15202
"""

from polyneat.algorithms.hyperneat.hyperneat_algorithm import HyperNEATAlgorithm

__all__ = ["HyperNEATAlgorithm"]
