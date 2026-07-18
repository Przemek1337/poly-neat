"""HyperNEAT: evolve a CPPN with NEAT and decode it into a substrate ANN.

References:
    Stanley, K. O., D'Ambrosio, D. B., & Gauci, J. (2009). A hypercube-based
    encoding for evolving large-scale neural networks. Artificial Life, 15(2),
    185-212.
"""

from polyneat.algorithms.hyperneat.hyperneat_algorithm import HyperNEATAlgorithm

__all__ = ["HyperNEATAlgorithm"]
