"""FD-NEAT (Tan et al., 2012): NEAT with automatic feature *de*selection.

References:
    Tan, M., Deklerck, R., Jansen, B., & Cornelis, J. (2012). Analysis of a Feature-Deselective
        Neuroevolution Classifier (FD-NEAT) in a Computer-Aided Lung Nodule Detection System
        for CT Images. *GECCO '12 Companion*, pp. 539-546. DOI: 10.1145/2330784.2330869
"""

from polyneat.algorithms.fdneat.fdneat_algorithm import FDNEATAlgorithm
from polyneat.algorithms.fdneat.mutations.delete_input_connection_mutation import (
    DeleteInputConnectionMutation,
)

__all__ = ["DeleteInputConnectionMutation", "FDNEATAlgorithm"]
