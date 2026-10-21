"""L-NEAT: NEAT hybridized with backpropagation learning for classification.

References:
    Chen, L., & Alahakoon, D. (2006). NeuroEvolution of Augmenting Topologies with Learning for
        Data Classification. *ICIA 2006: 2nd International Conference on Information and
        Automation*, pp. 367-371.
"""

from polyneat.algorithms.lneat.backpropagation_weight_trainer import (
    BackpropagationWeightTrainer,
)
from polyneat.algorithms.lneat.lneat_algorithm import LNEATAlgorithm
from polyneat.algorithms.lneat.recognizer_ensemble_phenotype import (
    RecognizerEnsemblePhenotype,
)
from polyneat.algorithms.lneat.trainable_torch_phenotype import (
    TrainableTorchFeedForwardPhenotype,
)

__all__ = [
    "BackpropagationWeightTrainer",
    "LNEATAlgorithm",
    "RecognizerEnsemblePhenotype",
    "TrainableTorchFeedForwardPhenotype",
]
