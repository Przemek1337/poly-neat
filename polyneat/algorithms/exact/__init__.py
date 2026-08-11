"""EXACT: evolutionary exploration of augmenting convolutional topologies.

References:
    Desell, T. (2017). Developing a Volunteer Computing Project to Evolve
        Convolutional Neural Networks and Their Hyperparameters. 2017 IEEE
        13th International Conference on e-Science, pp. 19-28.
"""

from polyneat.algorithms.exact.exact_algorithm import EXACTAlgorithm
from polyneat.algorithms.exact.exact_backpropagation_trainer import (
    EXACTBackpropagationTrainer,
)
from polyneat.algorithms.exact.exact_crossover import EXACTCrossover
from polyneat.algorithms.exact.exact_genome import (
    ConvolutionEdgeGene,
    EXACTGenome,
    FilterNodeGene,
    InvalidEXACTGenomeError,
)
from polyneat.algorithms.exact.exact_initial_population import (
    build_minimal_cnn_initial_population,
)
from polyneat.algorithms.exact.exact_phenotype_decoder import EXACTPhenotypeDecoder
from polyneat.algorithms.exact.exact_training_hyperparameters import (
    EXACTTrainingHyperparameters,
)
from polyneat.algorithms.exact.mutations.add_convolution_node_mutation import (
    AddConvolutionNodeMutation,
)
from polyneat.algorithms.exact.mutations.exact_composite_mutation import (
    EXACTCompositeMutation,
)
from polyneat.algorithms.exact.simplex_hyperparameter_optimizer import (
    SimplexHyperparameterOptimizer,
)
from polyneat.algorithms.exact.single_species_speciator import SingleSpeciesSpeciator
from polyneat.algorithms.exact.torch_convolutional_phenotype import (
    TorchConvolutionalPhenotype,
)

__all__ = [
    "AddConvolutionNodeMutation",
    "EXACTAlgorithm",
    "EXACTBackpropagationTrainer",
    "EXACTCompositeMutation",
    "EXACTCrossover",
    "EXACTGenome",
    "EXACTPhenotypeDecoder",
    "EXACTTrainingHyperparameters",
    "SimplexHyperparameterOptimizer",
    "ConvolutionEdgeGene",
    "FilterNodeGene",
    "InvalidEXACTGenomeError",
    "SingleSpeciesSpeciator",
    "TorchConvolutionalPhenotype",
    "build_minimal_cnn_initial_population",
]
