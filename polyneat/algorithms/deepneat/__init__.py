"""DeepNEAT: evolving deep neural network topologies layer by layer.

References:
    Miikkulainen, R., Liang, J., Meyerson, E., Rawal, A., Fink, D., Francon, O., Raju, B.,
        Shahrzad, H., Navruzyan, A., Duffy, N., & Hodjat, B. (2017). Evolving Deep Neural
        Networks. *arXiv:1703.00548*. Published in *Artificial Intelligence in the Age of
        Neural Networks and Brain Computing* (2019), pp. 293-312.
        DOI: 10.1016/B978-0-12-815480-9.00015-3
"""

from polyneat.algorithms.deepneat.deepneat_algorithm import DeepNEATAlgorithm
from polyneat.algorithms.deepneat.deepneat_crossover import DeepNEATCrossover
from polyneat.algorithms.deepneat.deepneat_genome import (
    DeepNEATGenome,
    DeepNEATGlobalHyperparameters,
    InvalidDeepNEATGenomeError,
    LayerNodeGene,
    TensorEdgeGene,
)
from polyneat.algorithms.deepneat.deepneat_initial_population import (
    build_deepneat_initial_population,
)
from polyneat.algorithms.deepneat.deepneat_innovation_tracker import (
    DeepNEATInnovationTracker,
)
from polyneat.algorithms.deepneat.deepneat_phenotype_decoder import (
    DeepNEATPhenotypeDecoder,
)
from polyneat.algorithms.deepneat.deepneat_speciator import DeepNEATSpeciator
from polyneat.algorithms.deepneat.mutations.add_layer_node_mutation import (
    AddLayerNodeMutation,
)
from polyneat.algorithms.deepneat.mutations.add_tensor_edge_mutation import (
    AddTensorEdgeMutation,
)
from polyneat.algorithms.deepneat.mutations.deepneat_composite_mutation import (
    DeepNEATCompositeMutation,
)
from polyneat.algorithms.deepneat.mutations.global_hyperparameter_mutation import (
    GlobalHyperparameterMutation,
)
from polyneat.algorithms.deepneat.mutations.layer_hyperparameter_mutation import (
    LayerHyperparameterMutation,
)
from polyneat.algorithms.deepneat.mutations.toggle_tensor_edge_mutation import (
    ToggleTensorEdgeMutation,
)
from polyneat.algorithms.deepneat.torch_layer_stack_phenotype import (
    TorchLayerStackPhenotype,
)

__all__ = [
    "AddLayerNodeMutation",
    "AddTensorEdgeMutation",
    "DeepNEATAlgorithm",
    "DeepNEATCompositeMutation",
    "DeepNEATCrossover",
    "DeepNEATGenome",
    "DeepNEATGlobalHyperparameters",
    "DeepNEATInnovationTracker",
    "DeepNEATPhenotypeDecoder",
    "DeepNEATSpeciator",
    "InvalidDeepNEATGenomeError",
    "GlobalHyperparameterMutation",
    "LayerHyperparameterMutation",
    "LayerNodeGene",
    "TensorEdgeGene",
    "ToggleTensorEdgeMutation",
    "TorchLayerStackPhenotype",
    "build_deepneat_initial_population",
]
