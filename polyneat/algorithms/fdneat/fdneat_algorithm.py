"""FD-NEAT: NEAT that deselects input features by deleting their connections.

References:
    Tan, M., Deklerck, R., Jansen, B., & Cornelis, J. (2012). Analysis of a Feature-Deselective
        Neuroevolution Classifier (FD-NEAT) in a Computer-Aided Lung Nodule Detection System
        for CT Images. *GECCO '12 Companion: Proceedings of the 14th Annual Conference Companion
        on Genetic and Evolutionary Computation*, pp. 539-546. DOI: 10.1145/2330784.2330869
"""

from __future__ import annotations

from dataclasses import dataclass

from polyneat.algorithms.fdneat.mutations.delete_input_connection_mutation import (
    DeleteInputConnectionMutation,
)
from polyneat.configs.neat.neat_config import NEATConfig
from polyneat.core.component_protocols import MutationOperator
from polyneat.core.neat.mutations.add_connection_mutation import AddConnectionMutation
from polyneat.core.neat.mutations.add_node_mutation import AddNodeMutation
from polyneat.core.neat.mutations.composite_neat_mutation import CompositeNEATMutation
from polyneat.core.neat.mutations.toggle_connection_enabled_mutation import (
    ToggleConnectionEnabledMutation,
)
from polyneat.core.neat.mutations.weight_modification_mutation import (
    WeightModificationMutation,
)
from polyneat.core.neat.neat_algorithm import NEATAlgorithm
from polyneat.core.neat.neat_genome import NEATGenome


@dataclass
class FDNEATAlgorithm(NEATAlgorithm):
    """FD-NEAT (Tan et al., 2012): NEAT with automatic feature *de*selection.

    FD-NEAT is the mirror image of FS-NEAT. Where FS-NEAT starts from a single
    random input connection and grows the useful ones, FD-NEAT starts fully
    connected - the ordinary NEAT minimal start - and *removes* the inputs that
    do not earn their place.

    The starting topology is therefore unchanged from vanilla NEAT, and so is
    everything else: crossover, speciation, selection and the generational loop
    are inherited untouched. The only difference is a fifth mutation operator,
    :class:`~polyneat.algorithms.fdneat.mutations.delete_input_connection_mutation.DeleteInputConnectionMutation`,
    appended to the standard pipeline.

    It runs **last**, after ``ToggleConnectionEnabledMutation``, so the additive
    operators see the pre-deletion topology and the deselection is the final
    word for that offspring.

    References:
        Tan, M., Deklerck, R., Jansen, B., & Cornelis, J. (2012). Analysis of a
            Feature-Deselective Neuroevolution Classifier (FD-NEAT) in a Computer-Aided
            Lung Nodule Detection System for CT Images. *GECCO '12 Companion*, pp. 539-546.
            DOI: 10.1145/2330784.2330869
    """

    @classmethod
    def _build_mutation(cls, config: NEATConfig) -> MutationOperator[NEATGenome]:
        """Compose the four canonical NEAT mutations plus FD-NEAT's deletion.

        Args:
            config: Hyperparameters; an
                :class:`~polyneat.configs.fdneat.fdneat_config.FDNEATConfig`
                supplies the deletion probability. A plain ``NEATConfig`` falls
                back to 0.0, which makes the operator inert and the algorithm
                behave as vanilla NEAT rather than failing.

        Returns:
            The five-operator mutation pipeline.
        """
        probability_of_deletion = getattr(
            config, "probability_of_deleting_input_connection", 0.0
        )
        return CompositeNEATMutation(
            ordered_individual_mutations=[
                WeightModificationMutation(
                    probability_of_genome_weight_mutation=(
                        config.probability_of_genome_weight_mutation
                    ),
                    probability_of_perturbation=config.probability_of_weight_perturbation,
                    probability_of_replacement=config.probability_of_weight_replacement,
                    weight_perturbation_magnitude=config.weight_perturbation_magnitude,
                    initial_weight_range_min=config.initial_weight_range_min,
                    initial_weight_range_max=config.initial_weight_range_max,
                ),
                AddConnectionMutation(
                    probability_of_application=config.probability_of_add_connection_mutation,
                    initial_weight_range_min=config.initial_weight_range_min,
                    initial_weight_range_max=config.initial_weight_range_max,
                ),
                AddNodeMutation(
                    probability_of_application=config.probability_of_add_node_mutation,
                    activation_function_name_for_new_hidden_node=(
                        config.default_activation_function_for_hidden_nodes
                    ),
                ),
                ToggleConnectionEnabledMutation(
                    probability_of_application=config.probability_of_toggle_connection_enabled,
                ),
                DeleteInputConnectionMutation(
                    probability_of_application=probability_of_deletion,
                ),
            ]
        )
