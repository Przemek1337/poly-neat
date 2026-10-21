"""Configuration for the FD-NEAT algorithm.

References:
    Tan, M., Deklerck, R., Jansen, B., & Cornelis, J. (2012). Analysis of a Feature-Deselective
        Neuroevolution Classifier (FD-NEAT) in a Computer-Aided Lung Nodule Detection System
        for CT Images. *GECCO '12 Companion: Proceedings of the 14th Annual Conference Companion
        on Genetic and Evolutionary Computation*, pp. 539-546. DOI: 10.1145/2330784.2330869
"""

from __future__ import annotations

from dataclasses import dataclass

from polyneat.configs.configuration_errors import ConfigurationError
from polyneat.configs.neat.neat_config import NEATConfig


@dataclass
class FDNEATConfig(NEATConfig):
    """Hyperparameters for FD-NEAT (Tan et al., 2012).

    FD-NEAT is vanilla NEAT plus one operator, so every inherited field keeps
    its meaning unchanged. In particular ``initial_population_strategy`` stays
    at ``"fully_connected"``: the fully connected start *is* FD-NEAT's start,
    and evolution removes what it does not need.

    Attributes:
        probability_of_deleting_input_connection: Chance that
            :class:`~polyneat.algorithms.fdneat.mutations.delete_input_connection_mutation.DeleteInputConnectionMutation`
            fires on a given genome. The default mirrors
            ``probability_of_add_connection_mutation`` (0.05), the operator
            working in the opposite direction; the source paper does not state a
            value.
    """

    probability_of_deleting_input_connection: float = 0.05

    def validate(self) -> None:
        """Validate the deletion probability on top of the inherited NEAT checks.

        Raises:
            ConfigurationError: If the deletion probability falls outside
                ``[0.0, 1.0]``, naming the field, the value and the reason.
        """
        super().validate()
        if not (0.0 <= self.probability_of_deleting_input_connection <= 1.0):
            raise ConfigurationError(
                f"probability_of_deleting_input_connection must be in [0.0, 1.0], "
                f"got {self.probability_of_deleting_input_connection}"
            )
