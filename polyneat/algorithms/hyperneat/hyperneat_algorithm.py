"""HyperNEAT as a ``NEATAlgorithm`` subclass (Stanley, D'Ambrosio & Gauci, 2009).

HyperNEAT evolves a CPPN with standard NEAT genetics, so it subclasses
:class:`~polyneat.core.neat.neat_algorithm.NEATAlgorithm` and overrides only two
component factories: ``_build_mutation`` (the add-node operator assigns a random
CPPN activation) and ``_build_phenotype_decoder`` (each CPPN is decoded into a
substrate ANN). The generational loop, crossover, speciation, parent selection,
innovation tracking, and the fully-connected initial population (a minimal
4-input + bias -> 1-output net is exactly a minimal CPPN) are inherited
unchanged. Requires a
:class:`~polyneat.config.hyperneat_config.HyperNEATConfig`.

References:
    Stanley, K. O., D'Ambrosio, D. B., & Gauci, J. (2009). A hypercube-based
    encoding for evolving large-scale neural networks. Artificial Life, 15(2),
    185-212.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch

from polyneat.algorithms.hyperneat.add_node_random_activation_mutation import (
    AddNodeWithRandomActivationMutation,
)
from polyneat.algorithms.hyperneat.hyperneat_phenotype_decoder import (
    HyperNEATPhenotypeDecoder,
)
from polyneat.algorithms.hyperneat.substrate import build_layered_substrate
from polyneat.config.hyperneat_config import HyperNEATConfig
from polyneat.config.neat_config import NEATConfig
from polyneat.core.component_protocols import MutationOperator, PhenotypeDecoder
from polyneat.core.neat.mutations.add_connection_mutation import AddConnectionMutation
from polyneat.core.neat.mutations.composite_neat_mutation import CompositeNEATMutation
from polyneat.core.neat.mutations.toggle_connection_enabled_mutation import (
    ToggleConnectionEnabledMutation,
)
from polyneat.core.neat.mutations.weight_modification_mutation import (
    WeightModificationMutation,
)
from polyneat.core.neat.neat_algorithm import NEATAlgorithm
from polyneat.core.neat.neat_genome import NEATGenome
from polyneat.core.neat.neat_phenotype_decoder import NEATPhenotypeDecoder


@dataclass
class HyperNEATAlgorithm(NEATAlgorithm):
    """HyperNEAT: evolve CPPNs with NEAT, decode each into a substrate ANN.

    Overrides only the two factories that differ from vanilla NEAT; the
    generational loop and every other component are inherited from
    :class:`~polyneat.core.neat.neat_algorithm.NEATAlgorithm`. Build it with the
    inherited ``HyperNEATAlgorithm.from_config(hyperneat_config)`` and give it a
    :class:`~polyneat.config.hyperneat_config.HyperNEATConfig`.
    """

    @classmethod
    def _build_mutation(cls, config: NEATConfig) -> MutationOperator[NEATGenome]:
        """Compose NEAT's mutations, swapping in the random-activation add-node.

        Every rate mirrors ``NEATAlgorithm._build_mutation``; only the add-node
        operator differs, assigning each newly inserted CPPN node a random
        activation from ``config.available_activation_functions`` so CPPNs
        compose heterogeneous functions (Stanley et al. 2009, Section 3.2). All
        fields read here exist on ``NEATConfig``, so no cast is needed.

        Args:
            config: Validated config supplying the mutation rates and CPPN
                function set (a ``HyperNEATConfig``, though only ``NEATConfig``
                fields are read here).

        Returns:
            The composite CPPN mutation operator.
        """
        return CompositeNEATMutation(
            ordered_individual_mutations=[
                WeightModificationMutation(
                    probability_of_perturbation=config.probability_of_weight_perturbation,
                    probability_of_replacement=config.probability_of_weight_replacement,
                    perturbation_strength_sigma=config.weight_perturbation_strength_sigma,
                    initial_weight_range_min=config.initial_weight_range_min,
                    initial_weight_range_max=config.initial_weight_range_max,
                ),
                AddConnectionMutation(
                    probability_of_application=config.probability_of_add_connection_mutation,
                    initial_weight_range_min=config.initial_weight_range_min,
                    initial_weight_range_max=config.initial_weight_range_max,
                ),
                AddNodeWithRandomActivationMutation(
                    probability_of_application=config.probability_of_add_node_mutation,
                    available_activation_function_names=config.available_activation_functions,
                ),
                ToggleConnectionEnabledMutation(
                    probability_of_application=config.probability_of_toggle_connection_enabled,
                ),
            ]
        )

    @classmethod
    def _build_phenotype_decoder(
        cls, config: NEATConfig, device: torch.device
    ) -> PhenotypeDecoder[NEATGenome]:
        """Build the CPPN-to-substrate decoder over a layered substrate.

        The config is narrowed to ``HyperNEATConfig`` to access the substrate
        fields; ``from_config`` must be called with a
        ``HyperNEATConfig``. A non-layered substrate (e.g. a grid sandwich) is
        supported by replacing this decoder via ``dataclasses.replace`` on the
        constructed algorithm.

        Args:
            config: Validated ``HyperNEATConfig`` describing the substrate and
                the CPPN-output-to-weight interpretation.
            device: Torch device the substrate phenotypes run on.

        Returns:
            The CPPN-to-substrate phenotype decoder.
        """
        hyperneat_config = cast("HyperNEATConfig", config)
        substrate = build_layered_substrate(
            input_layer_size=hyperneat_config.substrate_input_layer_size,
            hidden_layer_sizes=hyperneat_config.substrate_hidden_layer_sizes,
            output_layer_size=hyperneat_config.substrate_output_layer_size,
            coordinate_range_min=hyperneat_config.substrate_coordinate_range_min,
            coordinate_range_max=hyperneat_config.substrate_coordinate_range_max,
            include_bias_node=hyperneat_config.include_substrate_bias_node,
        )
        return HyperNEATPhenotypeDecoder(
            substrate=substrate,
            cppn_phenotype_decoder=NEATPhenotypeDecoder(device_for_computation=device),
            weight_expression_threshold=hyperneat_config.weight_expression_threshold,
            max_substrate_connection_weight_magnitude=(
                hyperneat_config.max_substrate_connection_weight_magnitude
            ),
            substrate_node_activation_function_name=(
                hyperneat_config.substrate_node_activation_function
            ),
            device_for_computation=device,
        )
