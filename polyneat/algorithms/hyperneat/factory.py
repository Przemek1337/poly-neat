"""Assembly factory for the HyperNEAT algorithm.

HyperNEAT keeps NEAT's genome and generation loop, so it is assembled by a
factory that returns a configured ``NEATAlgorithm`` (a component swap, not a
subclass), per the library's architecture contract.

References:
    Stanley, K. O., D'Ambrosio, D. B., & Gauci, J. (2009). A hypercube-based
    encoding for evolving large-scale neural networks. Artificial Life, 15(2),
    185-212.
"""
from __future__ import annotations

import dataclasses

import torch

from polyneat.algorithms.hyperneat.add_node_random_activation_mutation import (
    AddNodeWithRandomActivationMutation,
)
from polyneat.algorithms.hyperneat.hyperneat_phenotype_decoder import (
    HyperNEATPhenotypeDecoder,
)
from polyneat.algorithms.hyperneat.substrate import build_layered_substrate
from polyneat.algorithms.neat.mutations.add_connection_mutation import (
    AddConnectionMutation,
)
from polyneat.algorithms.neat.mutations.composite_neat_mutation import (
    CompositeNEATMutation,
)
from polyneat.algorithms.neat.mutations.toggle_connection_enabled_mutation import (
    ToggleConnectionEnabledMutation,
)
from polyneat.algorithms.neat.mutations.weight_modification_mutation import (
    WeightModificationMutation,
)
from polyneat.algorithms.neat.neat_algorithm import NEATAlgorithm
from polyneat.algorithms.neat.neat_phenotype_decoder import NEATPhenotypeDecoder
from polyneat.config.hyperneat_config import HyperNEATConfig


def _build_cppn_composite_mutation(
    config: HyperNEATConfig,
) -> CompositeNEATMutation:
    """Build NEAT's composite mutation with a random-activation add-node operator.

    Mirrors the weight/add-connection/toggle knobs of
    ``NEATAlgorithm.from_config`` exactly; only the add-node operator differs (it
    assigns a random activation from the CPPN function set). Kept as a helper,
    not inlined into the factory, so the one meaningful difference from vanilla
    NEAT is obvious.

    Args:
        config: HyperNEAT configuration supplying the mutation rates and CPPN
            function set.

    Returns:
        The assembled `CompositeNEATMutation` for evolving CPPNs.
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
                available_activation_function_names=(
                    config.available_activation_functions
                ),
            ),
            ToggleConnectionEnabledMutation(
                probability_of_application=config.probability_of_toggle_connection_enabled,
            ),
        ]
    )


def make_hyperneat_algorithm(
    config: HyperNEATConfig,
    device_for_phenotype_computation: torch.device | None = None,
) -> NEATAlgorithm:
    """Assemble HyperNEAT as a configured ``NEATAlgorithm`` (component swap, no subclass).

    HyperNEAT evolves CPPNs with standard NEAT genetics, so this starts from a
    vanilla ``NEATAlgorithm`` and replaces exactly two component fields:
    ``mutation`` (random-activation add-node so CPPNs compose heterogeneous
    functions) and ``_phenotype_decoder`` (maps each CPPN to a substrate ANN).
    Crossover, speciation, parent selection, innovation tracking, and the
    ``initial_population_factory`` are inherited untouched: a minimal
    fully-connected 4-inputs+bias->1-output network is exactly a minimal CPPN.

    Args:
        config: HyperNEAT configuration (CPPN genetics plus the substrate and
            weight-expression settings).
        device_for_phenotype_computation: Torch device for the substrate
            phenotypes. Defaults to the device named in the config.

    Returns:
        A `NEATAlgorithm` configured to evolve CPPNs and decode them into
        substrate ANNs. There is no ``HyperNEATAlgorithm`` type.
    """
    resolved_device = device_for_phenotype_computation or torch.device(
        config.device_for_phenotype_evaluation
    )

    substrate = build_layered_substrate(
        input_layer_size=config.substrate_input_layer_size,
        hidden_layer_sizes=config.substrate_hidden_layer_sizes,
        output_layer_size=config.substrate_output_layer_size,
        coordinate_range_min=config.substrate_coordinate_range_min,
        coordinate_range_max=config.substrate_coordinate_range_max,
        include_bias_node=config.include_substrate_bias_node,
    )
    hyperneat_decoder = HyperNEATPhenotypeDecoder(
        substrate=substrate,
        cppn_phenotype_decoder=NEATPhenotypeDecoder(
            device_for_computation=resolved_device
        ),
        weight_expression_threshold=config.weight_expression_threshold,
        max_substrate_connection_weight_magnitude=(
            config.max_substrate_connection_weight_magnitude
        ),
        substrate_node_activation_function_name=(
            config.substrate_node_activation_function
        ),
        device_for_computation=resolved_device,
    )

    base_algorithm = NEATAlgorithm.from_config(
        config, device_for_phenotype_computation
    )
    return dataclasses.replace(
        base_algorithm,
        mutation=_build_cppn_composite_mutation(config),
        _phenotype_decoder=hyperneat_decoder,
    )
