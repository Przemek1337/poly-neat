"""HyperNEAT-LEO as a ``HyperNEATAlgorithm`` subclass.

References:
    Verbancsics, P., & Stanley, K. O. (2011). Constraining Connectivity to Encourage
        Modularity in HyperNEAT. *GECCO '11: Proceedings of the 13th Annual Conference on
        Genetic and Evolutionary Computation*, pp. 1483-1490. DOI: 10.1145/2001576.2001776
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch

from polyneat.algorithms.hyperneat.hyperneat_algorithm import HyperNEATAlgorithm
from polyneat.algorithms.hyperneat.substrate import (
    build_layered_substrate,
    build_substrate_from_explicit_layer_coordinates,
)
from polyneat.algorithms.hyperneatleo.leo_phenotype_decoder import (
    HyperNEATLEOPhenotypeDecoder,
)
from polyneat.configs.hyperneatleo.hyperneatleo_config import HyperNEATLEOConfig
from polyneat.configs.neat.neat_config import NEATConfig
from polyneat.core.component_protocols import PhenotypeDecoder
from polyneat.core.neat.neat_genome import NEATGenome
from polyneat.core.neat.neat_phenotype_decoder import NEATPhenotypeDecoder


@dataclass
class HyperNEATLEOAlgorithm(HyperNEATAlgorithm):
    """HyperNEAT whose CPPN decides link expression on a dedicated output.

    Overrides only ``_build_phenotype_decoder``. The CPPN mutation operator
    (random activations on inserted nodes), crossover, speciation, parent
    selection and the generational loop are inherited from
    :class:`~polyneat.algorithms.hyperneat.hyperneat_algorithm.HyperNEATAlgorithm`
    and :class:`~polyneat.core.neat.neat_algorithm.NEATAlgorithm`. Generation 0
    comes from the ``leo_seeded`` strategy, selected by
    :class:`~polyneat.configs.hyperneatleo.hyperneatleo_config.HyperNEATLEOConfig`'s
    default ``initial_population_strategy``.

    Build it with ``HyperNEATLEOAlgorithm.from_config(leo_config)``.
    """

    @classmethod
    def _build_phenotype_decoder(
        cls, config: NEATConfig, device: torch.device
    ) -> PhenotypeDecoder[NEATGenome]:
        """Build the LEO decoder over the configured substrate.

        When ``substrate_layer_x_coordinates`` is set the substrate is built from
        those explicit coordinates, which is how a task separates groups of nodes
        by a gap. That is required whenever the locality seed has to distinguish
        groups: a distance threshold separates them only when every across-group
        distance exceeds every within-group one, and an even spread does not give
        that. Otherwise the substrate falls back to the evenly spread layered
        layout.

        Args:
            config: Validated ``HyperNEATLEOConfig``.
            device: Torch device the substrate phenotypes run on.

        Returns:
            The CPPN-to-substrate decoder using link expression outputs.
        """
        leo_config = cast("HyperNEATLEOConfig", config)

        if leo_config.substrate_layer_x_coordinates is not None:
            substrate = build_substrate_from_explicit_layer_coordinates(
                layer_x_coordinates=leo_config.substrate_layer_x_coordinates,
                coordinate_range_min=leo_config.substrate_coordinate_range_min,
                coordinate_range_max=leo_config.substrate_coordinate_range_max,
                bias_x_coordinate=(
                    leo_config.substrate_bias_x_coordinate
                    if leo_config.include_substrate_bias_node
                    else None
                ),
            )
        else:
            substrate = build_layered_substrate(
                input_layer_size=leo_config.substrate_input_layer_size,
                hidden_layer_sizes=leo_config.substrate_hidden_layer_sizes,
                output_layer_size=leo_config.substrate_output_layer_size,
                coordinate_range_min=leo_config.substrate_coordinate_range_min,
                coordinate_range_max=leo_config.substrate_coordinate_range_max,
                include_bias_node=leo_config.include_substrate_bias_node,
            )

        return HyperNEATLEOPhenotypeDecoder(
            substrate=substrate,
            cppn_phenotype_decoder=NEATPhenotypeDecoder(device_for_computation=device),
            link_expression_threshold=leo_config.link_expression_threshold,
            max_substrate_connection_weight_magnitude=(
                leo_config.max_substrate_connection_weight_magnitude
            ),
            substrate_node_activation_function_name=(
                leo_config.substrate_node_activation_function
            ),
            device_for_computation=device,
        )
