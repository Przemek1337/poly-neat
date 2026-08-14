"""Configuration for HyperNEAT-LEO (Link Expression Output).

References:
    Verbancsics, P., & Stanley, K. O. (2011). Constraining Connectivity to Encourage
        Modularity in HyperNEAT. *GECCO '11: Proceedings of the 13th Annual Conference on
        Genetic and Evolutionary Computation*, pp. 1483-1490. DOI: 10.1145/2001576.2001776
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from polyneat.configs.configuration_errors import ConfigurationError
from polyneat.configs.hyperneat.hyperneat_config import HyperNEATConfig

# x1, y1, x2, y2, dx, dy. The deltas are explicit inputs because a CPPN has no
# subtraction node, so locality cannot otherwise be expressed at all.
LEO_CPPN_INPUT_NODE_COUNT: int = 6
# Queried connection weight, plus the link expression output.
LEO_CPPN_OUTPUT_NODE_COUNT: int = 2

SEED_AXIS_NAME_TO_DELTA_INPUT_NODE_ID: dict[str, int] = {"x": 4, "y": 5}


@dataclass
class HyperNEATLEOConfig(HyperNEATConfig):
    """Hyperparameters for HyperNEAT-LEO (Verbancsics & Stanley, 2011).

    LEO separates *whether* a substrate connection exists from *how strong* it
    is. Classic HyperNEAT conflates the two: a connection exists when the queried
    weight's magnitude clears ``weight_expression_threshold``. Here a second CPPN
    output decides expression on its own, so a weak-but-present connection
    becomes expressible.

    ``weight_expression_threshold`` is inherited but unused. Setting it is
    rejected rather than ignored, in line with the library's strict-config rule.

    Attributes:
        link_expression_threshold: A connection is expressed when the CPPN's LEO
            output is strictly greater than this value.
        locality_seed_coordinate_axes: Which coordinate axes get a seeded
            Gaussian in generation 0. On a layered substrate ``y`` encodes the
            layer index, so its delta is constant between adjacent layers and
            seeding it adds only dead structure - hence the ``("x",)`` default.
        locality_seed_delta_weight: Weight ``w`` on the ``delta -> Gaussian``
            edge. With bias ``b`` the seeded rule is
            ``LEO = exp(-(w*delta)^2) + b``, expressed while
            ``|delta| < sqrt(ln(-1/b)) / w``.
        locality_seed_bias_weight: Weight ``b`` on the ``bias -> LEO`` edge. Must
            lie in ``(-1, 0)``: at ``b >= 0`` every connection is expressed, at
            ``b <= -1`` none is.
        substrate_layer_x_coordinates: Explicit per-layer x coordinates, or
            ``None`` to fall back to the evenly spread layered substrate. Needed
            whenever the locality seed has to tell groups of nodes apart, since a
            threshold separates them only when every across-group distance
            exceeds every within-group one - which an even spread does not give.
        substrate_bias_x_coordinate: Where the bias node sits when explicit
            coordinates are used. ``0.0`` keeps it out of both hemispheres for
            sign-based modularity metrics.
    """

    number_of_input_nodes: int = LEO_CPPN_INPUT_NODE_COUNT
    number_of_output_nodes: int = LEO_CPPN_OUTPUT_NODE_COUNT
    initial_population_strategy: str = "leo_seeded"

    link_expression_threshold: float = 0.0
    locality_seed_coordinate_axes: tuple[str, ...] = ("x",)
    locality_seed_delta_weight: float = 1.0
    locality_seed_bias_weight: float = -0.5

    substrate_layer_x_coordinates: tuple[tuple[float, ...], ...] | None = None
    substrate_bias_x_coordinate: float = 0.0

    def __post_init__(self) -> None:
        """Normalize nested coordinate sequences, then run the usual validation.

        ``AlgorithmConfig.from_dict`` coerces list to tuple only for fields whose
        default *is* a tuple, and only at the top level. This field defaults to
        ``None`` and is nested, so a YAML config would otherwise leave lists of
        lists behind and the declared type would be a lie.
        """
        if self.substrate_layer_x_coordinates is not None:
            self.substrate_layer_x_coordinates = tuple(
                tuple(float(x_coordinate) for x_coordinate in layer)
                for layer in self.substrate_layer_x_coordinates
            )
        if not isinstance(self.locality_seed_coordinate_axes, tuple):
            self.locality_seed_coordinate_axes = tuple(self.locality_seed_coordinate_axes)
        super().__post_init__()

    def _validate_cppn_input_output_counts(self) -> None:
        """Check the CPPN has 6 inputs (coordinates plus deltas) and 2 outputs.

        Raises:
            ConfigurationError: Naming the field, the value and the reason.
        """
        if self.number_of_input_nodes != LEO_CPPN_INPUT_NODE_COUNT:
            raise ConfigurationError(
                f"HyperNEAT-LEO feeds the CPPN both endpoint coordinates and their "
                f"deltas, so it must have exactly {LEO_CPPN_INPUT_NODE_COUNT} inputs "
                f"(x1, y1, x2, y2, dx, dy); got "
                f"number_of_input_nodes={self.number_of_input_nodes}"
            )
        if self.number_of_output_nodes != LEO_CPPN_OUTPUT_NODE_COUNT:
            raise ConfigurationError(
                f"HyperNEAT-LEO needs exactly {LEO_CPPN_OUTPUT_NODE_COUNT} CPPN outputs "
                f"(connection weight, link expression); got "
                f"number_of_output_nodes={self.number_of_output_nodes}"
            )

    def _validate_substrate_settings(self) -> None:
        """Validate substrate geometry, honouring explicit layer coordinates.

        When explicit coordinates are given the layer-size fields are unused, so
        validating them would reject sensible substrates over values nobody
        reads.

        Raises:
            ConfigurationError: Naming the field, the value and the reason.
        """
        if self.substrate_layer_x_coordinates is None:
            super()._validate_substrate_settings()
            return

        if len(self.substrate_layer_x_coordinates) < 2:
            raise ConfigurationError(
                f"substrate_layer_x_coordinates needs at least an input and an output "
                f"layer, got {len(self.substrate_layer_x_coordinates)}"
            )
        if any(len(layer) == 0 for layer in self.substrate_layer_x_coordinates):
            raise ConfigurationError(
                "every layer in substrate_layer_x_coordinates must have at least one node"
            )
        if self.max_substrate_connection_weight_magnitude <= 0.0:
            raise ConfigurationError(
                f"max_substrate_connection_weight_magnitude must be > 0.0, got "
                f"{self.max_substrate_connection_weight_magnitude}"
            )
        if self.substrate_coordinate_range_min >= self.substrate_coordinate_range_max:
            raise ConfigurationError(
                f"substrate_coordinate_range_min ({self.substrate_coordinate_range_min}) "
                f"must be < substrate_coordinate_range_max "
                f"({self.substrate_coordinate_range_max})"
            )

        from polyneat.nn.activation_functions import (
            ACTIVATION_FUNCTION_NAME_TO_CALLABLE,
        )

        if self.substrate_node_activation_function not in ACTIVATION_FUNCTION_NAME_TO_CALLABLE:
            raise ConfigurationError(
                f"substrate_node_activation_function "
                f"'{self.substrate_node_activation_function}' is not a registered "
                f"activation function. Known: "
                f"{sorted(ACTIVATION_FUNCTION_NAME_TO_CALLABLE.keys())}"
            )

    def _validate_connection_expression_settings(self) -> None:
        """Check the LEO expression rule and the locality seed parameters.

        Raises:
            ConfigurationError: Naming the field, the value and the reason.
        """
        if self.weight_expression_threshold != HyperNEATConfig.weight_expression_threshold:
            raise ConfigurationError(
                f"weight_expression_threshold is not used by HyperNEAT-LEO - expression "
                f"is decided by the CPPN's link expression output. Set "
                f"link_expression_threshold instead; got "
                f"weight_expression_threshold={self.weight_expression_threshold}"
            )
        if not math.isfinite(self.link_expression_threshold):
            raise ConfigurationError(
                f"link_expression_threshold must be finite, got "
                f"{self.link_expression_threshold}"
            )
        if not self.locality_seed_coordinate_axes:
            raise ConfigurationError(
                "locality_seed_coordinate_axes must name at least one axis; "
                f"known axes: {sorted(SEED_AXIS_NAME_TO_DELTA_INPUT_NODE_ID)}"
            )
        unknown_axes = [
            axis_name
            for axis_name in self.locality_seed_coordinate_axes
            if axis_name not in SEED_AXIS_NAME_TO_DELTA_INPUT_NODE_ID
        ]
        if unknown_axes:
            raise ConfigurationError(
                f"locality_seed_coordinate_axes contains unknown axes {unknown_axes}; "
                f"known axes: {sorted(SEED_AXIS_NAME_TO_DELTA_INPUT_NODE_ID)}"
            )
        if self.locality_seed_delta_weight <= 0.0:
            raise ConfigurationError(
                f"locality_seed_delta_weight must be > 0.0, got "
                f"{self.locality_seed_delta_weight}"
            )
        if not (-1.0 < self.locality_seed_bias_weight < 0.0):
            raise ConfigurationError(
                f"locality_seed_bias_weight must be in (-1.0, 0.0) - at >= 0.0 the seed "
                f"expresses every connection, at <= -1.0 it expresses none; got "
                f"{self.locality_seed_bias_weight}"
            )
