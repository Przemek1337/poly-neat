"""Configuration for HyperNEAT-LEO (Link Expression Output).

References:
    Verbancsics, P., & Stanley, K. O. (2011). Constraining Connectivity to Encourage
        Modularity in HyperNEAT. *GECCO '11: Proceedings of the 13th Annual Conference on
        Genetic and Evolutionary Computation*, pp. 1483-1490. DOI: 10.1145/2001576.2001776
    Huizinga, J., Mouret, J.-B., & Clune, J. (2014). Evolving Neural Networks That Are Both
        Modular and Regular: HyperNeat Plus the Connection Cost Technique. *GECCO '14*,
        pp. 697-704. DOI: 10.1145/2576768.2598232
        (Source of the seed constants used here; it reimplements LEO plus the Gaussian seed
        as its ``HyperNEAT-GS`` treatment.)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from polyneat.configs.configuration_errors import ConfigurationError
from polyneat.configs.hyperneat.hyperneat_config import (
    CPPN_INPUT_NODE_COUNT,
    HyperNEATConfig,
)

# The CPPN keeps HyperNEAT's four coordinate inputs (x1, y1, x2, y2). A coordinate
# *difference* needs no extra input: the Gaussian seed node receives an axis' two
# coordinates through equal and opposite weights, so its weighted sum already is
# that difference (Huizinga et al. 2014, section 2.1).
LEO_CPPN_INPUT_NODE_COUNT: int = CPPN_INPUT_NODE_COUNT
# Queried connection weight, plus the link expression output.
LEO_CPPN_OUTPUT_NODE_COUNT: int = 2

# Which two coordinate inputs form the difference along each axis.
SEED_AXIS_NAME_TO_COORDINATE_INPUT_NODE_IDS: dict[str, tuple[int, int]] = {
    "x": (0, 2),  # x1, x2
    "y": (1, 3),  # y1, y2
}


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
            output reaches this value. The source uses ``LEO >= 0``.
        locality_seed_coordinate_axes: Which coordinate axes get a seeded
            Gaussian in generation 0. The default ``("x",)`` follows the source,
            which reports the x-only variant as the most successful one.
        locality_seed_delta_weight: Magnitude ``w`` of the two equal-and-opposite
            weights running from an axis' two coordinate inputs into the Gaussian
            node. Their weighted sum is ``w * (x2 - x1)``, i.e. the coordinate
            difference - which is why no separate delta input is needed. The
            source uses 0.6.
        locality_seed_gaussian_to_leo_weight: Weight ``g`` from the Gaussian node
            to the LEO output. **The one seed constant the source leaves
            unspecified.** Since ``gaussian(v) = exp(-v^2)`` peaks at 1.0 and the
            LEO bias is negative, ``g`` must exceed ``-bias`` or nothing is ever
            expressed; the default 2.0 places the cutoff at the Gaussian's
            half-maximum.
        locality_seed_bias_weight: Weight ``b`` from the bias input to the LEO
            output. Must be negative, or every connection is expressed. The
            source uses -1.
        substrate_layer_x_coordinates: Explicit per-layer x coordinates, or
            ``None`` to fall back to the evenly spread layered substrate. Needed
            whenever the locality seed has to tell groups of nodes apart, since a
            threshold separates them only when every across-group distance
            exceeds every within-group one - which an even spread does not give.
        substrate_bias_x_coordinate: Where the bias node sits when explicit
            coordinates are used. ``0.0`` keeps it out of both hemispheres for
            sign-based modularity metrics.
    """

    number_of_output_nodes: int = LEO_CPPN_OUTPUT_NODE_COUNT
    initial_population_strategy: str = "leo_seeded"

    link_expression_threshold: float = 0.0
    locality_seed_coordinate_axes: tuple[str, ...] = ("x",)
    locality_seed_delta_weight: float = 0.6
    locality_seed_gaussian_to_leo_weight: float = 2.0
    locality_seed_bias_weight: float = -1.0

    substrate_layer_x_coordinates: tuple[tuple[float, ...], ...] | None = None
    substrate_bias_x_coordinate: float = 0.0

    @property
    def locality_seed_expression_cutoff(self) -> float:
        """The ``|coordinate difference|`` below which the seed expresses a link.

        Solving ``g * exp(-(w * d)^2) + b >= 0`` for ``d`` gives
        ``|d| <= sqrt(ln(g / -b)) / w``. The substrate geometry has to be chosen
        so this value falls between the largest within-group distance and the
        smallest across-group one; otherwise no threshold can separate the
        groups and the seed cannot produce modular structure.

        Returns:
            The cutoff distance, or ``0.0`` when the parameters express nothing.
        """
        expression_ratio = self.locality_seed_gaussian_to_leo_weight / (
            -self.locality_seed_bias_weight
        )
        if expression_ratio <= 1.0:
            return 0.0
        return math.sqrt(math.log(expression_ratio)) / self.locality_seed_delta_weight

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
        """Check the CPPN has HyperNEAT's 4 coordinate inputs and 2 outputs.

        Raises:
            ConfigurationError: Naming the field, the value and the reason.
        """
        if self.number_of_input_nodes != LEO_CPPN_INPUT_NODE_COUNT:
            raise ConfigurationError(
                f"HyperNEAT-LEO keeps HyperNEAT's {LEO_CPPN_INPUT_NODE_COUNT} coordinate "
                f"inputs (x1, y1, x2, y2) - the locality seed forms coordinate differences "
                f"from them through opposite-signed weights, so no delta input is needed; "
                f"got number_of_input_nodes={self.number_of_input_nodes}"
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
                f"known axes: {sorted(SEED_AXIS_NAME_TO_COORDINATE_INPUT_NODE_IDS)}"
            )
        unknown_axes = [
            axis_name
            for axis_name in self.locality_seed_coordinate_axes
            if axis_name not in SEED_AXIS_NAME_TO_COORDINATE_INPUT_NODE_IDS
        ]
        if unknown_axes:
            raise ConfigurationError(
                f"locality_seed_coordinate_axes contains unknown axes {unknown_axes}; "
                f"known axes: {sorted(SEED_AXIS_NAME_TO_COORDINATE_INPUT_NODE_IDS)}"
            )
        if self.locality_seed_delta_weight <= 0.0:
            raise ConfigurationError(
                f"locality_seed_delta_weight must be > 0.0, got "
                f"{self.locality_seed_delta_weight}"
            )
        if self.locality_seed_bias_weight >= 0.0:
            raise ConfigurationError(
                f"locality_seed_bias_weight must be < 0.0, or the seed expresses every "
                f"connection regardless of distance; got {self.locality_seed_bias_weight}"
            )
        if (
            self.locality_seed_gaussian_to_leo_weight
            <= -self.locality_seed_bias_weight
        ):
            raise ConfigurationError(
                f"locality_seed_gaussian_to_leo_weight "
                f"({self.locality_seed_gaussian_to_leo_weight}) must exceed "
                f"-locality_seed_bias_weight ({-self.locality_seed_bias_weight}); the "
                f"Gaussian peaks at 1.0, so otherwise the seed expresses nothing at all"
            )
