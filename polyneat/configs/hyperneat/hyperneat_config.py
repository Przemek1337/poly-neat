"""Configuration for the HyperNEAT algorithm.

References:
    Stanley, K. O., D'Ambrosio, D. B., & Gauci, J. (2009). A Hypercube-Based Encoding for
        Evolving Large-Scale Neural Networks. *Artificial Life*, 15(2), 185-212.
        DOI: 10.1162/artl.2009.15.2.15202
"""
from __future__ import annotations

from dataclasses import dataclass

from polyneat.configs.configuration_errors import ConfigurationError
from polyneat.configs.neat.neat_config import NEATConfig

# The CPPN has exactly these many coordinate inputs (x1, y1, x2, y2) and one
# output (the queried connection weight). 2-D substrate coordinates -> 4 inputs.
CPPN_INPUT_NODE_COUNT: int = 4
CPPN_OUTPUT_NODE_COUNT: int = 1


@dataclass
class HyperNEATConfig(NEATConfig):
    """Hyperparameters for HyperNEAT (Stanley, D'Ambrosio & Gauci, 2009).

    HyperNEAT evolves a CPPN with standard NEAT genetics, so every inherited
    NEAT field controls the *CPPN* population. ``number_of_input_nodes`` and
    ``number_of_output_nodes`` therefore describe the CPPN (fixed at 4 and 1);
    the task-facing network is the *substrate*, described by the fields below.
    """

    number_of_input_nodes: int = CPPN_INPUT_NODE_COUNT
    number_of_output_nodes: int = CPPN_OUTPUT_NODE_COUNT
    initial_weight_range_min: float = -3.0
    initial_weight_range_max: float = 3.0
    available_activation_functions: tuple[str, ...] = (
        "sigmoid",
        "gaussian",
        "sine",
        "absolute_value",
        "identity",
    )
    default_activation_function_for_hidden_nodes: str = "sigmoid"
    default_activation_function_for_output_nodes: str = "identity"

    substrate_input_layer_size: int = 2
    substrate_hidden_layer_sizes: tuple[int, ...] = (3,)
    substrate_output_layer_size: int = 1
    substrate_coordinate_range_min: float = -1.0
    substrate_coordinate_range_max: float = 1.0
    include_substrate_bias_node: bool = True
    substrate_node_activation_function: str = "steepened_sigmoid"

    weight_expression_threshold: float = 0.2
    max_substrate_connection_weight_magnitude: float = 3.0

    def validate(self) -> None:
        """Validate the CPPN and substrate settings.

        Split into three overridable parts so a variant that changes only one
        aspect - HyperNEAT-LEO changes the CPPN's input/output counts and the
        connection-expression rule - can reuse the rest instead of copying it.

        Raises:
            ConfigurationError: If the CPPN input/output counts are not 4/1, a
                substrate layer size is non-positive, the weight-expression
                threshold is outside ``[0, 1)``, the maximum weight magnitude is
                non-positive, the coordinate range is empty or inverted, or the
                substrate node activation is not a registered function.
        """
        super().validate()
        self._validate_cppn_input_output_counts()
        self._validate_substrate_settings()
        self._validate_connection_expression_settings()

    def _validate_cppn_input_output_counts(self) -> None:
        """Check the CPPN has exactly 4 coordinate inputs and 1 weight output.

        Raises:
            ConfigurationError: Naming the field, the value and the reason.
        """
        if self.number_of_input_nodes != CPPN_INPUT_NODE_COUNT:
            raise ConfigurationError(
                f"HyperNEAT uses 2-D substrate coordinates, so the CPPN must have "
                f"exactly {CPPN_INPUT_NODE_COUNT} inputs (x1, y1, x2, y2); got "
                f"number_of_input_nodes={self.number_of_input_nodes}"
            )
        if self.number_of_output_nodes != CPPN_OUTPUT_NODE_COUNT:
            raise ConfigurationError(
                f"The CPPN must have exactly {CPPN_OUTPUT_NODE_COUNT} output "
                f"(the queried connection weight); got "
                f"number_of_output_nodes={self.number_of_output_nodes}"
            )

    def _validate_substrate_settings(self) -> None:
        """Check substrate geometry and the substrate node activation.

        Raises:
            ConfigurationError: Naming the field, the value and the reason.
        """
        if self.substrate_input_layer_size < 1:
            raise ConfigurationError(
                f"substrate_input_layer_size must be >= 1, got "
                f"{self.substrate_input_layer_size}"
            )
        if self.substrate_output_layer_size < 1:
            raise ConfigurationError(
                f"substrate_output_layer_size must be >= 1, got "
                f"{self.substrate_output_layer_size}"
            )
        if any(layer_size < 1 for layer_size in self.substrate_hidden_layer_sizes):
            raise ConfigurationError(
                f"every substrate hidden layer size must be >= 1, got "
                f"{self.substrate_hidden_layer_sizes}"
            )
        if self.max_substrate_connection_weight_magnitude <= 0.0:
            raise ConfigurationError(
                f"max_substrate_connection_weight_magnitude must be > 0.0, got "
                f"{self.max_substrate_connection_weight_magnitude}"
            )
        if self.substrate_coordinate_range_min >= self.substrate_coordinate_range_max:
            raise ConfigurationError(
                f"substrate_coordinate_range_min "
                f"({self.substrate_coordinate_range_min}) must be < "
                f"substrate_coordinate_range_max "
                f"({self.substrate_coordinate_range_max})"
            )

        from polyneat.nn.activation_functions import (
            ACTIVATION_FUNCTION_NAME_TO_CALLABLE,
        )

        if (
            self.substrate_node_activation_function
            not in ACTIVATION_FUNCTION_NAME_TO_CALLABLE
        ):
            raise ConfigurationError(
                f"substrate_node_activation_function "
                f"'{self.substrate_node_activation_function}' is not a registered "
                f"activation function. Known: "
                f"{sorted(ACTIVATION_FUNCTION_NAME_TO_CALLABLE.keys())}"
            )

    def _validate_connection_expression_settings(self) -> None:
        """Check the rule deciding which substrate connections are expressed.

        Classic HyperNEAT expresses a connection when the queried weight's
        magnitude clears a threshold. HyperNEAT-LEO replaces that rule wholesale
        with a dedicated CPPN output, so it overrides this method rather than
        inheriting a check on a field it does not use.

        Raises:
            ConfigurationError: If ``weight_expression_threshold`` is outside
                ``[0.0, 1.0)``.
        """
        if not (0.0 <= self.weight_expression_threshold < 1.0):
            raise ConfigurationError(
                f"weight_expression_threshold must be in [0.0, 1.0), got "
                f"{self.weight_expression_threshold}"
            )
