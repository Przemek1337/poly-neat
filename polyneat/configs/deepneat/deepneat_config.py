"""Configuration for DeepNEAT.

References:
    Miikkulainen, R., Liang, J., Meyerson, E., Rawal, A., Fink, D., Francon, O., Raju, B.,
        Shahrzad, H., Navruzyan, A., Duffy, N., & Hodjat, B. (2017). Evolving Deep Neural
        Networks. *arXiv:1703.00548*. Published in *Artificial Intelligence in the Age of
        Neural Networks and Brain Computing* (2019), pp. 293-312.
        DOI: 10.1016/B978-0-12-815480-9.00015-3
"""

from __future__ import annotations

from dataclasses import dataclass

from polyneat.configs.configuration_errors import ConfigurationError
from polyneat.configs.neat.neat_config import NEATConfig


@dataclass
class DeepNEATConfig(NEATConfig):
    """Hyperparameters for DeepNEAT (Miikkulainen et al., 2017).

    Inherited NEAT fields that concern *connection weights* are unused: a
    DeepNEAT edge carries no weight, because weights live inside layers and are
    trained by gradient descent during fitness evaluation. The speciation
    coefficients are reused. ``c3`` weights an optional library-defined
    layer-hyperparameter distance; the publication does not define that term,
    so source profiles set it to zero.

    Attributes:
        input_image_channels: Channels of the task's input tensor.
        input_image_height: Height of the task's input tensor.
        input_image_width: Width of the task's input tensor.
        number_of_classes: Output layer width.
        available_filter_counts: Allowed ``filters`` values for conv layers.
        available_kernel_sizes: Allowed ``kernel_size`` values; all must be odd
            so ``padding="same"`` is symmetric.
        available_dense_unit_counts: Allowed ``units`` values for dense layers.
        dropout_rate_min: Lower end of the sampled dropout range.
        dropout_rate_max: Upper end of the sampled dropout range.
        probability_of_new_conv_layer: Chance that a newly inserted layer is a
            conv rather than a dense layer.
        maximum_total_parameter_count: Optional library safety limit. Genomes
            above it are degenerate and score zero. ``None`` disables the
            limit, matching the published algorithm, which reports no such
            fitness constraint.
        training_epochs_per_evaluation: Epochs each genome is trained for before
            being scored. Weights are discarded afterwards; DeepNEAT does not
            inherit them.
        training_learning_rate: Fallback learning rate for phenotypes without
            chromosome-wide DeepNEAT training genes.
        training_batch_size: Minibatch size used during evaluation.
        use_deterministic_training_algorithms: Whether to request deterministic
            cuDNN kernels. Costs throughput; off by default.
        number_of_generations: Number of populations evaluated by examples.
        final_training_epochs: Epochs used to retrain the selected architecture.
        maximum_training_samples: Optional cap inside the official train split.
        maximum_test_samples: Optional cap inside the official test split.
        validation_fraction: Fraction carved from official train for fitness.
    """

    input_image_channels: int = 1
    input_image_height: int = 28
    input_image_width: int = 28
    number_of_classes: int = 10

    available_filter_counts: tuple[int, ...] = (16, 32, 64, 128)
    available_kernel_sizes: tuple[int, ...] = (1, 3, 5)
    available_dense_unit_counts: tuple[int, ...] = (64, 128, 256)
    number_of_filters_min: int | None = None
    number_of_filters_max: int | None = None
    dropout_rate_min: float = 0.0
    dropout_rate_max: float = 0.5
    initial_weight_scaling_min: float = 0.0
    initial_weight_scaling_max: float = 2.0
    available_batch_normalization_options: tuple[bool, ...] = (False,)
    probability_of_new_conv_layer: float = 0.7
    gaussian_mutation_standard_deviation_fraction: float = 0.1

    global_learning_rate_min: float = 1e-4
    global_learning_rate_max: float = 0.1
    global_momentum_min: float = 0.68
    global_momentum_max: float = 0.99
    global_hue_shift_degrees_min: float = 0.0
    global_hue_shift_degrees_max: float = 0.0
    global_saturation_value_shift_min: float = 0.0
    global_saturation_value_shift_max: float = 0.0
    global_saturation_value_scale_min: float = 0.0
    global_saturation_value_scale_max: float = 0.0
    global_cropped_image_size_min: int = 0
    global_cropped_image_size_max: int = 0
    global_spatial_scaling_min: float = 0.0
    global_spatial_scaling_max: float = 0.0
    available_horizontal_flip_options: tuple[bool, ...] = (False,)
    available_variance_normalization_options: tuple[bool, ...] = (False,)
    available_nesterov_momentum_options: tuple[bool, ...] = (False, True)

    probability_of_add_layer_node_mutation: float = 0.15
    probability_of_add_tensor_edge_mutation: float = 0.15
    probability_of_toggle_tensor_edge_mutation: float = 0.05
    probability_of_layer_hyperparameter_mutation: float = 0.5
    probability_of_global_hyperparameter_mutation: float = 0.5

    # DeepNEAT's sources do not define a node-hyperparameter contribution to
    # compatibility distance. Keep the optional PolyNEAT extension off unless
    # a caller explicitly opts in.
    compatibility_distance_coefficient_weight_difference_c3: float = 0.0

    # The published algorithm reports no parameter-count fitness cutoff.
    maximum_total_parameter_count: int | None = None

    training_epochs_per_evaluation: int = 2
    training_learning_rate: float = 1e-3
    training_batch_size: int = 128
    use_deterministic_training_algorithms: bool = False

    number_of_generations: int = 25
    final_training_epochs: int = 20
    maximum_training_samples: int | None = 1_500
    maximum_test_samples: int | None = 1_000
    validation_fraction: float = 0.2

    def validate(self) -> None:
        """Validate the DeepNEAT fields on top of the inherited NEAT checks.

        Raises:
            ConfigurationError: Naming the field, the value and the reason.
        """
        super().validate()

        for probability_field_name in (
            "probability_of_add_layer_node_mutation",
            "probability_of_add_tensor_edge_mutation",
            "probability_of_toggle_tensor_edge_mutation",
            "probability_of_layer_hyperparameter_mutation",
            "probability_of_global_hyperparameter_mutation",
            "probability_of_new_conv_layer",
        ):
            value = getattr(self, probability_field_name)
            if not (0.0 <= value <= 1.0):
                raise ConfigurationError(
                    f"{probability_field_name} must be in [0.0, 1.0], got {value}"
                )

        for geometry_field_name in (
            "input_image_channels",
            "input_image_height",
            "input_image_width",
            "number_of_classes",
        ):
            value = getattr(self, geometry_field_name)
            if value < 1:
                raise ConfigurationError(f"{geometry_field_name} must be >= 1, got {value}")

        for search_space_field_name in (
            "available_filter_counts",
            "available_kernel_sizes",
            "available_dense_unit_counts",
        ):
            values = getattr(self, search_space_field_name)
            if not values:
                raise ConfigurationError(
                    f"{search_space_field_name} must offer at least one value, "
                    f"got {values}"
                )
            if any(value < 1 for value in values):
                raise ConfigurationError(
                    f"every value in {search_space_field_name} must be >= 1, "
                    f"got {values}"
                )

        if (self.number_of_filters_min is None) != (self.number_of_filters_max is None):
            raise ConfigurationError(
                "number_of_filters_min and number_of_filters_max must either both be "
                "set or both be null"
            )
        if self.number_of_filters_min is not None:
            assert self.number_of_filters_max is not None
            if (
                self.number_of_filters_min < 1
                or self.number_of_filters_max < self.number_of_filters_min
            ):
                raise ConfigurationError(
                    "number_of_filters_min/max must satisfy 1 <= min <= max, got "
                    f"{self.number_of_filters_min}/{self.number_of_filters_max}"
                )

        if any(kernel_size % 2 == 0 for kernel_size in self.available_kernel_sizes):
            raise ConfigurationError(
                f"available_kernel_sizes must contain only odd sizes so 'same' padding "
                f"is symmetric, got {self.available_kernel_sizes}"
            )

        if not (0.0 <= self.dropout_rate_min <= self.dropout_rate_max < 1.0):
            raise ConfigurationError(
                f"dropout rates must satisfy 0.0 <= dropout_rate_min "
                f"({self.dropout_rate_min}) <= dropout_rate_max "
                f"({self.dropout_rate_max}) < 1.0"
            )

        if not (
            0.0
            <= self.initial_weight_scaling_min
            <= self.initial_weight_scaling_max
        ):
            raise ConfigurationError(
                "initial weight scaling must satisfy 0.0 <= min <= max, got "
                f"{self.initial_weight_scaling_min}/{self.initial_weight_scaling_max}"
            )
        if not self.available_batch_normalization_options or any(
            not isinstance(value, bool)
            for value in self.available_batch_normalization_options
        ):
            raise ConfigurationError(
                "available_batch_normalization_options must contain at least one boolean"
            )
        for option_field_name in (
            "available_horizontal_flip_options",
            "available_variance_normalization_options",
            "available_nesterov_momentum_options",
        ):
            options = getattr(self, option_field_name)
            if not options or any(not isinstance(value, bool) for value in options):
                raise ConfigurationError(
                    f"{option_field_name} must contain at least one boolean"
                )
        if self.gaussian_mutation_standard_deviation_fraction <= 0.0:
            raise ConfigurationError(
                "gaussian_mutation_standard_deviation_fraction must be > 0.0, got "
                f"{self.gaussian_mutation_standard_deviation_fraction}"
            )

        for minimum_field_name, maximum_field_name in (
            ("global_learning_rate_min", "global_learning_rate_max"),
            ("global_momentum_min", "global_momentum_max"),
            ("global_hue_shift_degrees_min", "global_hue_shift_degrees_max"),
            (
                "global_saturation_value_shift_min",
                "global_saturation_value_shift_max",
            ),
            (
                "global_saturation_value_scale_min",
                "global_saturation_value_scale_max",
            ),
            ("global_spatial_scaling_min", "global_spatial_scaling_max"),
        ):
            minimum = getattr(self, minimum_field_name)
            maximum = getattr(self, maximum_field_name)
            if minimum < 0.0 or maximum < minimum:
                raise ConfigurationError(
                    f"{minimum_field_name}/{maximum_field_name} must satisfy "
                    f"0.0 <= min <= max, got {minimum}/{maximum}"
                )
        if self.global_learning_rate_min <= 0.0:
            raise ConfigurationError("global_learning_rate_min must be > 0.0")
        if self.global_momentum_max >= 1.0:
            raise ConfigurationError("global_momentum_max must be < 1.0")
        if not (
            0
            <= self.global_cropped_image_size_min
            <= self.global_cropped_image_size_max
        ):
            raise ConfigurationError(
                "global cropped image size must satisfy 0 <= min <= max, got "
                f"{self.global_cropped_image_size_min}/"
                f"{self.global_cropped_image_size_max}"
            )
        if self.global_cropped_image_size_max > min(
            self.input_image_height, self.input_image_width
        ):
            raise ConfigurationError(
                "global_cropped_image_size_max must fit inside the configured image, got "
                f"{self.global_cropped_image_size_max} for "
                f"{self.input_image_height}x{self.input_image_width}"
            )

        if (
            self.maximum_total_parameter_count is not None
            and self.maximum_total_parameter_count < 1
        ):
            raise ConfigurationError(
                f"maximum_total_parameter_count must be >= 1, got "
                f"{self.maximum_total_parameter_count}"
            )

        if self.training_epochs_per_evaluation < 1:
            raise ConfigurationError(
                f"training_epochs_per_evaluation must be >= 1, got "
                f"{self.training_epochs_per_evaluation}"
            )
        if self.training_learning_rate <= 0.0:
            raise ConfigurationError(
                f"training_learning_rate must be > 0.0, got {self.training_learning_rate}"
            )
        if self.training_batch_size < 1:
            raise ConfigurationError(
                f"training_batch_size must be >= 1, got {self.training_batch_size}"
            )
        if self.number_of_generations < 1:
            raise ConfigurationError(
                f"number_of_generations must be >= 1, got {self.number_of_generations}"
            )
        if self.final_training_epochs < 1:
            raise ConfigurationError(
                f"final_training_epochs must be >= 1, got {self.final_training_epochs}"
            )
        for sample_cap_field_name in (
            "maximum_training_samples",
            "maximum_test_samples",
        ):
            sample_cap = getattr(self, sample_cap_field_name)
            if sample_cap is not None and sample_cap < 2:
                raise ConfigurationError(
                    f"{sample_cap_field_name} must be >= 2 or null, got {sample_cap}"
                )
        if not (0.0 < self.validation_fraction < 1.0):
            raise ConfigurationError(
                f"validation_fraction must be in (0.0, 1.0), got "
                f"{self.validation_fraction}"
            )
