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
    coefficients are reused, with ``c3`` weighting the layer-hyperparameter
    distance instead of a weight difference.

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
        maximum_total_parameter_count: Genomes whose phenotype exceeds this are
            degenerate and score zero. The VRAM fuse - without it a single
            oversized genome can kill an entire run with an OOM.
        training_epochs_per_evaluation: Epochs each genome is trained for before
            being scored. Weights are discarded afterwards; DeepNEAT does not
            inherit them.
        training_learning_rate: Adam learning rate used during evaluation.
        training_batch_size: Minibatch size used during evaluation.
        use_deterministic_training_algorithms: Whether to request deterministic
            cuDNN kernels. Costs throughput; off by default.
    """

    input_image_channels: int = 1
    input_image_height: int = 28
    input_image_width: int = 28
    number_of_classes: int = 10

    available_filter_counts: tuple[int, ...] = (16, 32, 64, 128)
    available_kernel_sizes: tuple[int, ...] = (1, 3, 5)
    available_dense_unit_counts: tuple[int, ...] = (64, 128, 256)
    dropout_rate_min: float = 0.0
    dropout_rate_max: float = 0.5
    probability_of_new_conv_layer: float = 0.7

    probability_of_add_layer_node_mutation: float = 0.15
    probability_of_add_tensor_edge_mutation: float = 0.15
    probability_of_toggle_tensor_edge_mutation: float = 0.05
    probability_of_layer_hyperparameter_mutation: float = 0.5

    maximum_total_parameter_count: int = 20_000_000

    training_epochs_per_evaluation: int = 2
    training_learning_rate: float = 1e-3
    training_batch_size: int = 128
    use_deterministic_training_algorithms: bool = False

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

        if self.maximum_total_parameter_count < 1:
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
