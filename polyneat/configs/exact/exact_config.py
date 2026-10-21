"""Configuration for the EXACT algorithm.

References:
    Desell, T. (2017). Developing a Volunteer Computing Project to Evolve
        Convolutional Neural Networks and Their Hyperparameters. 2017 IEEE
        13th International Conference on e-Science, pp. 19-28.
"""
from __future__ import annotations

from dataclasses import dataclass

from polyneat.configs.configuration_errors import ConfigurationError
from polyneat.configs.neat.neat_config import NEATConfig

EXACT_INPUT_NODE_COUNT: int = 1

_MUTATION_PROBABILITY_SUM_TOLERANCE: float = 1e-6


@dataclass
class EXACTConfig(NEATConfig):
    """Hyperparameters for EXACT (Desell, 2017).

    EXACT rides PolyNEAT's generational loop, so the inherited NEAT fields
    that the loop reads (population size, elitism, survival fraction,
    ``probability_of_crossover_vs_mutation_only``, tournament size) stay
    meaningful. The NEAT weight-mutation, compatibility-distance, and
    activation-function fields are inert: EXACT does not evolve weights,
    does not speciate, and fixes its activations (clamped leaky ReLU on
    hidden filters, identity on the 1x1 softmax outputs).

    Defaults follow the paper where it states values: filter size
    modifications ``[-2, -1, +1, +2]`` (section III-B), the 20% crossover
    rate and the fixed-search backpropagation hyperparameters of section
    VIII-B1, and ``µ_max = 0.99``, ``η_min = 0.00001``, ``λ_min = 0.000001``
    (section VII). The paper does not state per-operator mutation rates or
    the two crossover inclusion rates; those defaults are this
    implementation's own. ``tournament_size_for_parent_selection = 1``
    reproduces the paper's uniform-random genome selection.
    """

    number_of_input_nodes: int = EXACT_INPUT_NODE_COUNT
    initial_population_strategy: str = "exact_minimal_cnn"
    tournament_size_for_parent_selection: int = 1
    probability_of_crossover_vs_mutation_only: float = 0.2

    # Input image geometry (the single input node's filter size).
    input_image_height: int = 28
    input_image_width: int = 28

    # Mutation (section III-B): a genome receives this many operator draws.
    number_of_mutations_per_genome: int = 3
    probability_of_disable_edge_mutation: float = 1.0 / 12.0
    probability_of_enable_edge_mutation: float = 2.0 / 12.0
    probability_of_split_edge_mutation: float = 2.0 / 12.0
    probability_of_add_edge_mutation: float = 4.0 / 12.0
    probability_of_add_node_mutation: float = 1.0 / 12.0
    probability_of_change_filter_size_mutation: float = 1.0 / 12.0
    probability_of_change_filter_size_x_mutation: float = 1.0 / 24.0
    probability_of_change_filter_size_y_mutation: float = 1.0 / 24.0
    filter_size_change_options: tuple[int, ...] = (-2, -1, 1, 2)
    minimum_hidden_filter_size: int = 1
    maximum_mutation_attempts_for_reachable_child: int = 10

    # Crossover (section III-C). Values are this implementation's defaults.
    more_fit_parent_edge_inclusion_rate: float = 0.8
    less_fit_parent_edge_inclusion_rate: float = 0.4

    # Epigenetic weight initialization (section III-D).
    use_epigenetic_weight_initialization: bool = True

    # Simplex hyperparameter optimization (section IV, eqs. 1-2).
    use_simplex_hyperparameter_optimization: bool = False
    simplex_number_of_selected_genomes: int = 5
    simplex_line_scale: float = 2.0
    simplex_line_offset: float = 0.5

    # Backpropagation (section VII; Section VIII-B1 fixed hyperparameters).
    leaky_relu_negative_slope: float = 0.1
    activation_clamp_maximum: float = 5.5
    number_of_training_epochs_per_genome: int = 5
    training_batch_size: int = 50
    velocity_reset_interval: int = 1000
    use_batch_normalization: bool = True
    input_dropout_probability: float = 0.0
    hidden_dropout_probability: float = 0.0
    batch_normalization_alpha: float = 0.1
    backpropagation_learning_rate: float = 0.0025
    learning_rate_decay_factor: float = 0.95
    minimum_learning_rate: float = 0.00001
    backpropagation_momentum: float = 0.5
    momentum_decay_factor: float = 0.95
    maximum_momentum: float = 0.99
    backpropagation_weight_decay: float = 0.0005
    weight_decay_decay_factor: float = 0.95
    minimum_weight_decay: float = 0.000001

    def validate(self) -> None:
        """Validate EXACT fields on top of the NEAT and shared validation.

        Raises:
            ConfigurationError: If the input node count is not 1, an image
                dimension is non-positive, the eight operator probabilities
                do not sum to 1, a size-change option is zero, a crossover
                rate leaves ``[0, 1]``, or a training hyperparameter is out
                of range (nesterov momentum requires ``0 < momentum < 1``).
        """
        super().validate()
        if self.number_of_input_nodes != EXACT_INPUT_NODE_COUNT:
            raise ConfigurationError(
                f"EXACT genomes have exactly {EXACT_INPUT_NODE_COUNT} input node (the "
                f"image); got number_of_input_nodes={self.number_of_input_nodes}"
            )
        if self.input_image_height < 1 or self.input_image_width < 1:
            raise ConfigurationError(
                f"input image dimensions must be >= 1, got "
                f"{self.input_image_height}x{self.input_image_width}"
            )
        if self.number_of_mutations_per_genome < 1:
            raise ConfigurationError(
                f"number_of_mutations_per_genome must be >= 1, "
                f"got {self.number_of_mutations_per_genome}"
            )
        mutation_probability_fields = (
            "probability_of_disable_edge_mutation",
            "probability_of_enable_edge_mutation",
            "probability_of_split_edge_mutation",
            "probability_of_add_edge_mutation",
            "probability_of_add_node_mutation",
            "probability_of_change_filter_size_mutation",
            "probability_of_change_filter_size_x_mutation",
            "probability_of_change_filter_size_y_mutation",
        )
        for probability_field in mutation_probability_fields:
            value = getattr(self, probability_field)
            if not (0.0 <= value <= 1.0):
                raise ConfigurationError(
                    f"{probability_field} must be in [0.0, 1.0], got {value}"
                )
        probability_sum = sum(
            getattr(self, probability_field)
            for probability_field in mutation_probability_fields
        )
        if abs(probability_sum - 1.0) > _MUTATION_PROBABILITY_SUM_TOLERANCE:
            raise ConfigurationError(
                f"the eight mutation operator probabilities must sum to 1.0, "
                f"got {probability_sum}"
            )
        if not self.filter_size_change_options:
            raise ConfigurationError("filter_size_change_options must not be empty")
        if any(option == 0 for option in self.filter_size_change_options):
            raise ConfigurationError(
                f"filter_size_change_options must not contain 0, "
                f"got {self.filter_size_change_options}"
            )
        if self.minimum_hidden_filter_size < 1:
            raise ConfigurationError(
                f"minimum_hidden_filter_size must be >= 1, "
                f"got {self.minimum_hidden_filter_size}"
            )
        if self.maximum_mutation_attempts_for_reachable_child < 1:
            raise ConfigurationError(
                f"maximum_mutation_attempts_for_reachable_child must be >= 1, "
                f"got {self.maximum_mutation_attempts_for_reachable_child}"
            )
        for rate_field in (
            "more_fit_parent_edge_inclusion_rate",
            "less_fit_parent_edge_inclusion_rate",
        ):
            value = getattr(self, rate_field)
            if not (0.0 <= value <= 1.0):
                raise ConfigurationError(f"{rate_field} must be in [0.0, 1.0], got {value}")
        if self.leaky_relu_negative_slope <= 0.0:
            raise ConfigurationError(
                f"leaky_relu_negative_slope must be > 0, got {self.leaky_relu_negative_slope}"
            )
        if self.activation_clamp_maximum <= 0.0:
            raise ConfigurationError(
                f"activation_clamp_maximum must be > 0, got {self.activation_clamp_maximum}"
            )
        if self.number_of_training_epochs_per_genome < 1:
            raise ConfigurationError(
                f"number_of_training_epochs_per_genome must be >= 1, "
                f"got {self.number_of_training_epochs_per_genome}"
            )
        if self.training_batch_size < 1:
            raise ConfigurationError(
                f"training_batch_size must be >= 1, got {self.training_batch_size}"
            )
        if self.simplex_number_of_selected_genomes < 2:
            raise ConfigurationError(
                f"simplex_number_of_selected_genomes must be >= 2, "
                f"got {self.simplex_number_of_selected_genomes}"
            )
        if self.simplex_line_scale <= 0.0:
            raise ConfigurationError(
                f"simplex_line_scale must be > 0, got {self.simplex_line_scale}"
            )
        if self.velocity_reset_interval < 0:
            raise ConfigurationError(
                f"velocity_reset_interval must be >= 0 (0 disables the reset), "
                f"got {self.velocity_reset_interval}"
            )
        for dropout_field in (
            "input_dropout_probability",
            "hidden_dropout_probability",
        ):
            value = getattr(self, dropout_field)
            if not (0.0 <= value < 1.0):
                raise ConfigurationError(f"{dropout_field} must be in [0.0, 1.0), got {value}")
        if not (0.0 < self.batch_normalization_alpha < 1.0):
            raise ConfigurationError(
                f"batch_normalization_alpha must be in (0.0, 1.0), "
                f"got {self.batch_normalization_alpha}"
            )
        if self.backpropagation_learning_rate <= 0.0:
            raise ConfigurationError(
                f"backpropagation_learning_rate must be > 0, "
                f"got {self.backpropagation_learning_rate}"
            )
        if not (0.0 < self.backpropagation_momentum < 1.0):
            raise ConfigurationError(
                f"backpropagation_momentum must be in (0.0, 1.0) (nesterov requires a "
                f"positive momentum), got {self.backpropagation_momentum}"
            )
        if not (0.0 < self.maximum_momentum < 1.0):
            raise ConfigurationError(
                f"maximum_momentum must be in (0.0, 1.0), got {self.maximum_momentum}"
            )
        if not (0.0 < self.momentum_decay_factor < 1.0):
            raise ConfigurationError(
                f"momentum_decay_factor must be in (0.0, 1.0), "
                f"got {self.momentum_decay_factor}"
            )
        for decay_field in ("learning_rate_decay_factor", "weight_decay_decay_factor"):
            value = getattr(self, decay_field)
            if not (0.0 < value <= 1.0):
                raise ConfigurationError(f"{decay_field} must be in (0.0, 1.0], got {value}")
        if self.minimum_learning_rate <= 0.0:
            raise ConfigurationError(
                f"minimum_learning_rate must be > 0, got {self.minimum_learning_rate}"
            )
        if self.backpropagation_weight_decay < 0.0:
            raise ConfigurationError(
                f"backpropagation_weight_decay must be >= 0, "
                f"got {self.backpropagation_weight_decay}"
            )
        if self.minimum_weight_decay < 0.0:
            raise ConfigurationError(
                f"minimum_weight_decay must be >= 0, got {self.minimum_weight_decay}"
            )
