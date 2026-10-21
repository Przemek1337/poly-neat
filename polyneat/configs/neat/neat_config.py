from __future__ import annotations

from dataclasses import dataclass

from polyneat.configs.algorithm_config import AlgorithmConfig
from polyneat.configs.configuration_errors import ConfigurationError


@dataclass
class NEATConfig(AlgorithmConfig):
    """All NEAT-specific hyperparameters.

    Defaults follow the experimental settings of Stanley & Miikkulainen
    (2002), section 4.1, wherever the paper specifies a value (mutation
    rates, compatibility coefficients c1/c2/c3 and threshold, interspecies
    mating rate, crossover fraction, stagnation limit). ``validate`` raises
    :class:`~polyneat.configs.configuration_errors.ConfigurationError` on any
    out-of-range value.
    """

    # Initial population. Name resolved against the strategy registry in
    # polyneat.core.neat.initial_population at NEATAlgorithm.from_config
    # time (not here, to keep config free of algorithm imports).
    initial_population_strategy: str = "fully_connected"

    # Mutation rates
    probability_of_add_node_mutation: float = 0.03
    probability_of_add_connection_mutation: float = 0.05
    # Section 4.1: 80% of genomes mutate their weights at all; within such a
    # genome each weight is perturbed with probability 0.9 and replaced with 0.1.
    probability_of_genome_weight_mutation: float = 0.8
    probability_of_weight_perturbation: float = 0.9
    probability_of_weight_replacement: float = 0.1
    probability_of_toggle_connection_enabled: float = 0.01

    # Weight initialization & perturbation
    initial_weight_range_min: float = -1.0
    initial_weight_range_max: float = 1.0
    # Half-width of the uniform perturbation, Stanley's ``power``. Not a Gaussian
    # sigma: section 4.1 perturbs weights uniformly.
    weight_perturbation_magnitude: float = 0.5

    # Speciation (compatibility distance)
    compatibility_distance_coefficient_excess_c1: float = 1.0
    compatibility_distance_coefficient_disjoint_c2: float = 1.0
    compatibility_distance_coefficient_weight_difference_c3: float = 0.4
    compatibility_distance_threshold: float = 3.0

    # Species management
    species_elitism_count: int = 1
    species_stagnation_generations_limit: int = 15
    # Section 4.1 copies the champion of each species with *more than* five
    # networks, so the inclusive threshold below is 6.
    minimum_species_size_for_elitism: int = 6

    # Crossover
    probability_of_crossover_vs_mutation_only: float = 0.75
    probability_of_inheriting_from_fitter_parent_for_matching_genes: float = 0.5
    probability_of_interspecies_mating: float = 0.001

    # Selection
    tournament_size_for_parent_selection: int = 3
    species_survival_fraction_for_reproduction: float = 0.2

    # Activation functions
    available_activation_functions: tuple[str, ...] = (
        "sigmoid",
        "steepened_sigmoid",
        "tanh",
        "relu",
    )
    default_activation_function_for_hidden_nodes: str = "steepened_sigmoid"
    default_activation_function_for_output_nodes: str = "steepened_sigmoid"

    def validate(self) -> None:
        """Check ranges of all NEAT fields on top of the base validation.

        Raises:
            ConfigurationError: Naming the field, the value, and the reason.
        """
        super().validate()
        for prob_field in (
            "probability_of_add_node_mutation",
            "probability_of_add_connection_mutation",
            "probability_of_genome_weight_mutation",
            "probability_of_weight_perturbation",
            "probability_of_weight_replacement",
            "probability_of_toggle_connection_enabled",
            "probability_of_crossover_vs_mutation_only",
            "probability_of_inheriting_from_fitter_parent_for_matching_genes",
            "probability_of_interspecies_mating",
        ):
            value = getattr(self, prob_field)
            if not (0.0 <= value <= 1.0):
                raise ConfigurationError(f"{prob_field} must be in [0.0, 1.0], got {value}")
        if not (0.0 < self.species_survival_fraction_for_reproduction <= 1.0):
            raise ConfigurationError(
                f"species_survival_fraction_for_reproduction must be in (0.0, 1.0], "
                f"got {self.species_survival_fraction_for_reproduction}"
            )
        if self.initial_weight_range_min >= self.initial_weight_range_max:
            raise ConfigurationError(
                f"initial_weight_range_min ({self.initial_weight_range_min}) must be "
                f"< initial_weight_range_max ({self.initial_weight_range_max})"
            )
        if (
            self.default_activation_function_for_hidden_nodes
            not in self.available_activation_functions
        ):
            raise ConfigurationError(
                f"default_activation_function_for_hidden_nodes "
                f"'{self.default_activation_function_for_hidden_nodes}' "
                f"not in available_activation_functions {self.available_activation_functions}"
            )
        if (
            self.default_activation_function_for_output_nodes
            not in self.available_activation_functions
        ):
            raise ConfigurationError(
                f"default_activation_function_for_output_nodes "
                f"'{self.default_activation_function_for_output_nodes}' "
                f"not in available_activation_functions {self.available_activation_functions}"
            )
