"""Configuration for the C-NEAT algorithm.

References:
    Alfaham, A., Van Raemdonck, S., & Mercelis, S. (2024). Genetic NEAT-Based Method for
        Multi-Class Classification. *ACAI 2024: 7th International Conference on Algorithms,
        Computing and Artificial Intelligence*.
        DOI: 10.1109/ACAI63924.2024.10899662
"""
from __future__ import annotations

from dataclasses import dataclass

from polyneat.configs.configuration_errors import ConfigurationError
from polyneat.configs.neat.neat_config import NEATConfig

_OUTPUT_ACTIVATIONS_BOUNDED_IN_UNIT_INTERVAL: tuple[str, ...] = (
    "sigmoid",
    "steepened_sigmoid",
)


@dataclass
class CNEATConfig(NEATConfig):
    """Hyperparameters for C-NEAT (Alfaham et al., 2024).

    C-NEAT keeps every NEAT hyperparameter unchanged and adds a container of
    size ``number_of_class_labels``: each organism is assigned the class label
    ``organism_index mod number_of_class_labels`` and evaluated only on
    recognizing that class, and the best genome per class is preserved across
    generations in a
    :class:`~polyneat.algorithms.cneat.class_genome_container.ClassGenomeContainer`.

    Attributes:
        number_of_class_labels: Number of classes in the classification task,
            which is also the container size.
    """

    number_of_class_labels: int = 3

    def validate(self) -> None:
        """Validate C-NEAT fields on top of the NEAT and shared validation.

        Raises:
            ConfigurationError: If there are fewer than two class labels, if
                genomes have more than one output node (each container cell is
                a single-output recognizer network), if the population is
                smaller than the container so some classes would never be
                assigned an organism, or if the output activation is not
                bounded in ``[0, 1]`` (required so the ``1 - MSE`` fitness
                stays non-negative for NEAT's offspring allocation).
        """
        super().validate()
        if self.number_of_class_labels < 2:
            raise ConfigurationError(
                f"number_of_class_labels must be >= 2, got {self.number_of_class_labels}"
            )
        if self.number_of_output_nodes != 1:
            raise ConfigurationError(
                "C-NEAT evolves one single-output recognizer network per class label; "
                f"number_of_output_nodes must be 1, got {self.number_of_output_nodes}"
            )
        if self.population_size < self.number_of_class_labels:
            raise ConfigurationError(
                f"population_size ({self.population_size}) must be >= "
                f"number_of_class_labels ({self.number_of_class_labels}) so every "
                f"class label is assigned at least one organism"
            )
        if (
            self.default_activation_function_for_output_nodes
            not in _OUTPUT_ACTIVATIONS_BOUNDED_IN_UNIT_INTERVAL
        ):
            raise ConfigurationError(
                "C-NEAT's 1 - MSE fitness requires output activations bounded in "
                f"[0, 1]; default_activation_function_for_output_nodes must be one of "
                f"{_OUTPUT_ACTIVATIONS_BOUNDED_IN_UNIT_INTERVAL}, got "
                f"'{self.default_activation_function_for_output_nodes}'"
            )
