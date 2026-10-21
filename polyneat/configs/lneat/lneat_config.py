"""Configuration for the L-NEAT algorithm.

References:
    Chen, L., & Alahakoon, D. (2006). NeuroEvolution of Augmenting Topologies with Learning for
        Data Classification. *ICIA 2006: 2nd International Conference on Information and
        Automation*, pp. 367-371.
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
class LNEATConfig(NEATConfig):
    """Hyperparameters for L-NEAT (Chen & Alahakoon, 2006).

    L-NEAT keeps every NEAT hyperparameter unchanged and adds the learning
    schedule of section IV.B of the paper: every
    ``learning_interval_generations`` (the paper's interval ``I``), networks
    that are not Type 1 undergo ``backpropagation_iterations_per_session``
    (the paper's amount ``L``) iterations of backpropagation on a fixed set of
    ``number_of_learning_samples`` (the paper's ``A``) training samples, and
    the trained weights are written back into the genomes.

    Attributes:
        number_of_class_labels: Number of classes; the divide-and-conquer
            split (section IV.A) runs one single-output recognizer evolution
            per class label.
        learning_interval_generations: Generations between backpropagation
            sessions (paper's ``I``, section IV.B.3).
        number_of_learning_samples: Size of the fixed learning sample subset
            presented during each backpropagation session (paper's ``A``).
        backpropagation_iterations_per_session: Gradient steps per session
            (paper's ``L``, the quality of the learning).
        backpropagation_learning_rate: SGD learning rate of the
            backpropagation sessions. Not specified by the paper.
        training_indicator: Maximum mean output distance for a correctly
            classifying network to count as Type 1 and skip backpropagation
            (section IV.B.2, selective learning).
        classification_threshold: Output level at or above which a single
            recognizer output counts as class membership.
    """

    number_of_class_labels: int = 3
    learning_interval_generations: int = 5
    number_of_learning_samples: int = 10
    backpropagation_iterations_per_session: int = 10
    backpropagation_learning_rate: float = 0.1
    training_indicator: float = 0.2
    classification_threshold: float = 0.5

    def validate(self) -> None:
        """Validate L-NEAT fields on top of the NEAT and shared validation.

        Raises:
            ConfigurationError: If there are fewer than two class labels, if
                genomes have more than one output node (each subtask network
                is a single-output recognizer), if any learning-schedule
                parameter is non-positive, if ``training_indicator`` leaves
                ``[0, 1]`` or ``classification_threshold`` leaves ``(0, 1)``,
                or if the output activation is not bounded in ``[0, 1]``
                (recognizer outputs are compared against binary targets and
                ``classification_threshold``, which assumes unit-interval
                outputs).
        """
        super().validate()
        if self.number_of_class_labels < 2:
            raise ConfigurationError(
                f"number_of_class_labels must be >= 2, got {self.number_of_class_labels}"
            )
        if self.number_of_output_nodes != 1:
            raise ConfigurationError(
                "L-NEAT evolves one single-output recognizer network per class label; "
                f"number_of_output_nodes must be 1, got {self.number_of_output_nodes}"
            )
        if self.learning_interval_generations < 1:
            raise ConfigurationError(
                f"learning_interval_generations must be >= 1, "
                f"got {self.learning_interval_generations}"
            )
        if self.number_of_learning_samples < 1:
            raise ConfigurationError(
                f"number_of_learning_samples must be >= 1, "
                f"got {self.number_of_learning_samples}"
            )
        if self.backpropagation_iterations_per_session < 1:
            raise ConfigurationError(
                f"backpropagation_iterations_per_session must be >= 1, "
                f"got {self.backpropagation_iterations_per_session}"
            )
        if self.backpropagation_learning_rate <= 0.0:
            raise ConfigurationError(
                f"backpropagation_learning_rate must be > 0, "
                f"got {self.backpropagation_learning_rate}"
            )
        if not (0.0 <= self.training_indicator <= 1.0):
            raise ConfigurationError(
                f"training_indicator must be in [0.0, 1.0], got {self.training_indicator}"
            )
        if not (0.0 < self.classification_threshold < 1.0):
            raise ConfigurationError(
                f"classification_threshold must be in (0.0, 1.0), "
                f"got {self.classification_threshold}"
            )
        if (
            self.default_activation_function_for_output_nodes
            not in _OUTPUT_ACTIVATIONS_BOUNDED_IN_UNIT_INTERVAL
        ):
            raise ConfigurationError(
                "L-NEAT recognizer outputs are compared against binary targets and "
                "classification_threshold, which requires output activations bounded "
                f"in [0, 1]; default_activation_function_for_output_nodes must be "
                f"one of {_OUTPUT_ACTIVATIONS_BOUNDED_IN_UNIT_INTERVAL}, got "
                f"'{self.default_activation_function_for_output_nodes}'"
            )
