"""Simplex hyperparameter optimization for EXACT (Desell, 2017, section IV).

References:
    Desell, T. (2017). Developing a Volunteer Computing Project to Evolve
        Convolutional Neural Networks and Their Hyperparameters. 2017 IEEE
        13th International Conference on e-Science, pp. 19-28.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

from numpy.random import Generator

from polyneat.algorithms.exact.exact_training_hyperparameters import (
    EXACTTrainingHyperparameters,
)
from polyneat.configs.exact.exact_config import EXACTConfig

# Per-hyperparameter bounds (section VIII-B1's initial ranges). Values are
# clamped here for the whole search; the paper's later widening of these
# constraints is intentionally not implemented.
HYPERPARAMETER_RANGES: dict[str, tuple[float, float]] = {
    "momentum": (0.4, 0.6),
    "momentum_decay_factor": (0.90, 0.99),
    "learning_rate": (0.001, 0.03),
    "learning_rate_decay_factor": (0.90, 0.99),
    "weight_decay": (0.0001, 0.001),
    "weight_decay_decay_factor": (0.90, 0.99),
    "velocity_reset_interval": (500.0, 3000.0),
    "input_dropout_probability": (0.0005, 0.002),
    "hidden_dropout_probability": (0.05, 0.15),
    "batch_size": (25.0, 150.0),
    "batch_normalization_alpha": (0.001, 0.2),
}

_INTEGER_HYPERPARAMETER_NAMES = frozenset({"velocity_reset_interval", "batch_size"})


@dataclass(frozen=True)
class SimplexHyperparameterOptimizer:
    """Co-evolves training hyperparameters via the simplex step of section IV.

    ``r = (rand(0, 1)·l1) - l2`` (eq. 1) and
    ``h_new,i = h_avg,i + r·(h_best,i - h_avg,i)`` (eq. 2), where ``h_best``
    belongs to the best of the selected genomes and ``h_avg`` averages the
    hyperparameters of the other selected genomes.

    Attributes:
        number_of_selected_genomes: How many distinct individuals to select
            (paper: N = 5).
        line_scale: ``l1`` in eq. 1 (paper: 2.0).
        line_offset: ``l2`` in eq. 1 (paper: 0.5).

    References:
        Desell, T. (2017). Developing a Volunteer Computing Project to Evolve
            Convolutional Neural Networks and Their Hyperparameters. 2017 IEEE
            13th International Conference on e-Science, pp. 19-28.
    """

    number_of_selected_genomes: int = 5
    line_scale: float = 2.0
    line_offset: float = 0.5

    @classmethod
    def from_config(cls, config: EXACTConfig) -> SimplexHyperparameterOptimizer:
        """Build from the config's SHO fields.

        Args:
            config: Validated EXACT hyperparameters.

        Returns:
            The configured optimizer.
        """
        return cls(
            number_of_selected_genomes=config.simplex_number_of_selected_genomes,
            line_scale=config.simplex_line_scale,
            line_offset=config.simplex_line_offset,
        )

    def draw_initial_hyperparameters(
        self, rng: Generator
    ) -> EXACTTrainingHyperparameters:
        """Uniform draw inside every hyperparameter's range (section VIII-B1).

        Args:
            rng: Source of randomness for the per-field draws.

        Returns:
            A hyperparameter vector for one generation-0 genome.
        """
        drawn_values = {
            field_name: float(rng.uniform(lower_bound, upper_bound))
            for field_name, (lower_bound, upper_bound) in HYPERPARAMETER_RANGES.items()
        }
        return self._build_hyperparameters(drawn_values)

    def generate_offspring_hyperparameters(
        self,
        candidate_hyperparameters_with_fitness: list[
            tuple[EXACTTrainingHyperparameters, float]
        ],
        rng: Generator,
    ) -> EXACTTrainingHyperparameters:
        """Eqs. 1-2 applied to the selected candidates, clamped to the ranges.

        Args:
            candidate_hyperparameters_with_fitness: The selected genomes'
                hyperparameter vectors with their fitness (higher is better).
                Must contain at least two entries.
            rng: Source of randomness for the single ``r`` draw.

        Returns:
            The offspring hyperparameter vector.
        """
        best_hyperparameters, _best_fitness = max(
            candidate_hyperparameters_with_fitness, key=lambda pair: pair[1]
        )
        other_hyperparameters = [
            hyperparameters
            for hyperparameters, _fitness in candidate_hyperparameters_with_fitness
            if hyperparameters is not best_hyperparameters
        ]
        random_line_position = (float(rng.random()) * self.line_scale) - self.line_offset

        best_values = dataclasses.asdict(best_hyperparameters)
        new_values: dict[str, float] = {}
        for field_name, (lower_bound, upper_bound) in HYPERPARAMETER_RANGES.items():
            average_value = sum(
                float(getattr(hyperparameters, field_name))
                for hyperparameters in other_hyperparameters
            ) / len(other_hyperparameters)
            new_value = average_value + random_line_position * (
                float(best_values[field_name]) - average_value
            )
            new_values[field_name] = min(max(new_value, lower_bound), upper_bound)
        return self._build_hyperparameters(new_values)

    @staticmethod
    def _build_hyperparameters(
        values: dict[str, float],
    ) -> EXACTTrainingHyperparameters:
        """Round the integer-valued hyperparameters and build the vector.

        Args:
            values: One float per :data:`HYPERPARAMETER_RANGES` key.

        Returns:
            The assembled hyperparameter vector.
        """
        prepared_values: dict[str, float | int] = {
            field_name: (
                round(value) if field_name in _INTEGER_HYPERPARAMETER_NAMES else value
            )
            for field_name, value in values.items()
        }
        return EXACTTrainingHyperparameters(**prepared_values)
