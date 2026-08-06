"""Tests for simplex hyperparameter optimization (Desell, 2017, section IV)."""

from __future__ import annotations

import numpy as np
import pytest

from polyneat.algorithms.exact.exact_training_hyperparameters import (
    EXACTTrainingHyperparameters,
)
from polyneat.algorithms.exact.simplex_hyperparameter_optimizer import (
    HYPERPARAMETER_RANGES,
    SimplexHyperparameterOptimizer,
)


def _hyperparameters_with(learning_rate: float) -> EXACTTrainingHyperparameters:
    return EXACTTrainingHyperparameters(
        learning_rate=learning_rate,
        learning_rate_decay_factor=0.95,
        momentum=0.5,
        momentum_decay_factor=0.95,
        weight_decay=0.0005,
        weight_decay_decay_factor=0.95,
        velocity_reset_interval=1000,
        input_dropout_probability=0.001,
        hidden_dropout_probability=0.1,
        batch_size=50,
        batch_normalization_alpha=0.1,
    )


def test_draw_initial_hyperparameters_stays_inside_ranges() -> None:
    optimizer = SimplexHyperparameterOptimizer()
    rng = np.random.default_rng(0)
    for _draw in range(20):
        hyperparameters = optimizer.draw_initial_hyperparameters(rng)
        for field_name, (lower_bound, upper_bound) in HYPERPARAMETER_RANGES.items():
            value = getattr(hyperparameters, field_name)
            assert lower_bound <= value <= upper_bound, field_name
    assert isinstance(hyperparameters.batch_size, int)
    assert isinstance(hyperparameters.velocity_reset_interval, int)


def test_generate_offspring_follows_equations_1_and_2() -> None:
    """h_new = h_avg + r*(h_best - h_avg) with r = rand*l1 - l2, h_avg over
    the non-best candidates. Verified by replaying the same seeded draw."""
    candidates = [
        (_hyperparameters_with(0.002), 0.90),  # best
        (_hyperparameters_with(0.010), 0.50),
        (_hyperparameters_with(0.020), 0.40),
    ]
    optimizer = SimplexHyperparameterOptimizer(number_of_selected_genomes=3)
    offspring = optimizer.generate_offspring_hyperparameters(
        candidates, rng=np.random.default_rng(42)
    )
    replay_rng = np.random.default_rng(42)
    expected_r = (
        float(replay_rng.random()) * optimizer.line_scale
    ) - optimizer.line_offset
    average_learning_rate = (0.010 + 0.020) / 2.0
    expected_learning_rate = average_learning_rate + expected_r * (
        0.002 - average_learning_rate
    )
    lower_bound, upper_bound = HYPERPARAMETER_RANGES["learning_rate"]
    expected_learning_rate = min(max(expected_learning_rate, lower_bound), upper_bound)
    assert offspring.learning_rate == pytest.approx(expected_learning_rate)


def test_generated_offspring_is_clamped_to_ranges() -> None:
    optimizer = SimplexHyperparameterOptimizer(number_of_selected_genomes=3)
    candidates = [
        (_hyperparameters_with(0.002), 0.90),
        (_hyperparameters_with(0.010), 0.50),
        (_hyperparameters_with(0.020), 0.40),
    ]
    for seed in range(30):
        offspring = optimizer.generate_offspring_hyperparameters(
            candidates, rng=np.random.default_rng(seed)
        )
        for field_name, (lower_bound, upper_bound) in HYPERPARAMETER_RANGES.items():
            value = getattr(offspring, field_name)
            assert lower_bound <= value <= upper_bound, field_name
