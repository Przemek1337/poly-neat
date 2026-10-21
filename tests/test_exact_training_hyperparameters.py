"""Tests for the per-genome training hyperparameter vector (section IV)."""

from __future__ import annotations

import pytest

from polyneat.algorithms.exact.exact_training_hyperparameters import (
    EXACTTrainingHyperparameters,
)
from polyneat.configs.exact.exact_config import EXACTConfig


def _example_hyperparameters() -> EXACTTrainingHyperparameters:
    return EXACTTrainingHyperparameters(
        learning_rate=0.0025,
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


def test_serialization_round_trip() -> None:
    original = _example_hyperparameters()
    restored = EXACTTrainingHyperparameters.from_serializable_dict(
        original.to_serializable_dict()
    )
    assert restored == original


def test_from_config_uses_config_training_fields() -> None:
    config = EXACTConfig()
    hyperparameters = EXACTTrainingHyperparameters.from_config(config)
    assert hyperparameters.learning_rate == pytest.approx(
        config.backpropagation_learning_rate
    )
    assert hyperparameters.momentum_decay_factor == pytest.approx(
        config.momentum_decay_factor
    )
    assert hyperparameters.batch_size == config.training_batch_size
    assert hyperparameters.velocity_reset_interval == config.velocity_reset_interval
