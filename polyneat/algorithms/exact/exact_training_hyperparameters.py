"""Per-genome training hyperparameters of EXACT (Desell, 2017, section IV).

References:
    Desell, T. (2017). Developing a Volunteer Computing Project to Evolve
        Convolutional Neural Networks and Their Hyperparameters. 2017 IEEE
        13th International Conference on e-Science, pp. 19-28.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

from polyneat.configs.exact.exact_config import EXACTConfig


@dataclass(frozen=True)
class EXACTTrainingHyperparameters:
    """The eleven backpropagation hyperparameters carried by a genome.

    Section IV co-evolves these alongside the CNN structure: every genome
    holds its own vector, and simplex hyperparameter optimization generates
    an offspring's vector from those of selected individuals. Genomes
    without a vector fall back to the trainer's config-derived defaults.

    Attributes:
        learning_rate: Initial ``η`` (eq. 8).
        learning_rate_decay_factor: Per-epoch ``Δη`` (< 1).
        momentum: Initial Nesterov ``µ`` (eq. 5).
        momentum_decay_factor: Per-epoch ``Δµ`` (eq. 7).
        weight_decay: Initial L2 ``λ`` (eq. 6).
        weight_decay_decay_factor: Per-epoch ``Δλ`` (< 1).
        velocity_reset_interval: ``ω`` — examples between momentum-buffer
            resets (section VII); 0 disables the reset.
        input_dropout_probability: Dropout on the input feature map.
        hidden_dropout_probability: Dropout on every hidden filter.
        batch_size: Samples per SGD step.
        batch_normalization_alpha: Momentum of the batch-norm running
            statistics (section VII).

    References:
        Desell, T. (2017). Developing a Volunteer Computing Project to Evolve
            Convolutional Neural Networks and Their Hyperparameters. 2017 IEEE
            13th International Conference on e-Science, pp. 19-28.
    """

    learning_rate: float
    learning_rate_decay_factor: float
    momentum: float
    momentum_decay_factor: float
    weight_decay: float
    weight_decay_decay_factor: float
    velocity_reset_interval: int
    input_dropout_probability: float
    hidden_dropout_probability: float
    batch_size: int
    batch_normalization_alpha: float

    @classmethod
    def from_config(cls, config: EXACTConfig) -> EXACTTrainingHyperparameters:
        """Snapshot the config's fixed training hyperparameters (section VII).

        Args:
            config: Validated EXACT hyperparameters.

        Returns:
            The vector every genome trains with when it carries none itself.
        """
        return cls(
            learning_rate=config.backpropagation_learning_rate,
            learning_rate_decay_factor=config.learning_rate_decay_factor,
            momentum=config.backpropagation_momentum,
            momentum_decay_factor=config.momentum_decay_factor,
            weight_decay=config.backpropagation_weight_decay,
            weight_decay_decay_factor=config.weight_decay_decay_factor,
            velocity_reset_interval=config.velocity_reset_interval,
            input_dropout_probability=config.input_dropout_probability,
            hidden_dropout_probability=config.hidden_dropout_probability,
            batch_size=config.training_batch_size,
            batch_normalization_alpha=config.batch_normalization_alpha,
        )

    def to_serializable_dict(self) -> dict:
        """Return a JSON-serializable dict of every field."""
        return dataclasses.asdict(self)

    @classmethod
    def from_serializable_dict(cls, payload: dict) -> EXACTTrainingHyperparameters:
        """Rebuild from ``to_serializable_dict`` output.

        Args:
            payload: Dict with one entry per field.

        Returns:
            The reconstructed hyperparameter vector.
        """
        return cls(**payload)
