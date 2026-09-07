"""One learning recipe, as data rather than as scattered constructor arguments.

Track B trains every selected topology the same way, so "the same way" has to
be a single object that can be frozen in the protocol lock, written into an
artifact directory and compared between runs. Track A keeps each algorithm's
own mechanism instead; this recipe is not imposed on it.

The schedule is deliberately small: constant, step or cosine. A benchmark that
freezes one recipe before the result series does not need an open-ended
scheduler zoo, and every option here has to be justified in the write-up.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

import torch


class OptimizerName(StrEnum):
    """Optimizers the shared trainer can build."""

    SGD = "sgd"
    ADAM = "adam"


class LearningRateSchedule(StrEnum):
    """Shapes the learning rate may follow across epochs."""

    CONSTANT = "constant"
    STEP = "step"
    COSINE = "cosine"


@dataclass(frozen=True)
class TrainingRecipe:
    """Everything the shared trainer needs to know about how to learn.

    Attributes:
        optimizer: Which optimizer to build.
        learning_rate: Initial learning rate.
        momentum: Momentum, used by SGD only.
        uses_nesterov_momentum: Whether SGD uses Nesterov momentum.
        weight_decay: L2 penalty passed to the optimizer.
        batch_size: Minibatch size.
        number_of_epochs: Passes over the training split.
        schedule: Learning-rate shape across epochs.
        step_schedule_gamma: Multiplier applied at each step boundary.
        step_schedule_epoch_interval: How many epochs between step boundaries.
        minimum_learning_rate: Floor for the cosine and step schedules.
        drops_singleton_minibatches: Whether a trailing minibatch of exactly
            one sample is skipped during training. Batch normalization cannot
            compute batch statistics from one sample and raises instead, so the
            default is to drop it - during training only, never in evaluation,
            where dropping a row would change the reported metric.
    """

    optimizer: OptimizerName = OptimizerName.SGD
    learning_rate: float = 0.01
    momentum: float = 0.9
    uses_nesterov_momentum: bool = True
    weight_decay: float = 5e-4
    batch_size: int = 32
    number_of_epochs: int = 30
    schedule: LearningRateSchedule = LearningRateSchedule.COSINE
    step_schedule_gamma: float = 0.1
    step_schedule_epoch_interval: int = 10
    minimum_learning_rate: float = 0.0
    drops_singleton_minibatches: bool = True

    def __post_init__(self) -> None:
        if self.learning_rate <= 0.0:
            raise ValueError(f"learning_rate must be > 0, got {self.learning_rate}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if self.number_of_epochs < 1:
            raise ValueError(f"number_of_epochs must be >= 1, got {self.number_of_epochs}")
        if self.minimum_learning_rate < 0.0:
            raise ValueError(
                f"minimum_learning_rate must be >= 0, got {self.minimum_learning_rate}"
            )
        if self.step_schedule_epoch_interval < 1:
            raise ValueError(
                "step_schedule_epoch_interval must be >= 1, got "
                f"{self.step_schedule_epoch_interval}"
            )

    def learning_rate_for_epoch(self, epoch_index: int) -> float:
        """Learning rate at the start of ``epoch_index`` (zero-based)."""
        if self.schedule is LearningRateSchedule.CONSTANT:
            return self.learning_rate
        if self.schedule is LearningRateSchedule.STEP:
            number_of_steps = epoch_index // self.step_schedule_epoch_interval
            decayed = self.learning_rate * (self.step_schedule_gamma**number_of_steps)
            return max(decayed, self.minimum_learning_rate)
        progress = epoch_index / max(self.number_of_epochs - 1, 1)
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.minimum_learning_rate + (
            self.learning_rate - self.minimum_learning_rate
        ) * cosine_factor

    def build_optimizer(self, parameters: list[torch.nn.Parameter]) -> torch.optim.Optimizer:
        """Build a fresh optimizer with no state carried from a previous session."""
        if self.optimizer is OptimizerName.ADAM:
            return torch.optim.Adam(
                parameters, lr=self.learning_rate, weight_decay=self.weight_decay
            )
        return torch.optim.SGD(
            parameters,
            lr=self.learning_rate,
            momentum=self.momentum,
            nesterov=self.uses_nesterov_momentum and self.momentum > 0.0,
            weight_decay=self.weight_decay,
        )

    def to_serializable_dict(self) -> dict:
        """Return the recipe as JSON-compatible data for the artifact directory."""
        return {
            "optimizer": str(self.optimizer),
            "learning_rate": self.learning_rate,
            "momentum": self.momentum,
            "uses_nesterov_momentum": self.uses_nesterov_momentum,
            "weight_decay": self.weight_decay,
            "batch_size": self.batch_size,
            "number_of_epochs": self.number_of_epochs,
            "schedule": str(self.schedule),
            "step_schedule_gamma": self.step_schedule_gamma,
            "step_schedule_epoch_interval": self.step_schedule_epoch_interval,
            "minimum_learning_rate": self.minimum_learning_rate,
            "drops_singleton_minibatches": self.drops_singleton_minibatches,
        }
