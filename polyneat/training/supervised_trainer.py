"""One supervised training session, against a model the trainer knows nothing about.

This is the shared loop: weighted cross-entropy, minibatches drawn from an
explicit stream, preprocessing applied exactly once per batch, and a stop check
at every batch boundary so a wall-clock budget can end a session without
corrupting it.

What it deliberately does *not* do is branch on which algorithm produced the
model. EXACT resets its optimizer velocities on a schedule of its own and
inherits kernels between generations; DeepNEAT reinitializes on every
evaluation and evolves its own learning rate. Those are properties of those
algorithms and live in their adapters. Putting them here as
``if algorithm_name == ...`` would make one algorithm's semantics the default
for every other.

Counters are returned rather than logged and forgotten: optimizer steps,
examples processed and completed epochs are what a budget report is built from,
and a session cut short by the deadline says so instead of looking finished.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import nn

from polyneat.logging_utils.custom_logger import get_logger
from polyneat.training.image_preprocessing import ImagePreprocessor
from polyneat.training.trainable_model import TrainableModel
from polyneat.training.training_recipe import TrainingRecipe

logger = get_logger(__name__)

DEADLINE_REACHED = "deadline_reached"
COMPLETED = "completed"


@dataclass(frozen=True)
class TrainingSessionResult:
    """What one training session actually did.

    Attributes:
        status: ``completed`` or ``deadline_reached``.
        completed_epochs: Epochs that ran to their end.
        optimizer_steps: Parameter updates applied.
        examples_processed: Training examples fed forward, counting repeats
            across epochs.
        wall_clock_seconds: Duration of the session.
        final_batch_loss: Loss of the last minibatch, or ``None`` if no batch
            ran.
        skipped_singleton_minibatches: Minibatches of exactly one sample that
            were dropped, so the count is visible rather than implied.
    """

    status: str
    completed_epochs: int
    optimizer_steps: int
    examples_processed: int
    wall_clock_seconds: float
    final_batch_loss: float | None
    skipped_singleton_minibatches: int

    @property
    def was_interrupted(self) -> bool:
        """Whether the session ended before its recipe was exhausted."""
        return self.status != COMPLETED

    def to_serializable_dict(self) -> dict:
        """Return the counters as JSON-compatible data."""
        return {
            "status": self.status,
            "completed_epochs": self.completed_epochs,
            "optimizer_steps": self.optimizer_steps,
            "examples_processed": self.examples_processed,
            "wall_clock_seconds": self.wall_clock_seconds,
            "final_batch_loss": self.final_batch_loss,
            "skipped_singleton_minibatches": self.skipped_singleton_minibatches,
        }


class TrainingDataError(ValueError):
    """Raised when a training split cannot support the requested session."""


class SupervisedTrainer:
    """Runs one training session against any :class:`TrainableModel`.

    The trainer owns the state of the session it is running - the optimizer,
    the epoch counter, the work counters - and nothing beyond it. It does not
    own the model, the data, the preprocessing statistics or the random
    streams; those are passed in by whoever composed the stage, which is what
    lets track A and track B use the same loop under different rules.
    """

    def __init__(
        self,
        *,
        recipe: TrainingRecipe,
        preprocessor: ImagePreprocessor,
        device_for_computation: torch.device,
        class_weights: torch.Tensor | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> None:
        """Assemble a trainer from the recipe and its collaborators.

        Args:
            recipe: How to learn. Frozen before the result series.
            preprocessor: Already-fitted image path. The trainer never fits it:
                fitting inside the loop would let augmented batches leak into
                the statistics.
            device_for_computation: Device batches and the model live on.
            class_weights: Per-class loss weights, typically from
                :func:`~polyneat.training.class_weights
                .compute_balanced_class_weights` on the split being trained on.
                ``None`` gives an unweighted cross-entropy.
            should_stop: Checked at every minibatch boundary. Returning ``True``
                ends the session with status ``deadline_reached`` instead of
                starting more work.
        """
        self._recipe = recipe
        self._preprocessor = preprocessor
        self._device_for_computation = device_for_computation
        self._class_weights = (
            None if class_weights is None else class_weights.to(device_for_computation)
        )
        self._should_stop = should_stop

    @property
    def recipe(self) -> TrainingRecipe:
        """The recipe this trainer applies."""
        return self._recipe

    def train(
        self,
        model: TrainableModel,
        *,
        images: torch.Tensor,
        labels: torch.Tensor,
        batch_order_generator: torch.Generator,
        augmentation_generator: torch.Generator,
    ) -> TrainingSessionResult:
        """Run the session and return what it did.

        Args:
            model: Model to update in place. Its parameters must already be in
                whatever state the caller wants training to start from; the
                trainer never reinitializes on its own, because track A and
                track B disagree about exactly that.
            images: Training images, ``NCHW``, unpreprocessed.
            labels: Long tensor of class indices.
            batch_order_generator: Stream owning the minibatch permutation.
            augmentation_generator: Stream owning the augmentation draws.

        Returns:
            The :class:`TrainingSessionResult` counters.

        Raises:
            TrainingDataError: If images and labels disagree in length, if the
                split is empty, or if every minibatch would be dropped as a
                singleton, which would mean reporting an untrained model as
                trained.
        """
        number_of_samples = int(images.shape[0])
        if number_of_samples != int(labels.shape[0]):
            raise TrainingDataError(
                f"images have {number_of_samples} rows but labels have {int(labels.shape[0])}"
            )
        if number_of_samples == 0:
            raise TrainingDataError("cannot train on an empty split")
        self._raise_if_no_usable_minibatch(number_of_samples)

        images = images.to(self._device_for_computation)
        labels = labels.to(torch.long).to(self._device_for_computation)
        loss_function = nn.CrossEntropyLoss(weight=self._class_weights)
        optimizer = self._recipe.build_optimizer(list(model.parameters()))
        model.train()

        session_started_at = time.perf_counter()
        completed_epochs = 0
        optimizer_steps = 0
        examples_processed = 0
        skipped_singleton_minibatches = 0
        final_batch_loss: float | None = None
        status = COMPLETED

        for epoch_index in range(self._recipe.number_of_epochs):
            epoch_learning_rate = self._recipe.learning_rate_for_epoch(epoch_index)
            for parameter_group in optimizer.param_groups:
                parameter_group["lr"] = epoch_learning_rate

            shuffled_positions = torch.randperm(
                number_of_samples, generator=batch_order_generator
            ).to(self._device_for_computation)

            for batch_start in range(0, number_of_samples, self._recipe.batch_size):
                if self._should_stop is not None and self._should_stop():
                    status = DEADLINE_REACHED
                    break
                batch_positions = shuffled_positions[
                    batch_start : batch_start + self._recipe.batch_size
                ]
                if (
                    self._recipe.drops_singleton_minibatches
                    and int(batch_positions.shape[0]) == 1
                ):
                    skipped_singleton_minibatches += 1
                    continue

                optimizer.zero_grad(set_to_none=True)
                batch_images = self._preprocessor.apply(
                    images[batch_positions], training=True, generator=augmentation_generator
                )
                batch_logits = model.forward_pass(batch_images)
                batch_loss = loss_function(batch_logits, labels[batch_positions])
                batch_loss.backward()
                optimizer.step()

                optimizer_steps += 1
                examples_processed += int(batch_positions.shape[0])
                final_batch_loss = float(batch_loss.detach())

            if status != COMPLETED:
                break
            completed_epochs += 1

        wall_clock_seconds = time.perf_counter() - session_started_at
        logger.info(
            "Training session %s after %d/%d epochs, %d optimizer steps, %.1fs",
            status,
            completed_epochs,
            self._recipe.number_of_epochs,
            optimizer_steps,
            wall_clock_seconds,
        )
        return TrainingSessionResult(
            status=status,
            completed_epochs=completed_epochs,
            optimizer_steps=optimizer_steps,
            examples_processed=examples_processed,
            wall_clock_seconds=wall_clock_seconds,
            final_batch_loss=final_batch_loss,
            skipped_singleton_minibatches=skipped_singleton_minibatches,
        )

    def _raise_if_no_usable_minibatch(self, number_of_samples: int) -> None:
        """Refuse a split where every minibatch would be dropped as a singleton."""
        if not self._recipe.drops_singleton_minibatches:
            return
        usable_minibatches = sum(
            1
            for batch_start in range(0, number_of_samples, self._recipe.batch_size)
            if min(self._recipe.batch_size, number_of_samples - batch_start) != 1
        )
        if usable_minibatches < 1:
            raise TrainingDataError(
                f"a split of {number_of_samples} samples at batch_size "
                f"{self._recipe.batch_size} leaves no minibatch larger than one sample; "
                "training would do nothing and report an untrained model as trained"
            )
