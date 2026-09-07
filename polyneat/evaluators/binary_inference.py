"""Batched inference that produces a traceable prediction set.

Kept apart from training on purpose. Inference must not fit preprocessing
statistics, must not touch class weights and must not update anything: a frozen
checkpoint has to reproduce its own predictions months later from its saved
state alone. Separating the two means that guarantee is enforced by which
function the caller reaches for, not by remembering to pass the right flag.

Batching is not an optimization here but a requirement: the protocol forbids
assuming a whole split or a whole population fits on the GPU at once.
"""

from __future__ import annotations

import torch

from polyneat.evaluators.binary_predictions import BinaryPredictions, build_binary_predictions
from polyneat.training.image_preprocessing import ImagePreprocessor
from polyneat.training.trainable_model import TrainableModel


def predict_binary_logits(
    model: TrainableModel,
    *,
    images: torch.Tensor,
    preprocessor: ImagePreprocessor,
    batch_size: int,
    device_for_computation: torch.device,
) -> torch.Tensor:
    """Run one model over ``images`` in evaluation mode and return raw logits.

    Args:
        model: Frozen model. Switched to evaluation mode and left there.
        images: ``NCHW`` batch, unpreprocessed.
        preprocessor: The model's own fitted preprocessing. It is applied with
            ``training=False``, so no augmentation runs and nothing is fitted.
        batch_size: Rows per forward pass.
        device_for_computation: Device to run on.

    Returns:
        ``(n, 2)`` ``float64`` tensor of logits on the CPU.

    Raises:
        ValueError: If ``batch_size`` is not positive, or the batch is empty.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    number_of_rows = int(images.shape[0])
    if number_of_rows == 0:
        raise ValueError("cannot run inference on an empty batch")

    model.eval()
    collected_logits: list[torch.Tensor] = []
    with torch.no_grad():
        for batch_start in range(0, number_of_rows, batch_size):
            batch_images = images[batch_start : batch_start + batch_size].to(
                device_for_computation
            )
            preprocessed = preprocessor.apply(batch_images, training=False)
            collected_logits.append(model.forward_pass(preprocessed).detach().cpu())
    return torch.cat(collected_logits).to(torch.float64)


def predict_binary(
    model: TrainableModel,
    *,
    images: torch.Tensor,
    labels: torch.Tensor,
    example_ids: tuple[str, ...],
    group_ids: tuple[str, ...],
    preprocessor: ImagePreprocessor,
    batch_size: int,
    device_for_computation: torch.device,
    model_id: str,
    stage: str,
    split_name: str,
) -> BinaryPredictions:
    """Produce a full :class:`BinaryPredictions` record for one model and split.

    The returned record carries the raw logits, the derived positive-class
    probability and the ids of every row, which is what the metrics, the
    threshold search, the grouped bootstrap and the paired comparison all need.

    Raises:
        NonFinitePredictionError: If the model emitted NaN or Inf. The protocol
            records that as a failed evaluation rather than a poor score.
    """
    logits = predict_binary_logits(
        model,
        images=images,
        preprocessor=preprocessor,
        batch_size=batch_size,
        device_for_computation=device_for_computation,
    )
    return build_binary_predictions(
        example_ids=example_ids,
        group_ids=group_ids,
        labels=labels,
        logits=logits,
        model_id=model_id,
        stage=stage,
        split_name=split_name,
    )
