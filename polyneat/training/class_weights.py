"""Balanced class weights for an imbalanced training split.

The weight of class ``c`` is ``N / (K * N_c)``: the number of training examples
divided by the number of classes times the count of that class. For two classes
this is the ``N / (2 * N_c)`` the pneumonia protocol specifies, and it leaves a
perfectly balanced split with all weights equal to one.

Weights are always computed on the split actually being trained on, which
differs between tracks: track A trains on ``train``, track B on
``train + search_validation``. Computing them once on the whole pool would
weight a loss by counts the model never sees.

A split missing one of the classes is an error rather than a zero weight. A
single-class training or validation split silently invalidates the weighted
loss, the threshold search and every ranking metric downstream, so it fails
here instead of producing a number that looks fine.
"""

from __future__ import annotations

import torch


def compute_balanced_class_weights(
    labels: torch.Tensor, number_of_classes: int
) -> torch.Tensor:
    """Return one weight per class, in class-index order.

    Args:
        labels: Long tensor of class indices for the split being trained on.
        number_of_classes: Total number of classes, so the returned tensor
            lines up with the model's output layer.

    Returns:
        ``(number_of_classes,)`` ``float32`` tensor of weights.

    Raises:
        ValueError: If the tensor is empty, if a label is out of range, or if
            some class does not occur in the split.
    """
    if number_of_classes < 2:
        raise ValueError(f"number_of_classes must be >= 2, got {number_of_classes}")
    if labels.numel() == 0:
        raise ValueError("cannot compute class weights from an empty split")

    label_indices = labels.reshape(-1).to(torch.long)
    if int(label_indices.min()) < 0 or int(label_indices.max()) >= number_of_classes:
        raise ValueError(
            f"labels must be in [0, {number_of_classes - 1}], got range "
            f"[{int(label_indices.min())}, {int(label_indices.max())}]"
        )

    counts = torch.bincount(label_indices, minlength=number_of_classes).to(torch.float32)
    missing_classes = torch.nonzero(counts == 0).reshape(-1).tolist()
    if missing_classes:
        raise ValueError(
            f"classes {missing_classes} do not occur in this split; a single-class split cannot "
            "carry a weighted loss, a threshold search or a ranking metric"
        )
    total_count = float(label_indices.numel())
    return total_count / (float(number_of_classes) * counts)
