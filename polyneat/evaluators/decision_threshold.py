"""Choosing one decision threshold, and binding it to what it was chosen for.

The rule is fixed before any test data exists: maximise Youden's J,
``sensitivity + specificity - 1``, on the threshold-selection split of one
already-frozen checkpoint. A row counts as positive when its score is greater
than or equal to the threshold.

The candidate set is every distinct observed score plus one value just above
the largest, which is the threshold that classifies everything negative.
Including it matters: without it a set where the best operating point is "call
nothing positive" cannot be expressed. Both ends therefore have a finite,
serializable value even when every score is identical, which is why the
above-maximum candidate is derived with ``nextafter`` rather than written as
infinity.

A threshold is bound to the exact artifacts it was derived from - the
checkpoint, its preprocessing state and the manifest of the split it was chosen
on. Carrying a threshold from track A to track B, or between seeds, changes all
three, and the binding digest makes that visible instead of plausible.

References:
    Youden, W. J. (1950). Index for rating diagnostic tests. *Cancer*, 3(1),
        32-35.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass

import torch

from polyneat.evaluators.binary_classification_metrics import (
    MetricInputError,
    compute_confusion_counts,
)
from polyneat.logging_utils.custom_logger import get_logger

logger = get_logger(__name__)

THRESHOLD_BINDING_SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class SelectedThreshold:
    """One chosen threshold and the evidence behind the choice.

    Attributes:
        threshold: The chosen value; positive means ``score >= threshold``.
        youden_j: Its ``sensitivity + specificity - 1`` on the selection split.
        sensitivity: Sensitivity at this threshold on that split.
        specificity: Specificity at this threshold on that split.
        number_of_candidates: How many thresholds were considered.
        model_id: Checkpoint the threshold belongs to.
        split_name: Split it was selected on, always the threshold split.
        binding_sha256: Digest of the checkpoint, preprocessing state and
            manifest this threshold is valid for.
    """

    threshold: float
    youden_j: float
    sensitivity: float
    specificity: float
    number_of_candidates: int
    model_id: str
    split_name: str
    binding_sha256: str

    def to_serializable_dict(self) -> dict:
        """Return the record as JSON-compatible data."""
        return {
            "threshold": self.threshold,
            "youden_j": self.youden_j,
            "sensitivity": self.sensitivity,
            "specificity": self.specificity,
            "number_of_candidates": self.number_of_candidates,
            "model_id": self.model_id,
            "split_name": self.split_name,
            "binding_sha256": self.binding_sha256,
        }


def compute_threshold_binding(
    *,
    checkpoint_sha256: str,
    preprocessing_state: dict,
    manifest_sha256: str,
) -> str:
    """Digest the three things a threshold is only valid for.

    Args:
        checkpoint_sha256: Digest of the frozen model weights.
        preprocessing_state: State dict of that model's preprocessing.
        manifest_sha256: Digest of the manifest whose threshold split was used.

    Returns:
        Lowercase hexadecimal SHA-256 over a canonical encoding of all three.
    """
    canonical_payload = json.dumps(
        {
            "schema_version": THRESHOLD_BINDING_SCHEMA_VERSION,
            "checkpoint_sha256": checkpoint_sha256,
            "preprocessing_state": preprocessing_state,
            "manifest_sha256": manifest_sha256,
        },
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(canonical_payload.encode("utf-8")).hexdigest()


def build_threshold_candidates(scores: torch.Tensor) -> torch.Tensor:
    """Every threshold worth evaluating, including the classify-nothing end.

    Args:
        scores: ``(n,)`` tensor of positive-class scores.

    Returns:
        Sorted ``float64`` tensor of candidates: each distinct score, which
        classifies that score and everything above it positive, plus one value
        immediately above the maximum, which classifies everything negative.

    Raises:
        MetricInputError: If the set is empty or holds a non-finite score.
    """
    if scores.numel() == 0:
        raise MetricInputError("cannot select a threshold from an empty set")
    if not torch.isfinite(scores).all():
        raise MetricInputError(
            "cannot select a threshold from non-finite scores; this is a failed evaluation"
        )
    distinct_scores = torch.unique(scores.to(torch.float64))
    above_maximum = torch.tensor(
        [math.nextafter(float(distinct_scores[-1]), math.inf)], dtype=torch.float64
    )
    return torch.cat([distinct_scores, above_maximum])


def select_threshold_by_youden_j(
    labels: torch.Tensor,
    scores: torch.Tensor,
    *,
    model_id: str,
    split_name: str,
    binding_sha256: str,
) -> SelectedThreshold:
    """Pick the threshold maximising Youden's J on the selection split.

    Ties are broken deterministically and in a stated order: first prefer the
    higher sensitivity, then the smaller threshold. On a screening task the
    first rule is the one with a clinical direction, and the second only exists
    so that two identical operating points cannot depend on iteration order.

    Args:
        labels: ``(n,)`` ground truth of the threshold-selection split.
        scores: ``(n,)`` positive-class scores of the frozen checkpoint on that
            split.
        model_id: Checkpoint being calibrated.
        split_name: Split the scores come from, recorded so a threshold taken
            from the wrong split is visible in the artifact.
        binding_sha256: Result of :func:`compute_threshold_binding`.

    Returns:
        The selected threshold and its operating point.

    Raises:
        MetricInputError: If the set is empty, misaligned, non-finite, or does
            not contain both classes. Youden's J is undefined without both.
    """
    if labels.shape[0] != scores.shape[0]:
        raise MetricInputError(
            f"labels and scores disagree in length: {labels.shape[0]} vs {scores.shape[0]}"
        )
    number_of_positives = int((labels == 1).sum())
    number_of_negatives = int((labels == 0).sum())
    if number_of_positives == 0 or number_of_negatives == 0:
        raise MetricInputError(
            "threshold selection needs both classes on the selection split, got "
            f"{number_of_positives} positive and {number_of_negatives} negative examples"
        )

    candidates = build_threshold_candidates(scores)
    scores_float64 = scores.to(torch.float64)

    best_threshold: float | None = None
    best_youden_j = -math.inf
    best_sensitivity = 0.0
    best_specificity = 0.0
    for candidate in candidates.tolist():
        confusion = compute_confusion_counts(labels, scores_float64, candidate)
        sensitivity = confusion.true_positives / number_of_positives
        specificity = confusion.true_negatives / number_of_negatives
        youden_j = sensitivity + specificity - 1.0
        is_better = (youden_j, sensitivity, -candidate) > (
            best_youden_j,
            best_sensitivity,
            -best_threshold if best_threshold is not None else -math.inf,
        )
        if is_better:
            best_threshold = candidate
            best_youden_j = youden_j
            best_sensitivity = sensitivity
            best_specificity = specificity

    assert best_threshold is not None  # candidates is never empty
    logger.info(
        "Threshold for %s selected on %s: %.6f (J=%.4f, sensitivity=%.4f, specificity=%.4f)",
        model_id,
        split_name,
        best_threshold,
        best_youden_j,
        best_sensitivity,
        best_specificity,
    )
    return SelectedThreshold(
        threshold=float(best_threshold),
        youden_j=float(best_youden_j),
        sensitivity=float(best_sensitivity),
        specificity=float(best_specificity),
        number_of_candidates=int(candidates.numel()),
        model_id=model_id,
        split_name=split_name,
        binding_sha256=binding_sha256,
    )
