"""Pure metric computations on a binary prediction set.

Ranking metrics are delegated to scikit-learn rather than reimplemented: a
hand-rolled AUROC that mishandles ties is a subtle way to make every number in
a thesis slightly wrong. The version actually used is recorded in the artifacts
so a later reader can reproduce it.

Two conventions are fixed here and stated rather than assumed:

* a prediction is positive when ``probability >= threshold``, so a threshold of
  ``0.0`` classifies everything positive;
* ``average_precision`` is scikit-learn's step-wise summary of the
  precision-recall curve. It is deliberately *not* called AUPRC and not
  equated with a trapezoidal integral of that curve, because the two differ.

A metric whose denominator is zero is ``None`` with a recorded reason, never a
zero that looks like a measurement. A set containing only one class, or an
empty one, is a data or call error: it raises, and the bootstrap that produced
it retries the replicate instead of averaging in a fabricated value.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

REFERENCE_THRESHOLD = 0.5


class MetricInputError(ValueError):
    """Raised when a metric is asked for on a set that cannot support it."""


@dataclass(frozen=True)
class ConfusionCounts:
    """The four cells of a binary confusion matrix at one threshold."""

    true_negatives: int
    false_positives: int
    false_negatives: int
    true_positives: int

    @property
    def number_of_actual_positives(self) -> int:
        """Rows whose ground-truth label is positive."""
        return self.true_positives + self.false_negatives

    @property
    def number_of_actual_negatives(self) -> int:
        """Rows whose ground-truth label is negative."""
        return self.true_negatives + self.false_positives

    @property
    def number_of_predicted_positives(self) -> int:
        """Rows the model called positive at this threshold."""
        return self.true_positives + self.false_positives

    @property
    def total(self) -> int:
        """Rows in the evaluated set."""
        return (
            self.true_negatives
            + self.false_positives
            + self.false_negatives
            + self.true_positives
        )


@dataclass(frozen=True)
class BinaryClassificationMetrics:
    """Every reported number for one model, one split and one threshold.

    Attributes:
        threshold: The decision threshold these counts were taken at.
        confusion: The four confusion cells.
        auroc: Area under the ROC curve, the protocol's ranking metric.
        average_precision: scikit-learn's ``average_precision_score``. Not an
            alias for a trapezoidal AUPRC.
        sensitivity: ``TP / (TP + FN)``.
        specificity: ``TN / (TN + FP)``.
        balanced_accuracy: Mean of sensitivity and specificity.
        accuracy: ``(TP + TN) / N``.
        precision: ``TP / (TP + FP)``.
        f1: Harmonic mean of precision and sensitivity.
        undefined_reasons: Why any of the above is ``None``, per metric name.
        metrics_backend: Identity of the ranking-metric implementation used.
    """

    threshold: float
    confusion: ConfusionCounts
    auroc: float | None
    average_precision: float | None
    sensitivity: float | None
    specificity: float | None
    balanced_accuracy: float | None
    accuracy: float | None
    precision: float | None
    f1: float | None
    undefined_reasons: dict[str, str] = field(default_factory=dict)
    metrics_backend: str = ""

    def to_serializable_dict(self) -> dict:
        """Return the metrics as JSON-compatible data, nulls included."""
        return {
            "threshold": self.threshold,
            "true_negatives": self.confusion.true_negatives,
            "false_positives": self.confusion.false_positives,
            "false_negatives": self.confusion.false_negatives,
            "true_positives": self.confusion.true_positives,
            "number_of_actual_positives": self.confusion.number_of_actual_positives,
            "number_of_actual_negatives": self.confusion.number_of_actual_negatives,
            "auroc": self.auroc,
            "average_precision": self.average_precision,
            "sensitivity": self.sensitivity,
            "specificity": self.specificity,
            "balanced_accuracy": self.balanced_accuracy,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "f1": self.f1,
            "undefined_reasons": dict(sorted(self.undefined_reasons.items())),
            "metrics_backend": self.metrics_backend,
        }


def _require_usable_ranking_input(labels: torch.Tensor, scores: torch.Tensor) -> None:
    """Reject the inputs a ranking metric has no defined answer for."""
    if labels.shape[0] == 0:
        raise MetricInputError("cannot compute a ranking metric on an empty set")
    if labels.shape[0] != scores.shape[0]:
        raise MetricInputError(
            f"labels and scores disagree in length: {labels.shape[0]} vs {scores.shape[0]}"
        )
    if not torch.isfinite(scores).all():
        raise MetricInputError(
            "scores contain NaN or Inf; this is a failed evaluation, not a ranking of zero"
        )
    unique_labels = torch.unique(labels)
    if unique_labels.numel() < 2:
        raise MetricInputError(
            f"a ranking metric needs both classes, got only label(s) {unique_labels.tolist()}"
        )


def describe_metrics_backend() -> str:
    """Return the pinned identity of the ranking-metric implementation."""
    import sklearn

    return f"scikit-learn=={sklearn.__version__}"


def compute_auroc(labels: torch.Tensor, scores: torch.Tensor) -> float:
    """Area under the ROC curve for the positive class.

    Constant finite scores over a set containing both classes give exactly
    ``0.5``: the model separated nothing, which is a real measurement rather
    than an undefined value.

    Args:
        labels: ``(n,)`` tensor of ``0``/``1`` ground truth.
        scores: ``(n,)`` tensor of positive-class scores.

    Returns:
        AUROC in ``[0, 1]``.

    Raises:
        MetricInputError: On an empty set, misaligned lengths, non-finite
            scores, or a set containing only one class.
    """
    from sklearn.metrics import roc_auc_score

    _require_usable_ranking_input(labels, scores)
    return float(roc_auc_score(labels.cpu().numpy(), scores.cpu().numpy()))


def compute_average_precision(labels: torch.Tensor, scores: torch.Tensor) -> float:
    """scikit-learn's ``average_precision_score`` for the positive class.

    This is the step-wise summary of the precision-recall curve, reported under
    the name ``average_precision``. It is not the trapezoidal area under that
    curve and the two should not be quoted interchangeably.

    Raises:
        MetricInputError: Same conditions as :func:`compute_auroc`.
    """
    from sklearn.metrics import average_precision_score

    _require_usable_ranking_input(labels, scores)
    return float(average_precision_score(labels.cpu().numpy(), scores.cpu().numpy()))


def compute_confusion_counts(
    labels: torch.Tensor, scores: torch.Tensor, threshold: float
) -> ConfusionCounts:
    """Count the four confusion cells, calling a row positive at ``score >= threshold``.

    Raises:
        MetricInputError: On an empty set, misaligned lengths or non-finite
            scores. A single-class set is allowed here: a confusion matrix is
            still well defined, and the metrics derived from the missing class
            come back as ``None`` with a reason.
    """
    if labels.shape[0] == 0:
        raise MetricInputError("cannot build a confusion matrix from an empty set")
    if labels.shape[0] != scores.shape[0]:
        raise MetricInputError(
            f"labels and scores disagree in length: {labels.shape[0]} vs {scores.shape[0]}"
        )
    if not torch.isfinite(scores).all():
        raise MetricInputError(
            "scores contain NaN or Inf; this is a failed evaluation, not a prediction"
        )
    predicted_positive = scores >= threshold
    actual_positive = labels == 1
    return ConfusionCounts(
        true_negatives=int((~predicted_positive & ~actual_positive).sum()),
        false_positives=int((predicted_positive & ~actual_positive).sum()),
        false_negatives=int((~predicted_positive & actual_positive).sum()),
        true_positives=int((predicted_positive & actual_positive).sum()),
    )


def _ratio_or_reason(
    numerator: int, denominator: int, reason_when_zero: str
) -> tuple[float | None, str | None]:
    """Return the ratio, or ``None`` plus why it has no value."""
    if denominator == 0:
        return None, reason_when_zero
    return numerator / denominator, None


def compute_binary_metrics(
    labels: torch.Tensor,
    scores: torch.Tensor,
    threshold: float,
    *,
    include_ranking_metrics: bool = True,
) -> BinaryClassificationMetrics:
    """Compute every reported metric for one threshold.

    Args:
        labels: ``(n,)`` tensor of ``0``/``1`` ground truth.
        scores: ``(n,)`` tensor of positive-class probabilities.
        threshold: Decision threshold; a row is positive at ``score >= threshold``.
        include_ranking_metrics: When ``False``, AUROC and average precision are
            skipped and recorded as undefined. Useful when the caller already
            knows the set has one class and only wants the threshold metrics.

    Returns:
        The full :class:`BinaryClassificationMetrics` record, with ``None`` and
        a stated reason wherever a denominator was zero.

    Raises:
        MetricInputError: On an empty set, misaligned lengths or non-finite
            scores.
    """
    confusion = compute_confusion_counts(labels, scores, threshold)
    undefined_reasons: dict[str, str] = {}

    sensitivity, sensitivity_reason = _ratio_or_reason(
        confusion.true_positives,
        confusion.number_of_actual_positives,
        "no positive examples in the evaluated set",
    )
    specificity, specificity_reason = _ratio_or_reason(
        confusion.true_negatives,
        confusion.number_of_actual_negatives,
        "no negative examples in the evaluated set",
    )
    precision, precision_reason = _ratio_or_reason(
        confusion.true_positives,
        confusion.number_of_predicted_positives,
        "the model predicted no positives at this threshold",
    )
    if sensitivity_reason:
        undefined_reasons["sensitivity"] = sensitivity_reason
    if specificity_reason:
        undefined_reasons["specificity"] = specificity_reason
    if precision_reason:
        undefined_reasons["precision"] = precision_reason

    if sensitivity is None or specificity is None:
        balanced_accuracy = None
        undefined_reasons["balanced_accuracy"] = "sensitivity or specificity is undefined"
    else:
        balanced_accuracy = 0.5 * (sensitivity + specificity)

    if precision is None or sensitivity is None:
        f1_score = None
        undefined_reasons["f1"] = "precision or sensitivity is undefined"
    elif precision + sensitivity == 0.0:
        f1_score = None
        undefined_reasons["f1"] = "precision and sensitivity are both zero"
    else:
        f1_score = 2.0 * precision * sensitivity / (precision + sensitivity)

    accuracy = (confusion.true_positives + confusion.true_negatives) / confusion.total

    auroc: float | None = None
    average_precision: float | None = None
    if include_ranking_metrics:
        try:
            auroc = compute_auroc(labels, scores)
            average_precision = compute_average_precision(labels, scores)
        except MetricInputError as ranking_error:
            undefined_reasons["auroc"] = str(ranking_error)
            undefined_reasons["average_precision"] = str(ranking_error)
    else:
        undefined_reasons["auroc"] = "ranking metrics were not requested"
        undefined_reasons["average_precision"] = "ranking metrics were not requested"

    return BinaryClassificationMetrics(
        threshold=float(threshold),
        confusion=confusion,
        auroc=auroc,
        average_precision=average_precision,
        sensitivity=sensitivity,
        specificity=specificity,
        balanced_accuracy=balanced_accuracy,
        accuracy=accuracy,
        precision=precision,
        f1=f1_score,
        undefined_reasons=undefined_reasons,
        metrics_backend=describe_metrics_backend(),
    )
