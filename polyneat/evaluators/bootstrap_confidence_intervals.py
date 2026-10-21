"""Bootstrap intervals for test metrics, and paired intervals for differences.

Three properties matter more than the arithmetic here.

**The resampling unit has to match the dependence in the data.** When every row
carries a trustworthy patient identifier, patients are resampled with all their
images; otherwise images are, and the result is labelled as such. Either way the
estimand stays the image-level metric under group dependence, not the mean of
per-patient metrics, which is a different quantity.

**The threshold does not move.** It was chosen once, on the threshold split, for
this frozen checkpoint. Re-tuning it inside each replicate would measure a
procedure that never ran.

**A replicate that cannot be scored is not scored as zero.** A draw containing
one class has no AUROC; it is retried, counted, and if too few valid replicates
are reached within the attempt limit the interval is reported as not estimated
rather than computed from whatever survived.

The intervals are conditional on the trained model and the chosen threshold.
They do not cover the uncertainty of the threshold search or the randomness of
the whole architecture search; seed-to-seed spread is reported separately.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import torch

from polyneat.evaluators.binary_classification_metrics import (
    MetricInputError,
    compute_auroc,
    compute_average_precision,
    compute_confusion_counts,
)
from polyneat.evaluators.binary_predictions import BinaryPredictions
from polyneat.logging_utils.custom_logger import get_logger

logger = get_logger(__name__)

GROUP_RESAMPLING = "group"
IMAGE_RESAMPLING = "image"

ESTIMATED = "estimated"
NOT_ESTIMATED = "not_estimated"

BOOTSTRAP_METRIC_NAMES: tuple[str, ...] = (
    "auroc",
    "average_precision",
    "sensitivity",
    "specificity",
)


@dataclass(frozen=True)
class BootstrapConfig:
    """How the bootstrap is run. Frozen before the result series starts.

    Attributes:
        number_of_replicates: Valid replicates required for an interval.
        maximum_attempts: Hard cap on draws, so a degenerate split cannot spin
            forever.
        confidence_level: Two-sided level; ``0.95`` gives the 2.5 and 97.5
            percentiles.
        resample_groups: Resample whole groups when the manifest established
            trustworthy identifiers, otherwise individual rows.
    """

    number_of_replicates: int = 2000
    maximum_attempts: int = 20_000
    confidence_level: float = 0.95
    resample_groups: bool = True

    def __post_init__(self) -> None:
        if self.number_of_replicates < 1:
            raise ValueError(
                f"number_of_replicates must be >= 1, got {self.number_of_replicates}"
            )
        if self.maximum_attempts < self.number_of_replicates:
            raise ValueError(
                f"maximum_attempts ({self.maximum_attempts}) must be at least "
                f"number_of_replicates ({self.number_of_replicates})"
            )
        if not 0.0 < self.confidence_level < 1.0:
            raise ValueError(f"confidence_level must be in (0, 1), got {self.confidence_level}")

    @property
    def percentiles(self) -> tuple[float, float]:
        """Lower and upper percentile of the interval, in percent."""
        tail = 100.0 * (1.0 - self.confidence_level) / 2.0
        return tail, 100.0 - tail


@dataclass(frozen=True)
class BootstrapConfidenceInterval:
    """One metric's interval, or an explicit statement that it has none.

    Attributes:
        metric_name: Which metric this describes.
        point_estimate: The metric on the observed test set, not a bootstrap
            mean.
        lower_bound: Lower percentile, ``None`` when not estimated.
        upper_bound: Upper percentile, ``None`` when not estimated.
        confidence_level: Level the bounds correspond to.
        resampling_unit: ``group`` or ``image``.
        number_of_valid_replicates: Replicates that could be scored.
        number_of_attempts: Draws taken, valid and invalid together.
        status: ``estimated`` or ``not_estimated``.
        reason: Why an interval is missing, when it is.
    """

    metric_name: str
    point_estimate: float | None
    lower_bound: float | None
    upper_bound: float | None
    confidence_level: float
    resampling_unit: str
    number_of_valid_replicates: int
    number_of_attempts: int
    status: str
    reason: str | None = None

    def to_serializable_dict(self) -> dict:
        """Return the interval as JSON-compatible data."""
        return {
            "metric_name": self.metric_name,
            "point_estimate": self.point_estimate,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "confidence_level": self.confidence_level,
            "resampling_unit": self.resampling_unit,
            "number_of_valid_replicates": self.number_of_valid_replicates,
            "number_of_attempts": self.number_of_attempts,
            "status": self.status,
            "reason": self.reason,
        }


def _rows_by_group(group_ids: tuple[str, ...]) -> tuple[tuple[str, ...], dict[str, np.ndarray]]:
    """Index rows by group, keeping a stable group order for reproducibility."""
    rows_of_group: dict[str, list[int]] = defaultdict(list)
    for row_index, group_id in enumerate(group_ids):
        rows_of_group[group_id].append(row_index)
    ordered_group_ids = tuple(sorted(rows_of_group))
    return ordered_group_ids, {
        group_id: np.asarray(rows_of_group[group_id], dtype=np.int64)
        for group_id in ordered_group_ids
    }


def _draw_replicate_rows(
    rng: np.random.Generator,
    *,
    number_of_rows: int,
    ordered_group_ids: tuple[str, ...] | None,
    rows_of_group: dict[str, np.ndarray] | None,
) -> np.ndarray:
    """Draw one replicate, either of whole groups or of individual rows."""
    if ordered_group_ids is None or rows_of_group is None:
        return rng.integers(0, number_of_rows, size=number_of_rows)
    drawn_group_positions = rng.integers(0, len(ordered_group_ids), size=len(ordered_group_ids))
    return np.concatenate(
        [rows_of_group[ordered_group_ids[position]] for position in drawn_group_positions]
    )


def _metrics_of_replicate(
    labels: torch.Tensor, scores: torch.Tensor, threshold: float
) -> dict[str, float]:
    """Score one replicate, or signal that it cannot be scored.

    Raises:
        MetricInputError: If the replicate holds only one class.
    """
    auroc = compute_auroc(labels, scores)
    average_precision = compute_average_precision(labels, scores)
    confusion = compute_confusion_counts(labels, scores, threshold)
    return {
        "auroc": auroc,
        "average_precision": average_precision,
        "sensitivity": confusion.true_positives / confusion.number_of_actual_positives,
        "specificity": confusion.true_negatives / confusion.number_of_actual_negatives,
    }


def bootstrap_metric_confidence_intervals(
    predictions: BinaryPredictions,
    *,
    threshold: float,
    config: BootstrapConfig,
    random_seed: int,
    resample_groups: bool | None = None,
) -> dict[str, BootstrapConfidenceInterval]:
    """Confidence intervals for AUROC, average precision, sensitivity, specificity.

    Args:
        predictions: Frozen model predictions on the evaluation split.
        threshold: The already-chosen decision threshold. It is held constant
            across every replicate.
        config: Frozen bootstrap settings.
        random_seed: Seed of the bootstrap stream, recorded in the artifacts.
        resample_groups: Overrides ``config.resample_groups``. Callers pass
            ``False`` when the manifest could not establish trustworthy
            identifiers, and the limitation is then recorded in the result.

    Returns:
        One :class:`BootstrapConfidenceInterval` per metric name. A metric
        whose point estimate does not exist, or for which too few valid
        replicates were drawn, comes back with status ``not_estimated`` and a
        reason rather than an invented number.
    """
    use_group_resampling = (
        config.resample_groups if resample_groups is None else resample_groups
    )
    resampling_unit = GROUP_RESAMPLING if use_group_resampling else IMAGE_RESAMPLING
    rng = np.random.default_rng(random_seed)

    try:
        point_estimates = _metrics_of_replicate(
            predictions.labels, predictions.positive_class_probabilities, threshold
        )
    except MetricInputError as estimate_error:
        return {
            metric_name: BootstrapConfidenceInterval(
                metric_name=metric_name,
                point_estimate=None,
                lower_bound=None,
                upper_bound=None,
                confidence_level=config.confidence_level,
                resampling_unit=resampling_unit,
                number_of_valid_replicates=0,
                number_of_attempts=0,
                status=NOT_ESTIMATED,
                reason=str(estimate_error),
            )
            for metric_name in BOOTSTRAP_METRIC_NAMES
        }

    ordered_group_ids, rows_of_group = (
        _rows_by_group(predictions.group_ids) if use_group_resampling else (None, None)
    )
    replicate_values: dict[str, list[float]] = {name: [] for name in BOOTSTRAP_METRIC_NAMES}
    number_of_attempts = 0
    while (
        len(replicate_values["auroc"]) < config.number_of_replicates
        and number_of_attempts < config.maximum_attempts
    ):
        number_of_attempts += 1
        drawn_rows = torch.from_numpy(
            _draw_replicate_rows(
                rng,
                number_of_rows=len(predictions),
                ordered_group_ids=ordered_group_ids,
                rows_of_group=rows_of_group,
            )
        )
        try:
            replicate_metrics = _metrics_of_replicate(
                predictions.labels[drawn_rows],
                predictions.positive_class_probabilities[drawn_rows],
                threshold,
            )
        except MetricInputError:
            continue
        for metric_name, value in replicate_metrics.items():
            replicate_values[metric_name].append(value)

    number_of_valid_replicates = len(replicate_values["auroc"])
    if number_of_valid_replicates < config.number_of_replicates:
        logger.warning(
            "Bootstrap for %s reached only %d of %d valid replicates in %d attempts",
            predictions.model_id,
            number_of_valid_replicates,
            config.number_of_replicates,
            number_of_attempts,
        )
    lower_percentile, upper_percentile = config.percentiles
    intervals: dict[str, BootstrapConfidenceInterval] = {}
    for metric_name in BOOTSTRAP_METRIC_NAMES:
        values = replicate_values[metric_name]
        if number_of_valid_replicates < config.number_of_replicates:
            intervals[metric_name] = BootstrapConfidenceInterval(
                metric_name=metric_name,
                point_estimate=point_estimates[metric_name],
                lower_bound=None,
                upper_bound=None,
                confidence_level=config.confidence_level,
                resampling_unit=resampling_unit,
                number_of_valid_replicates=number_of_valid_replicates,
                number_of_attempts=number_of_attempts,
                status=NOT_ESTIMATED,
                reason=(
                    f"only {number_of_valid_replicates} of {config.number_of_replicates} "
                    f"replicates could be scored within {config.maximum_attempts} attempts"
                ),
            )
            continue
        value_array = np.asarray(values, dtype=np.float64)
        intervals[metric_name] = BootstrapConfidenceInterval(
            metric_name=metric_name,
            point_estimate=point_estimates[metric_name],
            lower_bound=float(np.percentile(value_array, lower_percentile)),
            upper_bound=float(np.percentile(value_array, upper_percentile)),
            confidence_level=config.confidence_level,
            resampling_unit=resampling_unit,
            number_of_valid_replicates=number_of_valid_replicates,
            number_of_attempts=number_of_attempts,
            status=ESTIMATED,
        )
    return intervals


@dataclass(frozen=True)
class PairedAurocDifference:
    """The AUROC difference between two methods, with its own interval.

    Reporting two separate intervals is not the same test: the two methods were
    evaluated on the same test set, so their errors are correlated, and the
    interval of the difference is what the comparison actually needs.

    Attributes:
        first_model_id: Model whose AUROC is the minuend.
        second_model_id: Model whose AUROC is the subtrahend.
        observed_difference: ``AUROC(first) - AUROC(second)`` on the real test
            set.
        lower_bound: Lower percentile of the difference, ``None`` when not
            estimated.
        upper_bound: Upper percentile, same convention.
        confidence_level: Level the bounds correspond to.
        resampling_unit: ``group`` or ``image``.
        number_of_valid_replicates: Replicates both models could be scored on.
        number_of_attempts: Draws taken in total.
        status: ``estimated`` or ``not_estimated``.
        reason: Why an interval is missing, when it is.
    """

    first_model_id: str
    second_model_id: str
    observed_difference: float | None
    lower_bound: float | None
    upper_bound: float | None
    confidence_level: float
    resampling_unit: str
    number_of_valid_replicates: int
    number_of_attempts: int
    status: str
    reason: str | None = None

    def to_serializable_dict(self) -> dict:
        """Return the comparison as JSON-compatible data."""
        return {
            "first_model_id": self.first_model_id,
            "second_model_id": self.second_model_id,
            "observed_difference": self.observed_difference,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "confidence_level": self.confidence_level,
            "resampling_unit": self.resampling_unit,
            "number_of_valid_replicates": self.number_of_valid_replicates,
            "number_of_attempts": self.number_of_attempts,
            "status": self.status,
            "reason": self.reason,
        }


def paired_bootstrap_auroc_difference(
    first_predictions: BinaryPredictions,
    second_predictions: BinaryPredictions,
    *,
    config: BootstrapConfig,
    random_seed: int,
    resample_groups: bool | None = None,
) -> PairedAurocDifference:
    """Interval for the AUROC difference of two models on the same test rows.

    Both models are scored on the *same* drawn ids in every replicate, which is
    what makes the comparison paired.

    Args:
        first_predictions: Predictions of the first model.
        second_predictions: Predictions of the second model, on the same split,
            in the same row order.
        config: Frozen bootstrap settings.
        random_seed: Seed of the bootstrap stream.
        resample_groups: Overrides ``config.resample_groups``.

    Returns:
        The :class:`PairedAurocDifference`.

    Raises:
        ValueError: If the two prediction sets do not describe the same rows.
            Comparing different rows would not be a paired comparison.
    """
    if first_predictions.example_ids != second_predictions.example_ids:
        raise ValueError(
            "a paired bootstrap needs both models evaluated on the same rows in the same order; "
            f"{first_predictions.model_id!r} and {second_predictions.model_id!r} disagree"
        )
    if not torch.equal(first_predictions.labels, second_predictions.labels):
        raise ValueError("the two prediction sets carry different labels for the same rows")

    use_group_resampling = (
        config.resample_groups if resample_groups is None else resample_groups
    )
    resampling_unit = GROUP_RESAMPLING if use_group_resampling else IMAGE_RESAMPLING
    rng = np.random.default_rng(random_seed)
    ordered_group_ids, rows_of_group = (
        _rows_by_group(first_predictions.group_ids) if use_group_resampling else (None, None)
    )

    try:
        observed_difference = compute_auroc(
            first_predictions.labels, first_predictions.positive_class_probabilities
        ) - compute_auroc(
            second_predictions.labels, second_predictions.positive_class_probabilities
        )
    except MetricInputError as estimate_error:
        return PairedAurocDifference(
            first_model_id=first_predictions.model_id,
            second_model_id=second_predictions.model_id,
            observed_difference=None,
            lower_bound=None,
            upper_bound=None,
            confidence_level=config.confidence_level,
            resampling_unit=resampling_unit,
            number_of_valid_replicates=0,
            number_of_attempts=0,
            status=NOT_ESTIMATED,
            reason=str(estimate_error),
        )

    differences: list[float] = []
    number_of_attempts = 0
    while len(differences) < config.number_of_replicates and (
        number_of_attempts < config.maximum_attempts
    ):
        number_of_attempts += 1
        drawn_rows = torch.from_numpy(
            _draw_replicate_rows(
                rng,
                number_of_rows=len(first_predictions),
                ordered_group_ids=ordered_group_ids,
                rows_of_group=rows_of_group,
            )
        )
        try:
            first_auroc = compute_auroc(
                first_predictions.labels[drawn_rows],
                first_predictions.positive_class_probabilities[drawn_rows],
            )
            second_auroc = compute_auroc(
                second_predictions.labels[drawn_rows],
                second_predictions.positive_class_probabilities[drawn_rows],
            )
        except MetricInputError:
            continue
        differences.append(first_auroc - second_auroc)

    if len(differences) < config.number_of_replicates:
        return PairedAurocDifference(
            first_model_id=first_predictions.model_id,
            second_model_id=second_predictions.model_id,
            observed_difference=observed_difference,
            lower_bound=None,
            upper_bound=None,
            confidence_level=config.confidence_level,
            resampling_unit=resampling_unit,
            number_of_valid_replicates=len(differences),
            number_of_attempts=number_of_attempts,
            status=NOT_ESTIMATED,
            reason=(
                f"only {len(differences)} of {config.number_of_replicates} replicates could be "
                f"scored within {config.maximum_attempts} attempts"
            ),
        )

    lower_percentile, upper_percentile = config.percentiles
    difference_array = np.asarray(differences, dtype=np.float64)
    return PairedAurocDifference(
        first_model_id=first_predictions.model_id,
        second_model_id=second_predictions.model_id,
        observed_difference=observed_difference,
        lower_bound=float(np.percentile(difference_array, lower_percentile)),
        upper_bound=float(np.percentile(difference_array, upper_percentile)),
        confidence_level=config.confidence_level,
        resampling_unit=resampling_unit,
        number_of_valid_replicates=len(differences),
        number_of_attempts=number_of_attempts,
        status=ESTIMATED,
    )
