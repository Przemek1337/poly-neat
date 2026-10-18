"""Tests for binary metrics, threshold selection and the bootstrap."""

from __future__ import annotations

import math

import pytest
import torch

from polyneat.evaluators.binary_classification_metrics import (
    MetricInputError,
    compute_auroc,
    compute_average_precision,
    compute_binary_metrics,
    compute_confusion_counts,
)
from polyneat.evaluators.binary_predictions import (
    BinaryPredictions,
    NonFinitePredictionError,
    PredictionValidationError,
    build_binary_predictions,
    probabilities_from_logits,
)
from polyneat.evaluators.bootstrap_confidence_intervals import (
    ESTIMATED,
    GROUP_RESAMPLING,
    IMAGE_RESAMPLING,
    NOT_ESTIMATED,
    BootstrapConfig,
    bootstrap_metric_confidence_intervals,
    paired_bootstrap_auroc_difference,
)
from polyneat.evaluators.decision_threshold import (
    build_threshold_candidates,
    compute_threshold_binding,
    select_threshold_by_youden_j,
)


def _predictions(
    labels: list[int],
    scores: list[float],
    *,
    group_ids: list[str] | None = None,
    model_id: str = "model-a",
) -> BinaryPredictions:
    probabilities = torch.tensor(scores, dtype=torch.float64)
    logits = torch.stack([torch.zeros_like(probabilities), probabilities], dim=1)
    return BinaryPredictions(
        example_ids=tuple(f"e{index}" for index in range(len(labels))),
        group_ids=tuple(group_ids or [f"g{index}" for index in range(len(labels))]),
        labels=torch.tensor(labels, dtype=torch.long),
        positive_class_probabilities=probabilities,
        logits=logits,
        model_id=model_id,
        stage="track_a",
        split_name="official_test",
    )


class TestConfusionAndThresholdMetrics:
    def test_counts_follow_the_greater_or_equal_rule(self) -> None:
        labels = torch.tensor([0, 0, 1, 1])
        scores = torch.tensor([0.1, 0.5, 0.5, 0.9])
        confusion = compute_confusion_counts(labels, scores, 0.5)
        assert (confusion.true_negatives, confusion.false_positives) == (1, 1)
        assert (confusion.false_negatives, confusion.true_positives) == (0, 2)

    def test_metrics_match_hand_computed_counts(self) -> None:
        labels = torch.tensor([0, 0, 0, 1, 1, 1, 1])
        scores = torch.tensor([0.1, 0.2, 0.8, 0.4, 0.6, 0.7, 0.9])
        metrics = compute_binary_metrics(labels, scores, 0.5)
        assert metrics.confusion.true_positives == 3
        assert metrics.confusion.false_negatives == 1
        assert metrics.confusion.false_positives == 1
        assert metrics.confusion.true_negatives == 2
        assert metrics.sensitivity == pytest.approx(3 / 4)
        assert metrics.specificity == pytest.approx(2 / 3)
        assert metrics.precision == pytest.approx(3 / 4)
        assert metrics.accuracy == pytest.approx(5 / 7)
        assert metrics.balanced_accuracy == pytest.approx((3 / 4 + 2 / 3) / 2)
        assert metrics.f1 == pytest.approx(2 * (3 / 4) * (3 / 4) / (3 / 4 + 3 / 4))

    def test_zero_denominator_becomes_null_with_a_reason(self) -> None:
        labels = torch.tensor([0, 0, 1, 1])
        scores = torch.tensor([0.1, 0.2, 0.3, 0.4])
        metrics = compute_binary_metrics(labels, scores, 0.9)
        assert metrics.precision is None
        assert "predicted no positives" in metrics.undefined_reasons["precision"]
        assert metrics.f1 is None
        assert metrics.sensitivity == 0.0

    def test_single_class_set_leaves_ranking_metrics_undefined_with_a_reason(self) -> None:
        metrics = compute_binary_metrics(
            torch.tensor([1, 1, 1]), torch.tensor([0.2, 0.6, 0.9]), 0.5
        )
        assert metrics.auroc is None
        assert metrics.average_precision is None
        assert "both classes" in metrics.undefined_reasons["auroc"]
        assert metrics.specificity is None

    def test_metrics_record_the_backend_they_used(self) -> None:
        metrics = compute_binary_metrics(torch.tensor([0, 1]), torch.tensor([0.2, 0.8]), 0.5)
        assert metrics.metrics_backend.startswith("scikit-learn==")


class TestRankingMetrics:
    def test_perfect_ranking_scores_one(self) -> None:
        assert compute_auroc(
            torch.tensor([0, 0, 1, 1]), torch.tensor([0.1, 0.2, 0.8, 0.9])
        ) == pytest.approx(1.0)

    def test_reversed_ranking_scores_zero(self) -> None:
        assert compute_auroc(
            torch.tensor([0, 0, 1, 1]), torch.tensor([0.9, 0.8, 0.2, 0.1])
        ) == pytest.approx(0.0)

    def test_constant_finite_scores_score_one_half(self) -> None:
        assert compute_auroc(
            torch.tensor([0, 1, 0, 1]), torch.tensor([0.42, 0.42, 0.42, 0.42])
        ) == pytest.approx(0.5)

    def test_ties_are_handled_as_half_credit(self) -> None:
        # One positive and one negative share a score; the other pair is perfect.
        auroc = compute_auroc(
            torch.tensor([0, 0, 1, 1]), torch.tensor([0.1, 0.5, 0.5, 0.9])
        )
        assert auroc == pytest.approx(0.875)

    def test_average_precision_of_a_perfect_ranking_is_one(self) -> None:
        assert compute_average_precision(
            torch.tensor([0, 0, 1, 1]), torch.tensor([0.1, 0.2, 0.8, 0.9])
        ) == pytest.approx(1.0)

    def test_empty_and_single_class_and_non_finite_inputs_raise(self) -> None:
        with pytest.raises(MetricInputError, match="empty set"):
            compute_auroc(torch.tensor([], dtype=torch.long), torch.tensor([]))
        with pytest.raises(MetricInputError, match="both classes"):
            compute_auroc(torch.tensor([1, 1]), torch.tensor([0.2, 0.8]))
        with pytest.raises(MetricInputError, match="NaN or Inf"):
            compute_auroc(torch.tensor([0, 1]), torch.tensor([0.2, float("nan")]))

    def test_mismatched_lengths_raise(self) -> None:
        with pytest.raises(MetricInputError, match="disagree in length"):
            compute_auroc(torch.tensor([0, 1, 1]), torch.tensor([0.2, 0.8]))


class TestPredictionRecords:
    def test_ranking_score_is_the_positive_class_probability(self) -> None:
        logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
        probabilities = probabilities_from_logits(logits)
        assert probabilities[0] < 0.5 < probabilities[1]
        assert probabilities[1] == pytest.approx(1.0 / (1.0 + math.exp(-2.0)))

    def test_non_two_column_logits_are_a_call_error(self) -> None:
        with pytest.raises(PredictionValidationError, match="two-column"):
            probabilities_from_logits(torch.rand(4, 3))

    def test_non_finite_predictions_are_a_failed_evaluation(self) -> None:
        with pytest.raises(NonFinitePredictionError):
            build_binary_predictions(
                example_ids=("a", "b"),
                group_ids=("g", "g"),
                labels=torch.tensor([0, 1]),
                logits=torch.tensor([[0.0, float("inf")], [0.0, 1.0]]),
                model_id="m",
                stage="track_a",
                split_name="official_test",
            )

    def test_empty_and_misaligned_prediction_sets_are_rejected(self) -> None:
        with pytest.raises(PredictionValidationError, match="is empty"):
            _predictions([], [])
        with pytest.raises(PredictionValidationError, match="misaligned"):
            BinaryPredictions(
                example_ids=("a", "b"),
                group_ids=("g",),
                labels=torch.tensor([0, 1]),
                positive_class_probabilities=torch.tensor([0.1, 0.9]),
                logits=torch.zeros(2, 2),
                model_id="m",
                stage="track_a",
                split_name="official_test",
            )

    def test_selecting_rows_keeps_ids_aligned(self) -> None:
        predictions = _predictions([0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8])
        subset = predictions.select_rows(torch.tensor([3, 0, 3]))
        assert subset.example_ids == ("e3", "e0", "e3")
        assert subset.labels.tolist() == [1, 0, 1]


class TestThresholdSelection:
    def test_candidates_include_a_finite_classify_nothing_threshold(self) -> None:
        candidates = build_threshold_candidates(
            torch.tensor([0.3, 0.3, 0.7], dtype=torch.float64)
        )
        assert candidates.tolist()[:2] == [0.3, 0.7]
        assert math.isfinite(candidates[-1].item())
        assert candidates[-1].item() > 0.7

    def test_constant_scores_still_produce_two_finite_candidates(self) -> None:
        candidates = build_threshold_candidates(torch.full((6,), 0.5, dtype=torch.float64))
        assert candidates.numel() == 2
        assert all(math.isfinite(value) for value in candidates.tolist())

    def test_separable_scores_get_the_separating_threshold(self) -> None:
        selected = select_threshold_by_youden_j(
            torch.tensor([0, 0, 1, 1]),
            torch.tensor([0.1, 0.2, 0.8, 0.9]),
            model_id="m",
            split_name="threshold_validation",
            binding_sha256="deadbeef",
        )
        assert selected.threshold == pytest.approx(0.8)
        assert selected.youden_j == pytest.approx(1.0)

    def test_tie_prefers_higher_sensitivity_then_the_smaller_threshold(self) -> None:
        # Every threshold gives J = 0; the rule must pick the one calling
        # everything positive, which has sensitivity 1.
        selected = select_threshold_by_youden_j(
            torch.tensor([0, 1]),
            torch.tensor([0.4, 0.4]),
            model_id="m",
            split_name="threshold_validation",
            binding_sha256="deadbeef",
        )
        assert selected.threshold == pytest.approx(0.4)
        assert selected.sensitivity == pytest.approx(1.0)

    def test_selection_needs_both_classes(self) -> None:
        with pytest.raises(MetricInputError, match="both classes"):
            select_threshold_by_youden_j(
                torch.tensor([1, 1]),
                torch.tensor([0.4, 0.6]),
                model_id="m",
                split_name="threshold_validation",
                binding_sha256="deadbeef",
            )

    def test_binding_changes_with_checkpoint_preprocessing_or_manifest(self) -> None:
        base = dict(
            checkpoint_sha256="a" * 64,
            preprocessing_state={"fitted_mean": 0.5},
            manifest_sha256="b" * 64,
        )
        original = compute_threshold_binding(**base)
        assert original == compute_threshold_binding(**base)
        assert original != compute_threshold_binding(**{**base, "checkpoint_sha256": "c" * 64})
        assert original != compute_threshold_binding(
            **{**base, "preprocessing_state": {"fitted_mean": 0.6}}
        )
        assert original != compute_threshold_binding(**{**base, "manifest_sha256": "d" * 64})


class TestBootstrap:
    @staticmethod
    def _separable_predictions(number_per_class: int = 40, model_id: str = "model-a"):
        labels = [0] * number_per_class + [1] * number_per_class
        scores = [0.2 + 0.002 * index for index in range(number_per_class)] + [
            0.6 + 0.002 * index for index in range(number_per_class)
        ]
        groups = [f"p{index // 2}" for index in range(2 * number_per_class)]
        return _predictions(labels, scores, group_ids=groups, model_id=model_id)

    def test_intervals_bracket_the_point_estimate(self) -> None:
        predictions = self._separable_predictions()
        intervals = bootstrap_metric_confidence_intervals(
            predictions,
            threshold=0.5,
            config=BootstrapConfig(number_of_replicates=200, maximum_attempts=2000),
            random_seed=7,
        )
        for interval in intervals.values():
            assert interval.status == ESTIMATED
            assert interval.lower_bound <= interval.point_estimate <= interval.upper_bound
            assert interval.number_of_valid_replicates == 200

    def test_the_same_seed_reproduces_the_same_interval(self) -> None:
        predictions = self._separable_predictions()
        config = BootstrapConfig(number_of_replicates=100, maximum_attempts=1000)
        first = bootstrap_metric_confidence_intervals(
            predictions, threshold=0.5, config=config, random_seed=3
        )
        second = bootstrap_metric_confidence_intervals(
            predictions, threshold=0.5, config=config, random_seed=3
        )
        assert first["auroc"].lower_bound == second["auroc"].lower_bound
        assert first["auroc"].upper_bound == second["auroc"].upper_bound

    def test_resampling_unit_is_recorded_and_switchable(self) -> None:
        predictions = self._separable_predictions()
        config = BootstrapConfig(number_of_replicates=50, maximum_attempts=500)
        grouped = bootstrap_metric_confidence_intervals(
            predictions, threshold=0.5, config=config, random_seed=1, resample_groups=True
        )
        by_image = bootstrap_metric_confidence_intervals(
            predictions, threshold=0.5, config=config, random_seed=1, resample_groups=False
        )
        assert grouped["auroc"].resampling_unit == GROUP_RESAMPLING
        assert by_image["auroc"].resampling_unit == IMAGE_RESAMPLING

    def test_single_class_test_set_reports_not_estimated_rather_than_a_number(self) -> None:
        predictions = _predictions([1, 1, 1, 1], [0.2, 0.4, 0.6, 0.8])
        intervals = bootstrap_metric_confidence_intervals(
            predictions,
            threshold=0.5,
            config=BootstrapConfig(number_of_replicates=10, maximum_attempts=100),
            random_seed=1,
        )
        assert intervals["auroc"].status == NOT_ESTIMATED
        assert intervals["auroc"].point_estimate is None
        assert "both classes" in intervals["auroc"].reason

    def test_attempt_limit_yields_an_explicit_unestimated_status(self) -> None:
        # One positive in its own group: many draws miss it entirely, and the
        # low attempt cap makes the run give up rather than report a short CI.
        predictions = _predictions(
            [0] * 30 + [1], [0.1] * 30 + [0.9], group_ids=[f"g{i}" for i in range(31)]
        )
        intervals = bootstrap_metric_confidence_intervals(
            predictions,
            threshold=0.5,
            config=BootstrapConfig(number_of_replicates=50, maximum_attempts=60),
            random_seed=2,
        )
        assert intervals["auroc"].status == NOT_ESTIMATED
        assert intervals["auroc"].point_estimate is not None
        assert intervals["auroc"].number_of_attempts == 60
        assert "could be scored" in intervals["auroc"].reason

    def test_threshold_stays_fixed_across_replicates(self) -> None:
        predictions = self._separable_predictions()
        config = BootstrapConfig(number_of_replicates=100, maximum_attempts=1000)
        at_half = bootstrap_metric_confidence_intervals(
            predictions, threshold=0.5, config=config, random_seed=5
        )
        at_strict = bootstrap_metric_confidence_intervals(
            predictions, threshold=0.65, config=config, random_seed=5
        )
        assert at_half["sensitivity"].point_estimate == pytest.approx(1.0)
        assert at_strict["sensitivity"].point_estimate < 0.5
        # The ranking metric cannot depend on the threshold at all.
        assert at_half["auroc"].point_estimate == at_strict["auroc"].point_estimate
        assert at_half["average_precision"].point_estimate == (
            at_strict["average_precision"].point_estimate
        )


class TestPairedComparison:
    def test_identical_models_have_a_difference_interval_around_zero(self) -> None:
        first = TestBootstrap._separable_predictions(model_id="deepneat")
        second = TestBootstrap._separable_predictions(model_id="exact")
        comparison = paired_bootstrap_auroc_difference(
            first,
            second,
            config=BootstrapConfig(number_of_replicates=100, maximum_attempts=1000),
            random_seed=4,
        )
        assert comparison.status == ESTIMATED
        assert comparison.observed_difference == pytest.approx(0.0)
        assert comparison.lower_bound == pytest.approx(0.0)
        assert comparison.upper_bound == pytest.approx(0.0)

    def test_a_better_model_gets_a_positive_difference(self) -> None:
        labels = [0] * 20 + [1] * 20
        strong = _predictions(labels, [0.1] * 20 + [0.9] * 20, model_id="strong")
        weak = _predictions(
            labels, [0.5] * 19 + [0.9] + [0.1] + [0.5] * 19, model_id="weak"
        )
        comparison = paired_bootstrap_auroc_difference(
            strong,
            weak,
            config=BootstrapConfig(number_of_replicates=200, maximum_attempts=2000),
            random_seed=6,
        )
        assert comparison.observed_difference > 0.0
        assert comparison.lower_bound > 0.0

    def test_comparing_different_rows_is_refused(self) -> None:
        first = _predictions([0, 1], [0.2, 0.8], model_id="a")
        second = _predictions([0, 1, 1], [0.2, 0.8, 0.9], model_id="b")
        with pytest.raises(ValueError, match="same rows in the same order"):
            paired_bootstrap_auroc_difference(
                first, second, config=BootstrapConfig(number_of_replicates=10), random_seed=1
            )
