"""The statistics behind the result-series notebook."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.series_analysis import (
    compare_methods,
    compare_tracks,
    holm_adjust,
    load_series,
    mean_confidence_interval,
    official_test_interval,
    retraining_status_counts,
    vargha_delaney_a12,
    wilson_interval,
)
from benchmarks.summarize_series import SeriesSpec

_BINARY = SeriesSpec("RTG", "pneumonia/series-v2", "auroc", "search_validation_auroc", True)
_MULTICLASS = SeriesSpec("MNIST", "mnist/series-v1", "accuracy", "validation_accuracy")


def _write_run(
    root: Path,
    spec: SeriesSpec,
    method: str,
    seed: int,
    *,
    track_a: float,
    track_b: dict[int, float | None],
    maximum_test_samples: int | None = None,
) -> None:
    metric = spec.test_metric
    metric_values = {
        spec.selection_metric: 0.99,
        f"track_a_seed{seed}_test_{metric}": track_a,
        "selected_parameter_count": 1000.0,
        "failed_evaluation_fraction": 0.0,
    }
    for retraining_seed, value in track_b.items():
        if value is not None:
            metric_values[f"track_b_seed{retraining_seed}_test_{metric}"] = value
    model_id = f"{method}_track_a_seed{seed}"
    report = {
        "method": method,
        "effective_configuration": {
            "search_seed": seed,
            "retraining_seeds": list(track_b),
            "maximum_test_samples": maximum_test_samples,
        },
        "track_b_failed_retrainings": [
            {"retraining_seed": retraining_seed}
            for retraining_seed, value in track_b.items()
            if value is None
        ],
        "official_test": {
            model_id: {
                "confidence_intervals": {
                    "auroc": {"lower_bound": track_a - 0.02, "upper_bound": track_a + 0.02}
                }
            }
        },
        "summary": {"metric_values": metric_values, "number_of_generations": 3},
    }
    directory = root / spec.series_directory / "runs" / method / f"seed_{seed}"
    directory.mkdir(parents=True)
    (directory / "run_report.json").write_text(json.dumps(report), encoding="utf-8")


class TestPrimitives:
    def test_holm_multiplies_by_the_remaining_count_and_stays_monotone(self) -> None:
        assert holm_adjust([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])

    def test_holm_caps_at_one(self) -> None:
        assert holm_adjust([0.5, 0.6]) == [1.0, 1.0]

    def test_a12_is_one_for_complete_dominance_and_half_for_identical_samples(self) -> None:
        assert vargha_delaney_a12([1.0, 2.0, 3.0], [0.0, 0.0, 0.0]) == 1.0
        assert vargha_delaney_a12([1.0, 2.0], [1.0, 2.0]) == 0.5

    def test_wilson_interval_matches_the_textbook_value(self) -> None:
        lower, upper = wilson_interval(0.5, 100)
        assert lower == pytest.approx(0.4038, abs=1e-4)
        assert upper == pytest.approx(0.5962, abs=1e-4)

    def test_t_interval_uses_the_student_quantile(self) -> None:
        mean, lower, upper = mean_confidence_interval([1.0, 2.0, 3.0])
        assert mean == 2.0
        assert upper - mean == pytest.approx(4.3027 / 3**0.5, abs=1e-3)
        assert mean - lower == pytest.approx(upper - mean)


class TestLoading:
    def test_retrainings_are_ok_collapsed_or_diverged(self, tmp_path: Path) -> None:
        _write_run(
            tmp_path, _BINARY, "deepneat", 101, track_a=0.93,
            track_b={201: 0.91, 202: 0.52, 203: None},
        )

        results = load_series(tmp_path, _BINARY)

        (seed_result,) = results.by_method["deepneat"]
        assert [r.status for r in seed_result.retrainings] == ["ok", "collapsed", "diverged"]
        assert seed_result.track_b_mean == pytest.approx((0.91 + 0.52) / 2)
        assert retraining_status_counts(results)["deepneat"] == {
            "ok": 1, "collapsed": 1, "diverged": 1, "missing": 0,
        }

    def test_methods_come_in_the_fixed_order(self, tmp_path: Path) -> None:
        for method in ("fixed_cnn", "random_search", "exact", "deepneat"):
            _write_run(tmp_path, _BINARY, method, 101, track_a=0.9, track_b={201: 0.9})

        assert load_series(tmp_path, _BINARY).methods == [
            "deepneat", "exact", "random_search", "fixed_cnn",
        ]

    def test_a_test_subset_is_refused_for_the_wilson_interval(self, tmp_path: Path) -> None:
        _write_run(
            tmp_path, _MULTICLASS, "exact", 101, track_a=0.9, track_b={201: 0.9},
            maximum_test_samples=500,
        )

        with pytest.raises(ValueError, match="test subset"):
            load_series(tmp_path, _MULTICLASS)

    def test_the_test_interval_comes_from_the_stored_bootstrap(self, tmp_path: Path) -> None:
        _write_run(tmp_path, _BINARY, "exact", 101, track_a=0.9, track_b={201: 0.9})

        (seed_result,) = load_series(tmp_path, _BINARY).by_method["exact"]

        assert official_test_interval(seed_result, _BINARY) == pytest.approx((0.88, 0.92))


class TestComparisons:
    def test_exact_comparison_handles_tied_accuracy(self, tmp_path: Path) -> None:
        for seed in range(101, 106):
            _write_run(tmp_path, _MULTICLASS, "deepneat", seed,
                       track_a=0.99, track_b={201: 0.99})
            _write_run(tmp_path, _MULTICLASS, "exact", seed,
                       track_a=0.92, track_b={201: 0.92})
        (comparison,) = compare_methods(load_series(tmp_path, _MULTICLASS))
        assert comparison.mann_whitney_p == pytest.approx(2 / 252)

    def test_fully_separated_methods_reach_the_exact_mann_whitney_minimum(
        self, tmp_path: Path
    ) -> None:
        for offset, seed in enumerate(range(101, 106)):
            _write_run(tmp_path, _BINARY, "deepneat", seed, track_a=0.95 + offset / 1000,
                       track_b={201: 0.9})
            _write_run(tmp_path, _BINARY, "exact", seed, track_a=0.80 + offset / 1000,
                       track_b={201: 0.9})

        (comparison,) = compare_methods(load_series(tmp_path, _BINARY))

        assert comparison.a12 == 1.0
        assert comparison.mann_whitney_p == pytest.approx(2 / 252)
        assert comparison.holm_p == comparison.mann_whitney_p
        assert comparison.mean_difference == pytest.approx(0.15)
        lower, upper = comparison.difference_interval
        assert 0 < lower < 0.15 < upper

    def test_tracks_are_paired_by_seed_and_a_seed_without_track_b_is_left_out(
        self, tmp_path: Path
    ) -> None:
        values = {101: (0.80, 0.90), 102: (0.81, 0.93), 103: (0.82, 0.91), 104: (0.83, None)}
        for seed, (track_a, track_b) in values.items():
            _write_run(tmp_path, _BINARY, "exact", seed, track_a=track_a,
                       track_b={201: track_b})

        (comparison,) = compare_tracks(load_series(tmp_path, _BINARY))

        assert comparison.seeds_without_track_b == 1
        assert comparison.seeds_where_b_is_higher == 3
        assert comparison.mean_difference == pytest.approx((0.10 + 0.12 + 0.09) / 3)
        assert comparison.paired_t_p is not None and comparison.paired_t_p < 0.05
        assert comparison.holm_p == comparison.paired_t_p
