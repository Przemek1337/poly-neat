"""The series summary reads finished run reports into one table per problem."""

from __future__ import annotations

import json
from pathlib import Path

from benchmarks.summarize_series import SeriesSpec, build_summary, main

_BINARY = SeriesSpec("RTG", "pneumonia/series-v2", "auroc", "search_validation_auroc", True)
_MULTICLASS = SeriesSpec("MNIST", "mnist/series-v1", "accuracy", "validation_accuracy")


def _write_report(
    root: Path,
    spec: SeriesSpec,
    method: str,
    seed: int,
    *,
    track_a: float,
    track_b: float | None,
    diverged: int = 0,
    extra: dict | None = None,
) -> None:
    metric_values = {
        spec.selection_metric: track_a + 0.01,
        f"track_a_seed{seed}_test_{spec.test_metric}": track_a,
        "selected_parameter_count": 1000.0,
        "failed_evaluation_fraction": 0.25,
        **(extra or {}),
    }
    if track_b is not None:
        metric_values[f"track_b_mean_test_{spec.test_metric}"] = track_b
    directory = root / spec.series_directory / "runs" / method / f"seed_{seed}"
    directory.mkdir(parents=True)
    (directory / "run_report.json").write_text(
        json.dumps(
            {
                "method": method,
                "effective_configuration": {
                    "search_seed": seed,
                    "retraining_seeds": [201, 202, 203],
                },
                "track_b_failed_retrainings": [{"retraining_seed": 201}] * diverged,
                "summary": {"metric_values": metric_values, "number_of_generations": 4},
            }
        ),
        encoding="utf-8",
    )


def test_each_method_gets_mean_and_deviation_over_its_seeds(tmp_path: Path) -> None:
    _write_report(tmp_path, _MULTICLASS, "deepneat", 101, track_a=0.90, track_b=0.80)
    _write_report(tmp_path, _MULTICLASS, "deepneat", 102, track_a=0.92, track_b=0.82)

    summary = build_summary(tmp_path, (_MULTICLASS,))

    row = next(line for line in summary.splitlines() if line.startswith("| deepneat"))
    assert "| 2 |" in row
    assert "0.9100 ± 0.0141" in row
    assert "0.8100 ± 0.0141" in row
    assert "0/6" in row


def test_diverged_retrainings_are_counted_and_operating_point_is_reported(
    tmp_path: Path,
) -> None:
    for seed, diverged in ((101, 1), (102, 0)):
        _write_report(
            tmp_path,
            _BINARY,
            "deepneat",
            seed,
            track_a=0.95,
            track_b=0.93,
            diverged=diverged,
            extra={
                f"track_a_seed{seed}_test_sensitivity": 0.97,
                f"track_a_seed{seed}_test_specificity": 0.88,
            },
        )

    summary = build_summary(tmp_path, (_BINARY,))

    assert "1/6" in summary
    assert "| deepneat | 0.9700 ± 0.0000 | 0.8800 ± 0.0000 |" in summary


def test_a_missing_series_is_reported_rather_than_failing(tmp_path: Path) -> None:
    summary = build_summary(tmp_path, (_BINARY,))

    assert "No finished runs under pneumonia/series-v2." in summary


def test_the_command_writes_the_markdown_file(tmp_path: Path) -> None:
    _write_report(tmp_path, _MULTICLASS, "exact", 101, track_a=0.98, track_b=None)
    output = tmp_path / "summary.md"

    main(["--results-root", str(tmp_path), "--output", str(output)])

    text = output.read_text(encoding="utf-8")
    assert "| exact | 1 |" in text
    assert "n/a" in text
