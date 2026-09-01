"""The analysis pass, driven from a hand-built results directory."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from gpu_sweep.analyze import analyze_results_directory


def write_run_file(
    runs_directory: Path,
    dataset: str,
    algorithm: str,
    run_index: int,
    macro_f1: float,
    *,
    status: str = "ok",
    curve: list[float] | None = None,
) -> None:
    runs_directory.mkdir(parents=True, exist_ok=True)
    (runs_directory / f"{dataset}__{algorithm}__run{run_index}.json").write_text(
        json.dumps(
            {
                "dataset": dataset,
                "algorithm": algorithm,
                "run_index": run_index,
                "status": status,
                "test_macro_f1": macro_f1,
                "test_accuracy": macro_f1,
                "train_macro_f1": macro_f1,
                "train_accuracy": macro_f1,
                "runtime_seconds": 1.0,
                "generations_completed": 3,
                "plateau_generation": 1,
                "number_of_samples": 40,
                "number_of_features": 8,
                "number_of_classes": 2,
                "generation_best_fitnesses": curve or [0.1, 0.3, 0.4],
            }
        ),
        encoding="utf-8",
    )


def build_results_directory(tmp_path: Path) -> Path:
    """Six datasets by three algorithms, with a clear and consistent ordering."""
    results_directory = tmp_path / "results"
    runs_directory = results_directory / "runs"
    scores = {"alpha": 0.90, "beta": 0.60, "gamma": 0.30}
    for dataset_index in range(6):
        for algorithm_name, base_score in scores.items():
            for run_index in range(2):
                write_run_file(
                    runs_directory,
                    f"dataset{dataset_index}",
                    algorithm_name,
                    run_index,
                    base_score + 0.01 * run_index,
                )
    return results_directory


def test_analyze_writes_aggregates(tmp_path: Path) -> None:
    results_directory = build_results_directory(tmp_path)

    analyze_results_directory(results_directory)

    rows = list(
        csv.DictReader(
            (results_directory / "aggregates.csv").read_text(encoding="utf-8").splitlines()
        )
    )
    assert len(rows) == 18
    alpha_rows = [row for row in rows if row["algorithm"] == "alpha"]
    assert float(alpha_rows[0]["test_macro_f1_mean"]) == pytest.approx(0.905)


def test_analyze_writes_the_friedman_report_for_each_metric(tmp_path: Path) -> None:
    results_directory = build_results_directory(tmp_path)

    analyze_results_directory(results_directory)

    report = json.loads(
        (results_directory / "statistics" / "friedman_test_macro_f1.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["number_of_algorithms"] == 3
    assert report["number_of_problems"] == 6
    assert report["average_rank_by_algorithm"]["alpha"] == pytest.approx(1.0)
    assert report["average_rank_by_algorithm"]["gamma"] == pytest.approx(3.0)
    assert report["control_method"] == "alpha"
    assert report["omnibus_rejects_null_hypothesis"] is True
    assert (results_directory / "statistics" / "friedman_test_accuracy.json").exists()


def test_analyze_writes_both_holm_families(tmp_path: Path) -> None:
    results_directory = build_results_directory(tmp_path)

    analyze_results_directory(results_directory)

    control_rows = list(
        csv.DictReader(
            (results_directory / "statistics" / "holm_control_test_macro_f1.csv")
            .read_text(encoding="utf-8")
            .splitlines()
        )
    )
    all_pairs_rows = list(
        csv.DictReader(
            (results_directory / "statistics" / "holm_allpairs_test_macro_f1.csv")
            .read_text(encoding="utf-8")
            .splitlines()
        )
    )

    assert len(control_rows) == 2  # k - 1
    assert len(all_pairs_rows) == 3  # k(k - 1)/2
    assert {row["right_name"] for row in control_rows} == {"alpha"}


def test_analyze_writes_a_ranks_table(tmp_path: Path) -> None:
    results_directory = build_results_directory(tmp_path)

    analyze_results_directory(results_directory)

    rows = list(
        csv.DictReader(
            (results_directory / "statistics" / "ranks_test_macro_f1.csv")
            .read_text(encoding="utf-8")
            .splitlines()
        )
    )
    assert {row["algorithm"] for row in rows} == {"alpha", "beta", "gamma"}
    alpha_row = next(row for row in rows if row["algorithm"] == "alpha")
    assert float(alpha_row["average_rank"]) == pytest.approx(1.0)


def test_analyze_draws_a_figure_per_cell_and_per_dataset(tmp_path: Path) -> None:
    results_directory = build_results_directory(tmp_path)

    analyze_results_directory(results_directory)

    convergence_directory = results_directory / "convergence"
    assert (convergence_directory / "dataset0__alpha.png").exists()
    assert (convergence_directory / "dataset_dataset0.png").exists()


def test_analyze_records_datasets_dropped_from_the_matrix(tmp_path: Path) -> None:
    results_directory = build_results_directory(tmp_path)
    # Break one cell so its dataset cannot take part in a complete matrix.
    write_run_file(
        results_directory / "runs", "dataset0", "beta", 0, 0.0, status="timeout"
    )
    write_run_file(
        results_directory / "runs", "dataset0", "beta", 1, 0.0, status="timeout"
    )

    analyze_results_directory(results_directory)

    report = json.loads(
        (results_directory / "statistics" / "friedman_test_macro_f1.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["excluded_datasets"] == ["dataset0"]
    assert report["number_of_problems"] == 5


def test_analyze_reports_instead_of_failing_when_too_little_survives(
    tmp_path: Path,
) -> None:
    results_directory = tmp_path / "thin"
    write_run_file(results_directory / "runs", "only", "alpha", 0, 0.5)

    analyze_results_directory(results_directory)

    report = json.loads(
        (results_directory / "statistics" / "friedman_test_macro_f1.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["omnibus_rejects_null_hypothesis"] is None
    assert "not enough" in report["guidance"].lower()
