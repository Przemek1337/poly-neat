"""Turning stored run records into per-cell aggregates and a Friedman matrix."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from gpu_sweep.aggregation import (
    AGGREGATE_FIELD_NAMES,
    aggregate_run_records,
    build_metric_matrix,
    load_run_records,
    summarize_values,
    write_aggregates_csv,
)


def make_run_record(
    dataset: str, algorithm: str, run_index: int, macro_f1: float, status: str = "ok"
) -> dict[str, object]:
    return {
        "dataset": dataset,
        "algorithm": algorithm,
        "run_index": run_index,
        "status": status,
        "test_macro_f1": macro_f1,
        "test_accuracy": macro_f1 + 0.05,
        "train_macro_f1": macro_f1 + 0.1,
        "train_accuracy": macro_f1 + 0.15,
        "runtime_seconds": 1.0,
        "generations_completed": 3,
        "plateau_generation": 1,
        "number_of_features": 8,
        "number_of_classes": 2,
        "number_of_samples": 40,
    }


def test_summarize_values_reports_the_five_number_summary() -> None:
    summary = summarize_values([0.2, 0.4, 0.6])

    assert summary["mean"] == pytest.approx(0.4)
    assert summary["median"] == pytest.approx(0.4)
    assert summary["minimum"] == pytest.approx(0.2)
    assert summary["maximum"] == pytest.approx(0.6)
    assert summary["count"] == 3
    # population standard deviation of [0.2, 0.4, 0.6]
    assert summary["standard_deviation"] == pytest.approx(0.16329931, abs=1e-6)


def test_summarize_values_of_a_single_value_has_zero_spread() -> None:
    summary = summarize_values([0.5])

    assert summary["mean"] == pytest.approx(0.5)
    assert summary["standard_deviation"] == pytest.approx(0.0)


def test_summarize_values_of_nothing_is_empty_not_an_error() -> None:
    summary = summarize_values([])

    assert summary["count"] == 0
    assert summary["mean"] is None


def test_load_run_records_reads_every_json_file(tmp_path: Path) -> None:
    for run_index in range(2):
        (tmp_path / f"colon__neat__run{run_index}.json").write_text(
            json.dumps(make_run_record("colon", "neat", run_index, 0.5)), encoding="utf-8"
        )

    records = load_run_records(tmp_path)

    assert len(records) == 2
    assert {record["run_index"] for record in records} == {0, 1}


def test_aggregate_run_records_averages_each_cell_over_its_runs() -> None:
    records = [
        make_run_record("colon", "neat", 0, 0.4),
        make_run_record("colon", "neat", 1, 0.6),
    ]

    aggregates = aggregate_run_records(records)

    assert len(aggregates) == 1
    assert aggregates[0]["dataset"] == "colon"
    assert aggregates[0]["test_macro_f1_mean"] == pytest.approx(0.5)
    assert aggregates[0]["test_macro_f1_standard_deviation"] == pytest.approx(0.1)
    assert aggregates[0]["successful_runs"] == 2
    assert aggregates[0]["failed_runs"] == 0


def test_aggregate_run_records_ignores_failed_runs_but_counts_them() -> None:
    records = [
        make_run_record("colon", "neat", 0, 0.4),
        make_run_record("colon", "neat", 1, 0.0, status="timeout"),
    ]

    aggregates = aggregate_run_records(records)

    assert aggregates[0]["test_macro_f1_mean"] == pytest.approx(0.4)
    assert aggregates[0]["successful_runs"] == 1
    assert aggregates[0]["failed_runs"] == 1


def test_aggregate_records_use_only_declared_fields() -> None:
    aggregates = aggregate_run_records([make_run_record("colon", "neat", 0, 0.4)])

    assert set(aggregates[0]) <= set(AGGREGATE_FIELD_NAMES)


def test_write_aggregates_csv_writes_one_row_per_cell(tmp_path: Path) -> None:
    csv_path = tmp_path / "aggregates.csv"
    aggregates = aggregate_run_records(
        [make_run_record("colon", "neat", 0, 0.4), make_run_record("colon", "cneat", 0, 0.7)]
    )

    write_aggregates_csv(aggregates, csv_path)

    rows = list(csv.DictReader(csv_path.read_text(encoding="utf-8").splitlines()))
    assert {row["algorithm"] for row in rows} == {"neat", "cneat"}


def test_build_metric_matrix_lays_datasets_down_rows_and_algorithms_across() -> None:
    aggregates = aggregate_run_records(
        [
            make_run_record("colon", "neat", 0, 0.4),
            make_run_record("colon", "cneat", 0, 0.7),
            make_run_record("thyroid", "neat", 0, 0.6),
            make_run_record("thyroid", "cneat", 0, 0.5),
        ]
    )

    matrix = build_metric_matrix(aggregates, "test_macro_f1")

    assert matrix.dataset_keys == ["colon", "thyroid"]
    assert matrix.algorithm_names == ["cneat", "neat"]
    # pytest.approx refuses nested sequences, so the rows are compared one by one.
    assert matrix.values[0] == pytest.approx([0.7, 0.4])
    assert matrix.values[1] == pytest.approx([0.5, 0.6])
    assert matrix.excluded_dataset_keys == []


def test_build_metric_matrix_drops_a_dataset_missing_an_algorithm() -> None:
    aggregates = aggregate_run_records(
        [
            make_run_record("colon", "neat", 0, 0.4),
            make_run_record("colon", "cneat", 0, 0.7),
            make_run_record("thyroid", "neat", 0, 0.6),
            make_run_record("thyroid", "cneat", 0, 0.0, status="error"),
        ]
    )

    matrix = build_metric_matrix(aggregates, "test_macro_f1")

    assert matrix.dataset_keys == ["colon"]
    assert matrix.excluded_dataset_keys == ["thyroid"]
