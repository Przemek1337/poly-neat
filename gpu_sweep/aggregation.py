"""Reducing stored run records to per-cell aggregates and a Friedman matrix.

Aggregation is deliberately a separate pass over files on disk rather than
something the sweep accumulates in memory. Every number in ``aggregates.csv``
can therefore be recomputed by hand from the JSON files in ``runs/``, which is
the point: a mean nobody can check is a mean nobody should trust.
"""

from __future__ import annotations

import csv
import json
import math
import os
import statistics
from dataclasses import dataclass
from pathlib import Path

METRIC_NAMES: tuple[str, ...] = (
    "test_macro_f1",
    "test_accuracy",
    "train_macro_f1",
    "train_accuracy",
    "runtime_seconds",
    "generations_completed",
    "plateau_generation",
)
"""Per-run scalars that get a mean/spread summary in the aggregates."""

SUMMARY_SUFFIXES: tuple[str, ...] = (
    "mean",
    "standard_deviation",
    "median",
    "minimum",
    "maximum",
    "count",
)
"""``count`` is included on purpose: it is the denominator of ``mean``, and a
mean whose denominator is not written down cannot be checked by hand. It can
differ from ``successful_runs`` when a metric is missing on an otherwise
successful run - ``plateau_generation`` is ``None`` for a run with no
generations."""

STANDARD_DEVIATION_IS_POPULATION = True
"""Recorded in the output so a hand-check cannot silently disagree.

``statistics.pstdev`` divides by ``n``. Excel's ``STDEV()`` and numpy's default
``std(ddof=0)`` do not agree with each other on this, and at ``n = 5`` the two
conventions differ by about 12 percent - large enough to look like a bug when
someone recomputes a column by hand."""

AGGREGATE_FIELD_NAMES: tuple[str, ...] = (
    "dataset",
    "algorithm",
    "successful_runs",
    "failed_runs",
    "number_of_samples",
    "number_of_features",
    "number_of_classes",
    *[
        f"{metric_name}_{suffix}"
        for metric_name in METRIC_NAMES
        for suffix in SUMMARY_SUFFIXES
    ],
)


def write_json_atomically(payload: dict[str, object], json_path: Path) -> None:
    """Write ``payload`` to ``json_path`` via a temporary file and one rename.

    A run record is written by a child process that can be killed at any moment
    - by the sweep's own timeout, by Ctrl-C, by the OOM killer. Writing in place
    can leave a half-written file, and a single truncated record would make
    every future ``--analyze`` of that directory raise ``JSONDecodeError``. The
    rename is atomic on POSIX and on Windows via ``os.replace``, so a reader
    sees either the old file or the complete new one.
    """
    json_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = json_path.with_suffix(json_path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary_path, json_path)


def load_run_records(runs_directory: Path) -> list[dict[str, object]]:
    """Read every ``*.json`` run record in ``runs_directory``.

    A record that will not parse is reported and skipped rather than allowed to
    abort the load. The whole point of storing every run separately is that one
    damaged file costs one run, not the entire sweep's analysis.

    Args:
        runs_directory: Directory the sweep wrote its per-run files into.

    Returns:
        The records, sorted by dataset, algorithm and run index so that
        aggregation and CSV output are deterministic.
    """
    records: list[dict[str, object]] = []
    for json_path in sorted(runs_directory.glob("*.json")):
        try:
            records.append(json.loads(json_path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError) as read_error:
            print(f"skipping unreadable run record {json_path.name}: {read_error}")
    records.sort(
        key=lambda record: (
            str(record.get("dataset")),
            str(record.get("algorithm")),
            int(record.get("run_index") or 0),
        )
    )
    return records


def summarize_values(values: list[float]) -> dict[str, float | None]:
    """Mean, population standard deviation, median, min, max and count.

    The population standard deviation matches what ``benchmarks/run_benchmark.py``
    reports, so the two harnesses do not disagree about the same word.

    Args:
        values: Observations to summarise; may be empty.

    Non-finite observations are dropped before summarising: a NaN passes an
    ``is not None`` check and would otherwise flow into the Friedman matrix,
    producing exactly the corrupted comparison the completeness rule exists to
    prevent.

    Returns:
        A dict whose statistics are ``None`` when there is nothing to summarise.
        ``standard_deviation`` is the population standard deviation (divisor
        ``n``); see :data:`STANDARD_DEVIATION_IS_POPULATION`.
    """
    finite_values = [value for value in values if math.isfinite(value)]
    if not finite_values:
        return {
            "mean": None,
            "standard_deviation": None,
            "median": None,
            "minimum": None,
            "maximum": None,
            "count": 0,
        }
    values = finite_values
    return {
        "mean": statistics.fmean(values),
        "standard_deviation": statistics.pstdev(values),
        "median": statistics.median(values),
        "minimum": min(values),
        "maximum": max(values),
        "count": len(values),
    }


def aggregate_run_records(run_records: list[dict[str, object]]) -> list[dict[str, object]]:
    """Group runs by ``(dataset, algorithm)`` and summarise each group.

    Only runs with ``status == "ok"`` feed the summaries; failures are counted
    separately, so a cell whose mean rests on two of five runs is visibly
    different from one resting on all five.

    Args:
        run_records: Records as stored by the sweep.

    Returns:
        One aggregate record per cell, sorted by dataset then algorithm.
    """
    records_by_cell: dict[tuple[str, str], list[dict[str, object]]] = {}
    for record in run_records:
        key = (str(record.get("dataset")), str(record.get("algorithm")))
        records_by_cell.setdefault(key, []).append(record)

    aggregates: list[dict[str, object]] = []
    for (dataset_key, algorithm_name), cell_records in sorted(records_by_cell.items()):
        successful = [record for record in cell_records if record.get("status") == "ok"]
        aggregate: dict[str, object] = {
            "dataset": dataset_key,
            "algorithm": algorithm_name,
            "successful_runs": len(successful),
            "failed_runs": len(cell_records) - len(successful),
        }
        for shape_field in ("number_of_samples", "number_of_features", "number_of_classes"):
            observed_shapes = [
                record[shape_field]
                for record in cell_records
                if record.get(shape_field) is not None
            ]
            aggregate[shape_field] = observed_shapes[0] if observed_shapes else None

        for metric_name in METRIC_NAMES:
            observed = [
                float(record[metric_name])
                for record in successful
                if record.get(metric_name) is not None
            ]
            summary = summarize_values(observed)
            for suffix in SUMMARY_SUFFIXES:
                aggregate[f"{metric_name}_{suffix}"] = summary[suffix]
        aggregates.append(aggregate)
    return aggregates


def write_aggregates_csv(aggregate_records: list[dict[str, object]], csv_path: Path) -> None:
    """Write per-cell aggregates in :data:`AGGREGATE_FIELD_NAMES` order."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file, fieldnames=list(AGGREGATE_FIELD_NAMES), extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(aggregate_records)


@dataclass(frozen=True)
class MetricMatrix:
    """A complete algorithm-by-problem matrix, ready for the Friedman test.

    Attributes:
        dataset_keys: Row labels, in order.
        algorithm_names: Column labels, in order.
        values: ``values[row][column]`` is that cell's mean metric.
        excluded_dataset_keys: Datasets dropped because some algorithm has no
            successful run there. Friedman needs a complete matrix, and
            inventing a value for a run that never finished would be worse than
            saying which datasets could not take part.
    """

    dataset_keys: list[str]
    algorithm_names: list[str]
    values: list[list[float]]
    excluded_dataset_keys: list[str]


def build_metric_matrix(
    aggregate_records: list[dict[str, object]], metric_name: str
) -> MetricMatrix:
    """Lay the per-cell means out as a dataset-by-algorithm matrix.

    Args:
        aggregate_records: Output of :func:`aggregate_run_records`.
        metric_name: Base metric name, e.g. ``"test_macro_f1"``; that metric's
            ``_mean`` column is what lands in the matrix.

    Returns:
        The assembled :class:`MetricMatrix`.
    """
    mean_field = f"{metric_name}_mean"
    algorithm_names = sorted({str(record["algorithm"]) for record in aggregate_records})
    mean_by_cell = {
        (str(record["dataset"]), str(record["algorithm"])): record.get(mean_field)
        for record in aggregate_records
    }

    included_dataset_keys: list[str] = []
    excluded_dataset_keys: list[str] = []
    for dataset_key in sorted({str(record["dataset"]) for record in aggregate_records}):
        cell_values = [
            mean_by_cell.get((dataset_key, algorithm_name))
            for algorithm_name in algorithm_names
        ]
        # ``is None`` alone is not enough: a NaN satisfies it and would be
        # handed straight to the ranking, where every comparison against it is
        # False and the resulting ranks are meaningless.
        if any(
            value is None or not math.isfinite(float(value)) for value in cell_values
        ):
            excluded_dataset_keys.append(dataset_key)
        else:
            included_dataset_keys.append(dataset_key)

    values = [
        [
            float(mean_by_cell[(dataset_key, algorithm_name)])  # type: ignore[arg-type]
            for algorithm_name in algorithm_names
        ]
        for dataset_key in included_dataset_keys
    ]
    return MetricMatrix(
        dataset_keys=included_dataset_keys,
        algorithm_names=algorithm_names,
        values=values,
        excluded_dataset_keys=excluded_dataset_keys,
    )
