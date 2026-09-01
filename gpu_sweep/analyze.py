"""Turning a finished results directory into aggregates, figures and statistics.

This pass reads only files - the per-run JSON records the sweep stored - and
writes only files. Nothing here re-runs an evolution, so ``--analyze`` can be
pointed at a finished directory as often as needed: to change alpha, to check a
mean by hand, or to redraw a figure. That separation is what makes the reported
statistics auditable rather than merely asserted.

Statistical procedure follows Derrac et al. (2011); see
``gpu_sweep/statistical_tests.py`` for the formulas and their citations.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path

from gpu_sweep.aggregation import (
    aggregate_run_records,
    build_metric_matrix,
    load_run_records,
    write_aggregates_csv,
)
from gpu_sweep.convergence import (
    mean_curve,
    plot_cell_convergence,
    plot_dataset_convergence,
    running_best_curve,
)
from gpu_sweep.statistical_tests import (
    DEFAULT_ALPHA,
    Comparison,
    FriedmanResult,
    control_method_index,
    friedman_test,
    post_hoc_all_pairs_comparisons,
    post_hoc_control_comparisons,
)

STATISTICS_METRIC_NAMES: tuple[str, ...] = ("test_macro_f1", "test_accuracy")
"""Metrics the omnibus and post-hoc tests are run over, both higher-is-better."""

MINIMUM_ALGORITHMS_FOR_TESTING = 3
"""Two algorithms would make Friedman degenerate into a sign test, and leave
Holm with a single hypothesis to adjust - which adjusts nothing."""

MINIMUM_PROBLEMS_FOR_TESTING = 2

COMPARISON_FIELD_NAMES: tuple[str, ...] = (
    "left_name",
    "right_name",
    "z_statistic",
    "unadjusted_p_value",
    "holm_adjusted_p_value",
    "is_rejected",
)


def write_comparisons_csv(comparisons: list[Comparison], csv_path: Path) -> None:
    """Write one post-hoc family, already sorted by unadjusted p-value."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(COMPARISON_FIELD_NAMES))
        writer.writeheader()
        for comparison in comparisons:
            writer.writerow(asdict(comparison))


def write_ranks_csv(
    algorithm_names: list[str], friedman_result: FriedmanResult, csv_path: Path
) -> None:
    """Write each algorithm's average rank, best rank first."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {"algorithm": algorithm_name, "average_rank": average_rank}
        for algorithm_name, average_rank in zip(
            algorithm_names, friedman_result.average_ranks, strict=True
        )
    ]
    rows.sort(key=lambda row: float(row["average_rank"]))
    with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["algorithm", "average_rank"])
        writer.writeheader()
        writer.writerows(rows)


def algorithms_blocking_the_matrix(
    aggregate_records: list[dict[str, object]], metric_name: str
) -> list[str]:
    """Name the algorithms responsible for the most excluded datasets.

    One algorithm that fails everywhere excludes *every* dataset, because the
    completeness rule is per dataset. The resulting empty matrix looks like a
    data famine when it is really one bad column, so the report names the
    culprit instead of leaving the reader to work it out.

    Args:
        aggregate_records: Output of ``aggregate_run_records``.
        metric_name: Base metric name whose ``_mean`` column is checked.

    Returns:
        Algorithm names with at least one missing cell, worst first.
    """
    mean_field = f"{metric_name}_mean"
    missing_count_by_algorithm: dict[str, int] = {}
    for record in aggregate_records:
        algorithm_name = str(record["algorithm"])
        missing_count_by_algorithm.setdefault(algorithm_name, 0)
        if record.get(mean_field) is None:
            missing_count_by_algorithm[algorithm_name] += 1
    return [
        algorithm_name
        for algorithm_name, missing in sorted(
            missing_count_by_algorithm.items(), key=lambda item: (-item[1], item[0])
        )
        if missing > 0
    ]


def _write_insufficient_data_report(
    json_path: Path,
    *,
    metric_name: str,
    algorithm_names: list[str],
    dataset_keys: list[str],
    excluded_dataset_keys: list[str],
    blocking_algorithms: list[str],
) -> None:
    """Record why no test was run, rather than writing nothing at all."""
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(
            {
                "metric": metric_name,
                "number_of_algorithms": len(algorithm_names),
                "number_of_problems": len(dataset_keys),
                "algorithms": algorithm_names,
                "problems": dataset_keys,
                "excluded_datasets": excluded_dataset_keys,
                "algorithms_with_missing_cells": blocking_algorithms,
                "omnibus_rejects_null_hypothesis": None,
                "guidance": (
                    "not enough complete data to run the Friedman test: it needs at "
                    f"least {MINIMUM_ALGORITHMS_FOR_TESTING} algorithms scored on at "
                    f"least {MINIMUM_PROBLEMS_FOR_TESTING} problems, with every "
                    "algorithm having at least one successful run on each."
                    + (
                        " Every dataset was excluded because these algorithms have "
                        f"missing cells: {blocking_algorithms}. Re-run the sweep for "
                        "those, or exclude them with --algorithms, and the remaining "
                        "columns will form a complete matrix."
                        if blocking_algorithms
                        else ""
                    )
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def write_friedman_report(
    dataset_keys: list[str],
    algorithm_names: list[str],
    excluded_dataset_keys: list[str],
    friedman_result: FriedmanResult,
    json_path: Path,
    *,
    metric_name: str,
    alpha: float,
) -> None:
    """Write the omnibus result, its ranks, and the reading guidance.

    Args:
        dataset_keys: Problems that took part.
        algorithm_names: Algorithms compared, in matrix column order.
        excluded_dataset_keys: Problems dropped for having an incomplete row.
        friedman_result: Output of the omnibus test.
        json_path: Target file.
        metric_name: Metric the test ran over.
        alpha: Family-wise significance level.
    """
    control_index = control_method_index(friedman_result.average_ranks)
    rejects = friedman_result.iman_davenport_p_value <= alpha
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(
            {
                "metric": metric_name,
                "alpha": alpha,
                "number_of_algorithms": friedman_result.number_of_algorithms,
                "number_of_problems": friedman_result.number_of_problems,
                "problems": dataset_keys,
                "excluded_datasets": excluded_dataset_keys,
                "average_rank_by_algorithm": dict(
                    zip(algorithm_names, friedman_result.average_ranks, strict=True)
                ),
                "control_method": algorithm_names[control_index],
                "friedman_chi_squared_statistic": friedman_result.chi_squared_statistic,
                "friedman_chi_squared_p_value": friedman_result.chi_squared_p_value,
                "iman_davenport_statistic": friedman_result.iman_davenport_statistic,
                "iman_davenport_p_value": friedman_result.iman_davenport_p_value,
                "omnibus_rejects_null_hypothesis": rejects,
                "guidance": (
                    "The omnibus test rejects the null hypothesis of equal medians, "
                    "so the Holm post-hoc tables identify which specific differences "
                    "carry it."
                    if rejects
                    else (
                        "The omnibus test does NOT reject the null hypothesis at this "
                        "alpha. Derrac et al. (2011), section 6.2 advise checking that "
                        "rejection before reading post-hoc results; the Holm tables are "
                        "written for completeness but no difference should be claimed "
                        "from them."
                    )
                ),
                "reference": (
                    "Derrac, Garcia, Molina & Herrera (2011), Swarm and Evolutionary "
                    "Computation 1(1), 3-18. Friedman: eq. 2; Iman-Davenport: eq. 3; "
                    "post-hoc z: eq. 12; Holm: section 4.3."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def run_statistical_analysis(
    aggregate_records: list[dict[str, object]],
    statistics_directory: Path,
    *,
    alpha: float = DEFAULT_ALPHA,
) -> None:
    """Run Friedman plus both Holm families over every configured metric.

    Args:
        aggregate_records: Output of ``aggregate_run_records``.
        statistics_directory: Directory the reports are written into.
        alpha: Family-wise significance level.
    """
    for metric_name in STATISTICS_METRIC_NAMES:
        matrix = build_metric_matrix(aggregate_records, metric_name)
        report_path = statistics_directory / f"friedman_{metric_name}.json"

        if (
            len(matrix.algorithm_names) < MINIMUM_ALGORITHMS_FOR_TESTING
            or len(matrix.dataset_keys) < MINIMUM_PROBLEMS_FOR_TESTING
        ):
            _write_insufficient_data_report(
                report_path,
                metric_name=metric_name,
                algorithm_names=matrix.algorithm_names,
                dataset_keys=matrix.dataset_keys,
                excluded_dataset_keys=matrix.excluded_dataset_keys,
                blocking_algorithms=algorithms_blocking_the_matrix(
                    aggregate_records, metric_name
                ),
            )
            continue

        # Macro-F1 and accuracy are both higher-is-better, so rank 1 is the
        # largest value.
        friedman_result = friedman_test(matrix.values, higher_is_better=True)
        write_friedman_report(
            matrix.dataset_keys,
            matrix.algorithm_names,
            matrix.excluded_dataset_keys,
            friedman_result,
            report_path,
            metric_name=metric_name,
            alpha=alpha,
        )
        write_ranks_csv(
            matrix.algorithm_names,
            friedman_result,
            statistics_directory / f"ranks_{metric_name}.csv",
        )
        write_comparisons_csv(
            post_hoc_control_comparisons(
                friedman_result, matrix.algorithm_names, alpha=alpha
            ),
            statistics_directory / f"holm_control_{metric_name}.csv",
        )
        write_comparisons_csv(
            post_hoc_all_pairs_comparisons(
                friedman_result, matrix.algorithm_names, alpha=alpha
            ),
            statistics_directory / f"holm_allpairs_{metric_name}.csv",
        )


def draw_convergence_figures(
    run_records: list[dict[str, object]], convergence_directory: Path
) -> None:
    """Draw one figure per cell and one per dataset.

    Args:
        run_records: Records as stored by the sweep.
        convergence_directory: Directory the figures are written into.
    """
    curves_by_cell: dict[tuple[str, str], list[list[float]]] = {}
    for record in run_records:
        if record.get("status") != "ok":
            continue
        curve = record.get("generation_best_fitnesses") or []
        if not curve:
            continue
        key = (str(record["dataset"]), str(record["algorithm"]))
        curves_by_cell.setdefault(key, []).append([float(value) for value in curve])

    mean_curves_by_dataset: dict[str, dict[str, list[float]]] = {}
    for (dataset_key, algorithm_name), curves in sorted(curves_by_cell.items()):
        # The plateau caption is appended by plot_cell_convergence itself, from
        # the same describe_plateau call that labels the line. Do not build a
        # second one here - that is how a figure ends up titled "plateau at 12"
        # over a legend reading "still improving at 12".
        plot_cell_convergence(
            curves,
            convergence_directory / f"{dataset_key}__{algorithm_name}.png",
            title=f"{dataset_key} / {algorithm_name}",
        )
        mean_curves_by_dataset.setdefault(dataset_key, {})[algorithm_name] = mean_curve(
            [running_best_curve(curve) for curve in curves]
        )

    for dataset_key, mean_curve_by_algorithm in sorted(mean_curves_by_dataset.items()):
        plot_dataset_convergence(
            mean_curve_by_algorithm,
            convergence_directory / f"dataset_{dataset_key}.png",
            dataset_key=dataset_key,
        )


def analyze_results_directory(
    results_directory: Path, *, alpha: float = DEFAULT_ALPHA
) -> None:
    """Recompute every derived artifact from the stored run records.

    Safe to call repeatedly: it reads ``runs/`` and overwrites everything it
    produces, so changing alpha or fixing a figure never means re-running the
    sweep.

    Args:
        results_directory: A sweep output directory holding a ``runs/`` folder.
        alpha: Family-wise significance level for the post-hoc procedures.
    """
    results_directory = Path(results_directory)
    run_records = load_run_records(results_directory / "runs")
    if not run_records:
        print(f"no run records under {results_directory / 'runs'}; nothing to analyse")
        return

    aggregate_records = aggregate_run_records(run_records)
    write_aggregates_csv(aggregate_records, results_directory / "aggregates.csv")
    draw_convergence_figures(run_records, results_directory / "convergence")
    run_statistical_analysis(
        aggregate_records, results_directory / "statistics", alpha=alpha
    )
    print(
        f"analysed {len(run_records)} runs across {len(aggregate_records)} cells "
        f"into {results_directory}"
    )
