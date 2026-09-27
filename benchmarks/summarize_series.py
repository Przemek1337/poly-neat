"""Summarise the frozen result series of the image benchmarks in one table per problem.

Invoke as a module from the repository root::

    uv run python -m benchmarks.summarize_series [--results-root benchmarks/results]

Every finished run of a series leaves a ``run_report.json``; this reads them all
and reports, per method, the mean and sample standard deviation over search
seeds of: the selection metric the search optimised (optimistic by
construction), the track A test metric (the model the method delivers), the
track B test metric (the selected topology retrained from scratch under the
shared recipe, averaged over its retrainings), how many track B retrainings
diverged, the median selected parameter count, the mean number of generations
and the mean fraction of failed search evaluations. For the pneumonia problem it
adds track A sensitivity and specificity at the threshold chosen on the
threshold split.

The tables are written to a Markdown file next to the results, so the numbers in
the thesis can be traced back to the reports they came from. Nothing here reads
or writes a lock, and this module sits outside the digested implementation, so
running it cannot invalidate a frozen series.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

_DEFAULT_RESULTS_ROOT = Path(__file__).parent / "results"


@dataclass(frozen=True)
class SeriesSpec:
    """Where one problem's series lives and which metrics it reports.

    Attributes:
        problem: Display name of the problem.
        series_directory: Series directory relative to the results root.
        test_metric: Suffix of the per-model test metric, e.g. ``auroc``.
        selection_metric: Metric the search selected on.
        reports_operating_point: Whether the report carries sensitivity and
            specificity at a chosen threshold (binary problems only).
    """

    problem: str
    series_directory: str
    test_metric: str
    selection_metric: str
    reports_operating_point: bool = False


SERIES = (
    SeriesSpec("RTG", "pneumonia/series-v2", "auroc", "search_validation_auroc", True),
    SeriesSpec("MNIST", "mnist/series-v1", "accuracy", "validation_accuracy"),
    SeriesSpec("CIFAR-10", "cifar10/series-v1", "accuracy", "validation_accuracy"),
)


def load_series_reports(series_directory: Path) -> dict[str, list[dict]]:
    """Group every finished run of one series by method, in seed order."""
    reports: dict[str, list[dict]] = defaultdict(list)
    for path in sorted(series_directory.glob("runs/*/seed_*/run_report.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        reports[report["method"]].append(report)
    for method_reports in reports.values():
        method_reports.sort(key=lambda report: report["effective_configuration"]["search_seed"])
    return dict(reports)


def mean_and_deviation(values: list[float | None]) -> str:
    """``mean ± sample sd`` over the defined values, or ``n/a`` when there are none."""
    defined = [value for value in values if value is not None]
    if not defined:
        return "n/a"
    deviation = statistics.stdev(defined) if len(defined) > 1 else 0.0
    return f"{statistics.mean(defined):.4f} ± {deviation:.4f}"


def _track_a_values(reports: list[dict], metric_name: str) -> list[float | None]:
    return [
        report["summary"]["metric_values"].get(
            f"track_a_seed{report['effective_configuration']['search_seed']}_test_{metric_name}"
        )
        for report in reports
    ]


def summarize_series(spec: SeriesSpec, reports: dict[str, list[dict]]) -> str:
    """One Markdown section for one problem."""
    lines = [
        f"## {spec.problem}",
        "",
        f"Test metric: {spec.test_metric}; selection metric: {spec.selection_metric}. "
        "Values are mean ± sample standard deviation over search seeds.",
        "",
        "| method | seeds | selection (val) | track A test | track B test | track B diverged "
        "| params (median) | generations | failed evaluations |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for method, method_reports in sorted(reports.items()):
        metric_values = [report["summary"]["metric_values"] for report in method_reports]
        diverged = sum(
            len(report.get("track_b_failed_retrainings", [])) for report in method_reports
        )
        retrainings = sum(
            len(report["effective_configuration"]["retraining_seeds"])
            for report in method_reports
        )
        selection = [values[spec.selection_metric] for values in metric_values]
        track_b = [values.get(f"track_b_mean_test_{spec.test_metric}") for values in metric_values]
        parameters = statistics.median(
            values["selected_parameter_count"] for values in metric_values
        )
        generations = statistics.mean(
            report["summary"]["number_of_generations"] for report in method_reports
        )
        failed = statistics.mean(values["failed_evaluation_fraction"] for values in metric_values)
        cells = [
            method,
            str(len(method_reports)),
            mean_and_deviation(selection),
            mean_and_deviation(_track_a_values(method_reports, spec.test_metric)),
            mean_and_deviation(track_b),
            f"{diverged}/{retrainings}",
            f"{parameters:.0f}",
            f"{generations:.1f}",
            f"{failed:.3f}",
        ]
        lines.append("| " + " | ".join(cells) + " |")
    if spec.reports_operating_point:
        lines += [
            "",
            "Track A at the threshold chosen on the threshold split:",
            "",
            "| method | sensitivity | specificity |",
            "|---|---|---|",
        ]
        for method, method_reports in sorted(reports.items()):
            lines.append(
                f"| {method} "
                f"| {mean_and_deviation(_track_a_values(method_reports, 'sensitivity'))} "
                f"| {mean_and_deviation(_track_a_values(method_reports, 'specificity'))} |"
            )
    return "\n".join(lines)


def build_summary(results_root: Path, series: tuple[SeriesSpec, ...] = SERIES) -> str:
    """The whole Markdown document, skipping problems whose series is absent."""
    sections = ["# Result series summary"]
    for spec in series:
        reports = load_series_reports(results_root / spec.series_directory)
        if reports:
            sections.append(summarize_series(spec, reports))
        else:
            sections.append(f"## {spec.problem}\n\nNo finished runs under {spec.series_directory}.")
    return "\n\n".join(sections) + "\n"


def main(argument_list: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results-root", type=Path, default=_DEFAULT_RESULTS_ROOT)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Markdown file to write (default: <results-root>/series_summary.md)",
    )
    arguments = parser.parse_args(argument_list)
    output = arguments.output or arguments.results_root / "series_summary.md"
    summary = build_summary(arguments.results_root)
    output.write_text(summary, encoding="utf-8")
    print(summary)
    print(f"Summary written to {output}")


if __name__ == "__main__":
    main()
