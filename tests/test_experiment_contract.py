from __future__ import annotations

import dataclasses

import pytest

from examples._experiment import ExperimentReport, print_experiment_report


def _report() -> ExperimentReport:
    return ExperimentReport(
        metric_values={"test_accuracy": 0.94},
        number_of_generations=100,
        runtime_seconds=41.2,
    )


def test_experiment_report_is_frozen() -> None:
    report = _report()
    with pytest.raises(dataclasses.FrozenInstanceError):
        report.number_of_generations = 0  # type: ignore[misc]


def test_print_experiment_report_prints_every_metric_and_the_totals(
    capsys: pytest.CaptureFixture[str],
) -> None:
    print_experiment_report(_report())
    printed = capsys.readouterr().out
    assert "test_accuracy" in printed
    assert "0.9400" in printed
    assert "100" in printed
    assert "41.2" in printed
