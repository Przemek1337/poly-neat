"""Orchestrator behaviour: record shape, CSV writing, cell selection."""

from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path

import pytest

from gpu_sweep import __main__ as sweep_main
from gpu_sweep.__main__ import (
    CELL_RESULT_FIELD_NAMES,
    build_argument_parser,
    build_cell_record,
    select_cells,
    write_results_csv,
)


def test_select_cells_pairs_every_dataset_with_every_algorithm() -> None:
    cells = select_cells(["cleveland", "colon"], ["neat", "exact"])

    assert cells == [
        ("cleveland", "neat"),
        ("cleveland", "exact"),
        ("colon", "neat"),
        ("colon", "exact"),
    ]


def test_select_cells_rejects_an_unknown_dataset_key() -> None:
    with pytest.raises(SystemExit, match="unknown dataset keys"):
        select_cells(["not_a_dataset"], None)


def test_build_cell_record_fills_every_csv_column() -> None:
    record = build_cell_record("colon", "neat", status="ok", test_accuracy=0.75)

    assert set(record) == set(CELL_RESULT_FIELD_NAMES)
    assert record["dataset"] == "colon"
    assert record["status"] == "ok"
    assert record["test_accuracy"] == 0.75


def test_build_cell_record_reports_improvement_from_the_fitness_pair() -> None:
    record = build_cell_record(
        "colon",
        "neat",
        status="ok",
        first_generation_best_fitness=0.40,
        last_generation_best_fitness=0.55,
    )

    assert record["fitness_delta"] == pytest.approx(0.15)
    assert record["improved"] is True


def test_write_results_csv_writes_a_header_and_one_row_per_cell(tmp_path: Path) -> None:
    csv_path = tmp_path / "results.csv"

    write_results_csv(
        [
            build_cell_record("colon", "neat", status="ok"),
            build_cell_record("colon", "exact", status="timeout"),
        ],
        csv_path,
    )

    rows = list(csv.DictReader(csv_path.read_text(encoding="utf-8").splitlines()))
    assert [row["algorithm"] for row in rows] == ["neat", "exact"]
    assert rows[1]["status"] == "timeout"


def _sweep_arguments(**overrides: object) -> object:
    """Parse a default argument namespace, then apply overrides."""
    arguments = build_argument_parser().parse_args([])
    for name, value in overrides.items():
        setattr(arguments, name, value)
    return arguments


def test_child_process_command_carries_every_run_knob() -> None:
    arguments = _sweep_arguments(generations=3, population=7, train_fraction=0.5, seed=11)

    command = sweep_main._child_process_command(
        "colon", "neat", arguments, Path("out") / "colon__neat.json"
    )

    assert command[1:3] == ["-m", "gpu_sweep"]
    assert command[3:6] == ["--single", "colon", "neat"]
    for flag, value in (
        ("--generations", "3"),
        ("--population", "7"),
        ("--train-fraction", "0.5"),
        ("--seed", "11"),
    ):
        assert command[command.index(flag) + 1] == value


def test_run_sweep_collects_child_records_and_survives_a_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The child-process loop, with the subprocess call itself stubbed out.

    Covers what unit tests of the record helpers cannot: that a child's JSON is
    read back, that a timed-out cell still lands in the CSV, and that the run
    keeps going after one.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sweep_main, "resolve_cuda_device", lambda: "cuda")
    monkeypatch.setattr(sweep_main.torch.cuda, "get_device_name", lambda index: "NVIDIA L4")

    def fake_subprocess_run(command: list[str], **_: object) -> object:
        dataset_key, algorithm_name = command[4], command[5]
        if algorithm_name == "exact":
            raise subprocess.TimeoutExpired(command, 1)
        result_path = Path(command[command.index("--result-path") + 1])
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(
            json.dumps(
                build_cell_record(
                    dataset_key,
                    algorithm_name,
                    status="ok",
                    first_generation_best_fitness=0.2,
                    last_generation_best_fitness=0.4,
                )
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(sweep_main.subprocess, "run", fake_subprocess_run)

    output_directory = sweep_main.run_sweep(
        _sweep_arguments(), [("colon", "neat"), ("colon", "exact")]
    )

    rows = list(
        csv.DictReader((output_directory / "results.csv").read_text(encoding="utf-8").splitlines())
    )
    assert [row["status"] for row in rows] == ["ok", "timeout"]
    assert rows[0]["improved"] == "True"
    assert (output_directory / "cells" / "colon__exact.json").exists()
    assert json.loads((output_directory / "sweep_meta.json").read_text(encoding="utf-8"))[
        "cuda_device_name"
    ] == "NVIDIA L4"
