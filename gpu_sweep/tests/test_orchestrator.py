"""Orchestrator behaviour: record shape, CSV writing, run selection, the child loop."""

from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path

import pytest

from gpu_sweep import __main__ as sweep_main
from gpu_sweep.__main__ import (
    RUN_CSV_FIELD_NAMES,
    RUN_RECORD_FIELD_NAMES,
    build_argument_parser,
    build_run_record,
    select_cells,
    select_runs,
    write_runs_csv,
)


def test_select_cells_pairs_every_dataset_with_every_algorithm() -> None:
    cells = select_cells(["cleveland", "colon"], ["neat", "cneat"])

    assert cells == [
        ("cleveland", "neat"),
        ("cleveland", "cneat"),
        ("colon", "neat"),
        ("colon", "cneat"),
    ]


def test_select_cells_rejects_an_unknown_dataset_key() -> None:
    with pytest.raises(SystemExit, match="unknown dataset keys"):
        select_cells(["not_a_dataset"], None)


def test_exact_is_no_longer_a_selectable_algorithm() -> None:
    assert "exact" not in sweep_main.ALGORITHM_NAMES
    assert len(sweep_main.ALGORITHM_NAMES) == 6


def test_select_runs_expands_every_cell_into_numbered_runs() -> None:
    runs = select_runs([("colon", "neat"), ("colon", "cneat")], 3)

    assert runs == [
        ("colon", "neat", 0),
        ("colon", "neat", 1),
        ("colon", "neat", 2),
        ("colon", "cneat", 0),
        ("colon", "cneat", 1),
        ("colon", "cneat", 2),
    ]


def test_build_run_record_fills_every_record_field() -> None:
    record = build_run_record("colon", "neat", 2, status="ok", test_macro_f1=0.75)

    assert set(record) == set(RUN_RECORD_FIELD_NAMES)
    assert record["dataset"] == "colon"
    assert record["run_index"] == 2
    assert record["test_macro_f1"] == 0.75


def test_build_run_record_rejects_an_unknown_field() -> None:
    with pytest.raises(KeyError, match="not a run record field"):
        build_run_record("colon", "neat", 0, status="ok", nonsense_column=1)


def test_build_run_record_reports_improvement_from_the_fitness_pair() -> None:
    record = build_run_record(
        "colon",
        "neat",
        0,
        status="ok",
        first_generation_best_fitness=0.40,
        last_generation_best_fitness=0.55,
    )

    assert record["fitness_delta"] == pytest.approx(0.15)
    assert record["improved"] is True


def test_csv_columns_are_a_scalar_subset_of_the_record_fields() -> None:
    assert set(RUN_CSV_FIELD_NAMES) <= set(RUN_RECORD_FIELD_NAMES)
    assert "generation_best_fitnesses" not in RUN_CSV_FIELD_NAMES
    assert "per_class_f1_scores" not in RUN_CSV_FIELD_NAMES
    assert "test_macro_f1" in RUN_CSV_FIELD_NAMES


def test_write_runs_csv_writes_a_header_and_one_row_per_run(tmp_path: Path) -> None:
    csv_path = tmp_path / "runs.csv"

    write_runs_csv(
        [
            build_run_record("colon", "neat", 0, status="ok"),
            build_run_record("colon", "neat", 1, status="timeout"),
        ],
        csv_path,
    )

    rows = list(csv.DictReader(csv_path.read_text(encoding="utf-8").splitlines()))
    assert [row["run_index"] for row in rows] == ["0", "1"]
    assert rows[1]["status"] == "timeout"


def _sweep_arguments(**overrides: object) -> object:
    """Parse a default argument namespace, then apply overrides."""
    arguments = build_argument_parser().parse_args([])
    for name, value in overrides.items():
        setattr(arguments, name, value)
    return arguments


def test_child_process_command_carries_the_run_index_and_every_knob() -> None:
    arguments = _sweep_arguments(generations=3, population=7, train_fraction=0.5, seed=11)

    command = sweep_main._child_process_command(
        "colon", "neat", 2, arguments, Path("out") / "colon__neat__run2.json", None
    )

    assert command[3:7] == ["--single", "colon", "neat", "--run-index"]
    assert command[7] == "2"
    for flag, value in (
        ("--generations", "3"),
        ("--population", "7"),
        ("--train-fraction", "0.5"),
        ("--seed", "11"),
    ):
        assert command[command.index(flag) + 1] == value


def test_child_process_command_asks_for_topology_only_on_the_first_run() -> None:
    arguments = _sweep_arguments()

    with_topology = sweep_main._child_process_command(
        "colon", "neat", 0, arguments, Path("r.json"), Path("out") / "topology"
    )
    without_topology = sweep_main._child_process_command(
        "colon", "neat", 1, arguments, Path("r.json"), None
    )

    assert "--topology-dir" in with_topology
    assert "--topology-dir" not in without_topology


def test_run_sweep_collects_child_records_and_survives_a_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The child-process loop, with the subprocess call itself stubbed out.

    Covers what unit tests of the record helpers cannot: that a child's JSON is
    read back, that a timed-out run still lands in the CSV, and that the sweep
    keeps going after one.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sweep_main, "resolve_cuda_device", lambda: "cuda")
    monkeypatch.setattr(sweep_main.torch.cuda, "get_device_name", lambda index: "NVIDIA L4")
    monkeypatch.setattr(sweep_main, "analyze_results_directory", lambda *a, **k: None)

    def fake_subprocess_run(command: list[str], **_: object) -> object:
        dataset_key, algorithm_name = command[4], command[5]
        run_index = int(command[7])
        if run_index == 1:
            raise subprocess.TimeoutExpired(command, 1)
        result_path = Path(command[command.index("--result-path") + 1])
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(
            json.dumps(
                build_run_record(
                    dataset_key,
                    algorithm_name,
                    run_index,
                    status="ok",
                    first_generation_best_fitness=0.2,
                    last_generation_best_fitness=0.4,
                    generation_best_fitnesses=[0.2, 0.3, 0.4],
                )
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(sweep_main.subprocess, "run", fake_subprocess_run)

    output_directory = sweep_main.run_sweep(_sweep_arguments(runs=2), [("colon", "neat")])

    rows = list(
        csv.DictReader((output_directory / "runs.csv").read_text(encoding="utf-8").splitlines())
    )
    assert [row["status"] for row in rows] == ["ok", "timeout"]
    assert (output_directory / "runs" / "colon__neat__run1.json").exists()
    assert json.loads((output_directory / "sweep_meta.json").read_text(encoding="utf-8"))[
        "runs_per_cell"
    ] == 2
