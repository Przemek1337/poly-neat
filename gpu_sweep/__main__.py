"""Command line entry point of the GPU dataset sweep.

Run from the repository root on the target machine:

    uv run python -m gpu_sweep --list-cells
    uv run python -m gpu_sweep --generations 5 --population 50

Every (dataset, algorithm) cell runs in its own child process with a wall-clock
timeout, so one hang or CUDA out-of-memory does not end the sweep.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import subprocess
import sys
import traceback
from pathlib import Path

import torch

ALGORITHM_NAMES: tuple[str, ...] = (
    "neat",
    "fsneat",
    "neatdbm",
    "cneat",
    "lneat",
    "hyperneat",
    "exact",
)

DEFAULT_NUMBER_OF_GENERATIONS = 5
DEFAULT_POPULATION_SIZE = 50
DEFAULT_TIMEOUT_SECONDS = 600
DEFAULT_TRAIN_FRACTION = 0.66
DEFAULT_RANDOM_SEED = 42


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the sweep's argument parser."""
    parser = argparse.ArgumentParser(
        prog="gpu_sweep",
        description="Run every PolyNEAT algorithm against the paper's tabular datasets on CUDA",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="dataset keys to run (default: all in the catalog)",
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=None,
        choices=ALGORITHM_NAMES,
        help="algorithms to run (default: all seven)",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=DEFAULT_NUMBER_OF_GENERATIONS,
        help=f"generations per cell (default: {DEFAULT_NUMBER_OF_GENERATIONS})",
    )
    parser.add_argument(
        "--population",
        type=int,
        default=DEFAULT_POPULATION_SIZE,
        help=f"population size per cell (default: {DEFAULT_POPULATION_SIZE})",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
        help=f"wall-clock budget per cell (default: {DEFAULT_TIMEOUT_SECONDS})",
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=DEFAULT_TRAIN_FRACTION,
        help=f"share of rows used for training (default: {DEFAULT_TRAIN_FRACTION})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help=f"seed for the split and the evolution (default: {DEFAULT_RANDOM_SEED})",
    )
    parser.add_argument(
        "--list-cells",
        action="store_true",
        help="print the dataset/algorithm cells that would run, then exit",
    )
    parser.add_argument(
        "--single",
        nargs=2,
        metavar=("DATASET", "ALGORITHM"),
        default=None,
        help="internal: run exactly one cell in this process",
    )
    parser.add_argument(
        "--result-path",
        default=None,
        help="internal: where --single writes its JSON record",
    )
    return parser


def select_cells(
    dataset_keys: list[str] | None,
    algorithm_names: list[str] | None,
) -> list[tuple[str, str]]:
    """Return the ``(dataset_key, algorithm_name)`` pairs the sweep will run.

    Args:
        dataset_keys: Explicit dataset selection; ``None`` means the whole catalog.
        algorithm_names: Explicit algorithm selection; ``None`` means all seven.

    Returns:
        Cells in catalog order, algorithms in :data:`ALGORITHM_NAMES` order.

    Raises:
        SystemExit: With code 1 when a requested dataset key is not in the catalog.
    """
    from gpu_sweep.dataset_catalog import DATASET_SPECS

    selected_datasets = list(DATASET_SPECS) if dataset_keys is None else dataset_keys
    unknown_keys = [key for key in selected_datasets if key not in DATASET_SPECS]
    if unknown_keys:
        raise SystemExit(
            f"error: unknown dataset keys {unknown_keys}. Known keys: {sorted(DATASET_SPECS)}"
        )
    selected_algorithms = list(ALGORITHM_NAMES) if algorithm_names is None else algorithm_names
    return [
        (dataset_key, algorithm_name)
        for dataset_key in selected_datasets
        for algorithm_name in selected_algorithms
    ]


CELL_RESULT_FIELD_NAMES: tuple[str, ...] = (
    "dataset",
    "algorithm",
    "status",
    "device",
    "number_of_samples",
    "number_of_features",
    "number_of_classes",
    "generations_completed",
    "first_generation_best_fitness",
    "last_generation_best_fitness",
    "fitness_delta",
    "improved",
    "train_accuracy",
    "generalizability_accuracy",
    "test_accuracy",
    "runtime_seconds",
    "peak_gpu_memory_megabytes",
    "phenotype_output_device",
    "error",
)


def build_cell_record(
    dataset_key: str,
    algorithm_name: str,
    *,
    status: str,
    **field_values: object,
) -> dict[str, object]:
    """Build one fully populated result row.

    Every column in :data:`CELL_RESULT_FIELD_NAMES` is present in the returned
    record - missing values are ``None`` - so the CSV writer never has to guess
    and a failed cell lines up with a successful one.

    Args:
        dataset_key: Catalog key of the dataset.
        algorithm_name: Algorithm the cell ran.
        status: ``"ok"``, ``"error"``, or ``"timeout"``.
        **field_values: Any other column values to fill in.

    Returns:
        The record, with ``fitness_delta`` and ``improved`` derived from the
        first/last generation fitness pair when both are present.
    """
    record: dict[str, object] = dict.fromkeys(CELL_RESULT_FIELD_NAMES)
    record["dataset"] = dataset_key
    record["algorithm"] = algorithm_name
    record["status"] = status
    for field_name, value in field_values.items():
        if field_name not in record:
            raise KeyError(f"{field_name!r} is not a cell result column")
        record[field_name] = value

    first_fitness = record["first_generation_best_fitness"]
    last_fitness = record["last_generation_best_fitness"]
    if isinstance(first_fitness, float) and isinstance(last_fitness, float):
        record["fitness_delta"] = last_fitness - first_fitness
        record["improved"] = last_fitness > first_fitness
    return record


def write_results_csv(cell_records: list[dict[str, object]], csv_path: Path) -> None:
    """Write every cell record to ``csv_path`` in :data:`CELL_RESULT_FIELD_NAMES` order."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(CELL_RESULT_FIELD_NAMES))
        writer.writeheader()
        writer.writerows(cell_records)


def resolve_cuda_device() -> torch.device:
    """Return the CUDA device, or exit when the machine has none.

    Raises:
        SystemExit: With code 1 when CUDA is unavailable. The sweep exists to
            answer a GPU question, so a CPU fallback would be a wrong answer.
    """
    if not torch.cuda.is_available():
        raise SystemExit(
            "error: CUDA is not available. This sweep is GPU-only - install a "
            "CUDA-enabled torch build (see pyproject.toml's pytorch-cu126 index) "
            "and run on the target machine."
        )
    return torch.device("cuda")


def run_single_cell(
    dataset_key: str, algorithm_name: str, arguments: argparse.Namespace
) -> dict[str, object]:
    """Run one cell in this process and return its record. The child-process body."""
    from gpu_sweep.algorithm_runners import run_algorithm_on_dataset
    from gpu_sweep.dataset_catalog import DATASET_SPECS, load_tabular_dataset

    device = resolve_cuda_device()
    torch.cuda.reset_peak_memory_stats(device)
    dataset = load_tabular_dataset(
        DATASET_SPECS[dataset_key],
        train_fraction=arguments.train_fraction,
        random_seed=arguments.seed,
    )
    shape_fields = {
        "number_of_samples": dataset.number_of_samples,
        "number_of_features": dataset.number_of_features,
        "number_of_classes": dataset.number_of_classes,
        "device": str(device),
    }
    try:
        outcome = run_algorithm_on_dataset(
            algorithm_name,
            dataset,
            device=device,
            population_size=arguments.population,
            number_of_generations=arguments.generations,
            random_seed=arguments.seed,
        )
    except Exception as run_error:  # noqa: BLE001 - a failed cell is a result
        traceback.print_exc()
        return build_cell_record(
            dataset_key,
            algorithm_name,
            status="error",
            error=f"{type(run_error).__name__}: {run_error}",
            **shape_fields,
        )

    return build_cell_record(
        dataset_key,
        algorithm_name,
        status="ok",
        generations_completed=outcome.generations_completed,
        first_generation_best_fitness=outcome.first_generation_best_fitness,
        last_generation_best_fitness=outcome.last_generation_best_fitness,
        runtime_seconds=outcome.runtime_seconds,
        phenotype_output_device=outcome.phenotype_output_device,
        peak_gpu_memory_megabytes=torch.cuda.max_memory_allocated(device) / (1024 * 1024),
        train_accuracy=outcome.metric_values.get("train_accuracy"),
        generalizability_accuracy=outcome.metric_values.get("generalizability_accuracy"),
        test_accuracy=outcome.metric_values.get("test_accuracy"),
        **shape_fields,
    )


def _child_process_command(
    dataset_key: str,
    algorithm_name: str,
    arguments: argparse.Namespace,
    result_path: Path,
) -> list[str]:
    """Build the argv that re-invokes this module for exactly one cell."""
    return [
        sys.executable,
        "-m",
        "gpu_sweep",
        "--single",
        dataset_key,
        algorithm_name,
        "--result-path",
        str(result_path),
        "--generations",
        str(arguments.generations),
        "--population",
        str(arguments.population),
        "--train-fraction",
        str(arguments.train_fraction),
        "--seed",
        str(arguments.seed),
    ]


def run_sweep(arguments: argparse.Namespace, cells: list[tuple[str, str]]) -> Path:
    """Run every cell in a child process and write the results.

    Args:
        arguments: Parsed CLI arguments.
        cells: ``(dataset_key, algorithm_name)`` pairs to run.

    Returns:
        The directory the results were written to.
    """
    resolve_cuda_device()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_directory = Path("gpu_sweep_results") / timestamp
    cells_directory = output_directory / "cells"
    cells_directory.mkdir(parents=True, exist_ok=True)

    (output_directory / "sweep_meta.json").write_text(
        json.dumps(
            {
                "timestamp": timestamp,
                "torch_version": torch.__version__,
                "torch_cuda_version": torch.version.cuda,
                "cuda_device_name": torch.cuda.get_device_name(0),
                "generations": arguments.generations,
                "population": arguments.population,
                "timeout_seconds": arguments.timeout_seconds,
                "train_fraction": arguments.train_fraction,
                "seed": arguments.seed,
                "number_of_cells": len(cells),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    cell_records: list[dict[str, object]] = []
    for cell_index, (dataset_key, algorithm_name) in enumerate(cells, start=1):
        result_path = cells_directory / f"{dataset_key}__{algorithm_name}.json"
        print(
            f"\n=== [{cell_index}/{len(cells)}] {dataset_key}/{algorithm_name} "
            f"(timeout {arguments.timeout_seconds}s) ==="
        )
        command = _child_process_command(dataset_key, algorithm_name, arguments, result_path)
        try:
            completed = subprocess.run(command, timeout=arguments.timeout_seconds, check=False)
            if result_path.exists():
                record = json.loads(result_path.read_text(encoding="utf-8"))
            else:
                record = build_cell_record(
                    dataset_key,
                    algorithm_name,
                    status="error",
                    error=f"child process exited with code {completed.returncode}",
                )
        except subprocess.TimeoutExpired:
            record = build_cell_record(
                dataset_key,
                algorithm_name,
                status="timeout",
                error=f"exceeded {arguments.timeout_seconds}s",
            )
        if not result_path.exists():
            result_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        cell_records.append(record)
        print(f"-> {record['status']} (fitness delta {record['fitness_delta']})")

    write_results_csv(cell_records, output_directory / "results.csv")
    return output_directory


def main(argument_list: list[str] | None = None) -> None:
    """Parse arguments and list cells, run one cell, or run the whole sweep."""
    arguments = build_argument_parser().parse_args(argument_list)

    if arguments.single is not None:
        if arguments.result_path is None:
            raise SystemExit("error: --single requires --result-path")
        dataset_key, algorithm_name = arguments.single
        record = run_single_cell(dataset_key, algorithm_name, arguments)
        result_path = Path(arguments.result_path)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        print(json.dumps(record, indent=2))
        return

    cells = select_cells(arguments.datasets, arguments.algorithms)
    if arguments.list_cells:
        for dataset_key, algorithm_name in cells:
            print(f"{dataset_key}/{algorithm_name}")
        print(f"{len(cells)} cells")
        return

    output_directory = run_sweep(arguments, cells)
    print(f"\nResults written to {output_directory}")


if __name__ == "__main__":
    main()
