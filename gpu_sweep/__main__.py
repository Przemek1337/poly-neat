"""Command line entry point of the GPU dataset sweep.

Run from the repository root on the target machine:

    uv run python -m gpu_sweep --list-cells
    uv run python -m gpu_sweep --runs 5 --generations 30 --population 50
    uv run python -m gpu_sweep --resume gpu_sweep_results/<timestamp>
    uv run python -m gpu_sweep --analyze gpu_sweep_results/<timestamp>
    uv run python -m gpu_sweep --render-topology gpu_sweep_results/<timestamp>

Every run of every (dataset, algorithm) cell happens in its own child process
with a wall-clock timeout, so one hang or CUDA out-of-memory costs one run
rather than the sweep. Aggregation, statistics and figures are recomputed from
the stored records by --analyze; the network pictures are drawn separately by
--render-topology, which is a manual step on purpose.
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

from gpu_sweep.aggregation import load_run_records, write_json_atomically
from gpu_sweep.analyze import analyze_results_directory

ALGORITHM_NAMES: tuple[str, ...] = (
    "neat",
    "fsneat",
    "neatdbm",
    "cneat",
    "lneat",
    "hyperneat",
)

DEFAULT_NUMBER_OF_GENERATIONS = 5
DEFAULT_POPULATION_SIZE = 50
DEFAULT_NUMBER_OF_RUNS = 5
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
        "--runs",
        type=int,
        default=DEFAULT_NUMBER_OF_RUNS,
        help=f"repetitions of every cell (default: {DEFAULT_NUMBER_OF_RUNS})",
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
        help="internal: run exactly one run of one cell in this process",
    )
    parser.add_argument(
        "--run-index",
        type=int,
        default=0,
        help="internal: which repetition --single is performing",
    )
    parser.add_argument(
        "--result-path",
        default=None,
        help="internal: where --single writes its JSON record",
    )
    parser.add_argument(
        "--topology-dir",
        default=None,
        help="internal: where --single writes topology artifacts (first run only)",
    )
    parser.add_argument(
        "--analyze",
        default=None,
        metavar="RESULTS_DIR",
        help="recompute aggregates, figures and statistics from a finished results directory",
    )
    parser.add_argument(
        "--resume",
        default=None,
        metavar="RESULTS_DIR",
        help=(
            "continue a previous sweep: write into that directory and skip "
            "every run whose JSON record is already there"
        ),
    )
    parser.add_argument(
        "--render-topology",
        default=None,
        metavar="RESULTS_DIR",
        help=(
            "draw the network pictures from the topology records a finished "
            "sweep stored; safe to re-run as often as you like"
        ),
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


def select_runs(
    cells: list[tuple[str, str]], number_of_runs: int
) -> list[tuple[str, str, int]]:
    """Expand every cell into ``number_of_runs`` numbered repetitions.

    Runs of one cell stay adjacent, so a sweep stopped early has complete cells
    rather than one run of everything.

    Args:
        cells: ``(dataset_key, algorithm_name)`` pairs.
        number_of_runs: Repetitions per cell.

    Returns:
        ``(dataset_key, algorithm_name, run_index)`` triples.
    """
    return [
        (dataset_key, algorithm_name, run_index)
        for dataset_key, algorithm_name in cells
        for run_index in range(number_of_runs)
    ]


RUN_CSV_FIELD_NAMES: tuple[str, ...] = (
    "dataset",
    "algorithm",
    "run_index",
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
    "plateau_generation",
    "train_accuracy",
    "test_accuracy",
    "train_macro_f1",
    "test_macro_f1",
    "runtime_seconds",
    "peak_gpu_memory_megabytes",
    "phenotype_output_device",
    "evolution_seed",
    "split_seed",
    "error",
)

RUN_RECORD_FIELD_NAMES: tuple[str, ...] = (
    *RUN_CSV_FIELD_NAMES,
    "generation_best_fitnesses",
    "per_class_f1_scores",
)
"""Everything stored per run.

Only the two per-generation vectors stay out of ``runs.csv``, because a CSV
cell cannot hold a list; they live in the JSON records. The seeds *are* in the
CSV - they are plain integers, and they are the first thing anyone needs in
order to reproduce a single row."""


def build_run_record(
    dataset_key: str,
    algorithm_name: str,
    run_index: int,
    *,
    status: str,
    **field_values: object,
) -> dict[str, object]:
    """Build one fully populated run record.

    Every key in :data:`RUN_RECORD_FIELD_NAMES` is present - missing values are
    ``None`` - so a failed run lines up with a successful one and the CSV
    writer never has to guess.

    Args:
        dataset_key: Catalog key of the dataset.
        algorithm_name: Algorithm the run used.
        run_index: Which repetition of the cell this is, counting from zero.
        status: ``"ok"``, ``"error"``, or ``"timeout"``.
        **field_values: Any other record fields to fill in.

    Returns:
        The record, with ``fitness_delta`` and ``improved`` derived from the
        first/last generation fitness pair when both are present.

    Raises:
        KeyError: If a field name is not in :data:`RUN_RECORD_FIELD_NAMES`.
    """
    record: dict[str, object] = dict.fromkeys(RUN_RECORD_FIELD_NAMES)
    record["dataset"] = dataset_key
    record["algorithm"] = algorithm_name
    record["run_index"] = run_index
    record["status"] = status
    for field_name, value in field_values.items():
        if field_name not in record:
            raise KeyError(f"{field_name!r} is not a run record field")
        record[field_name] = value

    first_fitness = record["first_generation_best_fitness"]
    last_fitness = record["last_generation_best_fitness"]
    if isinstance(first_fitness, float) and isinstance(last_fitness, float):
        record["fitness_delta"] = last_fitness - first_fitness
        record["improved"] = last_fitness > first_fitness
    return record


def write_runs_csv(run_records: list[dict[str, object]], csv_path: Path) -> None:
    """Write every run record to ``csv_path`` in :data:`RUN_CSV_FIELD_NAMES` order."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file, fieldnames=list(RUN_CSV_FIELD_NAMES), extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(run_records)


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


def run_single_run(
    dataset_key: str,
    algorithm_name: str,
    run_index: int,
    arguments: argparse.Namespace,
) -> dict[str, object]:
    """Perform one run of one cell in this process. The child-process body.

    The split seed is ``--seed`` for every run, so all repetitions of a cell
    see identical data; the evolution seed is ``--seed + run_index``, so the
    spread between runs measures the search rather than the sampling.
    """
    from gpu_sweep.algorithm_runners import run_algorithm_on_dataset
    from gpu_sweep.convergence import find_plateau_generation
    from gpu_sweep.dataset_catalog import DATASET_SPECS, load_tabular_dataset
    from gpu_sweep.topology_report import write_topology_record

    device = resolve_cuda_device()
    torch.cuda.reset_peak_memory_stats(device)
    split_seed = arguments.seed
    evolution_seed = arguments.seed + run_index
    dataset = load_tabular_dataset(
        DATASET_SPECS[dataset_key],
        train_fraction=arguments.train_fraction,
        random_seed=split_seed,
    )
    shape_fields = {
        "number_of_samples": dataset.number_of_samples,
        "number_of_features": dataset.number_of_features,
        "number_of_classes": dataset.number_of_classes,
        "device": str(device),
        "evolution_seed": evolution_seed,
        "split_seed": split_seed,
    }
    try:
        outcome = run_algorithm_on_dataset(
            algorithm_name,
            dataset,
            device=device,
            population_size=arguments.population,
            number_of_generations=arguments.generations,
            random_seed=evolution_seed,
        )
    except Exception as run_error:  # noqa: BLE001 - a failed run is a result
        traceback.print_exc()
        return build_run_record(
            dataset_key,
            algorithm_name,
            run_index,
            status="error",
            error=f"{type(run_error).__name__}: {run_error}",
            **shape_fields,
        )

    if arguments.topology_dir is not None:
        # Store only. Drawing happens later, via --render-topology, so a run on
        # a wide dataset never spends its timeout inside matplotlib.
        for genome_label, genome in outcome.named_genomes.items():
            write_topology_record(
                genome,
                Path(arguments.topology_dir),
                f"{dataset_key}__{algorithm_name}__{genome_label}",
                title=f"{dataset_key} / {algorithm_name} / {genome_label}",
                structure_notes={
                    **outcome.structure_notes,
                    "dataset_features": dataset.number_of_features,
                    "dataset_classes": dataset.number_of_classes,
                    "evolution_seed": evolution_seed,
                },
            )

    return build_run_record(
        dataset_key,
        algorithm_name,
        run_index,
        status="ok",
        generations_completed=outcome.generations_completed,
        generation_best_fitnesses=outcome.generation_best_fitnesses,
        per_class_f1_scores=outcome.per_class_f1_scores,
        first_generation_best_fitness=outcome.first_generation_best_fitness,
        last_generation_best_fitness=outcome.last_generation_best_fitness,
        plateau_generation=find_plateau_generation(outcome.generation_best_fitnesses),
        runtime_seconds=outcome.runtime_seconds,
        phenotype_output_device=outcome.phenotype_output_device,
        peak_gpu_memory_megabytes=torch.cuda.max_memory_allocated(device) / (1024 * 1024),
        train_accuracy=outcome.metric_values.get("train_accuracy"),
        test_accuracy=outcome.metric_values.get("test_accuracy"),
        train_macro_f1=outcome.metric_values.get("train_macro_f1"),
        test_macro_f1=outcome.metric_values.get("test_macro_f1"),
        **shape_fields,
    )


def _child_process_command(
    dataset_key: str,
    algorithm_name: str,
    run_index: int,
    arguments: argparse.Namespace,
    result_path: Path,
    topology_directory: Path | None,
) -> list[str]:
    """Build the argv that re-invokes this module for exactly one run."""
    command = [
        sys.executable,
        "-m",
        "gpu_sweep",
        "--single",
        dataset_key,
        algorithm_name,
        "--run-index",
        str(run_index),
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
    if topology_directory is not None:
        command.extend(["--topology-dir", str(topology_directory)])
    return command


def run_sweep(arguments: argparse.Namespace, cells: list[tuple[str, str]]) -> Path:
    """Run every repetition of every cell in a child process, then analyse.

    Args:
        arguments: Parsed CLI arguments.
        cells: ``(dataset_key, algorithm_name)`` pairs to run.

    Returns:
        The directory the results were written to.
    """
    resolve_cuda_device()
    if arguments.resume is not None:
        output_directory = Path(arguments.resume)
        timestamp = output_directory.name
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_directory = Path("gpu_sweep_results") / timestamp
    runs_directory = output_directory / "runs"
    topology_directory = output_directory / "topology"
    runs_directory.mkdir(parents=True, exist_ok=True)

    runs = select_runs(cells, arguments.runs)
    (output_directory / "sweep_meta.json").write_text(
        json.dumps(
            {
                "timestamp": timestamp,
                "torch_version": torch.__version__,
                "torch_cuda_version": torch.version.cuda,
                "cuda_device_name": torch.cuda.get_device_name(0),
                "generations": arguments.generations,
                "population": arguments.population,
                "runs_per_cell": arguments.runs,
                "timeout_seconds": arguments.timeout_seconds,
                "train_fraction": arguments.train_fraction,
                "seed": arguments.seed,
                "number_of_cells": len(cells),
                "number_of_runs": len(runs),
                "seeding_rule": (
                    "split seed is --seed for every run; evolution seed is --seed + run_index"
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # Records are gathered from disk after the loop rather than accumulated
    # here, so a resumed sweep reports the runs it skipped alongside the ones
    # it just ran.
    for position, (dataset_key, algorithm_name, run_index) in enumerate(runs, start=1):
        result_path = runs_directory / f"{dataset_key}__{algorithm_name}__run{run_index}.json"
        if arguments.resume is not None and result_path.exists():
            print(
                f"[{position}/{len(runs)}] {dataset_key}/{algorithm_name} "
                f"run {run_index}: already recorded, skipping"
            )
            continue
        print(
            f"\n=== [{position}/{len(runs)}] {dataset_key}/{algorithm_name} "
            f"run {run_index} (timeout {arguments.timeout_seconds}s) ==="
        )
        command = _child_process_command(
            dataset_key,
            algorithm_name,
            run_index,
            arguments,
            result_path,
            topology_directory if run_index == 0 else None,
        )
        try:
            completed = subprocess.run(command, timeout=arguments.timeout_seconds, check=False)
            if result_path.exists():
                record = json.loads(result_path.read_text(encoding="utf-8"))
            else:
                record = build_run_record(
                    dataset_key,
                    algorithm_name,
                    run_index,
                    status="error",
                    error=f"child process exited with code {completed.returncode}",
                )
        except subprocess.TimeoutExpired:
            record = build_run_record(
                dataset_key,
                algorithm_name,
                run_index,
                status="timeout",
                error=f"exceeded {arguments.timeout_seconds}s",
            )
        if not result_path.exists():
            write_json_atomically(record, result_path)
        print(f"-> {record['status']} (test macro-F1 {record['test_macro_f1']})")

    # Rebuild runs.csv from the stored JSON rather than from an in-memory
    # list, so it is generated from exactly the source aggregates.csv reads.
    # Writing one from memory and the other from disk lets the two disagree
    # after a resumed or partially-failed sweep.
    write_runs_csv(load_run_records(runs_directory), output_directory / "runs.csv")
    analyze_results_directory(output_directory)
    return output_directory


def main(argument_list: list[str] | None = None) -> None:
    """Parse arguments and analyse, list cells, run one run, or run the sweep."""
    arguments = build_argument_parser().parse_args(argument_list)

    if arguments.render_topology is not None:
        from gpu_sweep.topology_report import render_topology_records

        topology_directory = Path(arguments.render_topology) / "topology"
        if not topology_directory.is_dir():
            raise SystemExit(f"error: no topology directory under {arguments.render_topology}")
        number_drawn = render_topology_records(topology_directory)
        print(f"drew {number_drawn} network pictures into {topology_directory}")
        return

    if arguments.analyze is not None:
        analyze_results_directory(Path(arguments.analyze))
        print(f"Analysis written into {arguments.analyze}")
        return

    if arguments.single is not None:
        if arguments.result_path is None:
            raise SystemExit("error: --single requires --result-path")
        dataset_key, algorithm_name = arguments.single
        record = run_single_run(dataset_key, algorithm_name, arguments.run_index, arguments)
        result_path = Path(arguments.result_path)
        write_json_atomically(record, result_path)
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
