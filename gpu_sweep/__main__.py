"""Command line entry point of the GPU dataset sweep.

Run from the repository root on the target machine:

    uv run python -m gpu_sweep --list-cells
    uv run python -m gpu_sweep --runs 5 --generations 30 --population 50
    uv run python -m gpu_sweep --resume gpu_sweep_results/<timestamp>
    uv run python -m gpu_sweep --analyze gpu_sweep_results/<timestamp>
    uv run python -m gpu_sweep --render-topology gpu_sweep_results/<timestamp>

The defaults are small on purpose. The thesis protocol needs every knob spelled
out:

    uv run python -u -m gpu_sweep --runs 15 --generations 150 --population 50 \
        --timeout-seconds 36000 --train-fraction 0.66 --seed 42

``--device cpu`` runs the same pipeline without CUDA, for smoke tests only.

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
import hashlib
import json
import subprocess
import sys
import traceback
from pathlib import Path

import numpy
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
SPLIT_SEED_OFFSET = 1000
"""Run ``i`` splits with seed ``--seed + SPLIT_SEED_OFFSET + i`` and evolves with
``--seed + i``. The offset keeps the two seeds apart: both feed
``numpy.random.default_rng``, and equal seeds would hand the split and the
evolution the same random stream."""

PROTOCOL_LABELS: dict[str, str] = {
    "feature_scaling_fit": "train",
    "split_seed_rule": "seed + 1000 + run_index",
    "prediction_rule": "argmax of the output activations, ties to the lowest class index",
}
"""Stored in every run record and in ``sweep_meta.json`` so records of this
protocol cannot be mixed up with the first sweep's, which scaled over all rows
and used one split for every run."""


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
        help=(
            f"base seed: run i evolves with seed+i and splits with "
            f"seed+{SPLIT_SEED_OFFSET}+i (default: {DEFAULT_RANDOM_SEED})"
        ),
    )
    parser.add_argument(
        "--device",
        choices=("cuda", "cpu"),
        default="cuda",
        help=(
            "device to evolve on (default: cuda). cpu exists for smoke tests only: "
            "its numbers are not comparable with a CUDA sweep"
        ),
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
        help="internal: where --single writes the scored network of its run",
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
    "device_name",
    "evolution_seed",
    "split_seed",
    "constant_train_feature_count",
    "train_tie_count",
    "test_tie_count",
    "output_sums_match_forward_pass",
    "feature_scaling_fit",
    "prediction_rule",
    "error",
)

RUN_RECORD_FIELD_NAMES: tuple[str, ...] = (
    *RUN_CSV_FIELD_NAMES,
    "generation_best_fitnesses",
    "generation_species_counts",
    "per_class_f1_scores",
)
"""Everything stored per run.

Only the per-generation and per-class vectors stay out of ``runs.csv``,
because a CSV cell cannot hold a list; they live in the JSON records. The seeds
*are* in the CSV - they are plain integers, and they are the first thing anyone
needs in order to reproduce a single row.

Per-patient arrays (output sums, outputs, predictions) are too large for a
record and go to ``predictions/<dataset>__<algorithm>__run<i>.npz``; the split
positions and fitted scaling go to ``splits/<dataset>__run<i>.json``."""


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


def resolve_device(device_name: str) -> torch.device:
    """Return the device ``--device`` asks for.

    ``cuda`` goes through :func:`resolve_cuda_device`, so a machine without
    CUDA still stops with an error instead of silently falling back.
    """
    if device_name == "cpu":
        return torch.device("cpu")
    return torch.device(resolve_cuda_device())


def device_display_name(device: torch.device) -> str:
    """Human-readable name of ``device`` for the records."""
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    return "cpu"


def split_seed_for_run(base_seed: int, run_index: int) -> int:
    """Seed of run ``run_index``'s train/test split; the same for every algorithm."""
    return base_seed + SPLIT_SEED_OFFSET + run_index


def write_split_record(dataset: object, split_seed: int, split_path: Path) -> None:
    """Store a run's split positions and fitted scaling, once per (dataset, run).

    Every algorithm of a run sees the same split, so the first one to get here
    writes the file and the others leave it alone.
    """
    if split_path.exists():
        return
    write_json_atomically(
        {
            "dataset": dataset.dataset_key,
            "split_seed": split_seed,
            "train_positions": [int(position) for position in dataset.train_positions],
            "test_positions": [int(position) for position in dataset.test_positions],
            "feature_scaling": dataset.feature_scaling.to_serializable_dict(),
        },
        split_path,
    )


def write_per_patient_outputs(per_patient_outputs: dict[str, object], output_path: Path) -> None:
    """Store the per-patient arrays of one run as a compressed ``.npz``."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(output_path.stem + ".partial.npz")
    numpy.savez_compressed(temporary_path, **per_patient_outputs)
    temporary_path.replace(output_path)


def run_single_run(
    dataset_key: str,
    algorithm_name: str,
    run_index: int,
    arguments: argparse.Namespace,
) -> dict[str, object]:
    """Perform one run of one cell in this process. The child-process body.

    The split seed is ``--seed + SPLIT_SEED_OFFSET + run_index``, so every run
    sees a different stratified split while all algorithms of the same run see
    the same one; the evolution seed is ``--seed + run_index``. The spread
    between runs therefore covers both the search and the sampling of the test
    patients.

    Besides the JSON record it writes, next to the ``runs`` directory that
    ``--result-path`` points into: the run's split and scaling
    (``splits/``), the per-patient outputs (``predictions/``) and, when
    ``--topology-dir`` is given, the scored network (``topology/``).
    """
    from gpu_sweep.algorithm_runners import run_algorithm_on_dataset
    from gpu_sweep.convergence import find_plateau_generation
    from gpu_sweep.dataset_catalog import DATASET_SPECS, load_tabular_dataset
    from gpu_sweep.topology_report import write_topology_record

    device = resolve_device(arguments.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    split_seed = split_seed_for_run(arguments.seed, run_index)
    evolution_seed = arguments.seed + run_index
    dataset = load_tabular_dataset(
        DATASET_SPECS[dataset_key],
        train_fraction=arguments.train_fraction,
        random_seed=split_seed,
    )
    # --result-path is <output>/runs/<name>.json; the other per-run artifacts
    # live beside the runs directory.
    output_directory = Path(arguments.result_path).parent.parent
    write_split_record(
        dataset, split_seed, output_directory / "splits" / f"{dataset_key}__run{run_index}.json"
    )
    shape_fields = {
        "number_of_samples": dataset.number_of_samples,
        "number_of_features": dataset.number_of_features,
        "number_of_classes": dataset.number_of_classes,
        "device": str(device),
        "device_name": device_display_name(device),
        "evolution_seed": evolution_seed,
        "split_seed": split_seed,
        "constant_train_feature_count": len(dataset.feature_scaling.constant_feature_indices),
        "feature_scaling_fit": PROTOCOL_LABELS["feature_scaling_fit"],
        "prediction_rule": PROTOCOL_LABELS["prediction_rule"],
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
                f"{dataset_key}__{algorithm_name}__run{run_index}__{genome_label}",
                title=f"{dataset_key} / {algorithm_name} / run {run_index} / {genome_label}",
                structure_notes={
                    **outcome.structure_notes,
                    "dataset_features": dataset.number_of_features,
                    "dataset_classes": dataset.number_of_classes,
                    "run_index": run_index,
                    "evolution_seed": evolution_seed,
                    "split_seed": split_seed,
                },
            )

    if outcome.per_patient_outputs:
        write_per_patient_outputs(
            outcome.per_patient_outputs,
            output_directory
            / "predictions"
            / f"{dataset_key}__{algorithm_name}__run{run_index}.npz",
        )

    return build_run_record(
        dataset_key,
        algorithm_name,
        run_index,
        status="ok",
        generations_completed=outcome.generations_completed,
        generation_best_fitnesses=outcome.generation_best_fitnesses,
        generation_species_counts=outcome.generation_species_counts,
        per_class_f1_scores=outcome.per_class_f1_scores,
        train_tie_count=outcome.tie_counts.get("train"),
        test_tie_count=outcome.tie_counts.get("test"),
        output_sums_match_forward_pass=outcome.output_sums_match_forward_pass,
        first_generation_best_fitness=outcome.first_generation_best_fitness,
        last_generation_best_fitness=outcome.last_generation_best_fitness,
        plateau_generation=find_plateau_generation(outcome.generation_best_fitnesses),
        runtime_seconds=outcome.runtime_seconds,
        phenotype_output_device=outcome.phenotype_output_device,
        peak_gpu_memory_megabytes=(
            torch.cuda.max_memory_allocated(device) / (1024 * 1024)
            if device.type == "cuda"
            else None
        ),
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
        "--device",
        arguments.device,
    ]
    if topology_directory is not None:
        command.extend(["--topology-dir", str(topology_directory)])
    return command


def recorded_status(result_path: Path) -> str | None:
    """Status stored in a run record, or ``None`` when it cannot be read."""
    try:
        return json.loads(result_path.read_text(encoding="utf-8")).get("status")
    except (json.JSONDecodeError, OSError):
        return None


def set_aside_failed_record(result_path: Path, failed_directory: Path) -> None:
    """Move a non-``ok`` record into ``failed_directory`` with a timestamp suffix."""
    failed_directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    result_path.replace(failed_directory / f"{result_path.stem}__{stamp}.json")


def _git_output(*git_arguments: str) -> str | None:
    """Output of a git command run in this repository, or ``None`` if git fails."""
    repository_root = Path(__file__).resolve().parent.parent
    try:
        completed = subprocess.run(
            ["git", "-C", str(repository_root), *git_arguments],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def dataset_file_checksums(dataset_keys: list[str]) -> dict[str, str]:
    """SHA-256 of every cached raw file the given datasets read."""
    from gpu_sweep.dataset_catalog import DATASET_SPECS, DEFAULT_CACHE_ROOT

    checksums: dict[str, str] = {}
    for dataset_key in dataset_keys:
        for file_name, _ in DATASET_SPECS[dataset_key].raw_files:
            file_path = DEFAULT_CACHE_ROOT / dataset_key / file_name
            if file_path.exists():
                checksums[f"{dataset_key}/{file_name}"] = hashlib.sha256(
                    file_path.read_bytes()
                ).hexdigest()
    return checksums


def update_sweep_meta(
    meta_path: Path,
    *,
    timestamp: str,
    arguments: argparse.Namespace,
    device: torch.device,
    cells: list[tuple[str, str]],
    number_of_runs: int,
) -> None:
    """Write ``sweep_meta.json``, appending to it on ``--resume`` instead of overwriting.

    The top-level fields describe the sweep as first started. Every invocation
    - the first one and each resume - adds an entry to ``invocations`` with its
    own arguments, device, library versions and repository state, so a sweep
    finished over several sessions keeps the full history.
    """
    status_lines = _git_output("status", "--porcelain")
    invocation = {
        "started_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "resume": arguments.resume is not None,
        "argv": sys.argv[1:],
        "device": str(device),
        "device_name": device_display_name(device),
        "python_version": sys.version.split()[0],
        "numpy_version": numpy.__version__,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "git_commit": _git_output("rev-parse", "HEAD"),
        "git_uncommitted_changes": status_lines.splitlines() if status_lines else [],
        "number_of_cells": len(cells),
        "number_of_runs": number_of_runs,
    }
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta.setdefault("invocations", []).append(invocation)
    else:
        dataset_keys = sorted({dataset_key for dataset_key, _ in cells})
        meta = {
            "timestamp": timestamp,
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "cuda_device_name": device_display_name(device),
            "device": str(device),
            "generations": arguments.generations,
            "population": arguments.population,
            "runs_per_cell": arguments.runs,
            "timeout_seconds": arguments.timeout_seconds,
            "train_fraction": arguments.train_fraction,
            "seed": arguments.seed,
            "number_of_cells": len(cells),
            "number_of_runs": number_of_runs,
            "seeding_rule": (
                f"split seed is --seed + {SPLIT_SEED_OFFSET} + run_index (the same for "
                "every algorithm of a run); evolution seed is --seed + run_index"
            ),
            **PROTOCOL_LABELS,
            "dataset_file_sha256": dataset_file_checksums(dataset_keys),
            "invocations": [invocation],
        }
    write_json_atomically(meta, meta_path)


def run_sweep(arguments: argparse.Namespace, cells: list[tuple[str, str]]) -> Path:
    """Run every repetition of every cell in a child process, then analyse.

    Args:
        arguments: Parsed CLI arguments.
        cells: ``(dataset_key, algorithm_name)`` pairs to run.

    Returns:
        The directory the results were written to.
    """
    device = resolve_device(arguments.device)
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
    update_sweep_meta(
        output_directory / "sweep_meta.json",
        timestamp=timestamp,
        arguments=arguments,
        device=device,
        cells=cells,
        number_of_runs=len(runs),
    )

    # Records are gathered from disk after the loop rather than accumulated
    # here, so a resumed sweep reports the runs it skipped alongside the ones
    # it just ran.
    for position, (dataset_key, algorithm_name, run_index) in enumerate(runs, start=1):
        result_path = runs_directory / f"{dataset_key}__{algorithm_name}__run{run_index}.json"
        if arguments.resume is not None and result_path.exists():
            if recorded_status(result_path) == "ok":
                print(
                    f"[{position}/{len(runs)}] {dataset_key}/{algorithm_name} "
                    f"run {run_index}: already recorded, skipping",
                    flush=True,
                )
                continue
            # A failed or timed-out run is retried. Its old record moves aside
            # rather than being deleted, and must not stay at result_path: the
            # loop below reads result_path to learn how the new attempt went.
            set_aside_failed_record(result_path, output_directory / "runs_failed")
        print(
            f"\n=== [{position}/{len(runs)}] {dataset_key}/{algorithm_name} "
            f"run {run_index} (timeout {arguments.timeout_seconds}s) ===",
            flush=True,
        )
        command = _child_process_command(
            dataset_key,
            algorithm_name,
            run_index,
            arguments,
            result_path,
            topology_directory,
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
        print(f"-> {record['status']} (test macro-F1 {record['test_macro_f1']})", flush=True)

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
