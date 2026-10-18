"""Run one example's experiment several times and record the spread.

Invoke as a module from the repository root (script invocation would put
``benchmarks/`` at ``sys.path[0]`` and leave ``examples`` unimportable):

    uv run python -m benchmarks.run_benchmark iris/cneat --repeats 5 [--cpu | --gpu] [--base-seed 0]

Each repeat calls the example's ``run_experiment`` with evolution seed
``base_seed + i``. Artifacts are skipped unless ``--artifacts-root`` is given,
in which case every seed writes into its own directory underneath it - which is
what a benchmark that must be reproducible from its own output needs, and what
the shared-directory default could not provide. One JSON document per
invocation lands in ``benchmarks/results/`` recording every run, its status and
artifact paths, the mean/std summary, the fraction of runs that failed, and the
full yaml config that produced it - after "edit the yaml, re-run, compare" the
old yaml is gone, so the result file itself must record what produced it.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import importlib
import inspect
import json
import statistics
import sys
from pathlib import Path
from typing import cast

from examples._example_cli import add_device_arguments, resolve_device
from examples._experiment import EXAMPLE_REGISTRY, ExampleModule

_REPOSITORY_ROOT = Path(__file__).parent.parent
_RESULTS_DIRECTORY = Path(__file__).parent / "results"

_import_example_module = importlib.import_module


def _parse_arguments(argument_list: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark a PolyNEAT example over repeated seeded runs",
        allow_abbrev=False,
    )
    parser.add_argument("example_id", help="example to benchmark, e.g. iris/cneat")
    parser.add_argument("--repeats", type=int, default=5, help="number of runs (default: 5)")
    parser.add_argument(
        "--base-seed", type=int, default=0, help="evolution seed of the first run (default: 0)"
    )
    parser.add_argument(
        "--artifacts-root",
        type=Path,
        default=None,
        help=(
            "write each repeat into its own directory under this root "
            "(<root>/<example>/seed_<n>). Omitted by default, which keeps the "
            "historical behaviour of running without artifacts; benchmarks that must be "
            "reproducible from their own output pass it."
        ),
    )
    parser.add_argument(
        "--data-directory",
        type=Path,
        default=None,
        help=(
            "dataset directory, for examples whose run_experiment accepts one. "
            "Full pediatric pneumonia profiles require it and refuse to run without it."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=(
            "profile yaml to run instead of the example's own, for examples whose "
            "run_experiment accepts one. Its text and digest are what the result "
            "document records."
        ),
    )
    add_device_arguments(parser)
    parser.add_argument("--mode", choices=("smoke", "pilot", "full"))
    parser.add_argument("--protocol-lock", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--lost-work-seconds", type=float)
    return parser.parse_args(argument_list)


def _profile_arguments(
    example_module: ExampleModule, arguments: argparse.Namespace, example_id: str
) -> dict:
    """Forward the profile-only flags to examples that declare them.

    Most examples take a device, a seed and an artifacts directory and nothing
    else. Passing a dataset directory or a config override to one of those
    would be a silent no-op at best, so a flag aimed at an example that cannot
    honour it stops the run instead.

    Args:
        example_module: The imported example.
        arguments: Parsed command line.
        example_id: Registry id, named in the error.

    Returns:
        The keyword arguments to add to the ``run_experiment`` call.

    Raises:
        SystemExit: Code 1 when a flag was given to an example that cannot use it.
    """
    accepted = inspect.signature(example_module.run_experiment).parameters
    forwarded: dict = {}
    if arguments.mode is not None:
        if "execution" not in accepted:
            raise SystemExit(f"{example_id} does not accept execution modes")
        from examples.pediatric_pneumonia._execution import ExecutionOptions

        forwarded["execution"] = ExecutionOptions(
            mode=arguments.mode,
            protocol_lock_path=arguments.protocol_lock,
            resume=arguments.resume,
            lost_work_seconds=arguments.lost_work_seconds,
        )
    elif (
        arguments.resume
        or arguments.protocol_lock is not None
        or arguments.lost_work_seconds is not None
    ):
        raise SystemExit("--resume, --protocol-lock and --lost-work-seconds require --mode")
    for name, value in (
        ("data_directory", arguments.data_directory),
        ("config_file_path", arguments.config),
    ):
        if value is None:
            continue
        if name not in accepted:
            flag = "--" + name.replace("_file_path", "").replace("_", "-")
            print(
                f"error: {example_id} does not accept {flag}",
                file=sys.stderr,
            )
            raise SystemExit(1)
        forwarded[name] = value
    return forwarded


def _artifacts_directory_for_seed(
    artifacts_root: Path | None, example_id: str, evolution_seed: int
) -> Path | None:
    """Give one repeat its own directory, or keep the historical no-artifacts run.

    Every seed writing into the same directory is why artifacts used to be
    skipped here at all: five runs would overwrite one another. A directory per
    seed removes that objection, so a benchmark that has to be reproducible
    from its own output can ask for one.
    """
    if artifacts_root is None:
        return None
    seed_directory = artifacts_root / example_id.replace("/", "_") / f"seed_{evolution_seed}"
    seed_directory.mkdir(parents=True, exist_ok=True)
    return seed_directory


def _summarize_runs(run_records: list[dict]) -> dict[str, dict[str, float]]:
    """Mean and population standard deviation for every recorded quantity."""
    summary: dict[str, dict[str, float]] = {}
    for metric_name in run_records[0]["metric_values"]:
        values = [run["metric_values"][metric_name] for run in run_records]
        summary[metric_name] = {
            "mean": statistics.fmean(values),
            "std": statistics.pstdev(values),
        }
    for field_name in ("number_of_generations", "runtime_seconds"):
        values = [run[field_name] for run in run_records]
        summary[field_name] = {
            "mean": statistics.fmean(values),
            "std": statistics.pstdev(values),
        }
    return summary


def main(argument_list: list[str] | None = None) -> None:
    arguments = _parse_arguments(argument_list)
    if arguments.repeats < 1:
        raise SystemExit("--repeats must be positive")
    if arguments.example_id not in EXAMPLE_REGISTRY:
        valid_ids = ", ".join(sorted(EXAMPLE_REGISTRY))
        print(
            f"error: unknown example id {arguments.example_id!r}. Valid ids: {valid_ids}",
            file=sys.stderr,
        )
        raise SystemExit(1)
    device = resolve_device(arguments)

    example_module = cast(
        ExampleModule, _import_example_module(EXAMPLE_REGISTRY[arguments.example_id])
    )
    profile_arguments = _profile_arguments(example_module, arguments, arguments.example_id)
    config_file_path = arguments.config or example_module.CONFIG_FILE_PATH
    config_text = config_file_path.read_text(encoding="utf-8")

    run_records: list[dict] = []
    for repeat_index in range(arguments.repeats):
        evolution_seed = arguments.base_seed + repeat_index
        print(
            f"\n=== Run {repeat_index + 1}/{arguments.repeats} "
            f"(evolution seed {evolution_seed}) ==="
        )
        seed_artifacts_directory = _artifacts_directory_for_seed(
            arguments.artifacts_root, arguments.example_id, evolution_seed
        )
        report = example_module.run_experiment(
            device=device,
            random_seed=evolution_seed,
            artifacts_directory=seed_artifacts_directory,
            **profile_arguments,
        )
        run_record = {
            "seed": evolution_seed,
            "metric_values": dict(report.metric_values),
            "number_of_generations": report.number_of_generations,
            "runtime_seconds": report.runtime_seconds,
            "status": getattr(report, "status", "succeeded"),
            "failure_reason": getattr(report, "failure_reason", None),
            "artifacts_directory": (
                None if seed_artifacts_directory is None else seed_artifacts_directory.as_posix()
            ),
            "artifact_paths": dict(getattr(report, "artifact_paths", {})),
            "undefined_metrics": dict(getattr(report, "undefined_metrics", {})),
            "effective_configuration": dict(getattr(report, "effective_configuration", {})),
        }
        run_records.append(run_record)
        print(f"Run result: {json.dumps(run_record)}")

    summary = _summarize_runs(run_records)

    try:
        config_file_value = config_file_path.relative_to(_REPOSITORY_ROOT).as_posix()
    except ValueError:
        config_file_value = config_file_path.as_posix()
    result_document = {
        "example": arguments.example_id,
        "config_file": config_file_value,
        "config_sha256": hashlib.sha256(config_text.encode("utf-8")).hexdigest(),
        "config_text": config_text,
        "device": str(device) if device is not None else "config_default",
        "repeats": arguments.repeats,
        "base_seed": arguments.base_seed,
        "runs": run_records,
        "summary": summary,
        "failed_run_count": sum(
            1 for run in run_records if run.get("status", "succeeded") != "succeeded"
        ),
        "failed_run_fraction": (
            sum(1 for run in run_records if run.get("status", "succeeded") != "succeeded")
            / len(run_records)
            if run_records
            else 0.0
        ),
    }

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file_name = f"{arguments.example_id.replace('/', '_')}_{timestamp}.json"
    _RESULTS_DIRECTORY.mkdir(parents=True, exist_ok=True)
    result_path = _RESULTS_DIRECTORY / result_file_name
    result_path.write_text(json.dumps(result_document, indent=2), encoding="utf-8")

    print(f"\nSummary over {arguments.repeats} runs:")
    print(json.dumps(summary, indent=2))
    print(f"Result written to {result_path}")


if __name__ == "__main__":
    main()
