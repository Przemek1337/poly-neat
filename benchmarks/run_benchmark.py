"""Run one example's experiment several times and record the spread.

Invoke as a module from the repository root (script invocation would put
``benchmarks/`` at ``sys.path[0]`` and leave ``examples`` unimportable):

    uv run python -m benchmarks.run_benchmark iris/cneat --repeats 5 [--cpu | --gpu] [--base-seed 0]

Each repeat calls the example's ``run_experiment`` with evolution seed
``base_seed + i`` and no artifacts directory. One JSON document per
invocation lands in ``benchmarks/results/`` recording every run, the
mean/std summary, and the full yaml config that produced it — after "edit
the yaml, re-run, compare" the old yaml is gone, so the result file itself
must record what produced it.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import importlib
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
    parser.add_argument(
        "--repeats", type=int, default=5, help="number of runs (default: 5)"
    )
    parser.add_argument(
        "--base-seed", type=int, default=0, help="evolution seed of the first run (default: 0)"
    )
    add_device_arguments(parser)
    return parser.parse_args(argument_list)


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
    config_file_path = example_module.CONFIG_FILE_PATH
    config_text = config_file_path.read_text(encoding="utf-8")

    run_records: list[dict] = []
    for repeat_index in range(arguments.repeats):
        evolution_seed = arguments.base_seed + repeat_index
        print(
            f"\n=== Run {repeat_index + 1}/{arguments.repeats} "
            f"(evolution seed {evolution_seed}) ==="
        )
        report = example_module.run_experiment(
            device=device, random_seed=evolution_seed, artifacts_directory=None
        )
        run_record = {
            "seed": evolution_seed,
            "metric_values": dict(report.metric_values),
            "number_of_generations": report.number_of_generations,
            "runtime_seconds": report.runtime_seconds,
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
