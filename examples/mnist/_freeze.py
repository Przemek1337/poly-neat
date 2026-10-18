"""Freeze verified MNIST profiles into an immutable v2 lock, after pilots, without testing.

A full result series runs against a frozen lock, and this command is the only
thing that writes one. The shared assembler in
:mod:`examples._benchmark.freeze_lock` does the work; this module only names
MNIST's required methods and its fitness metric.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from examples._benchmark.execution import ExecutionLockError
from examples._benchmark.freeze_lock import assemble_series_lock
from examples.mnist._execution import execution_environment
from examples.mnist._profiles import CONFIGS_DIRECTORY

DEFAULT_METHODS = ("deepneat", "exact")
_METRIC_KEY = "validation_accuracy"


def freeze_series(
    *,
    profiles_directory: Path,
    output_directory: Path,
    methods: list[str],
    seeds: list[int],
    protocol_id: str,
    dataset_release: str,
    dataset_license: str,
    pilot_reports: Sequence[Path],
    environment: dict,
) -> Path:
    """Freeze an MNIST series: both DeepNEAT and EXACT, on validation accuracy."""
    return assemble_series_lock(
        profiles_directory=profiles_directory,
        output_directory=output_directory,
        methods=methods,
        required_methods=DEFAULT_METHODS,
        seeds=seeds,
        protocol_id=protocol_id,
        dataset_release=dataset_release,
        dataset_license=dataset_license,
        pilot_reports=pilot_reports,
        environment=environment,
        metric_key=_METRIC_KEY,
    )


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--profiles-directory", type=Path, default=CONFIGS_DIRECTORY)
    parser.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS))
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--protocol-id", required=True)
    parser.add_argument("--dataset-release", required=True)
    parser.add_argument("--dataset-license", required=True)
    parser.add_argument("--pilot-reports", type=Path, nargs="+", required=True)
    from examples._example_cli import add_device_arguments, resolve_device

    add_device_arguments(parser)
    args = parser.parse_args()
    import torch

    device = resolve_device(args) or torch.device("cpu")
    try:
        path = freeze_series(
            profiles_directory=args.profiles_directory,
            output_directory=args.output_directory,
            methods=args.methods,
            seeds=args.seeds,
            protocol_id=args.protocol_id,
            dataset_release=args.dataset_release,
            dataset_license=args.dataset_license,
            pilot_reports=args.pilot_reports,
            environment=execution_environment(device),
        )
    except ExecutionLockError as error:
        parser.error(str(error))
    print(f"Frozen series: {path}")


if __name__ == "__main__":
    main()
