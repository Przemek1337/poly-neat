"""One command-line surface shared by every benchmark profile entry point.

Each family exposes one runnable module per method. They all accept the same
flags - device, an optional config override, the execution mode and its resume
and lock companions - so a person moving between the MNIST, CIFAR and pneumonia
benchmarks types the same thing. The family supplies only its ``run_experiment``
callable and the errors it wants turned into a clean exit rather than a
traceback.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

import torch

from examples._benchmark.execution import ExecutionOptions
from examples._example_cli import add_device_arguments, resolve_device
from examples._experiment import ExperimentReport, print_experiment_report
from polyneat.logging_utils.custom_logger import get_logger

logger = get_logger(__name__)


def parse_profile_cli(
    argument_list: list[str] | None = None,
    *,
    default_config_file_path: Path,
    description: str,
) -> tuple[torch.device | None, Path | None, Path, argparse.Namespace]:
    """Parse the flags every benchmark profile accepts.

    Args:
        argument_list: Arguments to parse instead of ``sys.argv[1:]``.
        default_config_file_path: The profile's own yaml, used when ``--config``
            is not given.
        description: Human description of the family, shown in ``--help``.

    Returns:
        ``(device, data_directory, config_file_path, execution_arguments)``.

    Raises:
        SystemExit: Code 1 for ``--gpu`` without CUDA, code 2 for a malformed
            command line.
    """
    parser = argparse.ArgumentParser(description=description, allow_abbrev=False)
    add_device_arguments(parser)
    parser.add_argument(
        "--data-directory",
        type=Path,
        default=None,
        help="dataset directory; families that auto-download a public set ignore it",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=f"profile yaml to run (default: {default_config_file_path.name})",
    )
    parser.add_argument("--mode", choices=("smoke", "pilot", "full"))
    parser.add_argument("--protocol-lock", type=Path)
    parser.add_argument("--artifacts-directory", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--lost-work-seconds", type=float)
    parsed_arguments = parser.parse_args(argument_list)
    if parsed_arguments.mode is None and (
        parsed_arguments.resume
        or parsed_arguments.protocol_lock is not None
        or parsed_arguments.lost_work_seconds is not None
    ):
        parser.error("--resume, --protocol-lock and --lost-work-seconds require an explicit --mode")
    if parsed_arguments.lost_work_seconds is not None and not parsed_arguments.resume:
        parser.error("--lost-work-seconds requires --resume")
    return (
        resolve_device(parsed_arguments),
        parsed_arguments.data_directory,
        parsed_arguments.config or default_config_file_path,
        parsed_arguments,
    )


def run_profile_main(
    run_experiment: Callable[..., ExperimentReport],
    *,
    default_config_file_path: Path,
    artifacts_directory: Path,
    description: str,
    expected_errors: tuple[type[Exception], ...],
    argument_list: list[str] | None = None,
) -> None:
    """Parse the command line, run one profile and print its report.

    Args:
        run_experiment: The profile's own ``run_experiment``.
        default_config_file_path: Used when ``--config`` is not given.
        artifacts_directory: Where this profile writes its artifacts.
        description: Human description of the family for ``--help``.
        expected_errors: Exception types turned into a code-1 exit with a
            logged message instead of a traceback (missing data, a broken lock).
        argument_list: Arguments to parse instead of ``sys.argv[1:]``.

    Raises:
        SystemExit: Code 1 when the run refuses to start for an expected reason.
    """
    device, data_directory, config_file_path, options = parse_profile_cli(
        argument_list,
        default_config_file_path=default_config_file_path,
        description=description,
    )
    try:
        report = run_experiment(
            device=device,
            artifacts_directory=options.artifacts_directory or artifacts_directory,
            data_directory=data_directory,
            config_file_path=config_file_path,
            random_seed=options.seed,
            execution=(
                None
                if options.mode is None
                else ExecutionOptions(
                    mode=options.mode,
                    protocol_lock_path=options.protocol_lock,
                    resume=options.resume,
                    lost_work_seconds=options.lost_work_seconds,
                )
            ),
        )
    except expected_errors as error:
        logger.error("%s", error)
        raise SystemExit(1) from error
    print_experiment_report(report)
