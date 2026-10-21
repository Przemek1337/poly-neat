"""Shared CLI helper for the PolyNEAT example scripts.

Every example accepts the same pair of mutually exclusive flags:

* ``--cpu`` - force phenotype evaluation on the CPU,
* ``--gpu`` - force phenotype evaluation on CUDA; exits with an error when
  CUDA is unavailable (no silent CPU fallback, so benchmarks stay honest).

When neither flag is given the helper returns ``None`` and the example falls
back to ``device_for_phenotype_evaluation`` from its YAML config, via
``Algorithm.from_config(config, device_for_phenotype_computation=device)``. The
benchmark harness composes the same flags into its own parser via
``add_device_arguments`` + ``resolve_device``.
"""

from __future__ import annotations

import argparse
import sys

import torch


def add_device_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the mutually exclusive ``--cpu``/``--gpu`` flags to ``parser``.

    Args:
        parser: Parser to extend; the harness composes these flags with its
            own arguments, the examples use them alone.
    """
    device_flag_group = parser.add_mutually_exclusive_group()
    device_flag_group.add_argument(
        "--cpu", action="store_true", help="evaluate phenotypes on the CPU"
    )
    device_flag_group.add_argument(
        "--gpu", action="store_true", help="evaluate phenotypes on CUDA (errors if unavailable)"
    )


def resolve_device(parsed_arguments: argparse.Namespace) -> torch.device | None:
    """Turn parsed ``--cpu``/``--gpu`` flags into the requested device.

    Args:
        parsed_arguments: Namespace produced by a parser that went through
            :func:`add_device_arguments`.

    Returns:
        ``torch.device("cpu")``, ``torch.device("cuda")``, or ``None`` when
        neither flag is present (caller uses the YAML config value).

    Raises:
        SystemExit: With code 1 when ``--gpu`` is requested but CUDA is not
            available (no silent CPU fallback, so benchmarks stay honest).
    """
    if parsed_arguments.gpu:
        if not torch.cuda.is_available():
            print(
                "error: --gpu requested but CUDA is not available on this machine. "
                "Run with --cpu or install a CUDA-enabled torch build.",
                file=sys.stderr,
            )
            raise SystemExit(1)
        return torch.device("cuda")
    if parsed_arguments.cpu:
        return torch.device("cpu")
    return None


def parse_device_from_cli(argument_list: list[str] | None = None) -> torch.device | None:
    """Parse ``--cpu``/``--gpu`` and return the requested evaluation device.

    Thin wrapper the example scripts call: builds a parser holding only the
    device flags and resolves it.

    Args:
        argument_list: Arguments to parse instead of ``sys.argv[1:]``.
            Intended for tests; examples call this with no arguments.

    Returns:
        See :func:`resolve_device`.

    Raises:
        SystemExit: Code 1 for ``--gpu`` without CUDA, code 2 when both flags
            are passed (argparse mutual exclusion).
    """
    parser = argparse.ArgumentParser(description="PolyNEAT example", allow_abbrev=False)
    add_device_arguments(parser)
    parsed_arguments = parser.parse_args(argument_list)
    return resolve_device(parsed_arguments)
