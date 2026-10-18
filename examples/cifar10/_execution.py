"""CIFAR-specific execution policy on top of the shared benchmark spine.

The mode ladder, the environment digest and the v2 lock format are the shared
ones; what is specific here is CIFAR's dataset identity - the release and split
parameters - which a full run must not change, and the libraries CIFAR pins.
"""

from __future__ import annotations

from pathlib import Path

import torch

from examples._benchmark.execution import (
    ExecutionLockError,
    ExecutionOptions,
    canonical_digest,
    read_execution_lock,
)

_CIFAR_DEPENDENCIES = ("numpy", "PyYAML")

__all__ = [
    "ExecutionLockError",
    "ExecutionOptions",
    "canonical_digest",
    "execution_environment",
    "read_execution_lock",
    "validate_execution_lock",
]


def execution_environment(device: torch.device) -> dict:
    """Pin a CIFAR run to its exact implementation, libraries and device."""
    from examples._benchmark.execution import execution_environment as _shared_environment

    return _shared_environment(
        device,
        family_directory=Path(__file__).parent,
        dependency_names=_CIFAR_DEPENDENCIES,
    )


def validate_execution_lock(
    path: Path,
    *,
    method: str,
    profile: dict,
    search_seed: int,
    device: torch.device,
    dataset_identity_sha256: str,
) -> str:
    """Verify the actual configuration, seed, environment and dataset identity."""
    payload = read_execution_lock(path)
    if payload["profiles"].get(method) != profile:
        raise ExecutionLockError(f"{method}: actual profile differs from the frozen profile")
    if search_seed not in payload["search_seeds"]:
        raise ExecutionLockError("search seed was not declared in the frozen series")
    if payload["environment"] != execution_environment(device):
        raise ExecutionLockError("execution environment differs from the frozen series")
    if payload["dataset_identity_sha256"] != dataset_identity_sha256:
        raise ExecutionLockError("current dataset or split differs from the frozen identity")
    return canonical_digest(payload)
