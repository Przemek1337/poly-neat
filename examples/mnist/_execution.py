"""MNIST-specific execution policy on top of the shared benchmark spine.

The mode ladder, the environment digest and the immutable v2 lock format are the
shared ones. What is specific here is what a full MNIST run must not change: the
dataset identity - the release and the split parameters - which stands in for the
pneumonia benchmark's audited manifest. MNIST is a fixed public set with a
deterministic split, so its identity is a digest rather than a file.
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

_MNIST_DEPENDENCIES = ("numpy", "PyYAML")

__all__ = [
    "ExecutionLockError",
    "ExecutionOptions",
    "canonical_digest",
    "execution_environment",
    "read_execution_lock",
    "validate_execution_lock",
]


def execution_environment(device: torch.device) -> dict:
    """Pin an MNIST run to its exact implementation, libraries and device."""
    from examples._benchmark.execution import execution_environment as _shared_environment

    return _shared_environment(
        device,
        family_directory=Path(__file__).parent,
        dependency_names=_MNIST_DEPENDENCIES,
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
