"""Pneumonia-specific execution policy on top of the shared benchmark spine.

The mode ladder, the environment digest and the immutable v2 lock format live
in :mod:`examples._benchmark.execution` and are shared with the MNIST and CIFAR
benchmarks. What stays here is the one thing only this benchmark can check: that
the frozen dataset manifest still describes the archive a full run is reading.
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
from examples.pediatric_pneumonia._dataset_manifest import DatasetManifest

# The pneumonia benchmark pins these distributions; the digit and object
# benchmarks pin their own. Kept here so the environment record names only the
# libraries this family relies on.
_PNEUMONIA_DEPENDENCIES = ("numpy", "scikit-learn", "Pillow", "torchvision", "PyYAML")

__all__ = [
    "ExecutionLockError",
    "ExecutionOptions",
    "canonical_digest",
    "execution_environment",
    "read_execution_lock",
    "validate_execution_lock",
]


def execution_environment(device: torch.device) -> dict:
    """Pin a pneumonia run to its exact implementation, libraries and device."""
    from examples._benchmark.execution import execution_environment as _shared_environment

    return _shared_environment(
        device,
        family_directory=Path(__file__).parent,
        dependency_names=_PNEUMONIA_DEPENDENCIES,
    )


def validate_execution_lock(
    path: Path,
    *,
    method: str,
    profile: dict,
    search_seed: int,
    device: torch.device,
    manifest: DatasetManifest | None = None,
) -> str:
    """Verify actual configuration, seed, environment and (when audited) data."""
    payload = read_execution_lock(path)
    if payload["profiles"].get(method) != profile:
        raise ExecutionLockError(f"{method}: actual profile differs from the frozen profile")
    if search_seed not in payload["search_seeds"]:
        raise ExecutionLockError("search seed was not declared in the frozen series")
    if payload["environment"] != execution_environment(device):
        raise ExecutionLockError("execution environment differs from the frozen series")
    locked_manifest, digest = DatasetManifest.read_json_file(path.parent / payload["manifest_path"])
    if digest != payload["manifest_sha256"] or locked_manifest.blocking_findings:
        raise ExecutionLockError("locked manifest differs or contains blocking audit findings")
    if manifest is not None and manifest.compute_sha256() != digest:
        raise ExecutionLockError("current archive/split differs from the frozen manifest")
    return canonical_digest(payload)
