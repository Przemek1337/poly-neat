"""Execution mode policy, environment digest and immutable v2 lock format.

These pieces are dataset-agnostic. A family passes in which of its own source
files should feed the environment digest and which distributions it pins; it
never has to reimplement the mode ladder, the canonical digest or the lock
reader. Family-specific validation (does the current manifest match the frozen
one, does the fixed split still hold) stays in the family and builds on
:func:`read_execution_lock` and :func:`execution_environment`.
"""

from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

import torch


class ExecutionLockError(RuntimeError):
    """Raised when a run's configuration, environment or data leaves the lock."""


# The examples layer and this shared spine feed every family's digest, so a
# change to shared execution code invalidates a frozen series it also governs.
_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_SHARED_DIGEST_SOURCES: tuple[Path, ...] = (
    *sorted(_REPOSITORY_ROOT.joinpath("polyneat").rglob("*.py")),
    *sorted(_REPOSITORY_ROOT.joinpath("examples").glob("_*.py")),
    *sorted(_REPOSITORY_ROOT.joinpath("examples/_benchmark").glob("*.py")),
    _REPOSITORY_ROOT / "pyproject.toml",
    _REPOSITORY_ROOT / "uv.lock",
)


@dataclass(frozen=True)
class ExecutionOptions:
    """How one run is allowed to behave.

    Attributes:
        mode: ``smoke`` (fast plumbing check), ``pilot`` (real data, never
            touches the official test) or ``full`` (a frozen result series).
        protocol_lock_path: The v2 lock a full run is bound to. Required in
            full mode, forbidden otherwise.
        resume: Continue a search from its last committed boundary.
        lost_work_seconds: Time to charge back after an unclean shutdown, taken
            from the scheduler or logs. Only meaningful with ``resume``.
    """

    mode: str = "smoke"
    protocol_lock_path: Path | None = None
    resume: bool = False
    lost_work_seconds: float | None = None

    def __post_init__(self) -> None:
        if self.mode not in {"smoke", "pilot", "full"}:
            raise ValueError("mode must be smoke, pilot or full")
        if self.mode == "full" and self.protocol_lock_path is None:
            raise ExecutionLockError(
                "full requires --protocol-lock; pilot never evaluates the test"
            )
        if self.mode != "full" and self.protocol_lock_path is not None:
            raise ExecutionLockError("--protocol-lock is only used in full mode")
        if self.lost_work_seconds is not None and not self.resume:
            raise ValueError("--lost-work-seconds requires --resume")


def canonical_digest(payload: dict) -> str:
    """SHA-256 of a payload in a stable key order, so equal payloads agree."""
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def resolve_profile_execution(
    protocol: dict, execution: ExecutionOptions | None
) -> ExecutionOptions:
    """Only explicitly designated smoke profiles may evaluate without a lock."""
    is_smoke = protocol.get("profile_kind") == "smoke"
    resolved = execution or ExecutionOptions(mode="smoke" if is_smoke else "pilot")
    if resolved.mode == "smoke" and not is_smoke:
        raise ExecutionLockError("smoke requires a smoke profile; use pilot for result profiles")
    if resolved.mode != "smoke" and is_smoke:
        raise ExecutionLockError("pilot/full requires a non-smoke profile")
    return resolved


def execution_environment(
    device: torch.device,
    *,
    family_directory: Path,
    dependency_names: tuple[str, ...],
) -> dict:
    """Pin a run to its exact implementation, interpreter, libraries and device.

    The digest covers the whole library, the shared examples layer, this shared
    spine and the family's own package, so a full result series is bound to the
    code it actually ran. Two runs whose environments differ here are not part
    of the same frozen study.

    Args:
        device: The one device the run uses.
        family_directory: The family package whose ``*.py`` files join the
            shared sources in the digest (e.g. ``examples/mnist``).
        dependency_names: Distributions to record versions of, chosen per
            family so the record names only libraries the family relies on.

    Returns:
        A JSON-serialisable environment description including the source digest.
    """
    source_files = (*_SHARED_DIGEST_SOURCES, *sorted(family_directory.glob("*.py")))
    digest = hashlib.sha256()
    for path in sorted(set(source_files)):
        digest.update(path.relative_to(_REPOSITORY_ROOT).as_posix().encode())
        digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
    return {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "dependencies": {name: version(name) for name in dependency_names},
        "cuda": torch.version.cuda,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "precision": "float32",
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "implementation_sha256": digest.hexdigest(),
    }


def read_execution_lock(path: Path) -> dict:
    """Read a v2 lock and refuse it if it was edited after freezing.

    Args:
        path: The ``protocol.lock.yaml`` written by a family's freeze command.

    Returns:
        The lock payload without its self-digest.

    Raises:
        ExecutionLockError: If the schema is not v2 or the recorded digest does
            not match the payload, which means the lock changed after freezing.
    """
    import yaml

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != "2.0":
        raise ExecutionLockError("executable full runs require a v2 lock generated by freeze")
    recorded_digest = payload.pop("lock_sha256", None)
    if recorded_digest != canonical_digest(payload):
        raise ExecutionLockError("protocol lock changed after freezing; create a new study")
    return payload
