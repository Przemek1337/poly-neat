"""The frozen protocol document a full benchmark series must run under.

Before the first result seed starts, every decision that could otherwise be
made after looking at a number has to be written down: which manifest, which
seeds, how much budget, which limits, which recipe, how many bootstrap
replicates. That document is the protocol lock. A full run loads it, checks it
against the manifest it was frozen for, and refuses to start when anything is
missing or still a placeholder.

The lock is deliberately a validated *document* rather than a deep tree of
typed fields. Its sections point at profiles and configurations that already
have their own typed loaders; duplicating those field-by-field would create a
second place to keep in sync without making the freeze any stronger. What this
module guarantees is the part the freeze actually needs: every required key is
present, no value is a leftover placeholder, and the manifest digest matches
the split the run is about to use.

Changing the lock after test predictions exist starts a new study. It does not
amend the original series.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from examples._benchmark.execution import ExecutionLockError
from polyneat.logging_utils.custom_logger import get_logger

logger = get_logger(__name__)

PROTOCOL_LOCK_SCHEMA_VERSION = "1.0"

# Keys every section must carry with a concrete value. The list is the contract
# between this module and the protocol document; adding a knob to the benchmark
# means adding it here so a lock written before that knob existed is rejected
# rather than silently defaulted.
REQUIRED_LOCK_KEYS: dict[str, tuple[str, ...]] = {
    "data": (
        "protocol_id",
        "manifest_path",
        "manifest_sha256",
        "dataset_release",
        "dataset_license",
        "image_side",
    ),
    "seeds": (
        "split_seed",
        "smoke_sampling_seed",
        "bootstrap_seed",
        "search_seeds",
        "retraining_seeds",
        "seed_derivation_rule",
    ),
    "budget": (
        "search_wall_clock_seconds",
        "pilot_wall_clock_seconds",
        "maximum_candidate_seconds",
        "maximum_phenotype_parameters",
        "checkpoint_interval_seconds",
    ),
    "search": (
        "methods",
        "population_size",
        "candidate_training_epochs",
        "candidate_batch_size",
        "fitness_metric",
    ),
    "random_search": (
        "graph_size_distribution",
        "gene_distributions",
    ),
    "fixed_cnn": (
        "architecture",
        "training_epochs",
    ),
    "track_b_recipe": (
        "optimizer",
        "learning_rate",
        "momentum",
        "weight_decay",
        "batch_size",
        "training_epochs",
        "learning_rate_schedule",
        "parameter_initialization",
    ),
    "transfer_learning": (
        "weights_identifier",
        "weights_sha256",
        "unfreezing_policy",
        "training_epochs",
        "preprocessing",
    ),
    "metrics": (
        "bootstrap_replicates",
        "maximum_bootstrap_attempts",
        "confidence_level",
        "threshold_rule",
        "reference_threshold",
    ),
    "environment": (
        "device",
        "precision",
        "deterministic_algorithms",
    ),
}

# Values that mean "not decided yet". A lock containing any of them is not
# frozen, whatever the file name says.
_PLACEHOLDER_STRINGS: frozenset[str] = frozenset(
    {"", "todo", "tbd", "fixme", "placeholder", "changeme", "?", "??", "???", "xxx", "n/a"}
)


# The pneumonia lock and the shared benchmark executor report the same kind of
# failure - a run leaving what was frozen - so they share one exception type.
# An error raised deep in the shared executor is then the very exception the
# pneumonia code raises and its tests catch.
ProtocolLockError = ExecutionLockError


@dataclass(frozen=True)
class ProtocolLock:
    """A loaded protocol lock and the provenance of the file it came from.

    Attributes:
        schema_version: Version of the lock layout.
        sections: The document itself, section by section.
        source_path: File the lock was read from.
        source_sha256: Digest of that file's bytes, recorded in every artifact
            directory so a reader can tell two series apart.
    """

    schema_version: str
    sections: Mapping[str, Mapping[str, Any]]
    source_path: Path
    source_sha256: str

    def section(self, section_name: str) -> Mapping[str, Any]:
        """Return one section.

        Raises:
            ProtocolLockError: If the section is absent.
        """
        if section_name not in self.sections:
            raise ProtocolLockError(
                f"protocol lock {self.source_path} has no section {section_name!r}; "
                f"present sections: {sorted(self.sections)}"
            )
        return self.sections[section_name]

    def value(self, section_name: str, key: str) -> Any:
        """Return one value from one section.

        Raises:
            ProtocolLockError: If the section or the key is absent.
        """
        section = self.section(section_name)
        if key not in section:
            raise ProtocolLockError(
                f"protocol lock {self.source_path} section {section_name!r} has no key {key!r}"
            )
        return section[key]

    @property
    def protocol_id(self) -> str:
        """Identifier of the protocol this lock freezes."""
        return str(self.value("data", "protocol_id"))

    @property
    def manifest_sha256(self) -> str:
        """Digest of the manifest this lock was frozen against."""
        return str(self.value("data", "manifest_sha256"))

    @property
    def search_seeds(self) -> tuple[int, ...]:
        """Result seeds of the search stage, in the order they were declared."""
        return tuple(int(seed) for seed in self.value("seeds", "search_seeds"))

    @property
    def retraining_seeds(self) -> tuple[int, ...]:
        """Track B retraining seeds applied to every selected topology."""
        return tuple(int(seed) for seed in self.value("seeds", "retraining_seeds"))

    def manifest_path_relative_to_lock(self) -> Path:
        """Resolve the manifest path recorded in the lock.

        A relative path is resolved against the directory holding the lock, so
        a series directory can be moved as a unit.
        """
        recorded_path = Path(str(self.value("data", "manifest_path")))
        if recorded_path.is_absolute():
            return recorded_path
        return (self.source_path.parent / recorded_path).resolve()


def _find_placeholder_values(payload: Any, path_prefix: str) -> list[str]:
    """Collect dotted paths of every value that is not actually decided.

    Args:
        payload: Any part of the lock document.
        path_prefix: Dotted path of ``payload`` inside the document.

    Returns:
        Dotted paths of placeholder values, empty when the subtree is concrete.
    """
    if payload is None:
        return [path_prefix]
    if isinstance(payload, str):
        return [path_prefix] if payload.strip().lower() in _PLACEHOLDER_STRINGS else []
    if isinstance(payload, Mapping):
        if not payload:
            return [path_prefix]
        return [
            offending_path
            for key, value in payload.items()
            for offending_path in _find_placeholder_values(value, f"{path_prefix}.{key}")
        ]
    if isinstance(payload, Sequence) and not isinstance(payload, str | bytes):
        if not payload:
            return [path_prefix]
        return [
            offending_path
            for index, value in enumerate(payload)
            for offending_path in _find_placeholder_values(value, f"{path_prefix}[{index}]")
        ]
    return []


def load_protocol_lock(lock_path: Path) -> ProtocolLock:
    """Read a protocol lock file without judging whether it is complete.

    Args:
        lock_path: The ``protocol.lock.yaml`` to read.

    Returns:
        The parsed lock, with the digest of the file it came from.

    Raises:
        ProtocolLockError: If the file is missing, unparseable, or is not a
            mapping of sections.
    """
    if not lock_path.is_file():
        raise ProtocolLockError(f"protocol lock {lock_path} does not exist")
    raw_bytes = lock_path.read_bytes()
    try:
        payload = yaml.safe_load(raw_bytes.decode("utf-8"))
    except yaml.YAMLError as parse_error:
        raise ProtocolLockError(f"protocol lock {lock_path} is not valid YAML") from parse_error
    if not isinstance(payload, Mapping):
        raise ProtocolLockError(
            f"protocol lock {lock_path} must be a mapping of sections, got {type(payload).__name__}"
        )

    schema_version = str(payload.get("schema_version", PROTOCOL_LOCK_SCHEMA_VERSION))
    sections = {
        str(section_name): section_payload
        for section_name, section_payload in payload.items()
        if section_name != "schema_version"
    }
    return ProtocolLock(
        schema_version=schema_version,
        sections=sections,
        source_path=lock_path,
        source_sha256=hashlib.sha256(raw_bytes).hexdigest(),
    )


def validate_protocol_lock_for_full_run(
    protocol_lock: ProtocolLock,
    *,
    manifest_sha256: str,
    required_keys: Mapping[str, Sequence[str]] | None = None,
) -> None:
    """Refuse a lock that is not actually frozen, or is frozen for other data.

    Three failures are treated identically, because all three mean the series
    is not reproducible from its own artifacts: a missing section or key, a
    value that is still a placeholder, and a manifest digest that does not
    match the split about to be used.

    Args:
        protocol_lock: The loaded lock.
        manifest_sha256: Digest of the manifest the run will actually read.
        required_keys: Sections and keys that must be present. Defaults to
            :data:`REQUIRED_LOCK_KEYS`.

    Raises:
        ProtocolLockError: Naming every missing key and placeholder path, so a
            single run tells the author everything left to decide instead of
            failing once per key.
    """
    schema = REQUIRED_LOCK_KEYS if required_keys is None else required_keys
    problems: list[str] = []

    for section_name, keys in schema.items():
        section_payload = protocol_lock.sections.get(section_name)
        if section_payload is None:
            problems.append(f"missing section {section_name!r}")
            continue
        if not isinstance(section_payload, Mapping):
            problems.append(
                f"section {section_name!r} must be a mapping, got {type(section_payload).__name__}"
            )
            continue
        problems.extend(
            f"missing key {section_name}.{key}"
            for key in keys
            if key not in section_payload
        )

    placeholder_paths = [
        path
        for section_name, section_payload in protocol_lock.sections.items()
        for path in _find_placeholder_values(section_payload, section_name)
    ]
    problems.extend(f"undecided placeholder value at {path}" for path in sorted(placeholder_paths))

    recorded_manifest_sha256 = str(
        protocol_lock.sections.get("data", {}).get("manifest_sha256", "")
    )
    if recorded_manifest_sha256 and recorded_manifest_sha256 != manifest_sha256:
        problems.append(
            f"lock was frozen for manifest {recorded_manifest_sha256[:12]} but the run loaded "
            f"manifest {manifest_sha256[:12]}; a different split is a different study"
        )

    if problems:
        raise ProtocolLockError(
            f"protocol lock {protocol_lock.source_path} is not ready for a full run:\n  - "
            + "\n  - ".join(problems)
        )
    logger.info(
        "Protocol lock %s validated for manifest %s (lock sha256 %s)",
        protocol_lock.protocol_id,
        manifest_sha256[:12],
        protocol_lock.source_sha256[:12],
    )
