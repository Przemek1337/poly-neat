"""The frozen split manifest every method and seed of the benchmark shares.

The manifest is the single source of truth about *which image is where*. It is
written once from an audit, hashed, and then read back by every search, every
retraining and the final test evaluation. Two runs that disagree about the
manifest hash are not comparable and the runner refuses to treat them as if
they were.

What it deliberately carries beyond the split itself:

* the observed class counts, not the expected ones, so a deviation from the
  pinned release stays visible in every artifact directory;
* the exclusion map, so a reader can see which images were dropped and why;
* whether patient independence was actually established, so a report cannot
  quietly imply a guarantee the file names never supported;
* the audit findings, including every blocking one, so a manifest built from a
  dirty archive announces that fact rather than hiding it.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

from examples._datasets import split_groups_into_stratified_parts
from examples.pediatric_pneumonia._dataset_audit import (
    BLOCKING,
    DEVELOPMENT_POOL,
    OFFICIAL_TEST_POOL,
    DatasetAudit,
    ImageRecord,
)
from polyneat.logging_utils.custom_logger import get_logger

logger = get_logger(__name__)

MANIFEST_SCHEMA_VERSION = "1.0"

TRAIN_SPLIT = "train"
SEARCH_VALIDATION_SPLIT = "search_validation"
THRESHOLD_VALIDATION_SPLIT = "threshold_validation"
OFFICIAL_TEST_SPLIT = "official_test"

DEVELOPMENT_SPLIT_NAMES: tuple[str, ...] = (
    TRAIN_SPLIT,
    SEARCH_VALIDATION_SPLIT,
    THRESHOLD_VALIDATION_SPLIT,
)
ALL_SPLIT_NAMES: tuple[str, ...] = (*DEVELOPMENT_SPLIT_NAMES, OFFICIAL_TEST_SPLIT)

# The protocol's target proportions for the development pool. Group
# disjointness outranks them, so the realised counts are recorded separately.
DEFAULT_SPLIT_FRACTIONS: dict[str, float] = {
    TRAIN_SPLIT: 0.70,
    SEARCH_VALIDATION_SPLIT: 0.15,
    THRESHOLD_VALIDATION_SPLIT: 0.15,
}


class ManifestError(RuntimeError):
    """Raised when a manifest is missing, malformed or does not match a run."""


@dataclass(frozen=True)
class ManifestEntry:
    """One image as the frozen split sees it.

    Attributes:
        example_id: Stable id predictions are exported against.
        relative_path: Archive-root-relative path of the file.
        class_name: ``NORMAL`` or ``PNEUMONIA``.
        label: ``0`` or ``1``.
        original_split: Shipped directory the image came from.
        pool: ``development`` or ``official_test``.
        target_split: Split this benchmark assigned it to.
        width: Decoded width in pixels.
        height: Decoded height in pixels.
        file_sha256: Digest of the raw bytes.
        pixel_sha256: Digest of the decoded pixels and their shape.
        group_id: Grouping key parsed from the file name.
        split_group_id: Group actually used by the split, after duplicate-driven
            merges.
        group_confidence: What the parsed key is known to mean.
        group_convention: Naming convention that produced it.
    """

    example_id: str
    relative_path: str
    class_name: str
    label: int
    original_split: str
    pool: str
    target_split: str
    width: int
    height: int
    file_sha256: str
    pixel_sha256: str
    group_id: str
    split_group_id: str
    group_confidence: str
    group_convention: str


@dataclass(frozen=True)
class ManifestFinding:
    """An audit observation carried into the manifest verbatim."""

    kind: str
    severity: str
    message: str
    example_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class ManifestExclusion:
    """One image the split does not use, and the record kept instead."""

    excluded_example_id: str
    kept_example_id: str
    reason: str


@dataclass(frozen=True)
class DatasetManifest:
    """The frozen assignment of every image to a split, plus its provenance.

    Attributes:
        schema_version: Version of this record layout.
        protocol_id: Identifier of the protocol that produced the split. A
            cleaned archive gets a new id and a new manifest; it never reuses
            the old one.
        dataset_release: Identifier of the archive actually used, e.g. the
            Kaggle slug and version string.
        dataset_license: License of that release, as stated by its publisher.
        archive_root_relative_path: Nesting level the audit resolved.
        decoder_version: Decoder used for the pixel digests.
        patient_identifier_scope: How far file-name counters were allowed to
            reach when grouping.
        perceptual_hash_threshold: Frozen similarity threshold of the audit.
        patient_independence_established: ``True`` only when every entry
            carries a documented patient identifier. When ``False`` the report
            must state that patient independence was not confirmed and the test
            bootstrap resamples images rather than patients.
        split_seed: Seed of the grouped split.
        requested_split_fractions: Target shares of the development pool.
        entries: Every usable image, in example-id order.
        counts_by_split_and_label: Realised counts, as ``"split|label"`` keys.
        exclusions: Images dropped from the development pool.
        findings: Audit observations, including any blocking ones.
        similar_pair_count: Size of the audit similarity report.
        cross_pool_similar_pair_count: How many of those span the pools.
    """

    schema_version: str
    protocol_id: str
    dataset_release: str
    dataset_license: str
    archive_root_relative_path: str
    decoder_version: str
    patient_identifier_scope: str
    perceptual_hash_threshold: int
    patient_independence_established: bool
    split_seed: int
    requested_split_fractions: dict[str, float]
    entries: tuple[ManifestEntry, ...]
    counts_by_split_and_label: dict[str, int]
    exclusions: tuple[ManifestExclusion, ...]
    findings: tuple[ManifestFinding, ...]
    similar_pair_count: int
    cross_pool_similar_pair_count: int

    @property
    def blocking_findings(self) -> tuple[ManifestFinding, ...]:
        """Findings that must be resolved before a full experiment may run."""
        return tuple(finding for finding in self.findings if finding.severity == BLOCKING)

    def entries_of_split(self, split_name: str) -> tuple[ManifestEntry, ...]:
        """Entries assigned to ``split_name``, in example-id order.

        Raises:
            ManifestError: If the split name is not one of the protocol's.
        """
        if split_name not in ALL_SPLIT_NAMES:
            raise ManifestError(
                f"unknown split {split_name!r}; expected one of {ALL_SPLIT_NAMES}"
            )
        return tuple(entry for entry in self.entries if entry.target_split == split_name)

    def example_ids_of_split(self, split_name: str) -> tuple[str, ...]:
        """Stable example ids of one split, in order."""
        return tuple(entry.example_id for entry in self.entries_of_split(split_name))

    def to_serializable_dict(self) -> dict:
        """Return the manifest as plain JSON-compatible data, hash excluded."""
        return {
            "schema_version": self.schema_version,
            "protocol_id": self.protocol_id,
            "dataset_release": self.dataset_release,
            "dataset_license": self.dataset_license,
            "archive_root_relative_path": self.archive_root_relative_path,
            "decoder_version": self.decoder_version,
            "patient_identifier_scope": self.patient_identifier_scope,
            "perceptual_hash_threshold": self.perceptual_hash_threshold,
            "patient_independence_established": self.patient_independence_established,
            "split_seed": self.split_seed,
            "requested_split_fractions": dict(sorted(self.requested_split_fractions.items())),
            "entries": [asdict(entry) for entry in self.entries],
            "counts_by_split_and_label": dict(sorted(self.counts_by_split_and_label.items())),
            "exclusions": [asdict(exclusion) for exclusion in self.exclusions],
            "findings": [
                {
                    "kind": finding.kind,
                    "severity": finding.severity,
                    "message": finding.message,
                    "example_ids": list(finding.example_ids),
                }
                for finding in self.findings
            ],
            "similar_pair_count": self.similar_pair_count,
            "cross_pool_similar_pair_count": self.cross_pool_similar_pair_count,
        }

    def compute_sha256(self) -> str:
        """Digest of the canonical JSON encoding of this manifest.

        The digest covers everything except itself, so it can be recomputed
        from any copy of the file and compared without trusting the stored
        value.
        """
        canonical_json = json.dumps(
            self.to_serializable_dict(), sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()

    def write_json_file(self, manifest_path: Path) -> str:
        """Write the manifest and its digest to ``manifest_path``.

        Args:
            manifest_path: Destination file. Parent directories are created.

        Returns:
            The SHA-256 digest that was written alongside the payload.
        """
        payload = self.to_serializable_dict()
        manifest_sha256 = self.compute_sha256()
        payload["manifest_sha256"] = manifest_sha256
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        logger.info("Manifest written to %s (sha256 %s)", manifest_path, manifest_sha256[:12])
        return manifest_sha256

    @classmethod
    def from_serializable_dict(cls, payload: Mapping) -> DatasetManifest:
        """Rebuild a manifest from :meth:`to_serializable_dict` output.

        Raises:
            ManifestError: If a required key is missing.
        """
        try:
            return cls(
                schema_version=payload["schema_version"],
                protocol_id=payload["protocol_id"],
                dataset_release=payload["dataset_release"],
                dataset_license=payload["dataset_license"],
                archive_root_relative_path=payload["archive_root_relative_path"],
                decoder_version=payload["decoder_version"],
                patient_identifier_scope=payload["patient_identifier_scope"],
                perceptual_hash_threshold=int(payload["perceptual_hash_threshold"]),
                patient_independence_established=bool(
                    payload["patient_independence_established"]
                ),
                split_seed=int(payload["split_seed"]),
                requested_split_fractions={
                    str(name): float(value)
                    for name, value in payload["requested_split_fractions"].items()
                },
                entries=tuple(
                    ManifestEntry(**entry_payload) for entry_payload in payload["entries"]
                ),
                counts_by_split_and_label={
                    str(key): int(value)
                    for key, value in payload["counts_by_split_and_label"].items()
                },
                exclusions=tuple(
                    ManifestExclusion(**exclusion_payload)
                    for exclusion_payload in payload["exclusions"]
                ),
                findings=tuple(
                    ManifestFinding(
                        kind=finding_payload["kind"],
                        severity=finding_payload["severity"],
                        message=finding_payload["message"],
                        example_ids=tuple(finding_payload.get("example_ids", ())),
                    )
                    for finding_payload in payload["findings"]
                ),
                similar_pair_count=int(payload["similar_pair_count"]),
                cross_pool_similar_pair_count=int(payload["cross_pool_similar_pair_count"]),
            )
        except KeyError as missing_key:
            raise ManifestError(f"manifest payload is missing key {missing_key}") from None

    @classmethod
    def read_json_file(cls, manifest_path: Path) -> tuple[DatasetManifest, str]:
        """Read a manifest file and verify its stored digest.

        Args:
            manifest_path: File written by :meth:`write_json_file`.

        Returns:
            ``(manifest, manifest_sha256)``.

        Raises:
            ManifestError: If the file is missing, unparseable, or its stored
                digest disagrees with the payload. A silently edited manifest
                would break the guarantee that every seed used the same split.
        """
        if not manifest_path.is_file():
            raise ManifestError(f"manifest file {manifest_path} does not exist")
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as decode_error:
            raise ManifestError(
                f"manifest file {manifest_path} is not valid JSON"
            ) from decode_error

        stored_sha256 = payload.pop("manifest_sha256", None)
        if stored_sha256 is None:
            raise ManifestError(f"manifest file {manifest_path} has no manifest_sha256 field")
        manifest = cls.from_serializable_dict(payload)
        recomputed_sha256 = manifest.compute_sha256()
        if recomputed_sha256 != stored_sha256:
            raise ManifestError(
                f"manifest file {manifest_path} was modified after it was written: stored digest "
                f"{stored_sha256[:12]} but its content hashes to {recomputed_sha256[:12]}"
            )
        return manifest, recomputed_sha256


def build_manifest_from_audit(
    audit: DatasetAudit,
    *,
    protocol_id: str,
    dataset_release: str,
    dataset_license: str,
    split_seed: int,
    split_fractions: Mapping[str, float] | None = None,
) -> DatasetManifest:
    """Turn an audit into the frozen split every run of the benchmark shares.

    The development pool is split once, by group, into train,
    search_validation and threshold_validation. The official test directory is
    carried over untouched: it is never re-split, re-balanced or trimmed.

    Args:
        audit: Result of :func:`~examples.pediatric_pneumonia._dataset_audit
            .audit_pediatric_pneumonia_dataset`. Its exclusion map is applied
            before splitting.
        protocol_id: Identifier of this protocol version. Cleaning the data
            later means a new id and a new manifest, not an edit of this one.
        dataset_release: Identifier of the archive actually used.
        dataset_license: License of that release.
        split_seed: Seed of the grouped split, separate from every other seed
            in the protocol.
        split_fractions: Target shares of the development pool. Defaults to the
            protocol's 70/15/15.

    Returns:
        The assembled manifest. Blocking audit findings are carried into it
        rather than raising here, so a caller can inspect and document them;
        the full-run entry point is what refuses to start.

    Raises:
        ManifestError: If the audit contains no usable development images, or
            if the fractions do not name exactly the three development splits.
    """
    resolved_fractions = dict(split_fractions or DEFAULT_SPLIT_FRACTIONS)
    if set(resolved_fractions) != set(DEVELOPMENT_SPLIT_NAMES):
        raise ManifestError(
            f"split_fractions must name exactly {DEVELOPMENT_SPLIT_NAMES}, "
            f"got {sorted(resolved_fractions)}"
        )

    usable_records = audit.usable_records
    development_records = [
        record for record in usable_records if record.pool == DEVELOPMENT_POOL
    ]
    official_test_records = [
        record for record in usable_records if record.pool == OFFICIAL_TEST_POOL
    ]
    if not development_records:
        raise ManifestError("audit produced no usable development images to split")

    grouped_split = split_groups_into_stratified_parts(
        group_ids=[audit.split_group_id_of(record) for record in development_records],
        labels=[record.label for record in development_records],
        split_fractions=resolved_fractions,
        random_seed=split_seed,
    )

    entries: list[ManifestEntry] = []
    counts_by_split_and_label: dict[str, int] = {
        f"{split_name}|{label}": 0 for split_name in ALL_SPLIT_NAMES for label in (0, 1)
    }
    for record in development_records:
        split_group_id = audit.split_group_id_of(record)
        target_split = grouped_split.split_of_group[split_group_id]
        entries.append(_manifest_entry_from_record(record, target_split, split_group_id))
        counts_by_split_and_label[f"{target_split}|{record.label}"] += 1
    for record in official_test_records:
        split_group_id = audit.split_group_id_of(record)
        entries.append(
            _manifest_entry_from_record(record, OFFICIAL_TEST_SPLIT, split_group_id)
        )
        counts_by_split_and_label[f"{OFFICIAL_TEST_SPLIT}|{record.label}"] += 1

    patient_independence_established = (
        audit.has_confirmed_patient_identifiers()
        and audit.patient_identifier_scope == "global"
    )
    manifest = DatasetManifest(
        schema_version=MANIFEST_SCHEMA_VERSION,
        protocol_id=protocol_id,
        dataset_release=dataset_release,
        dataset_license=dataset_license,
        archive_root_relative_path=audit.root_relative_path,
        decoder_version=audit.decoder_version,
        patient_identifier_scope=audit.patient_identifier_scope,
        perceptual_hash_threshold=audit.perceptual_hash_threshold,
        patient_independence_established=patient_independence_established,
        split_seed=split_seed,
        requested_split_fractions=resolved_fractions,
        entries=tuple(sorted(entries, key=lambda entry: entry.example_id)),
        counts_by_split_and_label=counts_by_split_and_label,
        exclusions=tuple(
            ManifestExclusion(
                excluded_example_id=exclusion.excluded_example_id,
                kept_example_id=exclusion.kept_example_id,
                reason=exclusion.reason,
            )
            for exclusion in audit.exclusions
        ),
        findings=tuple(
            ManifestFinding(
                kind=finding.kind,
                severity=finding.severity,
                message=finding.message,
                example_ids=finding.example_ids,
            )
            for finding in audit.findings
        ),
        similar_pair_count=len(audit.similarity_pairs),
        cross_pool_similar_pair_count=sum(
            1 for pair in audit.similarity_pairs if pair.crosses_pools
        ),
    )
    logger.info(
        "Manifest built: %s, split counts %s, patient independence established: %s",
        protocol_id,
        {key: value for key, value in sorted(counts_by_split_and_label.items()) if value},
        patient_independence_established,
    )
    return manifest


def _manifest_entry_from_record(
    record: ImageRecord, target_split: str, split_group_id: str
) -> ManifestEntry:
    """Project one audited record onto its manifest entry."""
    return ManifestEntry(
        example_id=record.example_id,
        relative_path=record.relative_path,
        class_name=record.class_name,
        label=record.label,
        original_split=record.original_split,
        pool=record.pool,
        target_split=target_split,
        width=record.width,
        height=record.height,
        file_sha256=record.file_sha256,
        pixel_sha256=record.pixel_sha256,
        group_id=record.group_id,
        split_group_id=split_group_id,
        group_confidence=record.group_confidence,
        group_convention=record.group_convention,
    )
