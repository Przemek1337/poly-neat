"""Integrity audit of the pediatric chest X-ray archive.

The audit walks the extracted archive once and answers, for every file it
finds, four questions the protocol depends on: which split and class it belongs
to, whether it decodes at all, what it hashes to, and which group it must not
be separated from. Nothing is skipped quietly - an unreadable JPEG, an
unexpected class directory, a count that disagrees with the pinned release or a
duplicate that crosses into the official test set all become recorded findings.

Severity decides what may happen next:

* ``blocking`` - the full experiment must not start until the finding is
  resolved and documented. Conflicting labels, a duplicate shared between the
  development pool and the official test set, and a confirmed patient on both
  sides are blocking by protocol.
* ``warning`` - recorded and reported, does not stop the run. Perceptual
  similarity is always at most a warning: it is a signal to review, never proof
  of a duplicate and never grounds for deleting an image.
* ``info`` - observations worth carrying into the manifest.

The official test directory is read here for integrity only. Nothing in this
module returns test tensors or test metrics to the search, the trainer or the
pilot.

References:
    Kermany, D. S., Goldbaum, M., Cai, W., et al. (2018). Identifying Medical
        Diagnoses and Treatable Diseases by Image-Based Deep Learning. *Cell*,
        172(5), 1122-1131. DOI: 10.1016/j.cell.2018.02.010
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from examples.pediatric_pneumonia._image_digests import (
    DEFAULT_PERCEPTUAL_HASH_HAMMING_THRESHOLD,
    CorruptImageError,
    compute_file_sha256,
    compute_perceptual_hash,
    compute_pixel_sha256,
    describe_decoder_version,
    load_grayscale_image,
)
from examples.pediatric_pneumonia._patient_identifiers import (
    GLOBAL_IDENTIFIER_SCOPE,
    GroupConfidence,
    parse_group_assignment,
)
from polyneat.logging_utils.custom_logger import get_logger

logger = get_logger(__name__)

CLASS_LABELS: dict[str, int] = {"NORMAL": 0, "PNEUMONIA": 1}
POSITIVE_CLASS_NAME = "PNEUMONIA"

ORIGINAL_SPLIT_DIRECTORY_NAMES: tuple[str, ...] = ("train", "val", "test")
REQUIRED_SPLIT_DIRECTORY_NAMES: tuple[str, ...] = ("train", "test")

DEVELOPMENT_POOL = "development"
OFFICIAL_TEST_POOL = "official_test"

# How far a file-name counter is allowed to reach. The shipped train and test
# directories number their patients independently, so ``pool`` is the honest
# default: keys never merge across the development/official-test boundary, and
# the audit says outright that file names cannot settle whether a patient
# appears on both sides.
POOL_IDENTIFIER_SCOPE = "pool"
SUPPORTED_IDENTIFIER_SCOPES: tuple[str, ...] = (POOL_IDENTIFIER_SCOPE, GLOBAL_IDENTIFIER_SCOPE)

# The pool each original split feeds. Only the original train and val are
# merged; the official test directory is never mixed into development.
_POOL_OF_ORIGINAL_SPLIT: dict[str, str] = {
    "train": DEVELOPMENT_POOL,
    "val": DEVELOPMENT_POOL,
    "test": OFFICIAL_TEST_POOL,
}

# Counts expected from the pinned Kaggle release. They are a cross-check
# against the downloaded archive, not a proof of integrity or of patient
# disjointness: a deviation is reported, and the manifest always records the
# counts that were actually observed.
EXPECTED_POOL_CLASS_COUNTS: dict[tuple[str, str], int] = {
    (DEVELOPMENT_POOL, "NORMAL"): 1349,
    (DEVELOPMENT_POOL, "PNEUMONIA"): 3883,
    (OFFICIAL_TEST_POOL, "NORMAL"): 234,
    (OFFICIAL_TEST_POOL, "PNEUMONIA"): 390,
}

SUPPORTED_IMAGE_SUFFIXES: frozenset[str] = frozenset({".jpeg", ".jpg", ".png"})

# Only these are ignored without comment: they are operating-system metadata,
# never image data. Anything else unexpected becomes a finding.
IGNORED_FILE_NAMES: frozenset[str] = frozenset(
    {".DS_Store", "Thumbs.db", "desktop.ini", ".gitkeep"}
)
IGNORED_DIRECTORY_NAMES: frozenset[str] = frozenset({"__MACOSX", ".ipynb_checkpoints"})

# Candidate nesting levels of the archive root, relative to the data directory.
# Kaggle has shipped this dataset both flat and doubly nested.
_SUPPORTED_ROOT_RELATIVE_PATHS: tuple[tuple[str, ...], ...] = (
    (),
    ("chest_xray",),
    ("chest_xray", "chest_xray"),
)

BLOCKING = "blocking"
WARNING = "warning"
INFO = "info"


class DatasetLayoutError(RuntimeError):
    """Raised when the archive root cannot be identified unambiguously."""


@dataclass(frozen=True)
class ImageRecord:
    """One audited image and everything the split and manifest need from it.

    Attributes:
        example_id: Stable identifier, the archive-root-relative POSIX path.
            Predictions are exported against this id.
        relative_path: The same path string, kept as its own field so a future
            id scheme can change without losing the location on disk.
        class_name: ``NORMAL`` or ``PNEUMONIA`` as found on disk.
        label: ``0`` for NORMAL, ``1`` for PNEUMONIA.
        original_split: ``train``, ``val`` or ``test`` as shipped.
        pool: ``development`` (original train+val) or ``official_test``.
        width: Decoded image width in pixels.
        height: Decoded image height in pixels.
        file_sha256: Digest of the raw bytes.
        pixel_sha256: Digest of the decoded pixels and their shape.
        perceptual_hash: 64-bit difference hash, for the similarity report.
        group_id: Namespaced grouping key parsed from the file name.
        group_confidence: What that key is known to mean.
        group_convention: Which naming convention matched.
    """

    example_id: str
    relative_path: str
    class_name: str
    label: int
    original_split: str
    pool: str
    width: int
    height: int
    file_sha256: str
    pixel_sha256: str
    perceptual_hash: int
    group_id: str
    group_confidence: str
    group_convention: str


@dataclass(frozen=True)
class AuditFinding:
    """A single recorded observation about the archive.

    Attributes:
        kind: Machine-readable category, e.g. ``label_conflict``.
        severity: ``blocking``, ``warning`` or ``info``.
        message: Human-readable explanation carried into the report.
        example_ids: Records the finding refers to, possibly empty.
    """

    kind: str
    severity: str
    message: str
    example_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class ExclusionRecord:
    """One image removed from the development pool, and why.

    Attributes:
        excluded_example_id: The record that will not be used.
        kept_example_id: The record kept in its place.
        reason: Why the exclusion happened.
    """

    excluded_example_id: str
    kept_example_id: str
    reason: str


@dataclass(frozen=True)
class SimilarityPair:
    """Two images whose perceptual hashes are within the frozen threshold.

    A pair here is a prompt to look, nothing more. It never triggers an
    automatic exclusion and never proves the images are the same study.

    Attributes:
        first_example_id: Lexicographically smaller id of the pair.
        second_example_id: The other id.
        hamming_distance: Differing bits between the two fingerprints.
        crosses_pools: Whether the pair spans development and official test,
            which is the case a reviewer must look at first.
    """

    first_example_id: str
    second_example_id: str
    hamming_distance: int
    crosses_pools: bool


@dataclass(frozen=True)
class DatasetAudit:
    """Everything one pass over the archive established.

    Attributes:
        archive_root: Directory identified as the archive root.
        root_relative_path: Which supported nesting level matched, as a string.
        decoder_version: Pinned decoder identity used for the pixel digests.
        records: Every successfully decoded image, in sorted example-id order.
        findings: All recorded observations.
        exclusions: Development-pool images dropped as exact duplicates.
        similarity_pairs: Perceptual-hash matches, for manual review.
        ignored_paths: Known technical files skipped on purpose.
        perceptual_hash_threshold: Frozen Hamming threshold used above.
        patient_identifier_scope: Whether file-name counters were treated as
            comparable across the whole archive or only inside one pool.
        group_merges: Parsed group ids merged because their images are exact
            duplicates, as ``{merged_group_id: canonical_group_id}``.
    """

    archive_root: Path
    root_relative_path: str
    decoder_version: str
    records: tuple[ImageRecord, ...]
    findings: tuple[AuditFinding, ...]
    exclusions: tuple[ExclusionRecord, ...]
    similarity_pairs: tuple[SimilarityPair, ...]
    ignored_paths: tuple[str, ...]
    perceptual_hash_threshold: int
    patient_identifier_scope: str = POOL_IDENTIFIER_SCOPE
    group_merges: dict[str, str] = field(default_factory=dict)

    @property
    def blocking_findings(self) -> tuple[AuditFinding, ...]:
        """Findings that must be resolved before a full experiment may run."""
        return tuple(finding for finding in self.findings if finding.severity == BLOCKING)

    @property
    def excluded_example_ids(self) -> frozenset[str]:
        """Ids dropped from the development pool by exact-duplicate reduction."""
        return frozenset(exclusion.excluded_example_id for exclusion in self.exclusions)

    @property
    def usable_records(self) -> tuple[ImageRecord, ...]:
        """Records that survive the exclusion map, in example-id order."""
        excluded = self.excluded_example_ids
        return tuple(record for record in self.records if record.example_id not in excluded)

    def split_group_id_of(self, record: ImageRecord) -> str:
        """Return the group id the split must keep together.

        Exact duplicates inside the development pool are merged before the
        split, so a record's effective group can differ from the one parsed
        from its file name.

        Args:
            record: Any record from this audit.

        Returns:
            The canonical group id after duplicate-driven merges.
        """
        return self.group_merges.get(record.group_id, record.group_id)

    def counts_by_pool_and_class(self) -> dict[tuple[str, str], int]:
        """Observed image counts per ``(pool, class_name)``, before exclusions."""
        counts: dict[tuple[str, str], int] = defaultdict(int)
        for record in self.records:
            counts[(record.pool, record.class_name)] += 1
        return dict(counts)

    def has_confirmed_patient_identifiers(self) -> bool:
        """Whether every usable record carries a documented patient identifier.

        Returns:
            ``True`` only when no record falls back to a study-series or
            unknown key. When this is ``False`` the report must state that
            patient independence was not established, and the test bootstrap
            resamples images rather than patients.
        """
        return all(
            record.group_confidence == GroupConfidence.CONFIRMED_PATIENT.value
            for record in self.usable_records
        )


def resolve_archive_root(data_directory: Path) -> Path:
    """Find the one directory that holds the shipped split directories.

    Args:
        data_directory: Directory the archive was extracted into.

    Returns:
        The archive root.

    Raises:
        DatasetLayoutError: If no supported nesting level contains the required
            split directories, or if more than one does. Two competing roots is
            an error, not permission to load the dataset twice.
    """
    if not data_directory.is_dir():
        raise DatasetLayoutError(f"data directory {data_directory} does not exist")

    matching_roots: list[Path] = []
    for relative_parts in _SUPPORTED_ROOT_RELATIVE_PATHS:
        candidate_root = data_directory.joinpath(*relative_parts)
        if not candidate_root.is_dir():
            continue
        if all(
            (candidate_root / split_name).is_dir()
            for split_name in REQUIRED_SPLIT_DIRECTORY_NAMES
        ):
            matching_roots.append(candidate_root)

    if not matching_roots:
        supported = ", ".join(
            repr("/".join(parts) or ".") for parts in _SUPPORTED_ROOT_RELATIVE_PATHS
        )
        raise DatasetLayoutError(
            f"no supported chest_xray layout under {data_directory}: expected the "
            f"{' and '.join(REQUIRED_SPLIT_DIRECTORY_NAMES)} directories at one of {supported}"
        )
    if len(matching_roots) > 1:
        competing = ", ".join(str(root) for root in matching_roots)
        raise DatasetLayoutError(
            f"ambiguous chest_xray layout: {competing} all look like archive roots; "
            "remove the redundant nesting instead of loading the dataset twice"
        )
    return matching_roots[0]


def _relative_posix_path(file_path: Path, archive_root: Path) -> str:
    """Archive-root-relative path with forward slashes, stable across platforms."""
    return file_path.relative_to(archive_root).as_posix()


def _scan_split_directory(
    split_directory: Path,
    archive_root: Path,
    original_split: str,
    findings: list[AuditFinding],
    ignored_paths: list[str],
) -> list[tuple[Path, str, str]]:
    """Collect candidate image files of one shipped split.

    Args:
        split_directory: ``<root>/train`` and friends.
        archive_root: Root the example ids are relative to.
        original_split: Name of the shipped split being scanned.
        findings: Mutable list the scan appends its observations to.
        ignored_paths: Mutable list of deliberately skipped technical files.

    Returns:
        ``(file_path, class_name, example_id)`` triples for every file that
        looks like an image of a known class.
    """
    candidate_files: list[tuple[Path, str, str]] = []
    for class_directory in sorted(split_directory.iterdir()):
        if class_directory.is_file():
            if class_directory.name in IGNORED_FILE_NAMES:
                ignored_paths.append(_relative_posix_path(class_directory, archive_root))
                continue
            findings.append(
                AuditFinding(
                    kind="unexpected_file",
                    severity=WARNING,
                    message=(
                        f"{_relative_posix_path(class_directory, archive_root)} sits directly "
                        f"in split {original_split!r} instead of a class directory"
                    ),
                )
            )
            continue
        if class_directory.name in IGNORED_DIRECTORY_NAMES:
            ignored_paths.append(_relative_posix_path(class_directory, archive_root))
            continue
        class_name = class_directory.name.upper()
        if class_name not in CLASS_LABELS:
            findings.append(
                AuditFinding(
                    kind="unknown_class_directory",
                    severity=BLOCKING,
                    message=(
                        f"split {original_split!r} contains class directory "
                        f"{class_directory.name!r}, which is not one of "
                        f"{sorted(CLASS_LABELS)}"
                    ),
                )
            )
            continue

        for file_path in sorted(class_directory.rglob("*")):
            if file_path.is_dir():
                continue
            if any(part in IGNORED_DIRECTORY_NAMES for part in file_path.parts):
                ignored_paths.append(_relative_posix_path(file_path, archive_root))
                continue
            if file_path.name in IGNORED_FILE_NAMES:
                ignored_paths.append(_relative_posix_path(file_path, archive_root))
                continue
            if file_path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
                findings.append(
                    AuditFinding(
                        kind="unexpected_file",
                        severity=WARNING,
                        message=(
                            f"{_relative_posix_path(file_path, archive_root)} has unsupported "
                            f"suffix {file_path.suffix!r} and was not loaded"
                        ),
                    )
                )
                continue
            candidate_files.append(
                (file_path, class_name, _relative_posix_path(file_path, archive_root))
            )
    return candidate_files


def _build_image_record(
    file_path: Path,
    class_name: str,
    example_id: str,
    original_split: str,
    identifier_scope: str,
) -> ImageRecord:
    """Decode one file and turn it into an :class:`ImageRecord`.

    Raises:
        CorruptImageError: Propagated from the decoder so the caller can record
            the file as a finding instead of dropping it silently.
    """
    grayscale_pixels = load_grayscale_image(file_path)
    height, width = grayscale_pixels.shape
    group_assignment = parse_group_assignment(
        file_path.stem, fallback_key=example_id, identifier_scope=identifier_scope
    )
    return ImageRecord(
        example_id=example_id,
        relative_path=example_id,
        class_name=class_name,
        label=CLASS_LABELS[class_name],
        original_split=original_split,
        pool=_POOL_OF_ORIGINAL_SPLIT[original_split],
        width=int(width),
        height=int(height),
        file_sha256=compute_file_sha256(file_path),
        pixel_sha256=compute_pixel_sha256(grayscale_pixels),
        perceptual_hash=compute_perceptual_hash(grayscale_pixels),
        group_id=group_assignment.group_id,
        group_confidence=group_assignment.confidence.value,
        group_convention=group_assignment.convention,
    )


def _check_expected_counts(
    records: tuple[ImageRecord, ...],
    expected_counts: dict[tuple[str, str], int],
    findings: list[AuditFinding],
) -> None:
    """Compare observed per-pool class counts with the pinned release."""
    observed_counts: dict[tuple[str, str], int] = defaultdict(int)
    for record in records:
        observed_counts[(record.pool, record.class_name)] += 1
    for pool_and_class, expected_count in sorted(expected_counts.items()):
        observed_count = observed_counts.get(pool_and_class, 0)
        if observed_count != expected_count:
            pool_name, class_name = pool_and_class
            findings.append(
                AuditFinding(
                    kind="count_deviation",
                    severity=WARNING,
                    message=(
                        f"{pool_name}/{class_name}: found {observed_count} images, the pinned "
                        f"release expects {expected_count}; the manifest records the observed "
                        "count and the report must state the deviation"
                    ),
                )
            )


class _GroupUnionFind:
    """Union-find over parsed group ids, used to merge duplicate-linked groups."""

    def __init__(self) -> None:
        self._parent_of: dict[str, str] = {}

    def find(self, group_id: str) -> str:
        """Return the canonical id of the set containing ``group_id``."""
        self._parent_of.setdefault(group_id, group_id)
        root = group_id
        while self._parent_of[root] != root:
            root = self._parent_of[root]
        while self._parent_of[group_id] != root:
            self._parent_of[group_id], group_id = root, self._parent_of[group_id]
        return root

    def union(self, first_group_id: str, second_group_id: str) -> None:
        """Merge two sets, keeping the lexicographically smaller id as root."""
        first_root = self.find(first_group_id)
        second_root = self.find(second_group_id)
        if first_root == second_root:
            return
        canonical, merged = sorted((first_root, second_root))
        self._parent_of[merged] = canonical

    def merge_map(self) -> dict[str, str]:
        """Return ``{group_id: canonical_id}`` for every id that actually moved."""
        return {
            group_id: self.find(group_id)
            for group_id in sorted(self._parent_of)
            if self.find(group_id) != group_id
        }


def _analyze_exact_duplicates(
    records: tuple[ImageRecord, ...],
    findings: list[AuditFinding],
    group_union: _GroupUnionFind,
) -> list[ExclusionRecord]:
    """Resolve pixel-identical records according to the protocol.

    Three cases, deliberately treated differently:

    * conflicting labels on identical pixels - blocking, because no split can
      be trusted while the same image carries two diagnoses;
    * identical pixels on both sides of the development/official-test boundary
      - blocking, and the test set is never edited automatically;
    * identical pixels inside the development pool - reduced to one
      deterministically chosen record, with the removed ones written into the
      exclusion map and their parsed groups merged before the split.

    Args:
        records: All audited records.
        findings: Mutable list the analysis appends to.
        group_union: Union-find that collects duplicate-driven group merges.

    Returns:
        The exclusion records for the development pool, in example-id order.
    """
    records_by_pixel_digest: dict[str, list[ImageRecord]] = defaultdict(list)
    for record in records:
        records_by_pixel_digest[record.pixel_sha256].append(record)

    exclusions: list[ExclusionRecord] = []
    for pixel_digest, duplicate_records in sorted(records_by_pixel_digest.items()):
        if len(duplicate_records) < 2:
            continue
        duplicate_records = sorted(duplicate_records, key=lambda item: item.example_id)
        duplicate_ids = tuple(record.example_id for record in duplicate_records)

        if len({record.label for record in duplicate_records}) > 1:
            findings.append(
                AuditFinding(
                    kind="label_conflict",
                    severity=BLOCKING,
                    message=(
                        f"pixel digest {pixel_digest[:12]} appears under more than one class: "
                        + ", ".join(
                            f"{record.example_id}={record.class_name}"
                            for record in duplicate_records
                        )
                    ),
                    example_ids=duplicate_ids,
                )
            )
            continue

        pools_present = {record.pool for record in duplicate_records}
        if len(pools_present) > 1:
            findings.append(
                AuditFinding(
                    kind="cross_pool_duplicate",
                    severity=BLOCKING,
                    message=(
                        f"pixel digest {pixel_digest[:12]} appears in both the development pool "
                        "and the official test set; resolve and document this before running "
                        "the full experiment - the test set is never edited automatically"
                    ),
                    example_ids=duplicate_ids,
                )
            )
            continue

        if pools_present == {OFFICIAL_TEST_POOL}:
            findings.append(
                AuditFinding(
                    kind="duplicate_inside_official_test",
                    severity=WARNING,
                    message=(
                        f"pixel digest {pixel_digest[:12]} appears {len(duplicate_records)} times "
                        "inside the official test set; recorded, not modified"
                    ),
                    example_ids=duplicate_ids,
                )
            )
            continue

        kept_record, *removed_records = duplicate_records
        for removed_record in removed_records:
            group_union.union(kept_record.group_id, removed_record.group_id)
            exclusions.append(
                ExclusionRecord(
                    excluded_example_id=removed_record.example_id,
                    kept_example_id=kept_record.example_id,
                    reason="exact_pixel_duplicate_in_development_pool",
                )
            )
        findings.append(
            AuditFinding(
                kind="duplicate_inside_development",
                severity=INFO,
                message=(
                    f"pixel digest {pixel_digest[:12]} appears {len(duplicate_records)} times in "
                    f"the development pool; kept {kept_record.example_id}"
                ),
                example_ids=duplicate_ids,
            )
        )
    return sorted(exclusions, key=lambda item: item.excluded_example_id)


def _analyze_group_overlap(
    records: tuple[ImageRecord, ...],
    group_merges: dict[str, str],
    excluded_example_ids: frozenset[str],
    findings: list[AuditFinding],
) -> None:
    """Report groups that straddle the development pool and the official test.

    A confirmed patient on both sides is blocking. A study-series key on both
    sides is a warning: the key was never documented as patient identity, so it
    can neither prove nor disprove leakage on its own.
    """
    records_by_group: dict[str, list[ImageRecord]] = defaultdict(list)
    for record in records:
        if record.example_id in excluded_example_ids:
            continue
        records_by_group[group_merges.get(record.group_id, record.group_id)].append(record)

    for group_id, group_records in sorted(records_by_group.items()):
        pools_present = {record.pool for record in group_records}
        if len(pools_present) < 2:
            continue
        group_ids = tuple(record.example_id for record in group_records)
        is_confirmed_patient = any(
            record.group_confidence == GroupConfidence.CONFIRMED_PATIENT.value
            for record in group_records
        )
        if is_confirmed_patient:
            findings.append(
                AuditFinding(
                    kind="shared_patient_across_pools",
                    severity=BLOCKING,
                    message=(
                        f"confirmed patient group {group_id!r} has images in both the "
                        "development pool and the official test set"
                    ),
                    example_ids=group_ids,
                )
            )
        else:
            findings.append(
                AuditFinding(
                    kind="shared_group_across_pools",
                    severity=WARNING,
                    message=(
                        f"group {group_id!r} spans the development pool and the official test "
                        "set, but its key is not a documented patient identifier, so this "
                        "neither proves nor rules out leakage"
                    ),
                    example_ids=group_ids,
                )
            )


def _report_identifier_confidence(
    records: tuple[ImageRecord, ...],
    findings: list[AuditFinding],
) -> None:
    """Summarise how much of the archive carries a documented patient id."""
    counts_by_confidence: dict[str, int] = defaultdict(int)
    for record in records:
        counts_by_confidence[record.group_confidence] += 1
    unknown_count = counts_by_confidence.get(GroupConfidence.UNKNOWN.value, 0)
    series_count = counts_by_confidence.get(GroupConfidence.STUDY_SERIES.value, 0)
    findings.append(
        AuditFinding(
            kind="identifier_confidence",
            severity=INFO if unknown_count == 0 else WARNING,
            message=(
                "patient identifier confidence: "
                + ", ".join(
                    f"{confidence}={count}"
                    for confidence, count in sorted(counts_by_confidence.items())
                )
                + (
                    "; patient independence is NOT established for the "
                    f"{series_count + unknown_count} images without a documented patient id"
                    if series_count or unknown_count
                    else ""
                )
            ),
        )
    )


def _resolve_identifier_scopes(patient_identifier_scope: str) -> dict[str, str]:
    """Map each shipped split to the scope its file-name counters belong to.

    Args:
        patient_identifier_scope: ``pool`` or ``global``.

    Returns:
        ``{original_split: identifier_scope}``. Under ``pool`` the original
        train and val share the development scope, so their counters merge with
        each other but never with the official test set.

    Raises:
        ValueError: If the scope is not one of the supported values.
    """
    if patient_identifier_scope == GLOBAL_IDENTIFIER_SCOPE:
        return dict.fromkeys(ORIGINAL_SPLIT_DIRECTORY_NAMES, GLOBAL_IDENTIFIER_SCOPE)
    if patient_identifier_scope == POOL_IDENTIFIER_SCOPE:
        return {
            split_name: _POOL_OF_ORIGINAL_SPLIT[split_name]
            for split_name in ORIGINAL_SPLIT_DIRECTORY_NAMES
        }
    raise ValueError(
        f"unsupported patient_identifier_scope {patient_identifier_scope!r}; "
        f"expected one of {SUPPORTED_IDENTIFIER_SCOPES}"
    )


def _report_identifier_scope(patient_identifier_scope: str, findings: list[AuditFinding]) -> None:
    """Record how far identifiers were allowed to reach, and what that costs."""
    if patient_identifier_scope == POOL_IDENTIFIER_SCOPE:
        findings.append(
            AuditFinding(
                kind="patient_identifier_scope",
                severity=WARNING,
                message=(
                    "file-name counters were treated as comparable only inside one pool, "
                    "because the shipped train and test directories number their patients "
                    "independently; consequently the file names can neither confirm nor rule "
                    "out that a patient appears in both the development pool and the official "
                    "test set, and the report must say so"
                ),
            )
        )
        return
    findings.append(
        AuditFinding(
            kind="patient_identifier_scope",
            severity=INFO,
            message=(
                "file-name counters were treated as comparable across the whole archive; this "
                "is only correct for a release that documents one archive-wide numbering"
            ),
        )
    )


def _build_similarity_report(
    records: tuple[ImageRecord, ...],
    hamming_threshold: int,
    maximum_pairs: int,
    findings: list[AuditFinding],
) -> tuple[SimilarityPair, ...]:
    """List perceptually similar image pairs, without acting on them.

    Pixel-identical pairs are left out: they are already handled as exact
    duplicates, and repeating them here would bury the near-duplicates this
    report exists to surface.

    Args:
        records: Records to compare, in example-id order.
        hamming_threshold: Frozen maximum differing-bit count.
        maximum_pairs: Cap on reported pairs, so a pathological archive cannot
            produce an unbounded report. Truncation is itself a finding.
        findings: Mutable list the report appends to.

    Returns:
        Similar pairs, ordered by distance and then by example id.
    """
    if len(records) < 2:
        return ()

    fingerprints = np.array([record.perceptual_hash for record in records], dtype=np.uint64)
    pixel_digests = [record.pixel_sha256 for record in records]
    pools = [record.pool for record in records]

    similar_pairs: list[SimilarityPair] = []
    was_truncated = False
    block_size = 256
    for block_start in range(0, len(records), block_size):
        block_end = min(block_start + block_size, len(records))
        differing_bits = np.bitwise_count(
            np.bitwise_xor(fingerprints[block_start:block_end, None], fingerprints[None, :])
        )
        for row_offset in range(block_end - block_start):
            first_index = block_start + row_offset
            candidate_indices = np.flatnonzero(
                differing_bits[row_offset] <= hamming_threshold
            )
            for second_index in candidate_indices.tolist():
                if second_index <= first_index:
                    continue
                if pixel_digests[first_index] == pixel_digests[second_index]:
                    continue
                if len(similar_pairs) >= maximum_pairs:
                    was_truncated = True
                    break
                similar_pairs.append(
                    SimilarityPair(
                        first_example_id=records[first_index].example_id,
                        second_example_id=records[second_index].example_id,
                        hamming_distance=int(differing_bits[row_offset, second_index]),
                        crosses_pools=pools[first_index] != pools[second_index],
                    )
                )
            if was_truncated:
                break
        if was_truncated:
            break

    if was_truncated:
        findings.append(
            AuditFinding(
                kind="similarity_report_truncated",
                severity=WARNING,
                message=(
                    f"perceptual similarity report stopped at {maximum_pairs} pairs; raise the "
                    "cap or tighten the threshold to see the rest"
                ),
            )
        )
    cross_pool_pairs = [pair for pair in similar_pairs if pair.crosses_pools]
    if cross_pool_pairs:
        findings.append(
            AuditFinding(
                kind="cross_pool_similarity",
                severity=WARNING,
                message=(
                    f"{len(cross_pool_pairs)} perceptually similar pairs span the development "
                    "pool and the official test set; review them - similarity is not proof of "
                    "a duplicate and never justifies deleting an image automatically"
                ),
                example_ids=tuple(
                    example_id
                    for pair in cross_pool_pairs[:50]
                    for example_id in (pair.first_example_id, pair.second_example_id)
                ),
            )
        )
    return tuple(
        sorted(
            similar_pairs,
            key=lambda pair: (pair.hamming_distance, pair.first_example_id, pair.second_example_id),
        )
    )


def audit_pediatric_pneumonia_dataset(
    data_directory: Path,
    *,
    perceptual_hash_threshold: int = DEFAULT_PERCEPTUAL_HASH_HAMMING_THRESHOLD,
    compute_similarity_report: bool = True,
    maximum_similarity_pairs: int = 5_000,
    expected_counts: dict[tuple[str, str], int] | None = None,
    patient_identifier_scope: str = POOL_IDENTIFIER_SCOPE,
) -> DatasetAudit:
    """Walk the extracted archive once and record everything the protocol needs.

    The official test directory is read here so its integrity can be checked
    before any training happens. Reading it for the audit is explicitly allowed;
    handing its tensors or metrics to the search, the trainer or the pilot is
    not, and no return value of this function carries test predictions.

    Args:
        data_directory: Directory the Kaggle archive was extracted into. Any of
            the supported nesting levels is accepted, exactly one of them.
        perceptual_hash_threshold: Frozen Hamming threshold for the similarity
            report. Changing it changes the audit and needs a new protocol id.
        compute_similarity_report: Set to ``False`` to skip the pairwise
            comparison when only the split-relevant checks are needed.
        maximum_similarity_pairs: Cap on similar pairs recorded.
        expected_counts: Per ``(pool, class_name)`` counts to cross-check
            against. Defaults to the pinned release counts.
        patient_identifier_scope: ``pool`` (the default) treats the counters in
            the development pool and the official test set as independent,
            which is what this release's naming actually supports. ``global``
            makes them comparable and is only correct for a release that
            documents one archive-wide numbering.

    Returns:
        The :class:`DatasetAudit`. Callers must inspect ``blocking_findings``
        before starting a full experiment.

    Raises:
        DatasetLayoutError: If the archive root is missing or ambiguous.
    """
    archive_root = resolve_archive_root(data_directory)
    root_relative_path = archive_root.relative_to(data_directory).as_posix() or "."
    logger.info("Auditing pediatric pneumonia archive at %s", archive_root)

    findings: list[AuditFinding] = []
    ignored_paths: list[str] = []
    candidate_files: list[tuple[Path, str, str, str]] = []

    for original_split in ORIGINAL_SPLIT_DIRECTORY_NAMES:
        split_directory = archive_root / original_split
        if not split_directory.is_dir():
            findings.append(
                AuditFinding(
                    kind="missing_split_directory",
                    severity=BLOCKING
                    if original_split in REQUIRED_SPLIT_DIRECTORY_NAMES
                    else INFO,
                    message=f"split directory {original_split!r} is missing under {archive_root}",
                )
            )
            continue
        for file_path, class_name, example_id in _scan_split_directory(
            split_directory, archive_root, original_split, findings, ignored_paths
        ):
            candidate_files.append((file_path, class_name, example_id, original_split))

    identifier_scope_of_split = _resolve_identifier_scopes(patient_identifier_scope)

    records: list[ImageRecord] = []
    for file_path, class_name, example_id, original_split in candidate_files:
        try:
            records.append(
                _build_image_record(
                    file_path,
                    class_name,
                    example_id,
                    original_split,
                    identifier_scope_of_split[original_split],
                )
            )
        except CorruptImageError:
            findings.append(
                AuditFinding(
                    kind="corrupt_image",
                    severity=WARNING,
                    message=f"{example_id} could not be decoded and was not loaded",
                    example_ids=(example_id,),
                )
            )

    sorted_records = tuple(sorted(records, key=lambda record: record.example_id))
    if not sorted_records:
        findings.append(
            AuditFinding(
                kind="empty_dataset",
                severity=BLOCKING,
                message=f"no decodable images found under {archive_root}",
            )
        )

    _check_expected_counts(
        sorted_records,
        EXPECTED_POOL_CLASS_COUNTS if expected_counts is None else expected_counts,
        findings,
    )
    _report_identifier_confidence(sorted_records, findings)
    _report_identifier_scope(patient_identifier_scope, findings)

    group_union = _GroupUnionFind()
    exclusions = _analyze_exact_duplicates(sorted_records, findings, group_union)
    group_merges = group_union.merge_map()
    _analyze_group_overlap(
        sorted_records,
        group_merges,
        frozenset(exclusion.excluded_example_id for exclusion in exclusions),
        findings,
    )

    similarity_pairs: tuple[SimilarityPair, ...] = ()
    if compute_similarity_report:
        similarity_pairs = _build_similarity_report(
            sorted_records, perceptual_hash_threshold, maximum_similarity_pairs, findings
        )

    audit = DatasetAudit(
        archive_root=archive_root,
        root_relative_path=root_relative_path,
        decoder_version=describe_decoder_version(),
        records=sorted_records,
        findings=tuple(findings),
        exclusions=tuple(exclusions),
        similarity_pairs=similarity_pairs,
        ignored_paths=tuple(sorted(ignored_paths)),
        perceptual_hash_threshold=perceptual_hash_threshold,
        patient_identifier_scope=patient_identifier_scope,
        group_merges=group_merges,
    )
    logger.info(
        "Audit finished: %d images, %d usable after exclusions, %d findings (%d blocking), "
        "%d similar pairs",
        len(audit.records),
        len(audit.usable_records),
        len(audit.findings),
        len(audit.blocking_findings),
        len(audit.similarity_pairs),
    )
    for blocking_finding in audit.blocking_findings:
        logger.warning("BLOCKING %s: %s", blocking_finding.kind, blocking_finding.message)
    return audit
