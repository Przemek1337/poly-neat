"""Tests for the pediatric pneumonia archive: identifiers, audit, split, manifest."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

from examples._datasets import split_groups_into_stratified_parts
from examples.pediatric_pneumonia._dataset_audit import (
    BLOCKING,
    DEVELOPMENT_POOL,
    OFFICIAL_TEST_POOL,
    WARNING,
    DatasetLayoutError,
    audit_pediatric_pneumonia_dataset,
    resolve_archive_root,
)
from examples.pediatric_pneumonia._dataset_manifest import (
    OFFICIAL_TEST_SPLIT,
    DatasetManifest,
    ManifestError,
    build_manifest_from_audit,
)
from examples.pediatric_pneumonia._image_digests import (
    compute_perceptual_hash,
    compute_pixel_sha256,
    perceptual_hash_hamming_distance,
)
from examples.pediatric_pneumonia._patient_identifiers import (
    GroupConfidence,
    parse_group_assignment,
)
from examples.pediatric_pneumonia._protocol_lock import (
    ProtocolLockError,
    load_protocol_lock,
    validate_protocol_lock_for_full_run,
)
from examples.pediatric_pneumonia._synthetic_archive import (
    write_synthetic_archive,
)


def _write_image(path: Path, pixels: np.ndarray, quality: int = 92) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels, mode="L").save(path, format="JPEG", quality=quality)


def _gradient_pixels(height: int = 32, width: int = 40, offset: int = 0) -> np.ndarray:
    rows = np.linspace(10 + offset, 200 + offset, height, dtype=np.float32)[:, None]
    columns = np.linspace(5, 60, width, dtype=np.float32)[None, :]
    return np.clip(rows + columns, 0, 255).astype(np.uint8)


@pytest.fixture
def synthetic_archive(tmp_path: Path) -> Path:
    """A tiny archive with the real nesting and naming conventions."""
    return write_synthetic_archive(tmp_path / "data", random_seed=4242)


class TestPatientIdentifiers:
    def test_person_prefix_is_the_only_confirmed_patient_convention(self) -> None:
        assignment = parse_group_assignment("person17_bacteria_3", fallback_key="x")
        assert assignment.confidence is GroupConfidence.CONFIRMED_PATIENT
        assert assignment.group_id.endswith("person:17")
        assert assignment.convention == "person_patient"

    @pytest.mark.parametrize(
        "file_stem",
        ["IM-0115-0001", "NORMAL2-IM-0222-0001", "IM-0115-0001-0001", "BACTERIA-1135262-0001"],
    )
    def test_series_names_never_claim_patient_identity(self, file_stem: str) -> None:
        assignment = parse_group_assignment(file_stem, fallback_key="x")
        assert assignment.confidence is GroupConfidence.STUDY_SERIES

    def test_unmatched_name_becomes_a_flagged_singleton(self) -> None:
        first = parse_group_assignment("mystery-file-a", fallback_key="train/NORMAL/a.jpeg")
        second = parse_group_assignment("mystery-file-b", fallback_key="train/NORMAL/b.jpeg")
        assert first.confidence is GroupConfidence.UNKNOWN
        assert first.group_id != second.group_id

    def test_different_collections_never_share_a_group(self) -> None:
        normal_two = parse_group_assignment("NORMAL2-IM-0222-0001", fallback_key="x")
        plain = parse_group_assignment("IM-0222-0001", fallback_key="y")
        assert normal_two.group_id != plain.group_id

    def test_scope_keeps_counters_from_different_pools_apart(self) -> None:
        development = parse_group_assignment(
            "person1_virus_1", fallback_key="x", identifier_scope="development"
        )
        official_test = parse_group_assignment(
            "person1_virus_1", fallback_key="y", identifier_scope="official_test"
        )
        assert development.group_id != official_test.group_id


class TestImageDigests:
    def test_pixel_digest_survives_reencoding_but_file_digest_does_not(
        self, tmp_path: Path
    ) -> None:
        pixels = _gradient_pixels()
        first_path = tmp_path / "a.jpeg"
        second_path = tmp_path / "b.jpeg"
        _write_image(first_path, pixels, quality=92)
        _write_image(second_path, pixels, quality=92)
        first_decoded = np.asarray(Image.open(first_path).convert("L"), dtype=np.uint8)
        second_decoded = np.asarray(Image.open(second_path).convert("L"), dtype=np.uint8)
        assert compute_pixel_sha256(first_decoded) == compute_pixel_sha256(second_decoded)

    def test_pixel_digest_includes_dimensions(self) -> None:
        pixels = np.arange(24, dtype=np.uint8)
        assert compute_pixel_sha256(pixels.reshape(4, 6)) != compute_pixel_sha256(
            pixels.reshape(6, 4)
        )

    def test_perceptual_hash_is_close_for_a_rescaled_copy(self) -> None:
        pixels = _gradient_pixels(64, 64)
        rescaled = np.asarray(
            Image.fromarray(pixels, mode="L").resize((48, 48)), dtype=np.uint8
        )
        distance = perceptual_hash_hamming_distance(
            compute_perceptual_hash(pixels), compute_perceptual_hash(rescaled)
        )
        assert distance <= 5


class TestArchiveRootResolution:
    def test_flat_layout_is_accepted(self, tmp_path: Path) -> None:
        for split in ("train", "test"):
            (tmp_path / split / "NORMAL").mkdir(parents=True)
        assert resolve_archive_root(tmp_path) == tmp_path

    def test_double_nesting_is_accepted(self, tmp_path: Path) -> None:
        nested = tmp_path / "chest_xray" / "chest_xray"
        for split in ("train", "test"):
            (nested / split / "NORMAL").mkdir(parents=True)
        assert resolve_archive_root(tmp_path) == nested

    def test_two_competing_roots_are_an_error(self, tmp_path: Path) -> None:
        for root in (tmp_path / "chest_xray", tmp_path / "chest_xray" / "chest_xray"):
            for split in ("train", "test"):
                (root / split / "NORMAL").mkdir(parents=True)
        with pytest.raises(DatasetLayoutError, match="ambiguous"):
            resolve_archive_root(tmp_path)

    def test_missing_layout_is_an_error(self, tmp_path: Path) -> None:
        (tmp_path / "something_else").mkdir()
        with pytest.raises(DatasetLayoutError, match="no supported"):
            resolve_archive_root(tmp_path)


class TestDatasetAudit:
    def test_synthetic_archive_audits_without_blocking_findings(
        self, synthetic_archive: Path
    ) -> None:
        audit = audit_pediatric_pneumonia_dataset(synthetic_archive, expected_counts={})
        assert audit.blocking_findings == ()
        assert len(audit.records) == len(audit.usable_records)
        pools = {record.pool for record in audit.records}
        assert pools == {DEVELOPMENT_POOL, OFFICIAL_TEST_POOL}

    def test_original_val_joins_development_and_test_stays_separate(
        self, synthetic_archive: Path
    ) -> None:
        audit = audit_pediatric_pneumonia_dataset(synthetic_archive, expected_counts={})
        pool_of_split = {
            record.original_split: record.pool for record in audit.records
        }
        assert pool_of_split["train"] == DEVELOPMENT_POOL
        assert pool_of_split["val"] == DEVELOPMENT_POOL
        assert pool_of_split["test"] == OFFICIAL_TEST_POOL

    def test_unknown_class_directory_blocks(self, synthetic_archive: Path) -> None:
        root = resolve_archive_root(synthetic_archive)
        _write_image(root / "train" / "TUBERCULOSIS" / "IM-9999-0001.jpeg", _gradient_pixels())
        audit = audit_pediatric_pneumonia_dataset(synthetic_archive, expected_counts={})
        kinds = {finding.kind for finding in audit.blocking_findings}
        assert "unknown_class_directory" in kinds

    def test_corrupt_image_is_reported_and_not_loaded(self, synthetic_archive: Path) -> None:
        root = resolve_archive_root(synthetic_archive)
        broken_path = root / "train" / "NORMAL" / "IM-8888-0001.jpeg"
        broken_path.write_bytes(b"this is not a jpeg")
        audit = audit_pediatric_pneumonia_dataset(synthetic_archive, expected_counts={})
        corrupt_findings = [f for f in audit.findings if f.kind == "corrupt_image"]
        assert len(corrupt_findings) == 1
        assert corrupt_findings[0].severity == WARNING
        assert all(
            record.example_id != "train/NORMAL/IM-8888-0001.jpeg" for record in audit.records
        )

    def test_count_deviation_is_reported_against_the_pinned_release(
        self, synthetic_archive: Path
    ) -> None:
        audit = audit_pediatric_pneumonia_dataset(synthetic_archive)
        deviations = [f for f in audit.findings if f.kind == "count_deviation"]
        assert deviations, "synthetic counts differ from the pinned release and must be reported"
        assert all(finding.severity == WARNING for finding in deviations)

    def test_known_technical_files_are_ignored_but_recorded(
        self, synthetic_archive: Path
    ) -> None:
        root = resolve_archive_root(synthetic_archive)
        (root / "train" / "NORMAL" / ".DS_Store").write_bytes(b"\x00")
        audit = audit_pediatric_pneumonia_dataset(synthetic_archive, expected_counts={})
        assert any(path.endswith(".DS_Store") for path in audit.ignored_paths)
        assert not [f for f in audit.findings if f.kind == "unexpected_file"]

    def test_unexpected_file_is_a_finding_not_a_silent_skip(
        self, synthetic_archive: Path
    ) -> None:
        root = resolve_archive_root(synthetic_archive)
        (root / "train" / "NORMAL" / "notes.txt").write_text("hello", encoding="utf-8")
        audit = audit_pediatric_pneumonia_dataset(synthetic_archive, expected_counts={})
        assert any(f.kind == "unexpected_file" for f in audit.findings)


class TestDuplicatesAndConflicts:
    def test_exact_duplicate_in_development_is_excluded_and_grouped(
        self, synthetic_archive: Path
    ) -> None:
        root = resolve_archive_root(synthetic_archive)
        original = sorted((root / "train" / "NORMAL").glob("*.jpeg"))[0]
        copy_path = root / "val" / "NORMAL" / "NORMAL2-IM-7777-0001.jpeg"
        copy_path.write_bytes(original.read_bytes())
        audit = audit_pediatric_pneumonia_dataset(synthetic_archive, expected_counts={})

        assert audit.blocking_findings == ()
        assert len(audit.exclusions) == 1
        exclusion = audit.exclusions[0]
        assert exclusion.reason == "exact_pixel_duplicate_in_development_pool"
        assert len(audit.usable_records) == len(audit.records) - 1

        kept = next(r for r in audit.records if r.example_id == exclusion.kept_example_id)
        removed = next(r for r in audit.records if r.example_id == exclusion.excluded_example_id)
        assert audit.split_group_id_of(kept) == audit.split_group_id_of(removed)

    def test_duplicate_across_pools_blocks_and_test_is_left_alone(
        self, synthetic_archive: Path
    ) -> None:
        root = resolve_archive_root(synthetic_archive)
        original = sorted((root / "train" / "NORMAL").glob("*.jpeg"))[0]
        leaked_path = root / "test" / "NORMAL" / "IM-7000-0001.jpeg"
        leaked_path.write_bytes(original.read_bytes())
        audit = audit_pediatric_pneumonia_dataset(synthetic_archive, expected_counts={})

        assert any(f.kind == "cross_pool_duplicate" for f in audit.blocking_findings)
        assert leaked_path.exists(), "the audit must never modify the official test set"
        assert audit.exclusions == ()

    def test_conflicting_labels_on_identical_pixels_block(
        self, synthetic_archive: Path
    ) -> None:
        root = resolve_archive_root(synthetic_archive)
        original = sorted((root / "train" / "NORMAL").glob("*.jpeg"))[0]
        conflicting = root / "train" / "PNEUMONIA" / "person9999_virus_1.jpeg"
        conflicting.write_bytes(original.read_bytes())
        audit = audit_pediatric_pneumonia_dataset(synthetic_archive, expected_counts={})
        assert any(f.kind == "label_conflict" for f in audit.blocking_findings)

    def test_similar_images_are_reported_but_never_excluded(
        self, synthetic_archive: Path
    ) -> None:
        root = resolve_archive_root(synthetic_archive)
        original_path = sorted((root / "train" / "NORMAL").glob("*.jpeg"))[0]
        original_pixels = np.asarray(Image.open(original_path).convert("L"), dtype=np.uint8)
        near_copy = original_pixels.copy()
        near_copy[0, 0] = np.uint8((int(near_copy[0, 0]) + 40) % 256)
        _write_image(root / "train" / "NORMAL" / "IM-6001-0001.jpeg", near_copy, quality=88)

        audit = audit_pediatric_pneumonia_dataset(synthetic_archive, expected_counts={})
        assert audit.exclusions == ()
        assert all(finding.severity != BLOCKING for finding in audit.findings)
        assert any(
            {pair.first_example_id, pair.second_example_id}
            == {original_path.relative_to(root).as_posix(), "train/NORMAL/IM-6001-0001.jpeg"}
            for pair in audit.similarity_pairs
        )

    def test_pool_scope_prevents_a_false_shared_patient_finding(
        self, synthetic_archive: Path
    ) -> None:
        audit = audit_pediatric_pneumonia_dataset(synthetic_archive, expected_counts={})
        assert not [f for f in audit.findings if f.kind == "shared_patient_across_pools"]
        scope_findings = [f for f in audit.findings if f.kind == "patient_identifier_scope"]
        assert len(scope_findings) == 1
        assert scope_findings[0].severity == WARNING

    def test_global_scope_makes_a_repeated_patient_number_blocking(
        self, synthetic_archive: Path
    ) -> None:
        audit = audit_pediatric_pneumonia_dataset(
            synthetic_archive, expected_counts={}, patient_identifier_scope="global"
        )
        assert any(f.kind == "shared_patient_across_pools" for f in audit.blocking_findings)


class TestGroupedStratifiedSplit:
    @staticmethod
    def _pool(number_of_groups: int = 120) -> tuple[list[str], list[int]]:
        group_ids: list[str] = []
        labels: list[int] = []
        for group_index in range(number_of_groups):
            label = 0 if group_index % 4 == 0 else 1
            for _ in range(2):
                group_ids.append(f"g{group_index}")
                labels.append(label)
        return group_ids, labels

    def test_split_is_deterministic_for_one_seed(self) -> None:
        group_ids, labels = self._pool()
        fractions = {"train": 0.7, "search_validation": 0.15, "threshold_validation": 0.15}
        first = split_groups_into_stratified_parts(
            group_ids=group_ids, labels=labels, split_fractions=fractions, random_seed=5
        )
        second = split_groups_into_stratified_parts(
            group_ids=group_ids, labels=labels, split_fractions=fractions, random_seed=5
        )
        assert first.split_of_group == second.split_of_group

    def test_no_group_is_split_and_every_split_has_both_classes(self) -> None:
        group_ids, labels = self._pool()
        result = split_groups_into_stratified_parts(
            group_ids=group_ids,
            labels=labels,
            split_fractions={"train": 0.7, "search_validation": 0.15, "threshold_validation": 0.15},
            random_seed=9,
        )
        assert set(result.split_of_group) == set(group_ids)
        assert all(count > 0 for count in result.counts_by_split_and_label.values())

    def test_realised_fractions_are_close_to_the_requested_ones(self) -> None:
        group_ids, labels = self._pool()
        result = split_groups_into_stratified_parts(
            group_ids=group_ids,
            labels=labels,
            split_fractions={"train": 0.7, "search_validation": 0.15, "threshold_validation": 0.15},
            random_seed=3,
        )
        for split_name, requested in result.requested_fractions.items():
            assert abs(result.realised_fraction(split_name, len(labels)) - requested) < 0.05

    def test_pool_too_small_for_every_label_raises(self) -> None:
        with pytest.raises(ValueError, match="cannot place label"):
            split_groups_into_stratified_parts(
                group_ids=["a", "a", "b", "c", "d", "e"],
                labels=[0, 0, 0, 1, 1, 1],
                split_fractions={"train": 0.5, "val": 0.25, "test": 0.25},
                random_seed=1,
            )

    def test_fractions_must_sum_to_one(self) -> None:
        group_ids, labels = self._pool(8)
        with pytest.raises(ValueError, match="sum to 1.0"):
            split_groups_into_stratified_parts(
                group_ids=group_ids,
                labels=labels,
                split_fractions={"train": 0.5, "val": 0.2},
                random_seed=1,
            )


class TestDatasetManifest:
    @staticmethod
    def _manifest(archive_path: Path, split_seed: int = 11) -> DatasetManifest:
        audit = audit_pediatric_pneumonia_dataset(archive_path, expected_counts={})
        return build_manifest_from_audit(
            audit,
            protocol_id="test-protocol-v1",
            dataset_release="synthetic/fixture",
            dataset_license="not-applicable-synthetic",
            split_seed=split_seed,
        )

    def test_official_test_split_keeps_every_shipped_test_image(
        self, synthetic_archive: Path
    ) -> None:
        audit = audit_pediatric_pneumonia_dataset(synthetic_archive, expected_counts={})
        manifest = self._manifest(synthetic_archive)
        shipped_test_ids = {
            record.example_id for record in audit.records if record.pool == OFFICIAL_TEST_POOL
        }
        assert set(manifest.example_ids_of_split(OFFICIAL_TEST_SPLIT)) == shipped_test_ids

    def test_development_groups_never_span_two_splits(self, synthetic_archive: Path) -> None:
        manifest = self._manifest(synthetic_archive)
        split_of_group: dict[str, str] = {}
        for entry in manifest.entries:
            if entry.target_split == OFFICIAL_TEST_SPLIT:
                continue
            assert split_of_group.setdefault(entry.split_group_id, entry.target_split) == (
                entry.target_split
            )

    def test_every_development_split_contains_both_classes(
        self, synthetic_archive: Path
    ) -> None:
        manifest = self._manifest(synthetic_archive)
        for split_name in ("train", "search_validation", "threshold_validation"):
            labels = {entry.label for entry in manifest.entries_of_split(split_name)}
            assert labels == {0, 1}

    def test_manifest_records_that_patient_independence_is_unproven(
        self, synthetic_archive: Path
    ) -> None:
        manifest = self._manifest(synthetic_archive)
        assert manifest.patient_independence_established is False

    def test_manifest_roundtrips_and_detects_tampering(
        self, synthetic_archive: Path, tmp_path: Path
    ) -> None:
        manifest = self._manifest(synthetic_archive)
        manifest_path = tmp_path / "manifest.json"
        written_digest = manifest.write_json_file(manifest_path)
        reloaded, read_digest = DatasetManifest.read_json_file(manifest_path)
        assert reloaded == manifest
        assert read_digest == written_digest

        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload["entries"][0]["label"] = 1 - payload["entries"][0]["label"]
        manifest_path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ManifestError, match="modified after it was written"):
            DatasetManifest.read_json_file(manifest_path)

    def test_a_different_split_seed_changes_the_manifest_digest(
        self, synthetic_archive: Path
    ) -> None:
        first = self._manifest(synthetic_archive, split_seed=11)
        second = self._manifest(synthetic_archive, split_seed=12)
        assert first.compute_sha256() != second.compute_sha256()

    def test_blocking_findings_are_carried_into_the_manifest(
        self, synthetic_archive: Path
    ) -> None:
        root = resolve_archive_root(synthetic_archive)
        original = sorted((root / "train" / "NORMAL").glob("*.jpeg"))[0]
        (root / "test" / "NORMAL" / "IM-7000-0001.jpeg").write_bytes(original.read_bytes())
        manifest = self._manifest(synthetic_archive)
        assert any(f.kind == "cross_pool_duplicate" for f in manifest.blocking_findings)

    def test_unknown_split_name_is_rejected(self, synthetic_archive: Path) -> None:
        manifest = self._manifest(synthetic_archive)
        with pytest.raises(ManifestError, match="unknown split"):
            manifest.entries_of_split("validation")


def _complete_lock_payload(manifest_sha256: str) -> dict:
    """A protocol lock with every required key filled in with a concrete value."""
    return {
        "schema_version": "1.0",
        "data": {
            "protocol_id": "test-protocol-v1",
            "manifest_path": "manifest.json",
            "manifest_sha256": manifest_sha256,
            "dataset_release": "andrewmvd/pediatric-pneumonia-chest-xray@1",
            "dataset_license": "CC BY 4.0 (per source Mendeley Data v2)",
            "image_side": 128,
        },
        "seeds": {
            "split_seed": 11,
            "smoke_sampling_seed": 12,
            "bootstrap_seed": 13,
            "search_seeds": [101, 102, 103, 104, 105],
            "retraining_seeds": [201, 202, 203],
            "seed_derivation_rule": "stream_seed = sha256(role, evaluation_id, seed) mod 2**63",
        },
        "budget": {
            "search_wall_clock_seconds": 7200,
            "pilot_wall_clock_seconds": 1800,
            "maximum_candidate_seconds": 300,
            "maximum_phenotype_parameters": 5_000_000,
            "checkpoint_interval_seconds": 300,
        },
        "search": {
            "methods": ["deepneat", "exact", "random_search_deepneat"],
            "population_size": 20,
            "candidate_training_epochs": 8,
            "candidate_batch_size": 32,
            "fitness_metric": "auroc_on_search_validation",
        },
        "random_search": {
            "graph_size_distribution": {"minimum_layers": 1, "maximum_layers": 6},
            "gene_distributions": {"layer_type": "uniform", "filters": "log_uniform"},
        },
        "fixed_cnn": {
            "architecture": "conv32-pool-conv64-pool-fc128",
            "training_epochs": 30,
        },
        "track_b_recipe": {
            "optimizer": "sgd",
            "learning_rate": 0.01,
            "momentum": 0.9,
            "weight_decay": 0.0005,
            "batch_size": 32,
            "training_epochs": 30,
            "learning_rate_schedule": "cosine",
            "parameter_initialization": "kaiming_normal_fan_out",
        },
        "transfer_learning": {
            "weights_identifier": "torchvision/resnet18/IMAGENET1K_V1",
            "weights_sha256": "0" * 64,
            "unfreezing_policy": "head_only_then_layer4",
            "training_epochs": 20,
            "preprocessing": "gray_to_three_channels_imagenet_normalization",
        },
        "metrics": {
            "bootstrap_replicates": 2000,
            "maximum_bootstrap_attempts": 20000,
            "confidence_level": 0.95,
            "threshold_rule": "youden_j_on_threshold_validation",
            "reference_threshold": 0.5,
        },
        "environment": {
            "device": "cuda",
            "precision": "float32",
            "deterministic_algorithms": True,
        },
    }


class TestProtocolLock:
    @staticmethod
    def _write_lock(lock_path: Path, payload: dict) -> None:
        lock_path.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")

    def test_a_complete_lock_validates(self, tmp_path: Path) -> None:
        lock_path = tmp_path / "protocol.lock.yaml"
        self._write_lock(lock_path, _complete_lock_payload("a" * 64))
        lock = load_protocol_lock(lock_path)
        validate_protocol_lock_for_full_run(lock, manifest_sha256="a" * 64)
        assert lock.search_seeds == (101, 102, 103, 104, 105)
        assert lock.retraining_seeds == (201, 202, 203)
        assert lock.protocol_id == "test-protocol-v1"

    def test_missing_key_is_refused(self, tmp_path: Path) -> None:
        payload = _complete_lock_payload("a" * 64)
        del payload["budget"]["maximum_candidate_seconds"]
        lock_path = tmp_path / "protocol.lock.yaml"
        self._write_lock(lock_path, payload)
        with pytest.raises(ProtocolLockError, match="missing key budget.maximum_candidate_seconds"):
            validate_protocol_lock_for_full_run(
                load_protocol_lock(lock_path), manifest_sha256="a" * 64
            )

    @pytest.mark.parametrize("placeholder", ["TODO", "", "TBD", None])
    def test_placeholder_values_are_refused(self, tmp_path: Path, placeholder) -> None:
        payload = _complete_lock_payload("a" * 64)
        payload["track_b_recipe"]["learning_rate_schedule"] = placeholder
        lock_path = tmp_path / "protocol.lock.yaml"
        self._write_lock(lock_path, payload)
        with pytest.raises(ProtocolLockError, match="undecided placeholder value"):
            validate_protocol_lock_for_full_run(
                load_protocol_lock(lock_path), manifest_sha256="a" * 64
            )

    def test_empty_seed_list_is_refused(self, tmp_path: Path) -> None:
        payload = _complete_lock_payload("a" * 64)
        payload["seeds"]["search_seeds"] = []
        lock_path = tmp_path / "protocol.lock.yaml"
        self._write_lock(lock_path, payload)
        with pytest.raises(ProtocolLockError, match="seeds.search_seeds"):
            validate_protocol_lock_for_full_run(
                load_protocol_lock(lock_path), manifest_sha256="a" * 64
            )

    def test_lock_frozen_for_another_manifest_is_refused(self, tmp_path: Path) -> None:
        lock_path = tmp_path / "protocol.lock.yaml"
        self._write_lock(lock_path, _complete_lock_payload("a" * 64))
        with pytest.raises(ProtocolLockError, match="different split is a different study"):
            validate_protocol_lock_for_full_run(
                load_protocol_lock(lock_path), manifest_sha256="b" * 64
            )

    def test_every_problem_is_reported_at_once(self, tmp_path: Path) -> None:
        payload = _complete_lock_payload("a" * 64)
        del payload["metrics"]["bootstrap_replicates"]
        payload["fixed_cnn"]["architecture"] = "TODO"
        lock_path = tmp_path / "protocol.lock.yaml"
        self._write_lock(lock_path, payload)
        with pytest.raises(ProtocolLockError) as raised:
            validate_protocol_lock_for_full_run(
                load_protocol_lock(lock_path), manifest_sha256="a" * 64
            )
        message = str(raised.value)
        assert "missing key metrics.bootstrap_replicates" in message
        assert "fixed_cnn.architecture" in message

    def test_lock_records_the_digest_of_its_own_file(self, tmp_path: Path) -> None:
        lock_path = tmp_path / "protocol.lock.yaml"
        self._write_lock(lock_path, _complete_lock_payload("a" * 64))
        first = load_protocol_lock(lock_path)
        self._write_lock(lock_path, _complete_lock_payload("a" * 64) | {"schema_version": "1.1"})
        assert load_protocol_lock(lock_path).source_sha256 != first.source_sha256
