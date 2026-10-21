"""Stage-separation tests: what each split is and is not allowed to influence.

The protocol gives every split a narrow permission. ``threshold_validation``
may only pick a decision threshold for an already frozen checkpoint;
``search_validation`` may drive model selection but never track A's training
statistics; the official test set may be read for the integrity audit and
nothing else. These tests change one split's *data* and assert that the
quantities it must not touch come out bit-identical.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from examples.pediatric_pneumonia._dataset_audit import (
    audit_pediatric_pneumonia_dataset,
    resolve_archive_root,
)
from examples.pediatric_pneumonia._dataset_manifest import (
    SEARCH_VALIDATION_SPLIT,
    THRESHOLD_VALIDATION_SPLIT,
    TRAIN_SPLIT,
    DatasetManifest,
    build_manifest_from_audit,
)
from examples.pediatric_pneumonia._synthetic_archive import write_synthetic_archive
from examples.pediatric_pneumonia.dataset import load_split_tensors
from polyneat.training.class_weights import compute_balanced_class_weights
from polyneat.training.image_preprocessing import (
    ImagePreprocessingConfig,
    ImagePreprocessor,
)

_TARGET_SIDE = 24


def _build_manifest(data_directory: Path) -> DatasetManifest:
    audit = audit_pediatric_pneumonia_dataset(
        data_directory, expected_counts={}, compute_similarity_report=False
    )
    return build_manifest_from_audit(
        audit,
        protocol_id="separation-test",
        dataset_release="synthetic/fixture",
        dataset_license="not-applicable-synthetic",
        split_seed=11,
    )


def _training_quantities(
    manifest: DatasetManifest, data_directory: Path, training_splits: tuple[str, ...]
) -> tuple[dict, torch.Tensor]:
    """Fitted preprocessing statistics and class weights for one training stage."""
    images = torch.cat(
        [
            load_split_tensors(
                manifest,
                data_directory=data_directory,
                split_name=split_name,
                target_side=_TARGET_SIDE,
            ).images
            for split_name in training_splits
        ]
    )
    labels = torch.cat(
        [
            load_split_tensors(
                manifest,
                data_directory=data_directory,
                split_name=split_name,
                target_side=_TARGET_SIDE,
            ).labels
            for split_name in training_splits
        ]
    )
    preprocessor = ImagePreprocessor(ImagePreprocessingConfig(target_side=_TARGET_SIDE))
    preprocessor.fit_standardization(images)
    return preprocessor.state_dict(), compute_balanced_class_weights(labels, 2)


def _overwrite_split_images(
    manifest: DatasetManifest, data_directory: Path, split_name: str
) -> None:
    """Replace every image of one split with pure noise, keeping names and labels."""
    archive_root = resolve_archive_root(data_directory)
    rng = np.random.default_rng(555)
    for entry in manifest.entries_of_split(split_name):
        image_path = archive_root / entry.relative_path
        noise = rng.integers(0, 256, size=(entry.height, entry.width), dtype=np.uint8)
        Image.fromarray(noise, mode="L").save(image_path, format="JPEG", quality=92)


@pytest.fixture
def archive(tmp_path: Path) -> Path:
    return write_synthetic_archive(tmp_path / "data", random_seed=808)


class TestThresholdValidationInfluence:
    def test_changing_threshold_validation_changes_neither_statistics_nor_weights(
        self, archive: Path
    ) -> None:
        manifest = _build_manifest(archive)
        before_state, before_weights = _training_quantities(manifest, archive, (TRAIN_SPLIT,))

        _overwrite_split_images(manifest, archive, THRESHOLD_VALIDATION_SPLIT)
        after_state, after_weights = _training_quantities(manifest, archive, (TRAIN_SPLIT,))

        assert before_state == after_state
        assert torch.equal(before_weights, after_weights)

    def test_changing_threshold_validation_does_not_change_track_b_training_data(
        self, archive: Path
    ) -> None:
        manifest = _build_manifest(archive)
        track_b_splits = (TRAIN_SPLIT, SEARCH_VALIDATION_SPLIT)
        before_state, before_weights = _training_quantities(manifest, archive, track_b_splits)

        _overwrite_split_images(manifest, archive, THRESHOLD_VALIDATION_SPLIT)
        after_state, after_weights = _training_quantities(manifest, archive, track_b_splits)

        assert before_state == after_state
        assert torch.equal(before_weights, after_weights)


class TestSearchValidationInfluence:
    def test_changing_search_validation_leaves_track_a_training_statistics_alone(
        self, archive: Path
    ) -> None:
        manifest = _build_manifest(archive)
        before_state, before_weights = _training_quantities(manifest, archive, (TRAIN_SPLIT,))

        _overwrite_split_images(manifest, archive, SEARCH_VALIDATION_SPLIT)
        after_state, after_weights = _training_quantities(manifest, archive, (TRAIN_SPLIT,))

        assert before_state == after_state
        assert torch.equal(before_weights, after_weights)

    def test_changing_search_validation_does_change_track_b_training_data(
        self, archive: Path
    ) -> None:
        manifest = _build_manifest(archive)
        track_b_splits = (TRAIN_SPLIT, SEARCH_VALIDATION_SPLIT)
        before_state, _ = _training_quantities(manifest, archive, track_b_splits)

        _overwrite_split_images(manifest, archive, SEARCH_VALIDATION_SPLIT)
        after_state, _ = _training_quantities(manifest, archive, track_b_splits)

        assert before_state != after_state, (
            "search_validation is part of track B training, so its data must reach the "
            "fitted statistics"
        )


class TestSplitLoadingIsScoped:
    def test_loading_one_split_never_returns_rows_of_another(self, archive: Path) -> None:
        manifest = _build_manifest(archive)
        loaded_ids = {
            split_name: set(
                load_split_tensors(
                    manifest,
                    data_directory=archive,
                    split_name=split_name,
                    target_side=_TARGET_SIDE,
                ).example_ids
            )
            for split_name in (TRAIN_SPLIT, SEARCH_VALIDATION_SPLIT, THRESHOLD_VALIDATION_SPLIT)
        }
        for first_split, first_ids in loaded_ids.items():
            for second_split, second_ids in loaded_ids.items():
                if first_split != second_split:
                    assert not (first_ids & second_ids)

    def test_loaded_rows_match_the_manifest_exactly(self, archive: Path) -> None:
        manifest = _build_manifest(archive)
        loaded = load_split_tensors(
            manifest,
            data_directory=archive,
            split_name=TRAIN_SPLIT,
            target_side=_TARGET_SIDE,
        )
        assert loaded.example_ids == manifest.example_ids_of_split(TRAIN_SPLIT)
        assert loaded.labels.tolist() == [
            entry.label for entry in manifest.entries_of_split(TRAIN_SPLIT)
        ]
