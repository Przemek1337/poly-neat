from __future__ import annotations

import numpy as np
import pytest
import torch

from examples._datasets import (
    ClassificationDataset,
    download_file_if_missing,
    split_features_and_labels,
    split_indices_into_train_and_test,
)
from examples.mnist.dataset import pool_features_to_grid


def test_split_returns_disjoint_indices_covering_every_sample() -> None:
    train_indices, test_indices = split_indices_into_train_and_test(
        number_of_samples=150, train_fraction=0.66, random_seed=42
    )
    assert set(train_indices).isdisjoint(test_indices)
    assert sorted(np.concatenate([train_indices, test_indices])) == list(range(150))


def test_split_honours_train_fraction() -> None:
    train_indices, test_indices = split_indices_into_train_and_test(
        number_of_samples=150, train_fraction=0.66, random_seed=42
    )
    assert len(train_indices) == int(0.66 * 150)
    assert len(test_indices) == 150 - int(0.66 * 150)


def test_split_is_deterministic_for_a_fixed_seed() -> None:
    first_train, first_test = split_indices_into_train_and_test(
        number_of_samples=150, train_fraction=0.66, random_seed=7
    )
    second_train, second_test = split_indices_into_train_and_test(
        number_of_samples=150, train_fraction=0.66, random_seed=7
    )
    assert np.array_equal(first_train, second_train)
    assert np.array_equal(first_test, second_test)


def test_split_differs_between_seeds() -> None:
    train_for_seed_one, _ = split_indices_into_train_and_test(
        number_of_samples=150, train_fraction=0.66, random_seed=1
    )
    train_for_seed_two, _ = split_indices_into_train_and_test(
        number_of_samples=150, train_fraction=0.66, random_seed=2
    )
    assert not np.array_equal(train_for_seed_one, train_for_seed_two)


def test_download_is_skipped_when_destination_already_exists(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    existing_path = tmp_path / "already_here.data"
    existing_path.write_text("cached", encoding="utf-8")

    def fail_if_called(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("urlretrieve must not run when the file already exists")

    monkeypatch.setattr("urllib.request.urlretrieve", fail_if_called)

    returned_path = download_file_if_missing("https://example.invalid/x", existing_path)

    assert returned_path == existing_path
    assert existing_path.read_text(encoding="utf-8") == "cached"


def test_download_creates_parent_directories_and_fetches_when_absent(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination_path = tmp_path / "nested" / "cache" / "payload.data"
    recorded_calls: list[tuple[str, object]] = []

    def record_call(source_url: str, target_path: object) -> None:
        recorded_calls.append((source_url, target_path))

    monkeypatch.setattr("urllib.request.urlretrieve", record_call)

    download_file_if_missing("https://example.invalid/payload", destination_path)

    assert destination_path.parent.is_dir()
    assert recorded_calls == [("https://example.invalid/payload", destination_path)]


def _labelled_features(
    number_of_samples: int, number_of_features: int, number_of_classes: int
) -> tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(0)
    features = rng.standard_normal(
        (number_of_samples, number_of_features), dtype=np.float32
    )
    labels = np.arange(number_of_samples, dtype=np.int64) % number_of_classes
    return torch.from_numpy(features), torch.from_numpy(labels)


def test_split_features_and_labels_returns_a_classification_dataset() -> None:
    features, labels = _labelled_features(150, 4, 3)

    dataset = split_features_and_labels(
        features, labels, train_fraction=0.66, random_seed=42
    )

    assert isinstance(dataset, ClassificationDataset)
    assert dataset.number_of_features == 4
    assert dataset.number_of_classes == 3  # inferred from labels
    assert dataset.train_features.shape[0] == int(0.66 * 150)
    assert dataset.test_features.shape[0] == 150 - int(0.66 * 150)
    assert dataset.train_features.dtype == torch.float32
    assert dataset.train_labels.dtype == torch.long


def test_split_features_and_labels_caps_each_split_at_the_requested_size() -> None:
    features, labels = _labelled_features(200, 3, 4)

    dataset = split_features_and_labels(
        features,
        labels,
        train_fraction=0.8,
        random_seed=1,
        max_train_samples=50,
        max_test_samples=20,
    )

    assert dataset.train_features.shape == (50, 3)
    assert dataset.test_features.shape == (20, 3)
    assert dataset.train_labels.shape == (50,)


def test_split_features_and_labels_honours_explicit_class_count() -> None:
    features, labels = _labelled_features(40, 2, 2)

    dataset = split_features_and_labels(
        features, labels, train_fraction=0.5, random_seed=0, number_of_classes=5
    )

    assert dataset.number_of_classes == 5


def test_average_pool_flattens_images_to_the_requested_grid() -> None:
    images = np.full((5, 28, 28), 255, dtype=np.uint8)

    pooled = pool_features_to_grid(images, grid_side=7)

    assert pooled.shape == (5, 49)
    assert pooled.dtype == np.float32
    assert np.allclose(pooled, 1.0)


def test_average_pool_preserves_row_major_pixel_order() -> None:
    images = np.zeros((1, 28, 28), dtype=np.uint8)
    images[0, 0:4, 0:4] = 255

    pooled = pool_features_to_grid(images, grid_side=7)

    assert pooled[0, 0] == pytest.approx(1.0)
    assert np.allclose(pooled[0, 1:], 0.0)


def test_full_resolution_grid_keeps_every_pixel() -> None:
    images = np.zeros((3, 28, 28), dtype=np.uint8)

    pooled = pool_features_to_grid(images, grid_side=28)

    assert pooled.shape == (3, 784)
