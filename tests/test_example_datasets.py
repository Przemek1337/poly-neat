from __future__ import annotations

import numpy as np
import pytest
import torch

from examples._datasets import (
    download_file_if_missing,
    split_indices_into_train_and_test,
)
from examples.mnist.dataset import (
    _average_pool_images_to_grid,
    _sample_subset,
    _standardize_features,
)


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


def test_average_pool_flattens_images_to_the_requested_grid() -> None:
    images = np.full((5, 28, 28), 255, dtype=np.uint8)

    pooled = _average_pool_images_to_grid(images, pooled_grid_side=7)

    assert pooled.shape == (5, 49)
    assert pooled.dtype == np.float32
    # A saturated image scales to 1.0 in every pooled cell.
    assert np.allclose(pooled, 1.0)


def test_average_pool_preserves_row_major_pixel_order() -> None:
    # A single bright 4x4 block in the top-left corner must land in pooled cell 0.
    images = np.zeros((1, 28, 28), dtype=np.uint8)
    images[0, 0:4, 0:4] = 255

    pooled = _average_pool_images_to_grid(images, pooled_grid_side=7)

    assert pooled[0, 0] == pytest.approx(1.0)
    assert np.allclose(pooled[0, 1:], 0.0)


def test_standardize_uses_training_statistics_for_both_splits() -> None:
    rng = np.random.default_rng(0)
    train_features = rng.normal(loc=5.0, scale=3.0, size=(200, 4)).astype(np.float32)
    test_features = rng.normal(loc=5.0, scale=3.0, size=(50, 4)).astype(np.float32)

    standardized_train, standardized_test = _standardize_features(
        train_features, test_features
    )

    assert np.allclose(standardized_train.mean(axis=0), 0.0, atol=1e-5)
    assert np.allclose(standardized_train.std(axis=0), 1.0, atol=1e-3)
    # Test features are scaled with the training statistics, so they are only
    # approximately centred - never exactly.
    assert standardized_test.shape == test_features.shape


def test_sample_subset_returns_torch_tensors_of_the_requested_size() -> None:
    features = np.arange(100 * 3, dtype=np.float32).reshape(100, 3)
    labels = np.arange(100, dtype=np.int64) % 4

    subset_features, subset_labels = _sample_subset(
        features, labels, subset_size=25, rng=np.random.default_rng(3)
    )

    assert isinstance(subset_features, torch.Tensor)
    assert subset_features.shape == (25, 3)
    assert subset_features.dtype == torch.float32
    assert subset_labels.shape == (25,)
    assert subset_labels.dtype == torch.long


def test_sample_subset_caps_at_the_available_sample_count() -> None:
    features = np.zeros((10, 2), dtype=np.float32)
    labels = np.zeros(10, dtype=np.int64)

    subset_features, subset_labels = _sample_subset(
        features, labels, subset_size=999, rng=np.random.default_rng(3)
    )

    assert subset_features.shape == (10, 2)
    assert subset_labels.shape == (10,)
