from __future__ import annotations

import gzip
import struct
from pathlib import Path

import numpy as np
import pytest
import torch

from examples.fashion_mnist.dataset import (
    load_fashion_mnist,
    load_fashion_mnist_features_and_labels,
    read_idx_gz_file,
)

_IDX_UBYTE_IMAGE_MAGIC = 0x00000803
_IDX_UBYTE_LABEL_MAGIC = 0x00000801


def _write_idx_gz(path: Path, magic: int, count: int, extra_dims: tuple[int, ...],
                   payload: np.ndarray) -> None:
    header = struct.pack(">II", magic, count)
    for dimension_size in extra_dims:
        header += struct.pack(">I", dimension_size)
    path.write_bytes(gzip.compress(header + payload.tobytes()))


def test_read_idx_gz_file_parses_a_3d_image_file(tmp_path: Path) -> None:
    images = np.random.default_rng(0).integers(0, 256, size=(5, 4, 4), dtype=np.uint8)
    idx_path = tmp_path / "images-idx3-ubyte.gz"
    _write_idx_gz(idx_path, _IDX_UBYTE_IMAGE_MAGIC, count=5, extra_dims=(4, 4), payload=images)

    result = read_idx_gz_file(idx_path)

    assert result.shape == (5, 4, 4)
    assert result.dtype == np.uint8
    assert np.array_equal(result, images)


def test_read_idx_gz_file_parses_a_1d_label_file(tmp_path: Path) -> None:
    labels = np.array([0, 1, 2, 3, 4, 5], dtype=np.uint8)
    idx_path = tmp_path / "labels-idx1-ubyte.gz"
    _write_idx_gz(idx_path, _IDX_UBYTE_LABEL_MAGIC, count=6, extra_dims=(), payload=labels)

    result = read_idx_gz_file(idx_path)

    assert result.shape == (6,)
    assert result.dtype == np.uint8
    assert np.array_equal(result, labels)


def test_read_idx_gz_file_rejects_an_unexpected_dtype_code(tmp_path: Path) -> None:
    bogus_magic = 0x00000705  # dtype code 0x07, not the supported 0x08
    idx_path = tmp_path / "bogus-idx.gz"
    _write_idx_gz(idx_path, bogus_magic, count=3, extra_dims=(), payload=np.zeros(3, np.uint8))

    with pytest.raises(ValueError):
        read_idx_gz_file(idx_path)


@pytest.fixture
def synthetic_fashion_mnist_archives(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> dict[str, int]:
    """Monkeypatch the download seam to serve small synthetic IDX archives.

    Keeps the dataset test independent of network access and of whether the
    real cache under ``examples/fashion_mnist/data/`` is populated, per this
    project's convention (see ``tests/test_example_datasets.py``).
    """
    train_count, test_count = 40, 20
    rng = np.random.default_rng(1234)
    train_images = rng.integers(0, 128, size=(train_count, 28, 28), dtype=np.uint8)
    train_labels = (np.arange(train_count) % 10).astype(np.uint8)
    test_images = rng.integers(128, 256, size=(test_count, 28, 28), dtype=np.uint8)
    test_labels = (np.arange(test_count) % 10).astype(np.uint8)

    def _images_bytes(images: np.ndarray) -> bytes:
        header = struct.pack(">IIII", _IDX_UBYTE_IMAGE_MAGIC, images.shape[0], 28, 28)
        return gzip.compress(header + images.tobytes())

    def _labels_bytes(labels: np.ndarray) -> bytes:
        header = struct.pack(">II", _IDX_UBYTE_LABEL_MAGIC, labels.shape[0])
        return gzip.compress(header + labels.tobytes())

    archive_bytes_by_file_name = {
        "train-images-idx3-ubyte.gz": _images_bytes(train_images),
        "train-labels-idx1-ubyte.gz": _labels_bytes(train_labels),
        "t10k-images-idx3-ubyte.gz": _images_bytes(test_images),
        "t10k-labels-idx1-ubyte.gz": _labels_bytes(test_labels),
    }

    def fake_download_file_if_missing(source_url: str, destination_path: Path) -> Path:
        fake_path = tmp_path / destination_path.name
        fake_path.write_bytes(archive_bytes_by_file_name[destination_path.name])
        return fake_path

    monkeypatch.setattr(
        "examples.fashion_mnist.dataset.download_file_if_missing",
        fake_download_file_if_missing,
    )
    return {"train_count": train_count, "test_count": test_count}


def test_load_features_and_labels_pools_and_concatenates_train_and_test(
    synthetic_fashion_mnist_archives: dict[str, int],
) -> None:
    total = (
        synthetic_fashion_mnist_archives["train_count"]
        + synthetic_fashion_mnist_archives["test_count"]
    )

    features, labels = load_fashion_mnist_features_and_labels(grid_side=7)

    assert features.shape == (total, 49)
    assert features.dtype == torch.float32
    assert labels.shape == (total,)
    assert labels.dtype == torch.long


def test_load_features_and_labels_fits_standardization_on_official_train_only(
    synthetic_fashion_mnist_archives: dict[str, int],
) -> None:
    """Guard the standardization step, which shape assertions alone cannot see.

    Raw pooled pixels sit in ``[0, 1]``; standardized ones are zero-mean and
    unit-variance per feature. Without this, dropping ``_standardize_features``
    from the loader passes every other test in this file.
    """
    features, _ = load_fashion_mnist_features_and_labels(grid_side=7)
    official_train_size = synthetic_fashion_mnist_archives["train_count"]
    train_features = features[:official_train_size]
    test_features = features[official_train_size:]

    per_feature_mean = train_features.mean(dim=0)
    per_feature_std = train_features.std(dim=0, unbiased=False)

    assert per_feature_mean.abs().max().item() == pytest.approx(0.0, abs=1e-4)
    assert per_feature_std.min().item() == pytest.approx(1.0, abs=1e-3)
    assert per_feature_std.max().item() == pytest.approx(1.0, abs=1e-3)
    assert test_features.mean().item() > 1.0


def test_load_fashion_mnist_reports_ten_classes(
    synthetic_fashion_mnist_archives: dict[str, int],
) -> None:
    dataset = load_fashion_mnist(random_seed=0)

    assert dataset.number_of_classes == 10


def test_load_fashion_mnist_is_reproducible_for_a_fixed_seed(
    synthetic_fashion_mnist_archives: dict[str, int],
) -> None:
    first = load_fashion_mnist(
        random_seed=5, max_train_samples=None, max_test_samples=None
    )
    second = load_fashion_mnist(
        random_seed=5, max_train_samples=None, max_test_samples=None
    )

    assert torch.equal(first.train_features, second.train_features)
    assert torch.equal(first.train_labels, second.train_labels)
    assert torch.equal(first.test_features, second.test_features)
    assert torch.equal(first.test_labels, second.test_labels)


def test_load_fashion_mnist_preserves_the_official_split_sizes(
    synthetic_fashion_mnist_archives: dict[str, int],
) -> None:
    dataset = load_fashion_mnist(
        random_seed=1, max_train_samples=None, max_test_samples=None
    )

    assert dataset.train_features.shape[0] == synthetic_fashion_mnist_archives["train_count"]
    assert dataset.test_features.shape[0] == synthetic_fashion_mnist_archives["test_count"]


def test_load_fashion_mnist_honours_the_sample_caps(
    synthetic_fashion_mnist_archives: dict[str, int],
) -> None:
    dataset = load_fashion_mnist(random_seed=2, max_train_samples=10, max_test_samples=5)

    assert dataset.train_features.shape == (10, 49)
    assert dataset.test_features.shape == (5, 49)


def test_load_fashion_mnist_full_resolution_grid_keeps_784_features(
    synthetic_fashion_mnist_archives: dict[str, int],
) -> None:
    dataset = load_fashion_mnist(
        random_seed=3, grid_side=28, max_train_samples=None, max_test_samples=None
    )

    assert dataset.number_of_features == 784


def test_pool_features_to_grid_is_reused_from_the_mnist_module() -> None:
    from examples.fashion_mnist import dataset as fashion_mnist_dataset
    from examples.mnist import dataset as mnist_dataset

    assert fashion_mnist_dataset.pool_features_to_grid is mnist_dataset.pool_features_to_grid
