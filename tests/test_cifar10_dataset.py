from __future__ import annotations

import io
import pickle
import tarfile
from pathlib import Path

import numpy as np
import pytest

from examples.cifar10.dataset import load_cifar10


def _add_pickled_batch(
    archive: tarfile.TarFile,
    member_name: str,
    images: np.ndarray,
    labels: np.ndarray,
) -> None:
    payload = pickle.dumps(
        {b"data": images.reshape(images.shape[0], -1), b"labels": labels.tolist()}
    )
    member = tarfile.TarInfo(f"cifar-10-batches-py/{member_name}")
    member.size = len(payload)
    archive.addfile(member, io.BytesIO(payload))


@pytest.fixture
def synthetic_cifar10_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> dict[str, int]:
    archive_path = tmp_path / "cifar-10-python.tar.gz"
    rng = np.random.default_rng(123)
    train_per_batch = 4
    test_count = 6
    with tarfile.open(archive_path, mode="w:gz") as archive:
        for batch_index in range(1, 6):
            train_images = rng.integers(
                0, 128, size=(train_per_batch, 3, 32, 32), dtype=np.uint8
            )
            train_labels = np.arange(train_per_batch, dtype=np.int64) % 10
            _add_pickled_batch(
                archive, f"data_batch_{batch_index}", train_images, train_labels
            )
        test_images = rng.integers(128, 256, size=(test_count, 3, 32, 32), dtype=np.uint8)
        test_labels = np.arange(test_count, dtype=np.int64) % 10
        _add_pickled_batch(archive, "test_batch", test_images, test_labels)

    monkeypatch.setattr(
        "examples.cifar10.dataset.download_file_if_missing",
        lambda _source_url, _destination_path: archive_path,
    )
    return {"train_count": train_per_batch * 5, "test_count": test_count}


def test_loader_preserves_official_split_and_shape(
    synthetic_cifar10_archive: dict[str, int],
) -> None:
    dataset = load_cifar10(
        random_seed=7, max_train_samples=None, max_test_samples=None, standardize=False
    )

    assert dataset.train_features.shape == (synthetic_cifar10_archive["train_count"], 3072)
    assert dataset.test_features.shape == (synthetic_cifar10_archive["test_count"], 3072)
    assert dataset.number_of_classes == 10
    assert 0.0 <= dataset.train_features.min() <= dataset.train_features.max() <= 1.0


def test_loader_normalization_uses_train_statistics_only(
    synthetic_cifar10_archive: dict[str, int],
) -> None:
    dataset = load_cifar10(
        random_seed=7, max_train_samples=None, max_test_samples=None, standardize=True
    )

    assert dataset.train_features.mean().item() == pytest.approx(0.0, abs=1e-5)
    assert dataset.test_features.mean().item() > 1.0


def test_loader_subsamples_each_official_split_independently(
    synthetic_cifar10_archive: dict[str, int],
) -> None:
    dataset = load_cifar10(
        random_seed=9, max_train_samples=8, max_test_samples=3, standardize=False
    )

    assert dataset.train_features.shape[0] == 8
    assert dataset.test_features.shape[0] == 3
