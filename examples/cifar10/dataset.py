"""Official CIFAR-10 Python-archive loader with leakage-free preprocessing."""

from __future__ import annotations

import pickle
import tarfile
from pathlib import Path

import numpy as np
import torch

from examples._datasets import (
    ClassificationDataset,
    build_dataset_from_official_splits,
    download_file_if_missing,
    standardize_feature_splits_from_training_statistics,
)

_CIFAR10_DATA_DIR = Path(__file__).parent / "data"
_CIFAR10_ARCHIVE_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
_CIFAR10_ARCHIVE_PATH = _CIFAR10_DATA_DIR / "cifar-10-python.tar.gz"
_CIFAR10_ARCHIVE_PREFIX = "cifar-10-batches-py"
_CIFAR10_NUMBER_OF_CLASSES = 10


def _read_batch_from_archive(
    archive: tarfile.TarFile, member_name: str
) -> tuple[np.ndarray, np.ndarray]:
    """Read one trusted official CIFAR batch without extracting files to disk."""
    member = archive.getmember(f"{_CIFAR10_ARCHIVE_PREFIX}/{member_name}")
    batch_file = archive.extractfile(member)
    if batch_file is None:
        raise ValueError(f"CIFAR-10 archive member {member_name!r} is not a file")
    payload = pickle.load(batch_file, encoding="bytes")  # noqa: S301 - trusted official archive
    flat_images = np.asarray(payload[b"data"], dtype=np.uint8)
    labels = np.asarray(payload[b"labels"], dtype=np.int64)
    images = flat_images.reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
    return images.reshape(images.shape[0], -1), labels


def load_cifar10(
    *,
    random_seed: int,
    max_train_samples: int | None = None,
    max_test_samples: int | None = None,
    standardize: bool = True,
) -> ClassificationDataset:
    """Load CIFAR-10 while preserving the official 50,000/10,000 split."""
    archive_path = download_file_if_missing(_CIFAR10_ARCHIVE_URL, _CIFAR10_ARCHIVE_PATH)
    with tarfile.open(archive_path, mode="r:gz") as archive:
        training_batches = [
            _read_batch_from_archive(archive, f"data_batch_{batch_index}")
            for batch_index in range(1, 6)
        ]
        test_features, test_labels = _read_batch_from_archive(archive, "test_batch")

    train_features = np.concatenate([batch[0] for batch in training_batches])
    train_labels = np.concatenate([batch[1] for batch in training_batches])
    dataset = build_dataset_from_official_splits(
        torch.from_numpy(train_features),
        torch.from_numpy(train_labels),
        torch.from_numpy(test_features),
        torch.from_numpy(test_labels),
        random_seed=random_seed,
        max_train_samples=max_train_samples,
        max_test_samples=max_test_samples,
        number_of_classes=_CIFAR10_NUMBER_OF_CLASSES,
    )
    if not standardize:
        return dataset
    standardized_train, standardized_test = standardize_feature_splits_from_training_statistics(
        dataset.train_features, dataset.test_features
    )
    return ClassificationDataset(
        train_features=standardized_train,
        train_labels=dataset.train_labels,
        test_features=standardized_test,
        test_labels=dataset.test_labels,
        number_of_classes=dataset.number_of_classes,
    )
