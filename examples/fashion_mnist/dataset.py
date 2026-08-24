"""Pooled Fashion-MNIST loading for the fashion_mnist examples.

Fashion-MNIST is shape-identical to MNIST: 60000 train + 10000 test images,
28x28 uint8 grayscale, 10 classes. It ships as four gzipped IDX-format
archives (two image files, two label files) rather than one ``.npz`` file, so
this module adds an IDX unpacker (:func:`read_idx_gz_file`) but otherwise
mirrors ``examples/mnist/dataset.py``: a raw
``load_fashion_mnist_features_and_labels`` returning ``(features, labels)``
for the whole dataset, and a top-level ``load_fashion_mnist`` returning the
shared bundle.

Pooling and standardization are identical to MNIST's (same image shape), so
they are imported rather than duplicated here -- see the import below.
"""

from __future__ import annotations

import gzip
import struct
from pathlib import Path

import numpy as np
import torch

from examples._datasets import (
    ClassificationDataset,
    download_file_if_missing,
    split_features_and_labels,
)

# Reused rather than duplicated: verbatim duplication of a logic block is an
# explicit defect in this project's review rubric, and a private-name import
# between sibling example modules is the lesser smell. Both helpers are
# shape-agnostic beyond "28x28 uint8 images", which Fashion-MNIST also is.
from examples.mnist.dataset import _standardize_features, pool_features_to_grid

_FASHION_MNIST_DATA_DIR = Path(__file__).parent / "data"
_FASHION_MNIST_BASE_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/"
_FASHION_MNIST_TRAIN_IMAGES_FILE_NAME = "train-images-idx3-ubyte.gz"
_FASHION_MNIST_TRAIN_LABELS_FILE_NAME = "train-labels-idx1-ubyte.gz"
_FASHION_MNIST_TEST_IMAGES_FILE_NAME = "t10k-images-idx3-ubyte.gz"
_FASHION_MNIST_TEST_LABELS_FILE_NAME = "t10k-labels-idx1-ubyte.gz"
_FASHION_MNIST_NUMBER_OF_CLASSES = 10
# 6/7 mirrors Fashion-MNIST's canonical 60,000 / 10,000 train/test split ratio.
_FASHION_MNIST_DEFAULT_TRAIN_FRACTION = 6 / 7

_IDX_UNSIGNED_BYTE_DTYPE_CODE = 0x08


def read_idx_gz_file(path: Path) -> np.ndarray:
    """Unpack one gzipped IDX-format file (an image archive or a label archive).

    The IDX format used by the original MNIST/Fashion-MNIST distributions
    starts with a big-endian ``magic`` uint32 whose two high bytes are zero,
    whose next byte is a dtype code (``0x08`` for unsigned byte, the only
    code these archives use) and whose low byte is the number of dimensions.
    A big-endian ``count`` uint32 follows, then one big-endian uint32 per
    extra dimension (row and column counts for a 3-D image file; none for a
    1-D label file), then the raw data itself.

    Args:
        path: Path to a ``.gz``-compressed IDX file.

    Returns:
        A ``uint8`` array of shape ``(count,)`` for a 1-D label file or
        ``(count, rows, cols)`` for a 3-D image file.

    Raises:
        ValueError: If the magic number encodes an unexpected dtype code.

    References:
        LeCun, Y., Cortes, C., Burges, C.J.C. THE MNIST DATABASE of
        handwritten digits. http://yann.lecun.com/exdb/mnist/ -- the IDX
        format Fashion-MNIST (Xiao et al., 2017) reuses byte-for-byte.
    """
    with gzip.open(path, "rb") as gz_file:
        raw_bytes = gz_file.read()

    magic, count = struct.unpack(">II", raw_bytes[:8])
    dtype_code = (magic >> 8) & 0xFF
    number_of_dimensions = magic & 0xFF
    if dtype_code != _IDX_UNSIGNED_BYTE_DTYPE_CODE:
        raise ValueError(
            f"Unexpected IDX dtype code 0x{dtype_code:02x} in {path} "
            f"(magic=0x{magic:08x}); only unsigned byte (0x08) is supported."
        )

    offset = 8
    extra_dimensions: list[int] = []
    for _ in range(number_of_dimensions - 1):
        (dimension_size,) = struct.unpack(">I", raw_bytes[offset : offset + 4])
        extra_dimensions.append(dimension_size)
        offset += 4

    data = np.frombuffer(raw_bytes[offset:], dtype=np.uint8)
    return data.reshape(count, *extra_dimensions)


def _load_idx_archive(file_name: str) -> np.ndarray:
    """Download (if missing) and unpack one Fashion-MNIST IDX archive."""
    archive_path = download_file_if_missing(
        _FASHION_MNIST_BASE_URL + file_name, _FASHION_MNIST_DATA_DIR / file_name
    )
    return read_idx_gz_file(archive_path)


def load_fashion_mnist_features_and_labels(
    grid_side: int = 7,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load the whole Fashion-MNIST dataset, pooled to ``grid_side`` and standardized.

    The train and test archives are concatenated into one
    ``(features, labels)`` pair, mirroring
    :func:`examples.mnist.dataset.load_mnist_features_and_labels`; the split
    is drawn later by :func:`load_fashion_mnist` / ``split_features_and_labels``.

    Args:
        grid_side: Side length of the pooled grid; must divide 28.
            ``grid_side == 28`` keeps full-resolution 784-pixel images.

    Returns:
        ``(features, labels)`` where features is a
        ``[70000, grid_side ** 2]`` float tensor and labels is a ``[70000]``
        long tensor holding class indices 0-9.
    """
    train_images = _load_idx_archive(_FASHION_MNIST_TRAIN_IMAGES_FILE_NAME)
    train_labels = _load_idx_archive(_FASHION_MNIST_TRAIN_LABELS_FILE_NAME)
    test_images = _load_idx_archive(_FASHION_MNIST_TEST_IMAGES_FILE_NAME)
    test_labels = _load_idx_archive(_FASHION_MNIST_TEST_LABELS_FILE_NAME)

    images = np.concatenate([train_images, test_images], axis=0)
    labels = np.concatenate([train_labels, test_labels]).astype(np.int64)

    pooled = pool_features_to_grid(images, grid_side)
    standardized = _standardize_features(pooled)
    return (
        torch.from_numpy(standardized).to(torch.float32),
        torch.from_numpy(labels),
    )


def load_fashion_mnist(
    *,
    train_fraction: float = _FASHION_MNIST_DEFAULT_TRAIN_FRACTION,
    random_seed: int,
    grid_side: int = 7,
    max_train_samples: int | None = 3000,
    max_test_samples: int | None = 2000,
) -> ClassificationDataset:
    """Load Fashion-MNIST, pool it to a small grid and split it into a bundle.

    The aligned top-level entrypoint, matching :func:`examples.mnist.dataset.load_mnist`
    in shape and return type. Vanilla NEAT grows topology from a minimal
    start, so the raw 28x28 images are pooled down to keep the search space
    tractable, and the split is optionally subsampled to keep evaluation fast.

    Args:
        train_fraction: Share of samples assigned to training.
        random_seed: Seed for the split and the subset draw.
        grid_side: Side length of the pooled grid; must divide 28.
        max_train_samples: Cap on training rows after the split; ``None`` keeps
            the whole training split.
        max_test_samples: Cap on test rows after the split.

    Returns:
        A :class:`ClassificationDataset` with ``grid_side ** 2`` features and 10
        classes.
    """
    features, labels = load_fashion_mnist_features_and_labels(grid_side)
    return split_features_and_labels(
        features,
        labels,
        train_fraction=train_fraction,
        random_seed=random_seed,
        max_train_samples=max_train_samples,
        max_test_samples=max_test_samples,
        number_of_classes=_FASHION_MNIST_NUMBER_OF_CLASSES,
    )
