"""Pooled MNIST loading for the mnist examples.

The standard Keras ``mnist.npz`` archive is downloaded once and cached under
``examples/mnist/data/``. Images are average-pooled to a small grid and
standardized, then split into a :class:`ClassificationDataset`.

The layout mirrors ``examples/iris/dataset.py``: a raw
``load_mnist_features_and_labels`` returning ``(features, labels)`` for the whole
dataset, and a top-level ``load_mnist`` returning the shared bundle. The pooling
grid is a tweakable knob (``grid_side``); ``grid_side == 28`` gives
full-resolution MNIST, and :func:`pool_features_to_grid` is exposed so a user can
re-pool raw images by hand.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from examples._datasets import (
    ClassificationDataset,
    download_file_if_missing,
    split_features_and_labels,
)

_MNIST_DATA_DIR = Path(__file__).parent / "data"
_MNIST_NPZ_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
_MNIST_IMAGE_SIDE = 28
_MNIST_NUMBER_OF_CLASSES = 10
# 6/7 mirrors MNIST's canonical 60,000 / 10,000 train/test split ratio.
_MNIST_DEFAULT_TRAIN_FRACTION = 6 / 7


def pool_features_to_grid(images_uint8: np.ndarray, grid_side: int) -> np.ndarray:
    """Average-pool ``(N, 28, 28)`` uint8 images to flattened float grids.

    Pixels are scaled to ``[0, 1]`` and each output value is the mean of a
    ``block x block`` patch of the original image. The row-major flatten matches
    the substrate's row-major input-node id assignment, so pooled cell ``i``
    corresponds to substrate input node ``i``.

    Args:
        images_uint8: Raw images, shape ``(N, 28, 28)``.
        grid_side: Side length of the pooled grid; must divide 28.

    Returns:
        Array of shape ``(N, grid_side ** 2)``.
    """
    number_of_images = images_uint8.shape[0]
    block = _MNIST_IMAGE_SIDE // grid_side
    scaled = images_uint8.astype(np.float32) / 255.0
    pooled = scaled.reshape(
        number_of_images, grid_side, block, grid_side, block
    ).mean(axis=(2, 4))
    return pooled.reshape(number_of_images, grid_side * grid_side)


def _standardize_features(features: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-variance each feature over the whole dataset.

    Standardizing keeps pre-activation sums small so that the networks' outputs
    do not saturate, which matters for a usable softmax and for gradient in the
    fitness signal. Statistics are taken over every returned sample, mirroring
    the full-data normalisation the iris loader applies.
    """
    feature_mean = features.mean(axis=0, keepdims=True)
    feature_std = features.std(axis=0, keepdims=True) + 1e-6
    return (features - feature_mean) / feature_std


def load_mnist_features_and_labels(
    grid_side: int = 7,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load the whole MNIST dataset, pooled to ``grid_side`` and standardized.

    The train and test halves of the Keras archive are concatenated into one
    ``(features, labels)`` pair, mirroring :func:`load_iris_features_and_labels`;
    the split is drawn later by :func:`load_mnist` / ``split_features_and_labels``.

    Args:
        grid_side: Side length of the pooled grid; must divide 28.
            ``grid_side == 28`` keeps full-resolution 784-pixel images.

    Returns:
        ``(features, labels)`` where features is a
        ``[70000, grid_side ** 2]`` float tensor and labels is a ``[70000]``
        long tensor holding digits 0-9.
    """
    npz_path = download_file_if_missing(_MNIST_NPZ_URL, _MNIST_DATA_DIR / "mnist.npz")
    with np.load(npz_path) as mnist_data:
        images = np.concatenate(
            [mnist_data["x_train"], mnist_data["x_test"]], axis=0
        )
        labels = np.concatenate(
            [mnist_data["y_train"], mnist_data["y_test"]]
        ).astype(np.int64)

    pooled = pool_features_to_grid(images, grid_side)
    standardized = _standardize_features(pooled)
    return (
        torch.from_numpy(standardized).to(torch.float32),
        torch.from_numpy(labels),
    )


def load_mnist(
    *,
    train_fraction: float = _MNIST_DEFAULT_TRAIN_FRACTION,
    random_seed: int,
    grid_side: int = 7,
    max_train_samples: int | None = 3000,
    max_test_samples: int | None = 2000,
) -> ClassificationDataset:
    """Load MNIST, pool it to a small grid and split it into a bundle.

    The aligned top-level entrypoint, matching :func:`load_iris` in shape and
    return type. Vanilla NEAT grows topology from a minimal start, so the raw
    28x28 images are pooled down to keep the search space tractable, and the
    split is optionally subsampled to keep evaluation fast.

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
    features, labels = load_mnist_features_and_labels(grid_side)
    return split_features_and_labels(
        features,
        labels,
        train_fraction=train_fraction,
        random_seed=random_seed,
        max_train_samples=max_train_samples,
        max_test_samples=max_test_samples,
        number_of_classes=_MNIST_NUMBER_OF_CLASSES,
    )
