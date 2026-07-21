"""Pooled MNIST loading for the mnist examples.

The standard Keras ``mnist.npz`` archive is downloaded once and cached under
``examples/mnist/data/``. Images are average-pooled to a small
grid, standardized with training statistics, and drawn into train/test
subsets.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from examples._datasets import download_file_if_missing

_MNIST_DATA_DIR = Path(__file__).parent / "data"
_MNIST_NPZ_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
_MNIST_IMAGE_SIDE = 28


def _average_pool_images_to_grid(
    images_uint8: np.ndarray, pooled_grid_side: int
) -> np.ndarray:
    """Average-pool ``(N, 28, 28)`` uint8 images to flattened float grids.

    Pixels are scaled to ``[0, 1]`` and each output value is the mean of a
    ``block x block`` patch of the original image. The row-major flatten matches
    the substrate's row-major input-node id assignment, so pooled cell ``i``
    corresponds to substrate input node ``i``.

    Args:
        images_uint8: Raw images, shape ``(N, 28, 28)``.
        pooled_grid_side: Side length of the pooled grid; must divide 28.

    Returns:
        Array of shape ``(N, pooled_grid_side ** 2)``.
    """
    number_of_images = images_uint8.shape[0]
    block = _MNIST_IMAGE_SIDE // pooled_grid_side
    scaled = images_uint8.astype(np.float32) / 255.0
    pooled = scaled.reshape(
        number_of_images, pooled_grid_side, block, pooled_grid_side, block
    ).mean(axis=(2, 4))
    return pooled.reshape(number_of_images, pooled_grid_side * pooled_grid_side)


def _standardize_features(
    train_features: np.ndarray, test_features: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Zero-mean, unit-variance each feature using training statistics.

    Standardizing keeps pre-activation sums small so that the networks' outputs
    do not saturate, which matters for a usable softmax and for gradient in the
    fitness signal. Test features are scaled with the training mean and standard
    deviation to avoid leaking test statistics.
    """
    feature_mean = train_features.mean(axis=0, keepdims=True)
    feature_std = train_features.std(axis=0, keepdims=True) + 1e-6
    return (
        (train_features - feature_mean) / feature_std,
        (test_features - feature_mean) / feature_std,
    )


def _sample_subset(
    features: np.ndarray,
    labels: np.ndarray,
    subset_size: int,
    rng: np.random.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw a random subset and return it as torch feature and label tensors.

    ``subset_size`` is capped at the number of available samples, so asking for
    more than the dataset holds returns the whole dataset rather than raising.
    """
    subset_size = min(subset_size, features.shape[0])
    selected_indices = rng.choice(features.shape[0], size=subset_size, replace=False)
    feature_tensor = torch.from_numpy(features[selected_indices]).to(torch.float32)
    label_tensor = torch.from_numpy(labels[selected_indices]).to(torch.long)
    return feature_tensor, label_tensor


@dataclass(frozen=True)
class MnistSubsets:
    """Pooled, standardized MNIST train and test subsets as one named bundle.

    A plain data carrier: four tensors travel together from the loader to the
    call sites, and field names replace the positional 4-tuple unpacking that
    every caller would otherwise have to get right by memory.
    """

    train_features: torch.Tensor
    train_labels: torch.Tensor
    test_features: torch.Tensor
    test_labels: torch.Tensor


def load_pooled_mnist_train_and_test(
    pooled_grid_side: int,
    training_subset_size: int,
    test_subset_size: int,
    random_seed: int,
) -> MnistSubsets:
    """Load MNIST, pool it to a small grid, standardize it and draw two subsets.

    Vanilla NEAT grows topology from a minimal start, so the raw 28x28 images
    are average-pooled down to ``pooled_grid_side x pooled_grid_side`` to keep
    the search space tractable; each pooled pixel becomes one input node.

    Args:
        pooled_grid_side: Side length of the pooled grid; must divide 28.
        training_subset_size: Number of training samples to draw.
        test_subset_size: Number of test samples to draw.
        random_seed: Seed for the subset draw.

    Returns:
        A :class:`MnistSubsets` with features of shape
        ``(subset_size, pooled_grid_side ** 2)``.
    """
    npz_path = download_file_if_missing(_MNIST_NPZ_URL, _MNIST_DATA_DIR / "mnist.npz")
    with np.load(npz_path) as mnist_data:
        train_images = _average_pool_images_to_grid(
            mnist_data["x_train"], pooled_grid_side
        )
        train_labels = mnist_data["y_train"].astype(np.int64)
        test_images = _average_pool_images_to_grid(
            mnist_data["x_test"], pooled_grid_side
        )
        test_labels = mnist_data["y_test"].astype(np.int64)

    train_images, test_images = _standardize_features(train_images, test_images)

    subset_rng = np.random.default_rng(random_seed)
    train_features, train_label_tensor = _sample_subset(
        train_images, train_labels, training_subset_size, subset_rng
    )
    test_features, test_label_tensor = _sample_subset(
        test_images, test_labels, test_subset_size, subset_rng
    )
    return MnistSubsets(
        train_features=train_features,
        train_labels=train_label_tensor,
        test_features=test_features,
        test_labels=test_label_tensor,
    )
