"""Cross-dataset primitives shared by the PolyNEAT example scripts.

Only what every dataset needs lives here: downloading and caching a raw file,
drawing a reproducible train/test split, and the shared bundle type every
classification example returns. Dataset-specific parsing lives next to the
examples that use it (``examples/<dataset>/dataset.py``).
"""

from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


def download_file_if_missing(source_url: str, destination_path: Path) -> Path:
    """Download ``source_url`` to ``destination_path`` unless it is already there.

    Args:
        source_url: URL to fetch on a cache miss.
        destination_path: Where the file is cached. Parent directories are
            created if needed.

    Returns:
        ``destination_path``, for chaining into a read.
    """
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    if not destination_path.exists():
        print(f"Downloading {source_url} to {destination_path} ...")
        urllib.request.urlretrieve(source_url, destination_path)
    return destination_path


def split_indices_into_train_and_test(
    number_of_samples: int,
    train_fraction: float,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Shuffle ``range(number_of_samples)`` and cut it into a train and test part.

    Indices are returned rather than the split data itself because some callers
    need to draw further subsets from within the training split.

    Args:
        number_of_samples: Total number of samples in the dataset.
        train_fraction: Share of samples assigned to the training split.
        random_seed: Seed for the shuffle, so that runs are reproducible.

    Returns:
        ``(train_indices, test_indices)`` as disjoint integer arrays whose union
        covers every sample index.
    """
    split_rng = np.random.default_rng(random_seed)
    shuffled_indices = split_rng.permutation(number_of_samples)
    train_size = int(train_fraction * number_of_samples)
    return shuffled_indices[:train_size], shuffled_indices[train_size:]


@dataclass(frozen=True)
class ClassificationDataset:
    """A train/test split of a classification dataset as one named bundle.

    Every classification example (iris, mnist) returns this exact type from its
    top-level loader, so their call sites read identically: reach for
    ``.train_features`` / ``.train_labels`` / ``.test_features`` /
    ``.test_labels`` rather than unpacking a positional tuple whose order every
    caller has to remember.
    """

    train_features: torch.Tensor  # [n_train, n_features] float32
    train_labels: torch.Tensor  # [n_train] long
    test_features: torch.Tensor  # [n_test, n_features] float32
    test_labels: torch.Tensor  # [n_test] long
    number_of_classes: int

    @property
    def number_of_features(self) -> int:
        """Length of one feature vector (the network's input width)."""
        return int(self.train_features.shape[1])


def split_features_and_labels(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    train_fraction: float,
    random_seed: int,
    max_train_samples: int | None = None,
    max_test_samples: int | None = None,
    number_of_classes: int | None = None,
) -> ClassificationDataset:
    """Shuffle-split ``(features, labels)`` into a :class:`ClassificationDataset`.

    The single seam every classification example splits through, so grid size,
    split fraction and subset caps are all tweakable knobs on one call.

    Args:
        features: ``[n_samples, n_features]`` float tensor.
        labels: ``[n_samples]`` long tensor of class label indices.
        train_fraction: Share of samples assigned to the training split.
        random_seed: Seed for both the split and the optional subset draw, so
            repeats see the same data.
        max_train_samples: If set and smaller than the training split, randomly
            subsample the training split to this many rows (for tractability on
            large sets like MNIST). ``None`` keeps the whole split.
        max_test_samples: As above, for the test split.
        number_of_classes: Number of classes. ``None`` infers it as
            ``int(labels.max()) + 1`` over the full label tensor.

    Returns:
        The assembled :class:`ClassificationDataset`.
    """
    train_indices, test_indices = split_indices_into_train_and_test(
        number_of_samples=len(labels),
        train_fraction=train_fraction,
        random_seed=random_seed,
    )

    subset_rng = np.random.default_rng(random_seed)
    if max_train_samples is not None and len(train_indices) > max_train_samples:
        train_indices = subset_rng.choice(
            train_indices, size=max_train_samples, replace=False
        )
    if max_test_samples is not None and len(test_indices) > max_test_samples:
        test_indices = subset_rng.choice(
            test_indices, size=max_test_samples, replace=False
        )

    if number_of_classes is None:
        number_of_classes = int(labels.max()) + 1

    return ClassificationDataset(
        train_features=features[train_indices],
        train_labels=labels[train_indices],
        test_features=features[test_indices],
        test_labels=labels[test_indices],
        number_of_classes=number_of_classes,
    )
