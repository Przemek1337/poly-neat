"""Cross-dataset primitives shared by the PolyNEAT example scripts.

Only what every dataset needs lives here: downloading and caching a raw file,
and drawing a reproducible train/test split. Dataset-specific parsing lives
next to the examples that use it (``examples/<dataset>/dataset.py``).
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np


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

    Indices are returned rather than the split data itself because some examples
    need to draw further subsets from within the training split - L-NEAT's fixed
    learning subset is drawn per class from the training indices.

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
