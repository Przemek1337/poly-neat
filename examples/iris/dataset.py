"""UCI Iris loading for the iris examples.

The raw file is downloaded once from the UCI repository and cached under
``examples/iris/data/``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from examples._datasets import download_file_if_missing

_IRIS_DATA_DIR = Path(__file__).parent / "data"
_IRIS_DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
)
_IRIS_CLASS_NAME_TO_INDEX = {
    "Iris-setosa": 0,
    "Iris-versicolor": 1,
    "Iris-virginica": 2,
}


def load_iris_features_and_labels() -> tuple[torch.Tensor, torch.Tensor]:
    """Load the UCI Iris dataset, downloading it once into the cache directory.

    Returns:
        ``(features, class_label_indices)`` where features is a ``[150, 4]``
        float tensor min-max normalised to ``[0, 1]`` per column, and
        class_label_indices is a ``[150]`` long tensor holding 0, 1 or 2.
    """
    data_path = download_file_if_missing(_IRIS_DATA_URL, _IRIS_DATA_DIR / "iris.data")
    rows = [
        line.strip().split(",")
        for line in data_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    features = np.array(
        [[float(value) for value in row[:4]] for row in rows], dtype=np.float32
    )
    labels = np.array(
        [_IRIS_CLASS_NAME_TO_INDEX[row[4]] for row in rows], dtype=np.int64
    )
    feature_minima = features.min(axis=0)
    feature_maxima = features.max(axis=0)
    features = (features - feature_minima) / (feature_maxima - feature_minima)
    return torch.from_numpy(features), torch.from_numpy(labels)
