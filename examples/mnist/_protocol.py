"""The MNIST comparison protocol: the shared multi-class controller, bound to MNIST.

Everything about the protocol - the fixed split, both tracks, the pilot that
never scores the test, the resumable search - lives in
:mod:`examples._benchmark.multiclass_protocol` and is shared with CIFAR. This
module only binds MNIST's pieces to it: the loader, the single-channel 28x28
geometry, and MNIST's own environment digest. The comparison and the
reproduction runner (``examples.mnist.deepneat``) stay separate.
"""

from __future__ import annotations

from pathlib import Path

import torch

from examples._benchmark.multiclass_protocol import (
    MulticlassBenchmarkSettings,
    PreparedData,
    SearchCallable,
    SearchContext,
    SelectedCandidate,
    run_multiclass_protocol,
)
from examples._benchmark.multiclass_protocol import prepare_data as _shared_prepare_data
from examples._experiment import ExperimentReport
from examples.mnist._execution import execution_environment
from examples.mnist.dataset import load_mnist

# MNIST is single-channel 28x28 with ten classes; these are fixed, not tunable.
CHANNELS = 1
IMAGE_SIDE = 28
NUMBER_OF_CLASSES = 10

# The comparison controller is shared, so its settings type is MNIST's too. The
# alias keeps the MNIST-facing name that the profiles and tests already use.
MnistBenchmarkSettings = MulticlassBenchmarkSettings

__all__ = [
    "CHANNELS",
    "IMAGE_SIDE",
    "NUMBER_OF_CLASSES",
    "MnistBenchmarkSettings",
    "PreparedData",
    "SearchCallable",
    "SearchContext",
    "SelectedCandidate",
    "prepare_data",
    "run_mnist_protocol",
]


def _load_mnist_dataset(
    settings: MulticlassBenchmarkSettings,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load the official MNIST split as flat rows, without normalizing.

    Normalization is left to the shared preprocessing, which fits its statistics
    after the validation split is carved out, so no validation pixel touches the
    statistics the training split is standardized by.
    """
    dataset = load_mnist(
        random_seed=settings.split_seed,
        grid_side=settings.image_side,
        max_train_samples=settings.maximum_training_samples,
        max_test_samples=settings.maximum_test_samples,
        standardize=False,
    )
    return (
        dataset.train_features,
        dataset.train_labels,
        dataset.test_features,
        dataset.test_labels,
    )


def prepare_data(settings: MnistBenchmarkSettings) -> PreparedData:
    """Load MNIST, carve the fixed validation split, and hold out the test set."""
    return _shared_prepare_data(settings, load_dataset=_load_mnist_dataset)


def run_mnist_protocol(
    settings: MnistBenchmarkSettings,
    search: SearchCallable,
    *,
    method_name: str,
    artifacts_directory: Path | None = None,
    lock_sha256: str = "unlocked",
) -> ExperimentReport:
    """Run one method through the MNIST protocol and report both tracks."""
    return run_multiclass_protocol(
        settings,
        search,
        method_name=method_name,
        load_dataset=_load_mnist_dataset,
        environment=execution_environment,
        artifacts_directory=artifacts_directory,
        lock_sha256=lock_sha256,
    )
