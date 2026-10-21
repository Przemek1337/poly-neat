"""The CIFAR-10 experiment protocol: the shared multi-class controller, bound to CIFAR.

The whole protocol - the fixed split, both tracks, the pilot that never scores
the test, the resumable search - lives in
:mod:`examples._benchmark.multiclass_protocol` and is shared with MNIST. This
module only binds CIFAR's pieces: the loader, the three-channel 32x32 geometry
and CIFAR's own environment digest.

CIFAR runs DeepNEAT only - it is the colour experiment, not a two-method
comparison, because EXACT is single-channel by construction. The comparison of
both algorithms lives on MNIST, and the source-experiment reproduction lives in
``examples.cifar10.deepneat_paper``; both stay separate from this.
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
from examples.cifar10 import _deepneat
from examples.cifar10._execution import execution_environment

# CIFAR-10 is three-channel 32x32 with ten classes; these are fixed, not tunable.
CHANNELS = 3
IMAGE_SIDE = 32
NUMBER_OF_CLASSES = 10
_PIXEL_SCALE = 255.0

Cifar10BenchmarkSettings = MulticlassBenchmarkSettings

__all__ = [
    "CHANNELS",
    "IMAGE_SIDE",
    "NUMBER_OF_CLASSES",
    "Cifar10BenchmarkSettings",
    "PreparedData",
    "SearchCallable",
    "SearchContext",
    "SelectedCandidate",
    "prepare_data",
    "run_cifar10_protocol",
]


def _load_cifar_dataset(
    settings: MulticlassBenchmarkSettings,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load the official CIFAR-10 split as ``[0, 255]`` flat rows, unnormalized.

    The loader is reached through :mod:`examples.cifar10._deepneat` so the test
    suite's redirect to a synthetic same-shape set applies here too and the suite
    never downloads. The loader returns ``[0, 1]`` floats, which are scaled back
    to the ``[0, 255]`` the shared preprocessing expects; normalization is fitted
    later, after the validation split is carved out.
    """
    dataset = _deepneat.load_cifar10(
        random_seed=settings.split_seed,
        max_train_samples=settings.maximum_training_samples,
        max_test_samples=settings.maximum_test_samples,
        standardize=False,
    )
    return (
        dataset.train_features * _PIXEL_SCALE,
        dataset.train_labels,
        dataset.test_features * _PIXEL_SCALE,
        dataset.test_labels,
    )


def prepare_data(settings: Cifar10BenchmarkSettings) -> PreparedData:
    """Load CIFAR-10, carve the fixed validation split, hold out the test set."""
    return _shared_prepare_data(settings, load_dataset=_load_cifar_dataset)


def run_cifar10_protocol(
    settings: Cifar10BenchmarkSettings,
    search: SearchCallable,
    *,
    method_name: str,
    artifacts_directory: Path | None = None,
    lock_sha256: str = "unlocked",
) -> ExperimentReport:
    """Run one method through the CIFAR-10 protocol and report both tracks."""
    return run_multiclass_protocol(
        settings,
        search,
        method_name=method_name,
        load_dataset=_load_cifar_dataset,
        environment=execution_environment,
        artifacts_directory=artifacts_directory,
        lock_sha256=lock_sha256,
    )
