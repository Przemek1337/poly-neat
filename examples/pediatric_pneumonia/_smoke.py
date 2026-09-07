"""Shared wiring for the pediatric pneumonia smoke profiles.

A smoke run proves the plumbing, not the science. It runs at 64 pixels on the
CPU with a tiny population, a handful of epochs and a synthetic archive, and it
exercises the parts that are easy to get quietly wrong: both tracks, threshold
selection, checkpoint save and reload, prediction export and the bootstrap.

Its numbers never appear in a results table. The synthetic images are not
radiographs, and a profile this small measures nothing about either algorithm.

The archive is generated once per directory and cached, so repeated smoke runs
and the test suite do not pay for it twice.
"""

from __future__ import annotations

from pathlib import Path

import torch
import yaml

from examples._experiment import ExperimentReport
from examples.pediatric_pneumonia._protocol import (
    BenchmarkSettings,
    SearchCallable,
    run_pneumonia_protocol,
)
from examples.pediatric_pneumonia._synthetic_archive import (
    SyntheticArchiveSpec,
    write_synthetic_archive,
)
from polyneat.evaluators.bootstrap_confidence_intervals import BootstrapConfig
from polyneat.logging_utils.custom_logger import get_logger
from polyneat.training.training_recipe import LearningRateSchedule, TrainingRecipe

logger = get_logger(__name__)

SMOKE_DATA_DIRECTORY = Path(__file__).parent / "data" / "synthetic_smoke"
SMOKE_PROTOCOL_ID = "pediatric-pneumonia-smoke-synthetic-v1"
SMOKE_DATASET_RELEASE = "synthetic/pediatric-pneumonia-smoke"
SMOKE_DATASET_LICENSE = "not-applicable-synthetic-fixture"


def ensure_smoke_archive(data_directory: Path | None = None) -> Path:
    """Create the synthetic archive if it is not already on disk.

    Args:
        data_directory: Where to write it. Defaults to a directory beside this
            package, which is gitignored along with every other dataset.

    Returns:
        The directory to hand to the loader.
    """
    destination = data_directory or SMOKE_DATA_DIRECTORY
    if (destination / "chest_xray" / "train").is_dir():
        return destination
    logger.info("Generating the synthetic smoke archive under %s", destination)
    return write_synthetic_archive(
        destination,
        random_seed=20260905,
        spec=SyntheticArchiveSpec(
            train_normal_patients=14,
            train_pneumonia_patients=20,
            validation_normal_patients=4,
            validation_pneumonia_patients=4,
            test_normal_patients=6,
            test_pneumonia_patients=8,
            images_per_patient=2,
        ),
    )


def load_smoke_settings(
    config_file_path: Path,
    *,
    data_directory: Path | None = None,
    random_seed: int | None = None,
) -> BenchmarkSettings:
    """Read one smoke profile yaml into :class:`BenchmarkSettings`.

    Args:
        config_file_path: The profile to read.
        data_directory: Override for the archive location.
        random_seed: Override for the search seed, used by the benchmark
            harness to run several seeds of the same profile.

    Returns:
        The settings this run will actually use, which is what gets recorded as
        the effective configuration rather than the yaml alone.
    """
    payload = yaml.safe_load(config_file_path.read_text(encoding="utf-8"))
    protocol_payload = payload["protocol"]
    resolved_data_directory = data_directory or ensure_smoke_archive()

    return BenchmarkSettings(
        data_directory=resolved_data_directory,
        protocol_id=protocol_payload.get("protocol_id", SMOKE_PROTOCOL_ID),
        dataset_release=protocol_payload.get("dataset_release", SMOKE_DATASET_RELEASE),
        dataset_license=protocol_payload.get("dataset_license", SMOKE_DATASET_LICENSE),
        image_side=int(protocol_payload["image_side"]),
        split_seed=int(protocol_payload["split_seed"]),
        search_seed=int(
            protocol_payload["search_seed"] if random_seed is None else random_seed
        ),
        retraining_seeds=tuple(int(seed) for seed in protocol_payload["retraining_seeds"]),
        bootstrap_seed=int(protocol_payload["bootstrap_seed"]),
        candidate_recipe=_recipe_from_payload(payload["candidate_recipe"]),
        track_b_recipe=_recipe_from_payload(payload["track_b_recipe"]),
        inference_batch_size=int(protocol_payload["inference_batch_size"]),
        bootstrap=BootstrapConfig(
            number_of_replicates=int(payload["bootstrap"]["number_of_replicates"]),
            maximum_attempts=int(payload["bootstrap"]["maximum_attempts"]),
            confidence_level=float(payload["bootstrap"]["confidence_level"]),
        ),
        maximum_phenotype_parameters=protocol_payload.get("maximum_phenotype_parameters"),
        search_budget_seconds=protocol_payload.get("search_budget_seconds"),
        uses_augmentation=bool(protocol_payload.get("uses_augmentation", True)),
        uses_identity_standardization=bool(
            protocol_payload.get("uses_identity_standardization", False)
        ),
        cache_directory=resolved_data_directory.parent / "cache",
    )


def _recipe_from_payload(payload: dict) -> TrainingRecipe:
    """Build one training recipe from its yaml section."""
    return TrainingRecipe(
        learning_rate=float(payload["learning_rate"]),
        momentum=float(payload["momentum"]),
        weight_decay=float(payload["weight_decay"]),
        batch_size=int(payload["batch_size"]),
        number_of_epochs=int(payload["number_of_epochs"]),
        schedule=LearningRateSchedule(payload["schedule"]),
        minimum_learning_rate=float(payload.get("minimum_learning_rate", 0.0)),
    )


def run_smoke_experiment(
    *,
    config_file_path: Path,
    method_name: str,
    build_search: SearchCallable,
    device: torch.device | None = None,
    random_seed: int | None = None,
    artifacts_directory: Path | None = None,
    data_directory: Path | None = None,
) -> ExperimentReport:
    """Run one smoke profile end to end through the shared protocol."""
    if device is not None and device.type != "cpu":
        logger.warning(
            "The smoke profile is a CPU plumbing check; %s was requested and will be used, "
            "but its numbers still mean nothing scientifically",
            device,
        )
    settings = load_smoke_settings(
        config_file_path, data_directory=data_directory, random_seed=random_seed
    )
    return run_pneumonia_protocol(
        settings,
        build_search,
        method_name=method_name,
        artifacts_directory=artifacts_directory,
    )
