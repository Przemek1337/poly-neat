"""Shared wiring behind the MNIST benchmark profiles, smoke, pilot and full.

One entry point per method reads one yaml. The mode decides what the run is: a
smoke profile runs on a small subset and exercises the whole path including a
quick test read, a pilot runs on more data but never touches the official test,
and a full run is bound to a frozen v2 lock. MNIST downloads itself, so unlike
the pneumonia benchmark there is no archive to point at and no synthetic
fallback to refuse; the mode alone separates a plumbing check from a measurement.
"""

from __future__ import annotations

from pathlib import Path

import torch
import yaml

from examples._benchmark.cli import parse_profile_cli as _shared_parse_profile_cli
from examples._benchmark.cli import run_profile_main as _shared_run_profile_main
from examples._benchmark.execution import resolve_profile_execution
from examples._experiment import ExperimentReport
from examples.mnist._execution import (
    ExecutionLockError,
    ExecutionOptions,
    validate_execution_lock,
)
from examples.mnist._protocol import (
    CHANNELS,
    IMAGE_SIDE,
    NUMBER_OF_CLASSES,
    MnistBenchmarkSettings,
    SearchCallable,
    prepare_data,
    run_mnist_protocol,
)
from polyneat.logging_utils.custom_logger import get_logger
from polyneat.training.training_recipe import LearningRateSchedule, TrainingRecipe

logger = get_logger(__name__)

CONFIGS_DIRECTORY = Path(__file__).parent / "configs"
_PROFILE_DESCRIPTION = "PolyNEAT MNIST benchmark profile"


class ProfileError(RuntimeError):
    """Raised when a profile cannot be run as asked."""


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


def load_profile_settings(
    config_file_path: Path,
    *,
    random_seed: int | None = None,
    device: torch.device | None = None,
    execution: ExecutionOptions | None = None,
) -> MnistBenchmarkSettings:
    """Read one MNIST profile yaml into :class:`MnistBenchmarkSettings`."""
    payload = yaml.safe_load(config_file_path.read_text(encoding="utf-8"))
    protocol = payload["protocol"]
    execution = resolve_profile_execution(protocol, execution)
    return MnistBenchmarkSettings(
        protocol_id=protocol["protocol_id"],
        dataset_release=protocol["dataset_release"],
        dataset_license=protocol["dataset_license"],
        image_side=int(protocol.get("image_side", IMAGE_SIDE)),
        channels=CHANNELS,
        number_of_classes=NUMBER_OF_CLASSES,
        split_seed=int(protocol["split_seed"]),
        validation_fraction=float(protocol["validation_fraction"]),
        search_seed=int(protocol["search_seed"] if random_seed is None else random_seed),
        retraining_seeds=tuple(int(seed) for seed in protocol["retraining_seeds"]),
        candidate_recipe=_recipe_from_payload(payload["candidate_recipe"]),
        track_b_recipe=_recipe_from_payload(payload["track_b_recipe"]),
        inference_batch_size=int(protocol.get("inference_batch_size", 64)),
        maximum_phenotype_parameters=protocol.get("maximum_phenotype_parameters"),
        search_budget_seconds=protocol.get("search_budget_seconds"),
        uses_augmentation=bool(protocol.get("uses_augmentation", True)),
        maximum_training_samples=protocol.get("maximum_training_samples"),
        maximum_test_samples=protocol.get("maximum_test_samples"),
        execution=execution,
        profile_payload=payload,
        device_for_computation=torch.device("cpu") if device is None else device,
    )


def run_profile_experiment(
    *,
    config_file_path: Path,
    method_name: str,
    build_search: SearchCallable,
    device: torch.device | None = None,
    random_seed: int | None = None,
    artifacts_directory: Path | None = None,
    data_directory: Path | None = None,
    execution: ExecutionOptions | None = None,
) -> ExperimentReport:
    """Run one MNIST profile end to end through the shared protocol.

    Raises:
        ExecutionLockError: If a full run's configuration, seed, environment or
            dataset identity does not match its frozen lock.
    """
    del data_directory  # MNIST downloads itself; there is no archive to point at.
    settings = load_profile_settings(
        config_file_path, random_seed=random_seed, device=device, execution=execution
    )
    lock_sha256 = "unlocked"
    if settings.execution.mode == "full":
        assert settings.execution.protocol_lock_path is not None
        data = prepare_data(settings)
        lock_sha256 = validate_execution_lock(
            settings.execution.protocol_lock_path,
            method=method_name,
            profile=settings.profile_payload,
            search_seed=settings.search_seed,
            device=settings.device_for_computation,
            dataset_identity_sha256=data.dataset_identity_sha256,
        )
    return run_mnist_protocol(
        settings,
        build_search,
        method_name=method_name,
        artifacts_directory=artifacts_directory,
        lock_sha256=lock_sha256,
    )


def run_profile_main(
    run_experiment,
    *,
    default_config_file_path: Path,
    artifacts_directory: Path,
    argument_list: list[str] | None = None,
) -> None:
    """Run one MNIST profile from the command line via the shared spine."""
    _shared_run_profile_main(
        run_experiment,
        default_config_file_path=default_config_file_path,
        artifacts_directory=artifacts_directory,
        description=_PROFILE_DESCRIPTION,
        expected_errors=(ProfileError, ExecutionLockError),
        argument_list=argument_list,
    )


def parse_profile_cli(argument_list: list[str] | None = None, *, default_config_file_path: Path):
    """Parse the MNIST profile command line via the shared spine."""
    return _shared_parse_profile_cli(
        argument_list,
        default_config_file_path=default_config_file_path,
        description=_PROFILE_DESCRIPTION,
    )
