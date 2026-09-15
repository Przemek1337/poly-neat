"""Shared wiring behind every pediatric pneumonia profile, smoke or full.

One entry point per method reads one yaml. Which yaml decides what the run is:
a smoke profile runs at 64 pixels against a generated synthetic archive and
proves only that the plumbing holds, while a full profile runs at 128 pixels
against the real Kaggle archive and produces numbers that may be reported.

The difference is declared in the yaml rather than inferred from a file name.
``protocol.allows_synthetic_archive`` says whether a run may fall back to
generating fake radiographs when no data directory is given. Smoke profiles set
it; full profiles do not, so a full run launched without ``--data-directory``
stops with an error instead of quietly measuring synthetic noise and writing it
into a run report that looks exactly like a real one.

Synthetic archives are generated once per directory and cached, so repeated
smoke runs and the test suite do not pay for the fixture twice.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

import torch
import yaml

from examples._benchmark.cli import parse_profile_cli as _shared_parse_profile_cli
from examples._benchmark.cli import run_profile_main as _shared_run_profile_main
from examples._experiment import ExperimentReport
from examples.pediatric_pneumonia._execution import ExecutionOptions, validate_execution_lock
from examples.pediatric_pneumonia._protocol import (
    BenchmarkSettings,
    SearchCallable,
    run_pneumonia_protocol,
)
from examples.pediatric_pneumonia._protocol_lock import ProtocolLockError
from examples.pediatric_pneumonia._synthetic_archive import (
    SyntheticArchiveSpec,
    write_synthetic_archive,
)
from polyneat.evaluators.bootstrap_confidence_intervals import BootstrapConfig
from polyneat.logging_utils.custom_logger import get_logger
from polyneat.training.training_recipe import LearningRateSchedule, TrainingRecipe

logger = get_logger(__name__)


class ProfileDataError(RuntimeError):
    """Raised when a profile cannot be given the archive it requires."""


_PROFILE_DESCRIPTION = "PolyNEAT pediatric pneumonia profile"


CONFIGS_DIRECTORY = Path(__file__).parent / "configs"
SMOKE_DATA_DIRECTORY = Path(__file__).parent / "data" / "synthetic_smoke"
SMOKE_PROTOCOL_ID = "pediatric-pneumonia-smoke-synthetic-v1"
SMOKE_DATASET_RELEASE = "synthetic/pediatric-pneumonia-smoke"
SMOKE_DATASET_LICENSE = "not-applicable-synthetic-fixture"


def config_path_for(profile_module_file: str) -> Path:
    """Locate the yaml belonging to one profile module.

    Configurations live in ``configs/`` rather than beside their modules, so
    the yaml a profile reads is named after the module rather than found next
    to it. Deriving the name from ``__file__`` keeps the pair together under a
    rename instead of leaving a path constant to drift.

    Args:
        profile_module_file: The profile module's ``__file__``.

    Returns:
        Path of that profile's yaml inside :data:`CONFIGS_DIRECTORY`.
    """
    return CONFIGS_DIRECTORY / f"{Path(profile_module_file).stem}.yaml"


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


def resolve_data_directory(
    protocol_payload: dict, data_directory: Path | None, *, config_file_path: Path
) -> Path:
    """Decide which archive a run reads, refusing to invent one for a full profile.

    Args:
        protocol_payload: The ``protocol`` section of the profile.
        data_directory: Directory given on the command line, or ``None``.
        config_file_path: The profile being loaded, named in the error.

    Returns:
        The directory the loader should read.

    Raises:
        ProfileDataError: If no directory was given and the profile does not
            allow the synthetic fixture. A full profile that silently generated
            fake radiographs would produce a run report indistinguishable from
            a real one.
    """
    if data_directory is not None:
        return data_directory
    if bool(protocol_payload.get("allows_synthetic_archive", False)):
        return ensure_smoke_archive()
    raise ProfileDataError(
        f"{config_file_path.name} is a full profile and needs a real archive: pass "
        "--data-directory pointing at the extracted Kaggle download. Only profiles "
        "that set protocol.allows_synthetic_archive may generate a synthetic one."
    )


def load_profile_settings(
    config_file_path: Path,
    *,
    data_directory: Path | None = None,
    random_seed: int | None = None,
    device: torch.device | None = None,
    execution: ExecutionOptions | None = None,
) -> BenchmarkSettings:
    """Read one profile yaml into :class:`BenchmarkSettings`.

    Args:
        config_file_path: The profile to read, smoke or full.
        data_directory: The extracted archive to run against. Required unless
            the profile allows the synthetic fixture.
        random_seed: Override for the search seed, used by the benchmark
            harness to run several seeds of the same profile.
        device: Device the run happens on, from ``--cpu``/``--gpu``. ``None``
            keeps the CPU, which is what the smoke profiles are sized for.

    Returns:
        The settings this run will actually use, which is what gets recorded as
        the effective configuration rather than the yaml alone.

    Raises:
        ProfileDataError: If a full profile was given no data directory.
    """
    payload = yaml.safe_load(config_file_path.read_text(encoding="utf-8"))
    protocol_payload = payload["protocol"]
    execution = execution or ExecutionOptions(
        mode="smoke" if protocol_payload.get("allows_synthetic_archive", False) else "pilot"
    )
    if execution.mode == "smoke" and not protocol_payload.get("allows_synthetic_archive", False):
        raise ProfileDataError("smoke requires a synthetic profile; use --mode pilot for real data")
    if execution.mode != "smoke" and (
        data_directory is None or protocol_payload.get("allows_synthetic_archive", False)
    ):
        raise ProfileDataError("pilot/full needs a real archive and a non-synthetic profile")
    resolved_data_directory = resolve_data_directory(
        protocol_payload, data_directory, config_file_path=config_file_path
    )

    return BenchmarkSettings(
        data_directory=resolved_data_directory,
        protocol_id=protocol_payload.get("protocol_id", SMOKE_PROTOCOL_ID),
        dataset_release=protocol_payload.get("dataset_release", SMOKE_DATASET_RELEASE),
        dataset_license=protocol_payload.get("dataset_license", SMOKE_DATASET_LICENSE),
        image_side=int(protocol_payload["image_side"]),
        split_seed=int(protocol_payload["split_seed"]),
        search_seed=int(protocol_payload["search_seed"] if random_seed is None else random_seed),
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
        device_for_computation=torch.device("cpu") if device is None else device,
        uses_augmentation=bool(protocol_payload.get("uses_augmentation", True)),
        uses_identity_standardization=bool(
            protocol_payload.get("uses_identity_standardization", False)
        ),
        cache_directory=resolved_data_directory.parent / "cache",
        execution=execution,
        profile_payload=payload,
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
    """Run one profile end to end through the shared protocol.

    Args:
        config_file_path: The profile yaml this run is defined by.
        method_name: Label recorded in the report and in artifact names.
        build_search: The method's search, built from the profile.
        device: Device from ``--cpu``/``--gpu``.
        random_seed: Override for the search seed.
        artifacts_directory: Where the manifest, checkpoints, predictions and
            run report are written.
        data_directory: The extracted archive, required by full profiles.

    Returns:
        The report the protocol produced.

    Raises:
        ProfileDataError: If a full profile was given no data directory.
    """
    settings = load_profile_settings(
        config_file_path,
        data_directory=data_directory,
        random_seed=random_seed,
        device=device,
        execution=execution,
    )
    if settings.execution.mode == "full":
        assert settings.execution.protocol_lock_path is not None
        validate_execution_lock(
            settings.execution.protocol_lock_path,
            method=method_name,
            profile=settings.profile_payload,
            search_seed=settings.search_seed,
            device=settings.device_for_computation,
        )
    if settings.dataset_release.startswith("synthetic/"):
        logger.warning(
            "%s runs against a synthetic archive: it checks the plumbing and its numbers "
            "mean nothing scientifically, whatever device it ran on",
            config_file_path.name,
        )
    return run_pneumonia_protocol(
        settings,
        build_search,
        method_name=method_name,
        artifacts_directory=artifacts_directory,
    )


def run_profile_main(
    run_experiment: Callable[..., ExperimentReport],
    *,
    default_config_file_path: Path,
    artifacts_directory: Path,
    argument_list: list[str] | None = None,
) -> None:
    """Run one pneumonia profile from the command line via the shared spine."""
    _shared_run_profile_main(
        run_experiment,
        default_config_file_path=default_config_file_path,
        artifacts_directory=artifacts_directory,
        description=_PROFILE_DESCRIPTION,
        expected_errors=(ProfileDataError, ProtocolLockError),
        argument_list=argument_list,
    )


def parse_profile_cli(
    argument_list: list[str] | None = None, *, default_config_file_path: Path
) -> tuple[torch.device | None, Path | None, Path, argparse.Namespace]:
    """Parse the pneumonia profile command line via the shared spine."""
    return _shared_parse_profile_cli(
        argument_list,
        default_config_file_path=default_config_file_path,
        description=_PROFILE_DESCRIPTION,
    )
