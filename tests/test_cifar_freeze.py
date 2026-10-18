"""Freezing a CIFAR-10 series binds it to its pilots, profiles, seeds and environment.

The freeze command turns verified pilot reports into an immutable v2 lock. These
tests build pilot reports by hand - the freeze logic is what is under test, not
the search - and check both directions: a consistent set of pilots freezes and
the resulting lock validates a matching full run, while a placeholder provenance,
a mismatched profile, a shared seed or an edited lock are all refused.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import yaml

from examples._benchmark.execution import ExecutionLockError
from examples.cifar10._execution import execution_environment, validate_execution_lock
from examples.cifar10._freeze import DEFAULT_METHODS, freeze_series
from examples.cifar10._profiles import CONFIGS_DIRECTORY

_DATASET_IDENTITY = "a" * 64
_PILOT_SEED = 7
_SERIES_SEEDS = [101, 102, 103]


def _write_pilot_reports(tmp_path: Path, profiles: dict, environment: dict) -> list[Path]:
    """One test-free pilot report per method, shaped as the freeze expects."""
    report_paths = []
    for method, profile in profiles.items():
        path = tmp_path / f"{method}_pilot.json"
        path.write_text(
            json.dumps(
                {
                    "method": method,
                    "mode": "pilot",
                    "official_test": {},
                    "environment": environment,
                    "dataset_identity_sha256": _DATASET_IDENTITY,
                    "summary": {"metric_values": {"validation_accuracy": 0.9}},
                    "effective_configuration": {
                        "search_seed": _PILOT_SEED,
                        "profile_payload": profile,
                    },
                }
            ),
            encoding="utf-8",
        )
        report_paths.append(path)
    return report_paths


@pytest.fixture
def profiles() -> dict:
    return {
        method: yaml.safe_load(
            (CONFIGS_DIRECTORY / f"{method}_full.yaml").read_text(encoding="utf-8")
        )
        for method in DEFAULT_METHODS
    }


@pytest.fixture
def environment() -> dict:
    return execution_environment(torch.device("cpu"))


def test_a_consistent_pilot_set_freezes_and_the_lock_validates(
    tmp_path: Path, profiles: dict, environment: dict
) -> None:
    reports = _write_pilot_reports(tmp_path, profiles, environment)

    lock_path = freeze_series(
        profiles_directory=CONFIGS_DIRECTORY,
        output_directory=tmp_path / "series",
        methods=list(DEFAULT_METHODS),
        seeds=_SERIES_SEEDS,
        protocol_id="cifar10-benchmark-full-v1",
        dataset_release="cifar10/official",
        dataset_license="cifar10-krizhevsky-2009-mit",
        pilot_reports=reports,
        environment=environment,
    )

    assert lock_path.is_file()
    frozen_profile = yaml.safe_load(lock_path.read_text(encoding="utf-8"))["profiles"]["deepneat"]
    # A matching full run validates against the frozen lock.
    lock_digest = validate_execution_lock(
        lock_path,
        method="deepneat",
        profile=frozen_profile,
        search_seed=_SERIES_SEEDS[0],
        device=torch.device("cpu"),
        dataset_identity_sha256=_DATASET_IDENTITY,
    )
    assert isinstance(lock_digest, str)


def test_a_full_run_with_a_different_dataset_is_refused(
    tmp_path: Path, profiles: dict, environment: dict
) -> None:
    reports = _write_pilot_reports(tmp_path, profiles, environment)
    lock_path = freeze_series(
        profiles_directory=CONFIGS_DIRECTORY,
        output_directory=tmp_path / "series",
        methods=list(DEFAULT_METHODS),
        seeds=_SERIES_SEEDS,
        protocol_id="cifar10-benchmark-full-v1",
        dataset_release="cifar10/official",
        dataset_license="cifar10-krizhevsky-2009-mit",
        pilot_reports=reports,
        environment=environment,
    )
    frozen_profile = yaml.safe_load(lock_path.read_text(encoding="utf-8"))["profiles"]["deepneat"]

    with pytest.raises(ExecutionLockError, match="dataset or split"):
        validate_execution_lock(
            lock_path,
            method="deepneat",
            profile=frozen_profile,
            search_seed=_SERIES_SEEDS[0],
            device=torch.device("cpu"),
            dataset_identity_sha256="b" * 64,
        )


def test_placeholder_provenance_is_refused(
    tmp_path: Path, profiles: dict, environment: dict
) -> None:
    reports = _write_pilot_reports(tmp_path, profiles, environment)
    with pytest.raises(ExecutionLockError, match="verified dataset provenance"):
        freeze_series(
            profiles_directory=CONFIGS_DIRECTORY,
            output_directory=tmp_path / "series",
            methods=list(DEFAULT_METHODS),
            seeds=_SERIES_SEEDS,
            protocol_id="TODO",
            dataset_release="cifar10/official",
            dataset_license="cifar10-krizhevsky-2009-mit",
            pilot_reports=reports,
            environment=environment,
        )


def test_a_pilot_seed_in_the_series_is_refused(
    tmp_path: Path, profiles: dict, environment: dict
) -> None:
    reports = _write_pilot_reports(tmp_path, profiles, environment)
    with pytest.raises(ExecutionLockError, match="disjoint"):
        freeze_series(
            profiles_directory=CONFIGS_DIRECTORY,
            output_directory=tmp_path / "series",
            methods=list(DEFAULT_METHODS),
            seeds=[_PILOT_SEED, 102, 103],
            protocol_id="cifar10-benchmark-full-v1",
            dataset_release="cifar10/official",
            dataset_license="cifar10-krizhevsky-2009-mit",
            pilot_reports=reports,
            environment=environment,
        )


def test_an_edited_lock_is_refused(tmp_path: Path, profiles: dict, environment: dict) -> None:
    reports = _write_pilot_reports(tmp_path, profiles, environment)
    lock_path = freeze_series(
        profiles_directory=CONFIGS_DIRECTORY,
        output_directory=tmp_path / "series",
        methods=list(DEFAULT_METHODS),
        seeds=_SERIES_SEEDS,
        protocol_id="cifar10-benchmark-full-v1",
        dataset_release="cifar10/official",
        dataset_license="cifar10-krizhevsky-2009-mit",
        pilot_reports=reports,
        environment=environment,
    )
    payload = yaml.safe_load(lock_path.read_text(encoding="utf-8"))
    payload["search_seeds"] = [*payload["search_seeds"], 999]
    lock_path.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")

    with pytest.raises(ExecutionLockError, match="changed after freezing"):
        validate_execution_lock(
            lock_path,
            method="deepneat",
            profile=payload["profiles"]["deepneat"],
            search_seed=_SERIES_SEEDS[0],
            device=torch.device("cpu"),
            dataset_identity_sha256=_DATASET_IDENTITY,
        )
