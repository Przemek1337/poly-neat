"""Assemble an immutable v2 lock from verified pilot reports, shared across families.

Both multi-class benchmarks freeze the same way: read the test-free pilot
reports, check they were produced by the exact profiles, seeds, environment and
dataset the series will use, refuse placeholder provenance, and write a
self-checksummed lock. The only family difference is which methods must be
present and which fitness metric a pilot reports, so those are parameters and
the rest lives here once.
"""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Sequence
from pathlib import Path

import yaml

from examples._benchmark.execution import ExecutionLockError, canonical_digest

_PLACEHOLDER_WORDS = ("todo", "tbd", "unverified", "placeholder", "changeme", "xxx", "fixme")
_COMMON_PROTOCOL_FIELDS = (
    "image_side",
    "split_seed",
    "validation_fraction",
    "uses_augmentation",
    "search_budget_seconds",
)


def assemble_series_lock(
    *,
    profiles_directory: Path,
    output_directory: Path,
    methods: Sequence[str],
    required_methods: Sequence[str],
    seeds: Sequence[int],
    protocol_id: str,
    dataset_release: str,
    dataset_license: str,
    pilot_reports: Sequence[Path],
    environment: dict,
    metric_key: str,
    minimum_seeds: int = 3,
) -> Path:
    """Produce a new series directory with a frozen lock; never overwrite a study.

    Args:
        profiles_directory: Where the ``<method>_full.yaml`` profiles live.
        output_directory: New directory to create; must not already exist.
        methods: Methods to freeze.
        required_methods: Methods that must all be present in ``methods``.
        seeds: The distinct nonnegative search seeds of the series.
        protocol_id: Concrete protocol identifier, not a placeholder.
        dataset_release: Verified dataset release string.
        dataset_license: Verified dataset license string.
        pilot_reports: One test-free pilot report per method.
        environment: The frozen execution environment.
        metric_key: The metric a successful pilot reports (e.g. accuracy).
        minimum_seeds: Fewest distinct seeds a series may declare.

    Returns:
        Path of the written ``protocol.lock.yaml``.

    Raises:
        ExecutionLockError: On any inconsistency, placeholder, or shared seed.
    """
    if output_directory.exists():
        raise ExecutionLockError("output directory already exists; a freeze creates a new study")
    if len(set(seeds)) != len(seeds) or len(seeds) < minimum_seeds or any(s < 0 for s in seeds):
        raise ExecutionLockError(
            f"declare at least {minimum_seeds} distinct nonnegative search seeds"
        )
    if len(set(methods)) != len(methods) or not set(required_methods).issubset(methods):
        raise ExecutionLockError(f"the series needs every method of {tuple(required_methods)}")
    for value in (protocol_id, dataset_release, dataset_license):
        if not value.strip() or any(word in value.lower() for word in _PLACEHOLDER_WORDS):
            raise ExecutionLockError(
                "supply verified dataset provenance and a concrete protocol ID"
            )

    profiles = {
        name: yaml.safe_load((profiles_directory / f"{name}_full.yaml").read_text("utf-8"))
        for name in methods
    }
    reference = profiles[methods[0]]
    for name, profile in profiles.items():
        for key in _COMMON_PROTOCOL_FIELDS:
            if profile["protocol"].get(key) != reference["protocol"].get(key):
                raise ExecutionLockError(f"inconsistent common protocol field: {name}/{key}")
        if (
            profile["track_b_recipe"] != reference["track_b_recipe"]
            or profile["protocol"]["retraining_seeds"] != reference["protocol"]["retraining_seeds"]
        ):
            raise ExecutionLockError("track B recipe and seeds must match across methods")

    reports: dict[str, dict] = {}
    dataset_identities: set[str] = set()
    for path in pilot_reports:
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("mode") != "pilot" or report.get("official_test"):
            raise ExecutionLockError("only test-free pilot reports may justify a freeze")
        if report.get("environment") != environment:
            raise ExecutionLockError("pilot ran with a different implementation or environment")
        score = report.get("summary", {}).get("metric_values", {}).get(metric_key)
        if not isinstance(score, int | float) or not math.isfinite(score):
            raise ExecutionLockError("pilot report has no successful selection result")
        dataset_identities.add(report["dataset_identity_sha256"])
        reports[report["method"]] = report
    if len(dataset_identities) != 1:
        raise ExecutionLockError("pilot reports disagree on the dataset identity")
    dataset_identity = dataset_identities.pop()

    for name, profile in profiles.items():
        if name not in reports:
            raise ExecutionLockError(f"missing successful pilot of the actual profile: {name}")
        if reports[name]["effective_configuration"]["profile_payload"] != profile:
            raise ExecutionLockError(f"{name}: pilot ran a different profile than the frozen one")
        if reports[name]["effective_configuration"]["search_seed"] in set(seeds):
            raise ExecutionLockError("pilot and result seeds must be disjoint")

    profiles = copy.deepcopy(profiles)
    for profile in profiles.values():
        profile["protocol"].update(
            protocol_id=protocol_id,
            dataset_release=dataset_release,
            dataset_license=dataset_license,
        )
    payload = {
        "schema_version": "2.0",
        "dataset_identity_sha256": dataset_identity,
        "profiles": profiles,
        "search_seeds": list(seeds),
        "environment": environment,
        "pilot_reports": {name: canonical_digest(report) for name, report in reports.items()},
        "checkpoint_policy": "each_completed_evaluation_and_generation_boundary",
        "interrupted_training": "rollback_and_charge_consumed_time",
    }
    payload["lock_sha256"] = canonical_digest(payload)

    output_directory.mkdir(parents=True)
    (output_directory / "profiles").mkdir()
    for name, profile in profiles.items():
        (output_directory / "profiles" / f"{name}.yaml").write_text(
            yaml.safe_dump(profile, sort_keys=True), encoding="utf-8"
        )
    (output_directory / "pilot_reports.json").write_text(
        json.dumps(reports, indent=2, sort_keys=True), encoding="utf-8"
    )
    destination = output_directory / "protocol.lock.yaml"
    destination.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")
    return destination
