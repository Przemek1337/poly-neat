"""Freeze verified profiles and an audited manifest after pilot runs, without testing."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import yaml

from examples._example_cli import add_device_arguments, resolve_device
from examples.pediatric_pneumonia._dataset_audit import audit_pediatric_pneumonia_dataset
from examples.pediatric_pneumonia._dataset_manifest import build_manifest_from_audit
from examples.pediatric_pneumonia._execution import canonical_digest, execution_environment
from examples.pediatric_pneumonia._profiles import CONFIGS_DIRECTORY
from examples.pediatric_pneumonia._protocol_lock import ProtocolLockError

DEFAULT_METHODS = ("deepneat", "exact", "random_search", "fixed_cnn")


def freeze_series(
    *,
    profiles_directory: Path,
    data_directory: Path,
    output_directory: Path,
    methods: list[str],
    seeds: list[int],
    protocol_id: str,
    dataset_release: str,
    dataset_license: str,
    pilot_reports: list[Path],
    environment: dict,
) -> Path:
    """Produce a new series directory; never overwrite an existing study."""
    if output_directory.exists():
        raise ProtocolLockError("output directory already exists; a freeze creates a new study")
    if len(set(seeds)) != len(seeds) or len(seeds) < 5 or any(seed < 0 for seed in seeds):
        raise ProtocolLockError("declare at least five distinct nonnegative search seeds")
    if len(set(methods)) != len(methods) or not set(DEFAULT_METHODS).issubset(methods):
        raise ProtocolLockError("the series needs DeepNEAT, EXACT, random_search and fixed_cnn")
    for value in (protocol_id, dataset_release, dataset_license):
        if not value.strip() or any(
            word in value.lower()
            for word in ("todo", "unverified", "tbd", "wpisz", "placeholder", "changeme")
        ):
            raise ProtocolLockError("supply verified dataset provenance and a concrete protocol ID")
    profiles = {
        name: yaml.safe_load((profiles_directory / f"{name}_full.yaml").read_text("utf-8"))
        for name in methods
    }
    reference = profiles["deepneat"]
    common_protocol_fields = ("image_side", "split_seed", "bootstrap_seed", "uses_augmentation")
    for name, profile in profiles.items():
        if profile["protocol"].get("allows_synthetic_archive", False):
            raise ProtocolLockError("cannot freeze a synthetic profile")
        for key in common_protocol_fields:
            if profile["protocol"].get(key) != reference["protocol"].get(key):
                raise ProtocolLockError(f"inconsistent common protocol field: {name}/{key}")
        if name in DEFAULT_METHODS and (
            profile["track_b_recipe"] != reference["track_b_recipe"]
            or profile["protocol"]["retraining_seeds"] != reference["protocol"]["retraining_seeds"]
        ):
            raise ProtocolLockError("track B recipes and seeds must match across methods")
    for name in ("exact", "random_search"):
        if profiles[name]["protocol"].get("search_budget_seconds") != (
            reference["protocol"].get("search_budget_seconds")
        ):
            raise ProtocolLockError("search methods need the same wall-clock budget")

    reports = {}
    for path in pilot_reports:
        report = json.loads(path.read_text(encoding="utf-8"))
        effective = report["effective_configuration"]
        if effective.get("mode") != "pilot" or report.get("official_test"):
            raise ProtocolLockError("only test-free pilot reports may justify a freeze")
        if report["blocking_audit_findings"]:
            raise ProtocolLockError("resolve pilot audit findings first")
        score = report.get("summary", {}).get("metric_values", {}).get("search_validation_auroc")
        if not isinstance(score, int | float) or not math.isfinite(score):
            raise ProtocolLockError("pilot report has no successful selection result")
        if report.get("environment") != environment:
            raise ProtocolLockError("pilot ran with a different implementation or environment")
        reports[report["method"]] = report
    for name, profile in profiles.items():
        if (
            name not in reports
            or reports[name]["effective_configuration"]["profile_payload"] != profile
        ):
            raise ProtocolLockError(f"missing successful pilot of the actual profile: {name}")
        if set(seeds) & {reports[name]["effective_configuration"]["search_seed"]}:
            raise ProtocolLockError("pilot and result seeds must be disjoint")

    audit = audit_pediatric_pneumonia_dataset(data_directory, compute_similarity_report=False)
    for name, profile in profiles.items():
        provenance = profile["protocol"]
        pilot_manifest = build_manifest_from_audit(
            audit,
            protocol_id=provenance["protocol_id"],
            dataset_release=provenance["dataset_release"],
            dataset_license=provenance["dataset_license"],
            split_seed=provenance["split_seed"],
        )
        if reports[name]["manifest_sha256"] != pilot_manifest.compute_sha256():
            raise ProtocolLockError("archive changed since pilot; audit and rerun the pilot")
    manifest = build_manifest_from_audit(
        audit,
        protocol_id=protocol_id,
        dataset_release=dataset_release,
        dataset_license=dataset_license,
        split_seed=reference["protocol"]["split_seed"],
    )
    if manifest.blocking_findings:
        raise ProtocolLockError(
            "dataset audit blocks freeze: "
            + "; ".join(finding.message for finding in manifest.blocking_findings)
        )
    profiles = copy.deepcopy(profiles)
    for profile in profiles.values():
        profile["protocol"].update(
            protocol_id=protocol_id,
            dataset_release=dataset_release,
            dataset_license=dataset_license,
        )
    payload = {
        "schema_version": "2.0",
        "manifest_path": "manifest.json",
        "manifest_sha256": manifest.compute_sha256(),
        "profiles": profiles,
        "search_seeds": seeds,
        "environment": environment,
        "pilot_reports": {name: canonical_digest(report) for name, report in reports.items()},
        "checkpoint_policy": "each_completed_evaluation_and_generation_boundary",
        "interrupted_training": "rollback_and_charge_consumed_time",
    }
    payload["lock_sha256"] = canonical_digest(payload)
    output_directory.mkdir(parents=True)
    manifest.write_json_file(output_directory / "manifest.json")
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--profiles-directory", type=Path, default=CONFIGS_DIRECTORY)
    parser.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS))
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--protocol-id", required=True)
    parser.add_argument("--dataset-release", required=True)
    parser.add_argument("--dataset-license", required=True)
    parser.add_argument("--pilot-reports", type=Path, nargs="+", required=True)
    add_device_arguments(parser)
    args = parser.parse_args()
    import torch

    device = resolve_device(args) or torch.device("cpu")
    try:
        path = freeze_series(
            profiles_directory=args.profiles_directory,
            data_directory=args.data_directory,
            output_directory=args.output_directory,
            methods=args.methods,
            seeds=args.seeds,
            protocol_id=args.protocol_id,
            dataset_release=args.dataset_release,
            dataset_license=args.dataset_license,
            pilot_reports=args.pilot_reports,
            environment=execution_environment(device),
        )
    except ProtocolLockError as error:
        parser.error(str(error))
    print(f"Frozen series: {path}")


if __name__ == "__main__":
    main()
