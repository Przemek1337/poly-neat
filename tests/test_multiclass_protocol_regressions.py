"""Data fidelity, safe modes, frozen identities and completed-run recovery."""

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
import yaml
from torch import nn

from examples._benchmark import multiclass_protocol as protocol
from examples._benchmark.execution import ExecutionLockError, ExecutionOptions
from examples._benchmark.freeze_lock import assemble_series_lock
from examples.cifar10 import _profiles as cifar_profiles
from examples.mnist import _profiles as mnist_profiles
from examples.mnist import _protocol as mnist_protocol
from examples.mnist._execution import execution_environment, validate_execution_lock


def settings():
    base = mnist_profiles.load_profile_settings(
        mnist_profiles.CONFIGS_DIRECTORY / "deepneat_full.yaml"
    )
    return replace(
        base,
        maximum_training_samples=20,
        maximum_test_samples=10,
        retraining_seeds=(201,),
        uses_augmentation=False,
        track_b_recipe=replace(base.track_b_recipe, number_of_epochs=1),
        execution=ExecutionOptions(mode="smoke"),
    )


def tensors():
    images = (torch.arange(20 * 784) % 256).reshape(20, 784).float()
    return images, torch.arange(20) % 10, images[:10].clone(), torch.arange(10)


def test_mnist_adapter_preserves_all_gray_levels(monkeypatch):
    pixels = torch.arange(256).repeat(4)[:784].reshape(1, 784).float()
    pixels = pixels.repeat(20, 1)
    monkeypatch.setattr(
        mnist_protocol,
        "load_mnist",
        lambda **_: SimpleNamespace(
            train_features=pixels / 255,
            train_labels=torch.arange(20) % 10,
            test_features=pixels[:10] / 255,
            test_labels=torch.arange(10),
        ),
    )
    data = mnist_protocol.prepare_data(settings())
    assert torch.equal(data.train.images[0].flatten(), pixels[0].to(torch.uint8))
    assert torch.unique(data.train.images).numel() == 256
    assert torch.equal(data.official_test.images[0].flatten(), pixels[0].to(torch.uint8))


@pytest.mark.parametrize("index", range(4))
def test_identity_changes_with_pixels_or_labels_in_either_split(index):
    original = tensors()
    changed = list(tensors())
    changed[index].flatten()[0] += 1
    before = protocol.prepare_data(settings(), load_dataset=lambda _: original)
    after = protocol.prepare_data(settings(), load_dataset=lambda _: tuple(changed))
    assert before.dataset_identity_sha256 != after.dataset_identity_sha256


def test_identity_tracks_row_order_and_split_but_not_provenance_label():
    original = tensors()
    before = protocol.prepare_data(settings(), load_dataset=lambda _: original)
    reordered = (original[0].flip(0), original[1].flip(0), *original[2:])
    after = protocol.prepare_data(settings(), load_dataset=lambda _: reordered)
    assert before.dataset_identity_sha256 != after.dataset_identity_sha256
    split = protocol.prepare_data(
        replace(settings(), split_seed=99), load_dataset=lambda _: original
    )
    assert before.dataset_identity_sha256 != split.dataset_identity_sha256
    relabeled = protocol.prepare_data(
        replace(settings(), dataset_release="verified/v2"), load_dataset=lambda _: original
    )
    assert before.dataset_identity_sha256 == relabeled.dataset_identity_sha256


@pytest.mark.parametrize("family", [mnist_profiles, cifar_profiles])
def test_result_profiles_default_to_pilot_and_cannot_be_smoke(family):
    path = family.CONFIGS_DIRECTORY / "deepneat_full.yaml"
    assert family.load_profile_settings(path).execution.mode == "pilot"
    with pytest.raises(ExecutionLockError, match="smoke requires"):
        family.load_profile_settings(path, execution=ExecutionOptions(mode="smoke"))
    smoke = family.CONFIGS_DIRECTORY / "deepneat_smoke.yaml"
    assert family.load_profile_settings(smoke).execution.mode == "smoke"
    with pytest.raises(ExecutionLockError, match="non-smoke"):
        family.load_profile_settings(smoke, execution=ExecutionOptions(mode="pilot"))


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(784, 10)

    def forward_pass(self, images):
        return self.linear(images.flatten(1))


def select(_):
    return protocol.SelectedCandidate(
        track_a_model=Model(),
        rebuild_model=Model,
        genome_kind="fixture",
        genome_payload={},
        selection_fitness=0.5,
        evaluation_records=(),
        number_of_generations=1,
        search_seconds=0.1,
        parameter_count=7850,
    )


@pytest.mark.parametrize("mode", ["smoke", "pilot", "full"])
def test_completed_resume_does_not_repeat_any_stage(mode, tmp_path, monkeypatch):
    execution = ExecutionOptions(
        mode=mode, protocol_lock_path=tmp_path / "lock.yaml" if mode == "full" else None
    )
    current = replace(settings(), execution=execution)
    arguments = dict(
        method_name="fixture",
        load_dataset=lambda _: tensors(),
        environment=lambda _: {"code": 1},
        artifacts_directory=tmp_path,
        lock_sha256="verified-lock",
    )
    first = protocol.run_multiclass_protocol(current, select, **arguments)
    report_bytes = (tmp_path / "run_report.json").read_bytes()

    def unexpected(*args, **kwargs):
        pytest.fail("completed resume must not search, retrain, freeze or test")

    for name in (
        "run_search_stage",
        "retrain_topology_for_track_b",
        "freeze_model",
        "evaluate_on_official_test",
    ):
        monkeypatch.setattr(protocol, name, unexpected)
    resumed = replace(current, execution=replace(execution, resume=True))
    assert protocol.run_multiclass_protocol(resumed, unexpected, **arguments) == first
    assert (tmp_path / "run_report.json").read_bytes() == report_bytes
    assert not (tmp_path / "run_report.partial").exists()
    changed = list(tensors())
    changed[0][0, 0] += 1
    with pytest.raises(ExecutionLockError, match="completed run differs"):
        protocol.run_multiclass_protocol(
            resumed, unexpected, **{**arguments, "load_dataset": lambda _: tuple(changed)}
        )
    with pytest.raises(ExecutionLockError, match="completed run differs"):
        protocol.run_multiclass_protocol(
            resumed, unexpected, **{**arguments, "lock_sha256": "another-lock"}
        )


def test_freeze_new_release_validates_against_actual_data(tmp_path):
    current = settings()
    identity = protocol.prepare_data(
        current, load_dataset=lambda _: tensors()
    ).dataset_identity_sha256
    environment = execution_environment(torch.device("cpu"))
    profiles_dir = tmp_path / "profiles"
    profiles_dir.mkdir()
    profile = current.profile_payload
    profile["protocol"]["maximum_training_samples"] = current.maximum_training_samples
    profile["protocol"]["maximum_test_samples"] = current.maximum_test_samples
    (profiles_dir / "deepneat_full.yaml").write_text(yaml.safe_dump(profile), encoding="utf-8")
    report = {
        "mode": "pilot",
        "official_test": {},
        "method": "deepneat",
        "environment": environment,
        "dataset_identity_sha256": identity,
        "summary": {"metric_values": {"validation_accuracy": 0.5}},
        "effective_configuration": {"profile_payload": profile, "search_seed": 7},
    }
    report_path = tmp_path / "pilot.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    lock = assemble_series_lock(
        profiles_directory=profiles_dir,
        output_directory=tmp_path / "series",
        methods=["deepneat"],
        required_methods=["deepneat"],
        seeds=[101, 102, 103],
        protocol_id="verified-series",
        dataset_release="verified/new-release",
        dataset_license="verified-license",
        pilot_reports=[report_path],
        environment=environment,
        metric_key="validation_accuracy",
    )
    frozen = mnist_profiles.load_profile_settings(lock.parent / "profiles/deepneat.yaml")
    actual = protocol.prepare_data(frozen, load_dataset=lambda _: tensors())
    assert validate_execution_lock(
        lock,
        method="deepneat",
        profile=frozen.profile_payload,
        search_seed=101,
        device=torch.device("cpu"),
        dataset_identity_sha256=actual.dataset_identity_sha256,
    )
