"""Executable policy: pilot isolation, audit gates and an exact series lock."""

import copy
import json
import shutil
from dataclasses import replace

import pytest
import torch
import yaml

from examples.pediatric_pneumonia import _protocol
from examples.pediatric_pneumonia._dataset_audit import audit_pediatric_pneumonia_dataset
from examples.pediatric_pneumonia._dataset_manifest import build_manifest_from_audit
from examples.pediatric_pneumonia._execution import ExecutionOptions, validate_execution_lock
from examples.pediatric_pneumonia._profiles import CONFIGS_DIRECTORY, load_profile_settings
from examples.pediatric_pneumonia._protocol_lock import ProtocolLockError
from examples.pediatric_pneumonia._synthetic_archive import write_synthetic_archive
from examples.pediatric_pneumonia.freeze import DEFAULT_METHODS, freeze_series
from polyneat.nn.fixed_convolutional_network import (
    FixedConvolutionalNetwork,
    FixedConvolutionalNetworkConfig,
)


def test_pilot_never_loads_threshold_or_test(tmp_path, monkeypatch):
    archive = write_synthetic_archive(tmp_path / "data", random_seed=741)
    settings = load_profile_settings(
        CONFIGS_DIRECTORY / "fixed_cnn_smoke.yaml",
        data_directory=archive,
    )
    settings = replace(settings, execution=ExecutionOptions(mode="pilot"))
    observed = []
    original_load = _protocol.load_split_tensors

    def guarded_load(*args, **kwargs):
        observed.append(kwargs["split_name"])
        assert kwargs["split_name"] in {"train", "search_validation"}
        return original_load(*args, **kwargs)

    monkeypatch.setattr(_protocol, "load_split_tensors", guarded_load)
    config = FixedConvolutionalNetworkConfig(convolution_channels=(2,))

    def search(context):
        return _protocol.SelectedCandidate(
            track_a_model=FixedConvolutionalNetwork(config),
            rebuild_model=lambda: pytest.fail("pilot must not retrain"),
            genome_kind="fixture",
            genome_payload={},
            selection_fitness=0.5,
            evaluation_records=(),
            number_of_generations=1,
            search_seconds=0.1,
            parameter_count=10,
        )

    report = _protocol.run_pneumonia_protocol(
        settings,
        search,
        method_name="fixture",
        artifacts_directory=tmp_path / "pilot",
    )
    assert observed == ["train", "search_validation"]
    assert not any("test" in key for key in report.metric_values)
    assert json.loads((tmp_path / "pilot/run_report.json").read_text())["official_test"] == {}


def test_blocking_audit_stops_before_search(tmp_path):
    archive = write_synthetic_archive(tmp_path / "data", random_seed=742)
    source = next(archive.glob("**/train/NORMAL/*.jpeg"))
    target = next(archive.glob("**/test/NORMAL/*.jpeg"))
    shutil.copyfile(source, target)
    settings = load_profile_settings(
        CONFIGS_DIRECTORY / "fixed_cnn_smoke.yaml", data_directory=archive
    )
    settings = replace(settings, execution=ExecutionOptions(mode="pilot"))
    with pytest.raises(ProtocolLockError, match="audit blocks"):
        _protocol.run_pneumonia_protocol(
            settings,
            lambda context: pytest.fail("must not train"),
            method_name="fixed_cnn",
        )


def test_full_requires_lock():
    with pytest.raises(ProtocolLockError, match="requires --protocol-lock"):
        ExecutionOptions(mode="full")


@pytest.mark.parametrize("method", ["deepneat", "exact", "random_search"])
def test_real_training_resume_restores_winning_weights(tmp_path, monkeypatch, method):
    from examples.pediatric_pneumonia._methods import (
        make_deepneat_search,
        make_exact_search,
        make_random_search,
    )
    from polyneat.configs.deepneat.deepneat_config import DeepNEATConfig
    from polyneat.configs.exact.exact_config import EXACTConfig
    from polyneat.evaluators.binary_auroc_evaluator import (
        PretrainedBinaryAurocEvaluator,
        TrainedBinaryAurocEvaluator,
    )
    from polyneat.training.model_checkpoint import ModelCheckpoint
    from polyneat.training.training_recipe import TrainingRecipe

    archive = write_synthetic_archive(tmp_path / "data", random_seed=99)
    recipe = TrainingRecipe(number_of_epochs=1, batch_size=32, learning_rate=0.01)
    settings = _protocol.BenchmarkSettings(
        data_directory=archive, protocol_id="resume-test", dataset_release="fixture",
        dataset_license="fixture", image_side=8, split_seed=11, search_seed=101,
        retraining_seeds=(), bootstrap_seed=13, candidate_recipe=recipe,
        track_b_recipe=recipe, uses_augmentation=True,
        execution=ExecutionOptions(mode="pilot"),
    )
    deep_config = DeepNEATConfig(
        population_size=2, number_of_input_nodes=1, number_of_output_nodes=2,
        number_of_classes=2, input_image_channels=1, input_image_height=8, input_image_width=8,
        available_filter_counts=(2,), available_dense_unit_counts=(4,),
    )
    if method == "exact":
        search = make_exact_search(EXACTConfig(
            population_size=2, number_of_input_nodes=1, number_of_output_nodes=2,
            input_image_height=8, input_image_width=8, training_batch_size=32,
            number_of_training_epochs_per_genome=1, use_simplex_hyperparameter_optimization=False,
        ), number_of_generations=2, image_side=8)
        evaluator_class = PretrainedBinaryAurocEvaluator
    elif method == "deepneat":
        search = make_deepneat_search(deep_config, number_of_generations=2)
        evaluator_class = TrainedBinaryAurocEvaluator
    else:
        search = make_random_search(deep_config, number_of_candidates=3,
                                    minimum_structural_mutations=1, maximum_structural_mutations=2)
        evaluator_class = TrainedBinaryAurocEvaluator
    continuous = _protocol.run_pneumonia_protocol(
        settings, search, method_name=method, artifacts_directory=tmp_path / "continuous",
    )
    original = evaluator_class._evaluate_one

    def interrupt_second(self, phenotype, evaluation_id):
        if len(self.evaluation_records) == 1:
            raise KeyboardInterrupt
        return original(self, phenotype, evaluation_id)

    with monkeypatch.context() as context:
        context.setattr(evaluator_class, "_evaluate_one", interrupt_second)
        with pytest.raises(KeyboardInterrupt):
            _protocol.run_pneumonia_protocol(
                settings, search, method_name=method, artifacts_directory=tmp_path / "resumed",
            )
    resumed = _protocol.run_pneumonia_protocol(
        replace(settings, execution=ExecutionOptions(mode="pilot", resume=True)), search,
        method_name=method, artifacts_directory=tmp_path / "resumed",
    )
    assert resumed.metric_values["search_validation_auroc"] == (
        continuous.metric_values["search_validation_auroc"]
    )
    expected, _ = ModelCheckpoint.read_file(tmp_path / "continuous/checkpoints/pilot_selected.pt")
    actual, _ = ModelCheckpoint.read_file(tmp_path / "resumed/checkpoints/pilot_selected.pt")
    assert expected.compute_sha256() == actual.compute_sha256()


@pytest.fixture
def frozen_series(tmp_path):
    from examples.pediatric_pneumonia._execution import execution_environment

    archive = write_synthetic_archive(tmp_path / "data", random_seed=743)
    profiles_directory = tmp_path / "configs"
    profiles_directory.mkdir()
    environment = execution_environment(torch.device("cpu"))
    reports = []
    audit = audit_pediatric_pneumonia_dataset(archive, compute_similarity_report=False)
    for method in DEFAULT_METHODS:
        profile = yaml.safe_load((CONFIGS_DIRECTORY / f"{method}_full.yaml").read_text())
        (profiles_directory / f"{method}_full.yaml").write_text(yaml.safe_dump(profile))
        provenance = profile["protocol"]
        manifest = build_manifest_from_audit(
            audit,
            protocol_id=provenance["protocol_id"],
            dataset_release=provenance["dataset_release"],
            dataset_license=provenance["dataset_license"],
            split_seed=provenance["split_seed"],
        )
        report_path = tmp_path / f"{method}_pilot.json"
        report_path.write_text(
            json.dumps(
                {
                    "method": method,
                    "official_test": {},
                    "blocking_audit_findings": [],
                    "environment": environment,
                    "manifest_sha256": manifest.compute_sha256(),
                    "summary": {"metric_values": {"search_validation_auroc": 0.7}},
                    "effective_configuration": {
                        "mode": "pilot",
                        "search_seed": 7,
                        "profile_payload": profile,
                    },
                }
            )
        )
        reports.append(report_path)
    path = freeze_series(
        profiles_directory=profiles_directory,
        data_directory=archive,
        output_directory=tmp_path / "series",
        methods=list(DEFAULT_METHODS),
        seeds=[101, 102, 103, 104, 105],
        protocol_id="fixture-v2",
        dataset_release="test-fixture-v1",
        dataset_license="test-fixture-license",
        pilot_reports=reports,
        environment=environment,
    )
    return path, archive


def test_frozen_profiles_load_and_reject_overrides(frozen_series):
    path, archive = frozen_series
    profile_path = path.parent / "profiles/deepneat.yaml"
    profile = yaml.safe_load(profile_path.read_text())
    settings = load_profile_settings(
        profile_path,
        data_directory=archive,
        execution=ExecutionOptions(mode="full", protocol_lock_path=path),
    )
    data = _protocol.prepare_data(settings)
    validate_execution_lock(
        path,
        method="deepneat",
        profile=profile,
        search_seed=101,
        device=torch.device("cpu"),
        manifest=data.manifest,
    )
    changed = copy.deepcopy(profile)
    changed["algorithm"]["population_size"] += 1
    with pytest.raises(ProtocolLockError, match="differs"):
        validate_execution_lock(
            path, method="deepneat", profile=changed, search_seed=101, device=torch.device("cpu")
        )
    with pytest.raises(ProtocolLockError, match="seed"):
        validate_execution_lock(
            path, method="deepneat", profile=profile, search_seed=999, device=torch.device("cpu")
        )


def test_changed_lock_and_archive_are_rejected(frozen_series):
    path, archive = frozen_series
    profile = yaml.safe_load((path.parent / "profiles/deepneat.yaml").read_text())
    settings = load_profile_settings(path.parent / "profiles/deepneat.yaml", data_directory=archive)
    data = _protocol.prepare_data(settings)
    changed_manifest = replace(data.manifest, split_seed=data.manifest.split_seed + 1)
    with pytest.raises(ProtocolLockError, match="archive/split"):
        validate_execution_lock(
            path,
            method="deepneat",
            profile=profile,
            search_seed=101,
            device=torch.device("cpu"),
            manifest=changed_manifest,
        )
    payload = yaml.safe_load(path.read_text())
    payload["search_seeds"].append(999)
    path.write_text(yaml.safe_dump(payload))
    with pytest.raises(ProtocolLockError, match="changed after freezing"):
        validate_execution_lock(
            path, method="deepneat", profile=profile, search_seed=101, device=torch.device("cpu")
        )
