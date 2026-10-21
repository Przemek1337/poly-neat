from __future__ import annotations

import hashlib
import importlib
import json
import statistics
from pathlib import Path

import pytest
import torch

from benchmarks import run_benchmark
from examples._experiment import EXAMPLE_REGISTRY, ExperimentReport


def test_every_registry_entry_satisfies_the_example_contract() -> None:
    """The runtime half of the ExampleModule protocol; mypy checks signatures."""
    for module_path in EXAMPLE_REGISTRY.values():
        example_module = importlib.import_module(module_path)
        assert callable(example_module.run_experiment), module_path
        assert isinstance(example_module.CONFIG_FILE_PATH, Path), module_path
        assert example_module.CONFIG_FILE_PATH.is_file(), module_path


class _StubExampleModule:
    """Records run_experiment calls and returns seed-dependent reports."""

    def __init__(self, config_file_path: Path) -> None:
        self.CONFIG_FILE_PATH = config_file_path
        self.recorded_calls: list[dict] = []

    def run_experiment(
        self,
        device: torch.device | None = None,
        random_seed: int | None = None,
        artifacts_directory: Path | None = None,
    ) -> ExperimentReport:
        self.recorded_calls.append(
            {
                "device": device,
                "random_seed": random_seed,
                "artifacts_directory": artifacts_directory,
            }
        )
        return ExperimentReport(
            metric_values={"test_accuracy": 0.9 + 0.01 * random_seed},
            number_of_generations=10 + random_seed,
            runtime_seconds=1.5,
        )


@pytest.fixture
def stubbed_benchmark(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> _StubExampleModule:
    """Stub the example module import and redirect results into tmp_path."""
    config_file_path = tmp_path / "stub.yaml"
    config_file_path.write_text("population_size: 8\n", encoding="utf-8")
    stub_module = _StubExampleModule(config_file_path)
    monkeypatch.setattr(
        run_benchmark, "_import_example_module", lambda module_path: stub_module
    )
    monkeypatch.setattr(run_benchmark, "_RESULTS_DIRECTORY", tmp_path / "results")
    return stub_module


def test_repeats_run_with_consecutive_seeds_and_no_artifacts(
    stubbed_benchmark: _StubExampleModule,
) -> None:
    run_benchmark.main(["iris/cneat", "--repeats", "3", "--base-seed", "7", "--cpu"])

    assert [call["random_seed"] for call in stubbed_benchmark.recorded_calls] == [7, 8, 9]
    assert all(
        call["artifacts_directory"] is None for call in stubbed_benchmark.recorded_calls
    )
    assert all(
        call["device"] == torch.device("cpu") for call in stubbed_benchmark.recorded_calls
    )


def test_result_file_matches_the_schema_and_summary_is_correct(
    stubbed_benchmark: _StubExampleModule,
) -> None:
    run_benchmark.main(["iris/cneat", "--repeats", "2", "--cpu"])

    result_files = list((stubbed_benchmark.CONFIG_FILE_PATH.parent / "results").glob("*.json"))
    assert len(result_files) == 1
    assert result_files[0].name.startswith("iris_cneat_")
    result_document = json.loads(result_files[0].read_text(encoding="utf-8"))

    config_text = stubbed_benchmark.CONFIG_FILE_PATH.read_text(encoding="utf-8")
    assert result_document["example"] == "iris/cneat"
    assert result_document["config_text"] == config_text
    assert result_document["config_sha256"] == hashlib.sha256(
        config_text.encode("utf-8")
    ).hexdigest()
    assert result_document["device"] == "cpu"
    assert result_document["repeats"] == 2
    assert result_document["base_seed"] == 0

    assert [run["seed"] for run in result_document["runs"]] == [0, 1]
    accuracies = [run["metric_values"]["test_accuracy"] for run in result_document["runs"]]
    assert result_document["summary"]["test_accuracy"]["mean"] == pytest.approx(
        statistics.fmean(accuracies)
    )
    assert result_document["summary"]["test_accuracy"]["std"] == pytest.approx(
        statistics.pstdev(accuracies)
    )
    assert set(result_document["summary"]) == {
        "test_accuracy",
        "number_of_generations",
        "runtime_seconds",
    }


def test_unknown_example_id_exits_nonzero_and_lists_valid_ids(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as raised:
        run_benchmark.main(["no/such_example", "--cpu"])
    assert raised.value.code == 1
    assert "iris/cneat" in capsys.readouterr().err
