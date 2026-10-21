from __future__ import annotations

from pathlib import Path

import torch

from examples._experiment import EXAMPLE_REGISTRY
from examples.xor import fdneat


def test_example_exposes_the_experiment_contract() -> None:
    assert isinstance(fdneat.CONFIG_FILE_PATH, Path)
    assert fdneat.CONFIG_FILE_PATH.exists()
    assert callable(fdneat.run_experiment)


def test_example_is_registered() -> None:
    assert EXAMPLE_REGISTRY["xor/fdneat"] == "examples.xor.fdneat"


def test_config_declares_the_fully_connected_start() -> None:
    import polyneat as pn

    config = pn.FDNEATConfig.load_from_yaml_file(fdneat.CONFIG_FILE_PATH)
    assert config.initial_population_strategy == "fully_connected"
    assert config.probability_of_deleting_input_connection > 0.0
    assert config.number_of_input_nodes == 8


def test_short_run_reports_both_metrics() -> None:
    report = fdneat.run_experiment(
        device=torch.device("cpu"),
        random_seed=0,
        artifacts_directory=None,
    )
    assert "best_fitness" in report.metric_values
    assert "number_of_connected_input_features" in report.metric_values
    assert 0.0 <= report.metric_values["number_of_connected_input_features"] <= 8.0
    assert report.number_of_generations > 0
    assert report.runtime_seconds > 0.0


def test_both_feature_selection_examples_report_the_same_metric() -> None:
    # The comparison FS-NEAT vs FD-NEAT is the point of this example, so the two
    # must expose an identically named metric computed by the same function.
    from examples.xor import fsneat

    fdneat_report = fdneat.run_experiment(device=torch.device("cpu"), random_seed=0)
    fsneat_report = fsneat.run_experiment(device=torch.device("cpu"), random_seed=0)
    assert (
        "number_of_connected_input_features" in fdneat_report.metric_values
        and "number_of_connected_input_features" in fsneat_report.metric_values
    )
