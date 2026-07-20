from __future__ import annotations

import pytest

from polyneat.configs.configuration_errors import ConfigurationError
from polyneat.configs.lneat.lneat_config import LNEATConfig


def test_defaults_are_valid_and_inherit_neat_fields() -> None:
    config = LNEATConfig(number_of_input_nodes=4, number_of_class_labels=3)
    assert config.number_of_class_labels == 3
    assert config.learning_interval_generations == 5
    assert config.number_of_learning_samples == 10
    assert config.backpropagation_iterations_per_session == 10
    assert config.backpropagation_learning_rate == 0.1
    assert config.training_indicator == 0.2
    assert config.classification_threshold == 0.5
    # spot-check NEAT inheritance
    assert config.probability_of_add_node_mutation == 0.03
    assert config.compatibility_distance_threshold == 3.0
    assert config.initial_population_strategy == "fully_connected"


def test_rejects_fewer_than_two_class_labels() -> None:
    with pytest.raises(ConfigurationError):
        LNEATConfig(number_of_class_labels=1)


def test_rejects_multi_output_networks() -> None:
    # each subtask network is a single-output recognizer (paper, section IV.A)
    with pytest.raises(ConfigurationError):
        LNEATConfig(number_of_output_nodes=3)


def test_rejects_non_positive_learning_interval() -> None:
    with pytest.raises(ConfigurationError):
        LNEATConfig(learning_interval_generations=0)


def test_rejects_non_positive_learning_sample_count() -> None:
    with pytest.raises(ConfigurationError):
        LNEATConfig(number_of_learning_samples=0)


def test_rejects_non_positive_backpropagation_iterations() -> None:
    with pytest.raises(ConfigurationError):
        LNEATConfig(backpropagation_iterations_per_session=0)


def test_rejects_non_positive_learning_rate() -> None:
    with pytest.raises(ConfigurationError):
        LNEATConfig(backpropagation_learning_rate=0.0)


def test_rejects_training_indicator_outside_unit_interval() -> None:
    with pytest.raises(ConfigurationError):
        LNEATConfig(training_indicator=1.5)
    with pytest.raises(ConfigurationError):
        LNEATConfig(training_indicator=-0.1)


def test_rejects_classification_threshold_outside_open_unit_interval() -> None:
    with pytest.raises(ConfigurationError):
        LNEATConfig(classification_threshold=0.0)
    with pytest.raises(ConfigurationError):
        LNEATConfig(classification_threshold=1.0)


def test_rejects_unbounded_output_activation() -> None:
    # recognizer outputs are compared against binary targets and the
    # classification threshold, which assumes outputs in [0, 1]
    with pytest.raises(ConfigurationError):
        LNEATConfig(default_activation_function_for_output_nodes="tanh")

def test_yaml_round_trip(tmp_path) -> None:
    config = LNEATConfig(
        number_of_input_nodes=4,
        number_of_class_labels=3,
        learning_interval_generations=7,
    )
    yaml_path = tmp_path / "lneat.yaml"
    config.save_to_yaml_file(yaml_path)
    reloaded = LNEATConfig.load_from_yaml_file(yaml_path)
    assert reloaded == config
