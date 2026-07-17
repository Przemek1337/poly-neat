from __future__ import annotations

import pytest

from polyneat.config.cneat_config import CNEATConfig
from polyneat.config.configuration_errors import ConfigurationError


def test_defaults_are_valid_and_inherit_neat_fields() -> None:
    config = CNEATConfig(number_of_input_nodes=4, number_of_class_labels=3)
    assert config.number_of_class_labels == 3
    # spot-check NEAT inheritance
    assert config.probability_of_add_node_mutation == 0.03
    assert config.compatibility_distance_threshold == 3.0
    assert config.initial_population_strategy == "fully_connected"


def test_rejects_fewer_than_two_class_labels() -> None:
    with pytest.raises(ConfigurationError):
        CNEATConfig(number_of_class_labels=1)


def test_rejects_multi_output_networks() -> None:
    # each container cell is a single-output recognizer network
    with pytest.raises(ConfigurationError):
        CNEATConfig(number_of_output_nodes=3)


def test_rejects_population_smaller_than_class_count() -> None:
    with pytest.raises(ConfigurationError):
        CNEATConfig(population_size=2, number_of_class_labels=3)


def test_rejects_output_activation_not_bounded_in_unit_interval() -> None:
    # tanh outputs reach [-1, 1], so 1 - MSE could go negative and break
    # NEAT's clamped offspring allocation
    with pytest.raises(ConfigurationError):
        CNEATConfig(default_activation_function_for_output_nodes="tanh")


def test_yaml_round_trip(tmp_path) -> None:
    config = CNEATConfig(number_of_input_nodes=8, number_of_class_labels=10)
    yaml_path = tmp_path / "cneat.yaml"
    config.save_to_yaml_file(yaml_path)
    loaded = CNEATConfig.load_from_yaml_file(yaml_path)
    assert loaded == config
