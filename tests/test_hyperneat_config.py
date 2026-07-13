from __future__ import annotations

import pytest

from polyneat.config.configuration_errors import ConfigurationError
from polyneat.config.hyperneat_config import HyperNEATConfig


def test_defaults_are_valid_and_describe_a_cppn():
    config = HyperNEATConfig()
    # CPPN has 4 coordinate inputs and 1 weight output
    assert config.number_of_input_nodes == 4
    assert config.number_of_output_nodes == 1
    # substrate description is independent of the CPPN's own I/O
    assert config.substrate_input_layer_size == 2
    assert config.substrate_output_layer_size == 1
    assert config.weight_expression_threshold == 0.2
    assert config.max_substrate_connection_weight_magnitude == 3.0
    # inherits NEAT genetics unchanged
    assert config.probability_of_add_node_mutation == 0.03


def test_rejects_cppn_input_count_other_than_four():
    with pytest.raises(ConfigurationError):
        HyperNEATConfig(number_of_input_nodes=2)


def test_rejects_cppn_output_count_other_than_one():
    with pytest.raises(ConfigurationError):
        HyperNEATConfig(number_of_output_nodes=2)


def test_rejects_threshold_outside_unit_interval():
    with pytest.raises(ConfigurationError):
        HyperNEATConfig(weight_expression_threshold=1.0)


def test_rejects_nonpositive_max_magnitude():
    with pytest.raises(ConfigurationError):
        HyperNEATConfig(max_substrate_connection_weight_magnitude=0.0)


def test_rejects_substrate_activation_not_in_registry():
    with pytest.raises(ConfigurationError):
        HyperNEATConfig(substrate_node_activation_function="not_a_function")


def test_yaml_round_trip(tmp_path):
    config = HyperNEATConfig(substrate_hidden_layer_sizes=(4, 4))
    yaml_path = tmp_path / "hyperneat.yaml"
    config.save_to_yaml_file(yaml_path)
    loaded = HyperNEATConfig.load_from_yaml_file(yaml_path)
    assert loaded == config
