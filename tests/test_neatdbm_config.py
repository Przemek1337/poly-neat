from __future__ import annotations

import pytest

from polyneat.config.configuration_errors import ConfigurationError
from polyneat.config.neatdbm_config import NEATDBMConfig


def test_defaults_are_valid_and_inherit_neat_fields() -> None:
    config = NEATDBMConfig()
    assert config.probability_of_difference_based_mutation == 0.1
    assert config.difference_mutation_scaling_factor == 0.5
    assert config.probability_of_add_node_mutation == 0.03
    assert config.compatibility_distance_threshold == 3.0
    assert config.initial_population_strategy == "fully_connected"


def test_rejects_difference_mutation_probability_above_one() -> None:
    with pytest.raises(ConfigurationError):
        NEATDBMConfig(probability_of_difference_based_mutation=1.5)


def test_rejects_negative_difference_mutation_probability() -> None:
    with pytest.raises(ConfigurationError):
        NEATDBMConfig(probability_of_difference_based_mutation=-0.1)


def test_rejects_non_positive_scaling_factor() -> None:
    with pytest.raises(ConfigurationError):
        NEATDBMConfig(difference_mutation_scaling_factor=0.0)


def test_yaml_round_trip_preserves_dbm_fields(tmp_path) -> None:
    original_config = NEATDBMConfig(
        probability_of_difference_based_mutation=0.2,
        difference_mutation_scaling_factor=0.7,
    )
    yaml_file_path = tmp_path / "neatdbm.yaml"
    original_config.save_to_yaml_file(yaml_file_path)
    reloaded_config = NEATDBMConfig.load_from_yaml_file(yaml_file_path)
    assert reloaded_config == original_config
