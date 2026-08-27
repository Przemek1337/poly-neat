from __future__ import annotations

import pytest

from polyneat.configs.configuration_errors import ConfigurationError
from polyneat.configs.fdneat.fdneat_config import FDNEATConfig


def test_default_probability_is_in_valid_range() -> None:
    config = FDNEATConfig(number_of_input_nodes=2, number_of_output_nodes=1)
    config.validate()
    assert 0.0 <= config.probability_of_deleting_input_connection <= 1.0


def test_fdneat_config_is_a_neat_config() -> None:
    from polyneat.configs.neat.neat_config import NEATConfig

    assert issubclass(FDNEATConfig, NEATConfig)


def test_default_start_is_the_vanilla_neat_fully_connected_one() -> None:
    # The fully connected start *is* FD-NEAT's start: evolution deselects from
    # it. Anything else would make the algorithm something other than FD-NEAT.
    assert FDNEATConfig().initial_population_strategy == "fully_connected"


@pytest.mark.parametrize("invalid_probability", [-0.01, 1.01, 2.0])
def test_probability_outside_unit_interval_is_rejected(invalid_probability: float) -> None:
    # AlgorithmConfig.__post_init__ calls validate(), so an invalid config
    # cannot even be constructed - the error surfaces here, not on a later call.
    with pytest.raises(ConfigurationError) as raised:
        FDNEATConfig(
            number_of_input_nodes=2,
            number_of_output_nodes=1,
            probability_of_deleting_input_connection=invalid_probability,
        )
    assert "probability_of_deleting_input_connection" in str(raised.value)


def test_boundary_probabilities_are_accepted() -> None:
    for boundary_value in (0.0, 1.0):
        config = FDNEATConfig(probability_of_deleting_input_connection=boundary_value)
        assert config.probability_of_deleting_input_connection == boundary_value


def test_inherited_neat_validation_still_runs() -> None:
    with pytest.raises(ConfigurationError) as raised:
        FDNEATConfig(
            number_of_input_nodes=2,
            number_of_output_nodes=1,
            probability_of_add_node_mutation=5.0,
        )
    assert "probability_of_add_node_mutation" in str(raised.value)


def test_unknown_yaml_key_is_rejected() -> None:
    with pytest.raises(ConfigurationError):
        FDNEATConfig.from_dict({"number_of_input_nodes": 2, "not_a_real_field": 1})


def test_known_key_is_accepted_by_from_dict() -> None:
    config = FDNEATConfig.from_dict(
        {
            "number_of_input_nodes": 8,
            "number_of_output_nodes": 1,
            "probability_of_deleting_input_connection": 0.2,
        }
    )
    assert config.probability_of_deleting_input_connection == 0.2
    assert config.number_of_input_nodes == 8


def test_subpackage_reexports_the_config_class() -> None:
    from polyneat.configs.fdneat import FDNEATConfig as ReexportedConfig

    assert ReexportedConfig is FDNEATConfig
