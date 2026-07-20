from __future__ import annotations

import pytest

from polyneat.configs.algorithm_config import AlgorithmConfig
from polyneat.configs.hyperneat.hyperneat_config import HyperNEATConfig
from polyneat.configs.neat.neat_config import NEATConfig


@pytest.mark.parametrize(
    "config",
    [
        AlgorithmConfig(),
        NEATConfig(),
        HyperNEATConfig(),
        NEATConfig(
            available_activation_functions=("tanh", "relu"),
            default_activation_function_for_hidden_nodes="tanh",
            default_activation_function_for_output_nodes="relu",
        ),
        HyperNEATConfig(substrate_hidden_layer_sizes=(4, 4)),
    ],
)
def test_yaml_save_load_round_trips_exactly(tmp_path, config):
    """Every config must survive save -> load unchanged.

    Regression guard for a base-class bug: ``yaml.dump`` emitted
    ``!!python/tuple`` for tuple-valued fields, which ``yaml.safe_load`` then
    refused to parse - so any config with a tuple field (e.g. NEAT's
    ``available_activation_functions``) could not be reloaded at all.
    """
    yaml_path = tmp_path / "config.yaml"
    config.save_to_yaml_file(yaml_path)
    loaded_config = type(config).load_from_yaml_file(yaml_path)
    assert loaded_config == config


def test_saved_yaml_has_no_python_specific_tags():
    """Saved YAML must be plain (safe_load-able), not Python-tagged."""
    config = NEATConfig()
    dumped_yaml = "\n".join(
        f"{key}: {value}" for key, value in config.to_dict().items()
    )
    # to_dict itself must not surface tuples (which yaml.dump tags as
    # !!python/tuple); tuple fields are listed instead.
    assert all(
        not isinstance(value, tuple) for value in config.to_dict().values()
    )
    assert "python/tuple" not in dumped_yaml


def test_tuple_fields_are_restored_as_tuples_after_load(tmp_path):
    config = NEATConfig()
    yaml_path = tmp_path / "config.yaml"
    config.save_to_yaml_file(yaml_path)
    loaded_config = NEATConfig.load_from_yaml_file(yaml_path)
    assert isinstance(loaded_config.available_activation_functions, tuple)
    assert (
        loaded_config.available_activation_functions
        == config.available_activation_functions
    )
