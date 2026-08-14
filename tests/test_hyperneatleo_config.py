from __future__ import annotations

import pytest

from polyneat.configs.configuration_errors import ConfigurationError
from polyneat.configs.hyperneat.hyperneat_config import HyperNEATConfig
from polyneat.configs.hyperneatleo.hyperneatleo_config import (
    LEO_CPPN_INPUT_NODE_COUNT,
    LEO_CPPN_OUTPUT_NODE_COUNT,
    HyperNEATLEOConfig,
)

_RETINA_LAYERS = (
    (-1.00, -0.85, -0.70, -0.55, 0.55, 0.70, 0.85, 1.00),
    (-1.00, -0.85, -0.70, -0.55, 0.55, 0.70, 0.85, 1.00),
    (-0.775, 0.775),
)


def test_defaults_describe_a_six_input_two_output_cppn() -> None:
    config = HyperNEATLEOConfig()
    assert config.number_of_input_nodes == LEO_CPPN_INPUT_NODE_COUNT == 6
    assert config.number_of_output_nodes == LEO_CPPN_OUTPUT_NODE_COUNT == 2
    config.validate()


def test_leo_config_is_a_hyperneat_config() -> None:
    assert issubclass(HyperNEATLEOConfig, HyperNEATConfig)


def test_four_input_cppn_is_rejected() -> None:
    with pytest.raises(ConfigurationError) as raised:
        HyperNEATLEOConfig(number_of_input_nodes=4)
    assert "number_of_input_nodes" in str(raised.value)


def test_single_output_cppn_is_rejected() -> None:
    with pytest.raises(ConfigurationError) as raised:
        HyperNEATLEOConfig(number_of_output_nodes=1)
    assert "number_of_output_nodes" in str(raised.value)


def test_substrate_validation_is_inherited() -> None:
    with pytest.raises(ConfigurationError) as raised:
        HyperNEATLEOConfig(substrate_input_layer_size=0)
    assert "substrate_input_layer_size" in str(raised.value)


def test_explicitly_set_weight_expression_threshold_is_rejected() -> None:
    # LEO replaces the magnitude threshold entirely; silently ignoring the field
    # would violate the library's strict-config rule.
    with pytest.raises(ConfigurationError) as raised:
        HyperNEATLEOConfig(weight_expression_threshold=0.4)
    assert "link_expression_threshold" in str(raised.value)


def test_inherited_default_weight_expression_threshold_is_accepted() -> None:
    config = HyperNEATLEOConfig()
    assert config.weight_expression_threshold == HyperNEATConfig.weight_expression_threshold
    config.validate()


@pytest.mark.parametrize("bad_axes", [("z",), ("x", "z"), ()])
def test_unknown_or_empty_seed_axes_are_rejected(bad_axes: tuple[str, ...]) -> None:
    with pytest.raises(ConfigurationError) as raised:
        HyperNEATLEOConfig(locality_seed_coordinate_axes=bad_axes)
    assert "locality_seed_coordinate_axes" in str(raised.value)


def test_both_axes_are_accepted() -> None:
    HyperNEATLEOConfig(locality_seed_coordinate_axes=("x", "y")).validate()


@pytest.mark.parametrize("bad_bias_weight", [0.0, 0.5, -1.0, -2.0])
def test_seed_bias_weight_outside_the_open_unit_interval_is_rejected(
    bad_bias_weight: float,
) -> None:
    # At b >= 0 exp(-x^2) + b is positive everywhere, so the seed expresses every
    # connection; at b <= -1 it is non-positive everywhere and expresses none.
    with pytest.raises(ConfigurationError) as raised:
        HyperNEATLEOConfig(locality_seed_bias_weight=bad_bias_weight)
    assert "locality_seed_bias_weight" in str(raised.value)


def test_non_positive_seed_delta_weight_is_rejected() -> None:
    with pytest.raises(ConfigurationError) as raised:
        HyperNEATLEOConfig(locality_seed_delta_weight=0.0)
    assert "locality_seed_delta_weight" in str(raised.value)


def test_explicit_layer_coordinates_replace_the_layer_size_checks() -> None:
    # With explicit coordinates the layer-size fields are unused, so their
    # values must stop being validated - otherwise a sensible explicit substrate
    # could be rejected over a field nobody reads.
    config = HyperNEATLEOConfig(
        substrate_layer_x_coordinates=_RETINA_LAYERS,
        substrate_input_layer_size=0,
    )
    config.validate()


def test_explicit_layer_coordinates_are_normalized_to_nested_tuples() -> None:
    # AlgorithmConfig.from_dict only coerces top-level list -> tuple, and only
    # for fields whose default is a tuple. This field defaults to None and is
    # nested, so YAML would otherwise leave lists of lists behind.
    config = HyperNEATLEOConfig(
        substrate_layer_x_coordinates=[[-1.0, 1.0], [-0.5, 0.5]]  # type: ignore[arg-type]
    )
    assert config.substrate_layer_x_coordinates == ((-1.0, 1.0), (-0.5, 0.5))
    assert isinstance(config.substrate_layer_x_coordinates, tuple)
    assert all(isinstance(layer, tuple) for layer in config.substrate_layer_x_coordinates)


def test_explicit_coordinates_with_a_single_layer_are_rejected() -> None:
    with pytest.raises(ConfigurationError) as raised:
        HyperNEATLEOConfig(substrate_layer_x_coordinates=((-1.0, 1.0),))
    assert "substrate_layer_x_coordinates" in str(raised.value)


def test_explicit_coordinates_with_an_empty_layer_are_rejected() -> None:
    with pytest.raises(ConfigurationError) as raised:
        HyperNEATLEOConfig(substrate_layer_x_coordinates=((-1.0, 1.0), ()))
    assert "substrate_layer_x_coordinates" in str(raised.value)


def test_unknown_yaml_key_is_rejected() -> None:
    with pytest.raises(ConfigurationError):
        HyperNEATLEOConfig.from_dict({"not_a_real_field": 1})


def test_yaml_round_trip_keeps_nested_coordinates_as_tuples() -> None:
    config = HyperNEATLEOConfig.from_dict(
        {
            "substrate_layer_x_coordinates": [list(layer) for layer in _RETINA_LAYERS],
            "locality_seed_coordinate_axes": ["x"],
        }
    )
    assert config.substrate_layer_x_coordinates == _RETINA_LAYERS
    assert config.locality_seed_coordinate_axes == ("x",)


def test_subpackage_reexports_the_config_class() -> None:
    from polyneat.configs.hyperneatleo import HyperNEATLEOConfig as ReexportedConfig

    assert ReexportedConfig is HyperNEATLEOConfig
