from __future__ import annotations

import math

import pytest

from polyneat.configs.configuration_errors import ConfigurationError
from polyneat.configs.hyperneat.hyperneat_config import (
    CPPN_INPUT_NODE_COUNT,
    HyperNEATConfig,
)
from polyneat.configs.hyperneatleo.hyperneatleo_config import (
    LEO_CPPN_INPUT_NODE_COUNT,
    LEO_CPPN_OUTPUT_NODE_COUNT,
    HyperNEATLEOConfig,
)

# Two clusters of four, tight enough that the seed cutoff falls in the gap.
_RETINA_LAYERS = (
    (-1.0, -0.9167, -0.8333, -0.75, 0.75, 0.8333, 0.9167, 1.0),
    (-1.0, -0.9167, -0.8333, -0.75, 0.75, 0.8333, 0.9167, 1.0),
    (-0.875, 0.875),
)


def test_cppn_keeps_hyperneats_four_coordinate_inputs() -> None:
    # The locality seed forms coordinate differences from x1 and x2 through
    # opposite-signed weights, so no separate delta input is needed.
    assert LEO_CPPN_INPUT_NODE_COUNT == CPPN_INPUT_NODE_COUNT == 4


def test_defaults_describe_a_four_input_two_output_cppn() -> None:
    config = HyperNEATLEOConfig()
    assert config.number_of_input_nodes == 4
    assert config.number_of_output_nodes == LEO_CPPN_OUTPUT_NODE_COUNT == 2
    config.validate()


def test_leo_config_is_a_hyperneat_config() -> None:
    assert issubclass(HyperNEATLEOConfig, HyperNEATConfig)


def test_six_input_cppn_is_rejected() -> None:
    with pytest.raises(ConfigurationError) as raised:
        HyperNEATLEOConfig(number_of_input_nodes=6)
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


def test_seed_defaults_follow_the_published_constants() -> None:
    config = HyperNEATLEOConfig()
    assert config.locality_seed_delta_weight == 0.6
    assert config.locality_seed_bias_weight == -1.0
    assert config.locality_seed_coordinate_axes == ("x",)


@pytest.mark.parametrize("bad_bias_weight", [0.0, 0.5])
def test_non_negative_seed_bias_weight_is_rejected(bad_bias_weight: float) -> None:
    with pytest.raises(ConfigurationError) as raised:
        HyperNEATLEOConfig(locality_seed_bias_weight=bad_bias_weight)
    assert "locality_seed_bias_weight" in str(raised.value)


def test_gaussian_weight_not_exceeding_the_bias_is_rejected() -> None:
    # The Gaussian peaks at 1.0, so g <= -b means the seed expresses nothing.
    with pytest.raises(ConfigurationError) as raised:
        HyperNEATLEOConfig(
            locality_seed_gaussian_to_leo_weight=1.0, locality_seed_bias_weight=-1.0
        )
    assert "locality_seed_gaussian_to_leo_weight" in str(raised.value)


def test_non_positive_seed_delta_weight_is_rejected() -> None:
    with pytest.raises(ConfigurationError) as raised:
        HyperNEATLEOConfig(locality_seed_delta_weight=0.0)
    assert "locality_seed_delta_weight" in str(raised.value)


def test_expression_cutoff_matches_the_closed_form() -> None:
    config = HyperNEATLEOConfig()
    expected = math.sqrt(math.log(2.0)) / 0.6
    assert abs(config.locality_seed_expression_cutoff - expected) < 1e-12
    assert abs(config.locality_seed_expression_cutoff - 1.3877) < 1e-3


def test_expression_cutoff_scales_with_the_delta_weight() -> None:
    narrow = HyperNEATLEOConfig(locality_seed_delta_weight=1.2)
    wide = HyperNEATLEOConfig(locality_seed_delta_weight=0.3)
    assert narrow.locality_seed_expression_cutoff < wide.locality_seed_expression_cutoff


def test_retina_geometry_puts_the_cutoff_inside_the_hemisphere_gap() -> None:
    # The property the whole experiment depends on.
    config = HyperNEATLEOConfig(substrate_layer_x_coordinates=_RETINA_LAYERS)
    xs = config.substrate_layer_x_coordinates[0]
    within = [abs(a - b) for a in xs for b in xs if a * b > 0]
    across = [abs(a - b) for a in xs for b in xs if a * b < 0]
    assert max(within) < config.locality_seed_expression_cutoff < min(across)


def test_explicit_layer_coordinates_replace_the_layer_size_checks() -> None:
    HyperNEATLEOConfig(
        substrate_layer_x_coordinates=_RETINA_LAYERS, substrate_input_layer_size=0
    ).validate()


def test_explicit_layer_coordinates_are_normalized_to_nested_tuples() -> None:
    config = HyperNEATLEOConfig(
        substrate_layer_x_coordinates=[[-1.0, 1.0], [-0.5, 0.5]]  # type: ignore[arg-type]
    )
    assert config.substrate_layer_x_coordinates == ((-1.0, 1.0), (-0.5, 0.5))
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
