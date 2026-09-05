from __future__ import annotations

import pytest

from polyneat.configs.configuration_errors import ConfigurationError
from polyneat.configs.deepneat.deepneat_config import DeepNEATConfig


def test_defaults_validate() -> None:
    config = DeepNEATConfig()
    config.validate()
    assert config.compatibility_distance_coefficient_weight_difference_c3 == 0.0
    assert config.maximum_total_parameter_count is None


def test_generic_search_space_defaults_are_valid() -> None:
    config = DeepNEATConfig()
    assert config.available_filter_counts == (16, 32, 64, 128)
    assert config.available_kernel_sizes == (1, 3, 5)
    assert config.available_dense_unit_counts == (64, 128, 256)


@pytest.mark.parametrize(
    "field_name",
    [
        "probability_of_add_layer_node_mutation",
        "probability_of_add_tensor_edge_mutation",
        "probability_of_toggle_tensor_edge_mutation",
        "probability_of_layer_hyperparameter_mutation",
        "probability_of_global_hyperparameter_mutation",
    ],
)
def test_probabilities_outside_unit_interval_are_rejected(field_name: str) -> None:
    with pytest.raises(ConfigurationError) as raised:
        DeepNEATConfig(**{field_name: 1.5})
    assert field_name in str(raised.value)


def test_empty_search_space_is_rejected() -> None:
    with pytest.raises(ConfigurationError) as raised:
        DeepNEATConfig(available_filter_counts=())
    assert "available_filter_counts" in str(raised.value)


def test_even_kernel_size_is_rejected() -> None:
    # Even kernels have no symmetric "same" padding.
    with pytest.raises(ConfigurationError) as raised:
        DeepNEATConfig(available_kernel_sizes=(2,))
    assert "available_kernel_sizes" in str(raised.value)


def test_dropout_range_must_be_ordered_and_within_unit_interval() -> None:
    with pytest.raises(ConfigurationError):
        DeepNEATConfig(dropout_rate_min=0.6, dropout_rate_max=0.5)
    with pytest.raises(ConfigurationError):
        DeepNEATConfig(dropout_rate_max=1.0)


def test_non_positive_parameter_budget_is_rejected() -> None:
    with pytest.raises(ConfigurationError) as raised:
        DeepNEATConfig(maximum_total_parameter_count=0)
    assert "maximum_total_parameter_count" in str(raised.value)


def test_parameter_budget_can_be_disabled_for_source_only_runs() -> None:
    assert DeepNEATConfig(maximum_total_parameter_count=None).maximum_total_parameter_count is None


def test_input_geometry_must_be_positive() -> None:
    with pytest.raises(ConfigurationError):
        DeepNEATConfig(input_image_height=0)
    with pytest.raises(ConfigurationError):
        DeepNEATConfig(input_image_channels=0)


def test_training_block_is_validated() -> None:
    with pytest.raises(ConfigurationError):
        DeepNEATConfig(training_epochs_per_evaluation=0)
    with pytest.raises(ConfigurationError):
        DeepNEATConfig(training_learning_rate=0.0)
    with pytest.raises(ConfigurationError):
        DeepNEATConfig(training_batch_size=0)


@pytest.mark.parametrize(
    "keyword_arguments",
    [
        {"number_of_generations": 0},
        {"final_training_epochs": 0},
        {"maximum_training_samples": 1},
        {"maximum_test_samples": 1},
        {"validation_fraction": 0.0},
        {"validation_fraction": 1.0},
    ],
)
def test_example_protocol_fields_are_validated(
    keyword_arguments: dict[str, int | float],
) -> None:
    with pytest.raises(ConfigurationError):
        DeepNEATConfig(**keyword_arguments)


def test_unknown_yaml_key_is_rejected() -> None:
    with pytest.raises(ConfigurationError):
        DeepNEATConfig.from_dict({"not_a_real_field": 1})


def test_global_search_ranges_and_crop_geometry_are_validated() -> None:
    with pytest.raises(ConfigurationError):
        DeepNEATConfig(global_learning_rate_min=0.0)
    with pytest.raises(ConfigurationError):
        DeepNEATConfig(global_momentum_max=1.0)
    with pytest.raises(ConfigurationError):
        DeepNEATConfig(global_cropped_image_size_max=29)
    with pytest.raises(ConfigurationError):
        DeepNEATConfig(number_of_filters_min=32, number_of_filters_max=None)
