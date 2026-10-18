from __future__ import annotations

from pathlib import Path

import pytest

from polyneat import DeepNEATConfig


@pytest.mark.parametrize(
    "profile_name,expected_population,expected_generations,expected_epochs",
    [
        ("deepneat_smoke", 12, 8, 2),
        ("deepneat_paper", 100, 60, 8),
    ],
)
def test_cifar10_profile_has_expected_evolution_budget(
    profile_name: str,
    expected_population: int,
    expected_generations: int,
    expected_epochs: int,
) -> None:
    config_path = (
        Path(__file__).parent.parent / "examples" / "cifar10" / f"{profile_name}.yaml"
    )
    config = DeepNEATConfig.load_from_yaml_file(config_path)

    assert config.population_size == expected_population
    assert config.number_of_generations == expected_generations
    assert config.training_epochs_per_evaluation == expected_epochs


def test_smoke_profile_keeps_the_official_test_set_and_small_search_budget() -> None:
    config_path = (
        Path(__file__).parent.parent / "examples" / "cifar10" / "deepneat_smoke.yaml"
    )
    config = DeepNEATConfig.load_from_yaml_file(config_path)

    assert config.population_size * config.number_of_generations == 96
    assert config.maximum_training_samples == 10_000
    assert config.maximum_test_samples == 10_000
    assert config.final_training_epochs == 30


def test_paper_profile_encodes_table_3_1_search_intervals() -> None:
    config_path = (
        Path(__file__).parent.parent / "examples" / "cifar10" / "deepneat_paper.yaml"
    )
    config = DeepNEATConfig.load_from_yaml_file(config_path)

    assert (config.number_of_filters_min, config.number_of_filters_max) == (32, 256)
    assert config.available_kernel_sizes == (1, 3)
    assert (config.dropout_rate_min, config.dropout_rate_max) == (0.0, 0.7)
    assert (
        config.initial_weight_scaling_min,
        config.initial_weight_scaling_max,
    ) == (0.0, 2.0)
    assert (config.global_learning_rate_min, config.global_learning_rate_max) == (
        0.0001,
        0.1,
    )
    assert (config.global_momentum_min, config.global_momentum_max) == (0.68, 0.99)
    assert config.global_hue_shift_degrees_max == 45.0
    assert config.global_saturation_value_shift_max == 0.5
    assert config.global_saturation_value_scale_max == 0.5
    assert (
        config.global_cropped_image_size_min,
        config.global_cropped_image_size_max,
    ) == (26, 32)
    assert config.global_spatial_scaling_max == 0.3
    assert config.available_horizontal_flip_options == (False, True)
    assert config.available_variance_normalization_options == (False, True)
    assert config.available_nesterov_momentum_options == (False, True)
    assert config.probability_of_new_conv_layer == 1.0
    assert config.available_batch_normalization_options == (False,)
    assert config.compatibility_distance_coefficient_weight_difference_c3 == 0.0
    assert config.maximum_total_parameter_count is None
