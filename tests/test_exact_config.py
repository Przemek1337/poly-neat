from __future__ import annotations

import pytest

from polyneat.configs.configuration_errors import ConfigurationError
from polyneat.configs.exact.exact_config import EXACTConfig


def test_defaults_are_valid_and_follow_the_paper() -> None:
    config = EXACTConfig()
    assert config.number_of_input_nodes == 1
    assert config.input_image_height == 28
    assert config.input_image_width == 28
    assert config.initial_population_strategy == "exact_minimal_cnn"
    assert config.number_of_mutations_per_genome == 3
    # Eight operator rates: 1/12, 2/12, 2/12, 4/12, 1/12, 1/12, 1/24, 1/24.
    assert config.probability_of_disable_edge_mutation == pytest.approx(1 / 12)
    assert config.probability_of_enable_edge_mutation == pytest.approx(2 / 12)
    assert config.probability_of_split_edge_mutation == pytest.approx(2 / 12)
    assert config.probability_of_add_edge_mutation == pytest.approx(4 / 12)
    assert config.probability_of_add_node_mutation == pytest.approx(1 / 12)
    assert config.probability_of_change_filter_size_mutation == pytest.approx(1 / 12)
    assert config.probability_of_change_filter_size_x_mutation == pytest.approx(1 / 24)
    assert config.probability_of_change_filter_size_y_mutation == pytest.approx(1 / 24)
    assert config.filter_size_change_options == (-2, -1, 1, 2)
    # Section III-B: 20% crossover rate; tournament size 1 = the paper's uniform draw.
    assert config.probability_of_crossover_vs_mutation_only == 0.2
    assert config.tournament_size_for_parent_selection == 1
    assert config.use_epigenetic_weight_initialization is True
    # Section VII / VIII-B1 backpropagation hyperparameters.
    assert config.leaky_relu_negative_slope == 0.1
    assert config.activation_clamp_maximum == 5.5
    assert config.training_batch_size == 50
    assert config.velocity_reset_interval == 1000
    assert config.input_dropout_probability == 0.0
    assert config.hidden_dropout_probability == 0.0
    assert config.batch_normalization_alpha == pytest.approx(0.1)
    assert config.backpropagation_learning_rate == pytest.approx(0.0025)
    assert config.learning_rate_decay_factor == pytest.approx(0.95)
    assert config.minimum_learning_rate == 0.00001
    assert config.backpropagation_momentum == 0.5
    assert config.momentum_decay_factor == pytest.approx(0.95)
    assert config.maximum_momentum == 0.99
    assert config.backpropagation_weight_decay == pytest.approx(0.0005)
    assert config.weight_decay_decay_factor == pytest.approx(0.95)
    assert config.minimum_weight_decay == 0.000001


def test_momentum_decay_factor_must_be_in_open_unit_interval() -> None:
    with pytest.raises(ConfigurationError, match="momentum_decay_factor"):
        EXACTConfig(momentum_decay_factor=1.5)


def test_rejects_more_than_one_input_node() -> None:
    with pytest.raises(ConfigurationError):
        EXACTConfig(number_of_input_nodes=2)


def test_rejects_non_positive_image_dimensions() -> None:
    with pytest.raises(ConfigurationError):
        EXACTConfig(input_image_height=0)
    with pytest.raises(ConfigurationError):
        EXACTConfig(input_image_width=-1)


def test_rejects_mutation_probabilities_that_do_not_sum_to_one() -> None:
    with pytest.raises(ConfigurationError, match="eight mutation operator"):
        EXACTConfig(probability_of_add_edge_mutation=0.9)
    with pytest.raises(ConfigurationError, match="eight mutation operator"):
        EXACTConfig(probability_of_add_node_mutation=0.5)


def test_rejects_bad_training_hyperparameters() -> None:
    with pytest.raises(ConfigurationError):
        EXACTConfig(number_of_training_epochs_per_genome=0)
    with pytest.raises(ConfigurationError):
        EXACTConfig(training_batch_size=0)
    with pytest.raises(ConfigurationError):
        EXACTConfig(backpropagation_learning_rate=0.0)
    with pytest.raises(ConfigurationError):
        EXACTConfig(backpropagation_momentum=0.0)  # nesterov needs momentum > 0
    with pytest.raises(ConfigurationError):
        EXACTConfig(maximum_momentum=1.0)
    with pytest.raises(ConfigurationError):
        EXACTConfig(momentum_decay_factor=0.0)
    with pytest.raises(ConfigurationError):
        EXACTConfig(learning_rate_decay_factor=1.5)
    with pytest.raises(ConfigurationError):
        EXACTConfig(minimum_hidden_filter_size=0)
    with pytest.raises(ConfigurationError):
        EXACTConfig(filter_size_change_options=())
    with pytest.raises(ConfigurationError):
        EXACTConfig(filter_size_change_options=(-2, 0, 2))
    with pytest.raises(ConfigurationError):
        EXACTConfig(maximum_mutation_attempts_for_reachable_child=0)
    with pytest.raises(ConfigurationError, match="velocity_reset_interval"):
        EXACTConfig(velocity_reset_interval=-1)
    with pytest.raises(ConfigurationError, match="hidden_dropout_probability"):
        EXACTConfig(hidden_dropout_probability=1.0)
    with pytest.raises(ConfigurationError, match="batch_normalization_alpha"):
        EXACTConfig(batch_normalization_alpha=0.0)


def test_crossover_rates_must_be_probabilities() -> None:
    with pytest.raises(ConfigurationError):
        EXACTConfig(more_fit_parent_edge_inclusion_rate=1.5)
    with pytest.raises(ConfigurationError):
        EXACTConfig(less_fit_parent_edge_inclusion_rate=-0.1)
