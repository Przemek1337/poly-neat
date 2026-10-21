from __future__ import annotations

from pathlib import Path

import pytest
import torch

from examples._experiment import EXAMPLE_REGISTRY
from examples.retina import hyperneat as retina_hyperneat
from examples.retina import leo as retina_leo

_REQUIRED_METRICS = {
    "best_fitness",
    "number_of_expressed_connections",
    "number_of_cross_hemisphere_connections",
    "fraction_of_cross_hemisphere_connections",
}


@pytest.mark.parametrize("module", [retina_hyperneat, retina_leo])
def test_example_exposes_the_experiment_contract(module) -> None:  # noqa: ANN001
    assert isinstance(module.CONFIG_FILE_PATH, Path)
    assert module.CONFIG_FILE_PATH.exists()
    assert callable(module.run_experiment)


def test_both_examples_are_registered() -> None:
    assert EXAMPLE_REGISTRY["retina/hyperneat"] == "examples.retina.hyperneat"
    assert EXAMPLE_REGISTRY["retina/leo"] == "examples.retina.leo"


def test_the_two_configs_differ_only_where_the_algorithms_do() -> None:
    # Any other divergence would make the comparison meaningless.
    import polyneat as pn

    leo_config = pn.HyperNEATLEOConfig.load_from_yaml_file(retina_leo.CONFIG_FILE_PATH)
    baseline_config = pn.HyperNEATConfig.load_from_yaml_file(
        retina_hyperneat.CONFIG_FILE_PATH
    )
    for shared_field_name in (
        "population_size",
        "random_seed",
        "initial_weight_range_min",
        "initial_weight_range_max",
        "probability_of_add_node_mutation",
        "probability_of_add_connection_mutation",
        "probability_of_genome_weight_mutation",
        "compatibility_distance_threshold",
        "species_elitism_count",
        "tournament_size_for_parent_selection",
        "substrate_node_activation_function",
        "max_substrate_connection_weight_magnitude",
        "available_activation_functions",
    ):
        assert getattr(leo_config, shared_field_name) == getattr(
            baseline_config, shared_field_name
        ), f"{shared_field_name} differs between the two retina configs"


def test_both_substrates_have_the_same_shape() -> None:
    import polyneat as pn

    leo_algorithm = pn.HyperNEATLEOAlgorithm.from_config(
        pn.HyperNEATLEOConfig.load_from_yaml_file(retina_leo.CONFIG_FILE_PATH),
        device_for_phenotype_computation=torch.device("cpu"),
    )
    baseline_algorithm = pn.HyperNEATAlgorithm.from_config(
        pn.HyperNEATConfig.load_from_yaml_file(retina_hyperneat.CONFIG_FILE_PATH),
        device_for_phenotype_computation=torch.device("cpu"),
    )
    leo_substrate = leo_algorithm.phenotype_decoder.substrate
    baseline_substrate = baseline_algorithm.phenotype_decoder.substrate
    assert len(leo_substrate.input_layer.nodes) == len(baseline_substrate.input_layer.nodes)
    assert len(leo_substrate.output_layer.nodes) == len(baseline_substrate.output_layer.nodes)
    assert len(leo_substrate.hidden_layers[0].nodes) == len(
        baseline_substrate.hidden_layers[0].nodes
    )


@pytest.mark.parametrize("module", [retina_hyperneat, retina_leo])
def test_short_run_reports_every_metric(module) -> None:  # noqa: ANN001
    report = module.run_experiment(
        device=torch.device("cpu"),
        random_seed=0,
        artifacts_directory=None,
        max_generations=3,
    )
    assert _REQUIRED_METRICS <= set(report.metric_values)
    assert 0.0 <= report.metric_values["best_fitness"] <= 256.0
    assert report.metric_values["number_of_cross_hemisphere_connections"] >= 0.0
    assert (
        report.metric_values["number_of_cross_hemisphere_connections"]
        <= report.metric_values["number_of_expressed_connections"]
    )
    assert report.number_of_generations > 0


def test_leo_generation_zero_starts_fully_modular() -> None:
    # Not a claim about the final result - only that the seed does its job on the
    # substrate this example actually configures.
    import numpy as np

    import polyneat as pn

    config = pn.HyperNEATLEOConfig.load_from_yaml_file(retina_leo.CONFIG_FILE_PATH)
    algorithm = pn.HyperNEATLEOAlgorithm.from_config(
        config, device_for_phenotype_computation=torch.device("cpu")
    )
    population = algorithm.create_initial_population(np.random.default_rng(0))
    decoder = algorithm.phenotype_decoder
    for genome in population.genomes[:5]:
        decoded = decoder.decode_substrate_genome(genome)
        assert pn.count_cross_hemisphere_connections(decoder.substrate, decoded) == 0
        assert pn.count_expressed_connections(decoded) > 0


def test_baseline_generation_zero_is_not_modular() -> None:
    # The contrast that makes the experiment worth running: plain HyperNEAT has
    # no reason to keep the hemispheres apart, so generation 0 crosses freely.
    import numpy as np

    import polyneat as pn

    config = pn.HyperNEATConfig.load_from_yaml_file(retina_hyperneat.CONFIG_FILE_PATH)
    algorithm = pn.HyperNEATAlgorithm.from_config(
        config, device_for_phenotype_computation=torch.device("cpu")
    )
    population = algorithm.create_initial_population(np.random.default_rng(0))
    decoder = algorithm.phenotype_decoder
    crossings = [
        pn.count_cross_hemisphere_connections(
            decoder.substrate, decoder.decode_substrate_genome(genome)
        )
        for genome in population.genomes[:20]
    ]
    assert max(crossings) > 0
