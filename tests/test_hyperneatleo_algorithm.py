from __future__ import annotations

import numpy as np
import torch

from polyneat.algorithms.hyperneat.add_node_random_activation_mutation import (
    AddNodeWithRandomActivationMutation,
)
from polyneat.algorithms.hyperneat.hyperneat_algorithm import HyperNEATAlgorithm
from polyneat.algorithms.hyperneatleo.hyperneatleo_algorithm import HyperNEATLEOAlgorithm
from polyneat.algorithms.hyperneatleo.leo_phenotype_decoder import (
    HyperNEATLEOPhenotypeDecoder,
)
from polyneat.configs.hyperneatleo.hyperneatleo_config import HyperNEATLEOConfig
from polyneat.core.neat.neat_algorithm import NEATAlgorithm

_RETINA_X = (-1.0, -0.9167, -0.8333, -0.75, 0.75, 0.8333, 0.9167, 1.0)


def _config(**overrides) -> HyperNEATLEOConfig:
    base = {
        "population_size": 6,
        "random_seed": 0,
        "substrate_layer_x_coordinates": (_RETINA_X, _RETINA_X, (-0.875, 0.875)),
    }
    base.update(overrides)
    return HyperNEATLEOConfig(**base)


def test_leo_is_a_hyperneat_algorithm() -> None:
    assert issubclass(HyperNEATLEOAlgorithm, HyperNEATAlgorithm)


def test_only_the_decoder_factory_is_overridden() -> None:
    assert (
        HyperNEATLEOAlgorithm._build_phenotype_decoder.__func__
        is not HyperNEATAlgorithm._build_phenotype_decoder.__func__
    )
    assert (
        HyperNEATLEOAlgorithm._build_mutation.__func__
        is HyperNEATAlgorithm._build_mutation.__func__
    )
    assert HyperNEATLEOAlgorithm.advance_one_generation is NEATAlgorithm.advance_one_generation


def test_decoder_is_the_leo_decoder() -> None:
    algorithm = HyperNEATLEOAlgorithm.from_config(_config())
    assert isinstance(algorithm.phenotype_decoder, HyperNEATLEOPhenotypeDecoder)


def test_explicit_coordinates_reach_the_substrate() -> None:
    algorithm = HyperNEATLEOAlgorithm.from_config(_config())
    substrate = algorithm.phenotype_decoder.substrate
    assert tuple(n.x_coordinate for n in substrate.input_layer.nodes) == _RETINA_X
    assert substrate.bias_node is not None
    assert substrate.bias_node.x_coordinate == 0.0


def test_falls_back_to_the_even_layered_substrate_without_explicit_coordinates() -> None:
    algorithm = HyperNEATLEOAlgorithm.from_config(
        _config(
            substrate_layer_x_coordinates=None,
            substrate_input_layer_size=4,
            substrate_hidden_layer_sizes=(3,),
            substrate_output_layer_size=2,
        )
    )
    substrate = algorithm.phenotype_decoder.substrate
    assert len(substrate.input_layer.nodes) == 4
    assert len(substrate.hidden_layers[0].nodes) == 3


def test_cppn_mutation_operator_is_inherited_from_hyperneat() -> None:
    algorithm = HyperNEATLEOAlgorithm.from_config(_config())
    operator_types = [type(op) for op in algorithm.mutation._ordered_individual_mutations]
    assert AddNodeWithRandomActivationMutation in operator_types


def test_initial_population_uses_the_leo_seed() -> None:
    algorithm = HyperNEATLEOAlgorithm.from_config(_config())
    population = algorithm.create_initial_population(np.random.default_rng(0))
    for genome in population.genomes:
        assert any(node.activation_function_name == "gaussian" for node in genome.node_genes)


def test_decoded_phenotype_produces_two_outputs() -> None:
    algorithm = HyperNEATLEOAlgorithm.from_config(_config())
    population = algorithm.create_initial_population(np.random.default_rng(0))
    phenotype = algorithm.phenotype_decoder.build_phenotype_from_genome(population.genomes[0])
    outputs = phenotype.forward_pass(torch.zeros(3, 8))
    assert outputs.shape == (3, 2)


def test_evolution_advances_without_error() -> None:
    algorithm = HyperNEATLEOAlgorithm.from_config(_config())
    rng = np.random.default_rng(0)
    population = algorithm.create_initial_population(rng)
    for _generation in range(5):
        fitnesses = list(rng.uniform(0.0, 1.0, size=len(population.genomes)))
        population, _statistics = algorithm.advance_one_generation(
            current_population=population,
            fitnesses_of_current_population=fitnesses,
            rng=rng,
        )
    assert len(population.genomes) == 6


def test_every_evolved_genome_still_decodes() -> None:
    algorithm = HyperNEATLEOAlgorithm.from_config(_config())
    rng = np.random.default_rng(1)
    population = algorithm.create_initial_population(rng)
    for _generation in range(5):
        fitnesses = list(rng.uniform(0.0, 1.0, size=len(population.genomes)))
        population, _statistics = algorithm.advance_one_generation(
            current_population=population,
            fitnesses_of_current_population=fitnesses,
            rng=rng,
        )
    for genome in population.genomes:
        phenotype = algorithm.phenotype_decoder.build_phenotype_from_genome(genome)
        assert phenotype.forward_pass(torch.zeros(2, 8)).shape == (2, 2)
