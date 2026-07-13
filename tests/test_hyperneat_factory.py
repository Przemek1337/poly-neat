from __future__ import annotations

import numpy as np
import torch

from polyneat.algorithms.hyperneat.add_node_random_activation_mutation import (
    AddNodeWithRandomActivationMutation,
)
from polyneat.algorithms.hyperneat.factory import make_hyperneat_algorithm
from polyneat.algorithms.hyperneat.hyperneat_phenotype_decoder import (
    HyperNEATPhenotypeDecoder,
)
from polyneat.algorithms.neat.mutations.composite_neat_mutation import (
    CompositeNEATMutation,
)
from polyneat.algorithms.neat.neat_algorithm import NEATAlgorithm
from polyneat.algorithms.neat.neat_genome import NEATGenome
from polyneat.config.hyperneat_config import HyperNEATConfig


def _small_config() -> HyperNEATConfig:
    return HyperNEATConfig(
        population_size=12,
        substrate_input_layer_size=2,
        substrate_hidden_layer_sizes=(2,),
        substrate_output_layer_size=1,
        random_seed=7,
    )


def test_factory_returns_a_plain_neat_algorithm_not_a_subclass():
    algorithm = make_hyperneat_algorithm(_small_config())
    # architecture Case 1: the factory returns a configured NEATAlgorithm,
    # and there is no HyperNEATAlgorithm subclass at all.
    assert type(algorithm) is NEATAlgorithm


def test_factory_installs_hyperneat_decoder_and_random_activation_mutation():
    algorithm = make_hyperneat_algorithm(_small_config())
    assert isinstance(algorithm.phenotype_decoder, HyperNEATPhenotypeDecoder)
    assert isinstance(algorithm.mutation, CompositeNEATMutation)
    has_random_activation_add_node = any(
        isinstance(operator, AddNodeWithRandomActivationMutation)
        for operator in algorithm.mutation._ordered_individual_mutations
    )
    assert has_random_activation_add_node


def test_initial_population_genomes_are_cppns_with_four_inputs():
    algorithm = make_hyperneat_algorithm(_small_config())
    population = algorithm.create_initial_population(np.random.default_rng(0))
    assert len(population.genomes) == 12
    first_genome = population.genomes[0]
    assert isinstance(first_genome, NEATGenome)
    input_nodes = [n for n in first_genome.node_genes if n.node_type == "input"]
    output_nodes = [n for n in first_genome.node_genes if n.node_type == "output"]
    assert len(input_nodes) == 4  # x1, y1, x2, y2
    assert len(output_nodes) == 1  # weight


def test_initial_cppn_phenotype_maps_to_a_runnable_substrate():
    algorithm = make_hyperneat_algorithm(_small_config())
    population = algorithm.create_initial_population(np.random.default_rng(0))
    substrate_phenotype = algorithm.phenotype_decoder.build_phenotype_from_genome(
        population.genomes[0]
    )
    # substrate takes 2 task inputs and yields 1 output
    output_tensor = substrate_phenotype.forward_pass(torch.tensor([[1.0, 0.0]]))
    assert output_tensor.shape == (1, 1)
    assert torch.isfinite(output_tensor).all()


def test_advance_one_generation_runs_and_preserves_population_size():
    algorithm = make_hyperneat_algorithm(_small_config())
    rng = np.random.default_rng(1)
    population = algorithm.create_initial_population(rng)
    fitnesses = [float(index) for index in range(len(population.genomes))]
    next_population, statistics = algorithm.advance_one_generation(
        current_population=population,
        fitnesses_of_current_population=fitnesses,
        rng=rng,
    )
    assert len(next_population.genomes) == 12
    assert statistics.generation_number == 0
