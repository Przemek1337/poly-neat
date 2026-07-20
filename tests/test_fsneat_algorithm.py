from __future__ import annotations

import dataclasses

import numpy as np

from polyneat.algorithms.fsneat import FSNEATAlgorithm
from polyneat.configs.neat.neat_config import NEATConfig
from polyneat.core.neat.compatibility_distance_speciator import (
    CompatibilityDistanceSpeciator,
)
from polyneat.core.neat.mutations.composite_neat_mutation import CompositeNEATMutation
from polyneat.core.neat.neat_algorithm import NEATAlgorithm
from polyneat.core.neat.neat_crossover import NEATCrossover
from polyneat.core.neat.neat_genome import NEATGenome
from polyneat.core.neat.tournament_parent_selection import TournamentParentSelection


def test_fsneat_is_a_neat_algorithm() -> None:
    assert issubclass(FSNEATAlgorithm, NEATAlgorithm)


def test_from_config_builds_inherited_standard_components(
    small_neat_config: NEATConfig,
) -> None:
    algorithm = FSNEATAlgorithm.from_config(small_neat_config)
    assert isinstance(algorithm, FSNEATAlgorithm)
    assert isinstance(algorithm.mutation, CompositeNEATMutation)
    assert isinstance(algorithm.crossover, NEATCrossover)
    assert isinstance(algorithm.parent_selection, TournamentParentSelection)
    assert isinstance(algorithm.speciator, CompatibilityDistanceSpeciator)


def test_initial_population_is_fs_neat_style_single_connection(
    small_neat_config: NEATConfig, rng: np.random.Generator
) -> None:
    algorithm = FSNEATAlgorithm.from_config(small_neat_config)
    population = algorithm.create_initial_population(rng)
    assert population.size() == small_neat_config.population_size
    assert population.generation_number == 0
    for genome in population.genomes:
        assert isinstance(genome, NEATGenome)
        assert len(genome.connection_genes) == 1
        assert genome.connection_genes[0].is_enabled


def test_yaml_initial_population_strategy_is_ignored(
    small_neat_config: NEATConfig, rng: np.random.Generator
) -> None:
    """FS-NEAT defines its start topology by definition of the algorithm:
    even an explicit ``fully_connected`` strategy in the config is ignored."""
    config = dataclasses.replace(small_neat_config, initial_population_strategy="fully_connected")
    algorithm = FSNEATAlgorithm.from_config(config)
    population = algorithm.create_initial_population(rng)
    for genome in population.genomes:
        assert len(genome.connection_genes) == 1


def test_one_full_generation_runs_end_to_end(
    small_neat_config: NEATConfig, rng: np.random.Generator
) -> None:
    algorithm = FSNEATAlgorithm.from_config(small_neat_config)
    population = algorithm.create_initial_population(rng)
    fitnesses = [float(index) for index in range(population.size())]
    next_population, statistics = algorithm.advance_one_generation(population, fitnesses, rng)
    assert next_population.size() == small_neat_config.population_size
    assert statistics.generation_number == 0
