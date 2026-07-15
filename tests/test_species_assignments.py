from __future__ import annotations

import numpy as np

from polyneat.config.neat_config import NEATConfig
from polyneat.core.neat.neat_algorithm import NEATAlgorithm


def test_offspring_population_carries_species_assignments(
    small_neat_config: NEATConfig, rng: np.random.Generator
) -> None:
    algorithm = NEATAlgorithm.from_config(small_neat_config)
    population = algorithm.create_initial_population(rng)
    assert population.species_assignments is None  # generation 0: not yet speciated

    fitnesses = [float(index) for index in range(population.size())]
    next_population, _statistics = algorithm.advance_one_generation(population, fitnesses, rng)

    assert next_population.species_assignments is not None
    assert len(next_population.species_assignments) == small_neat_config.population_size
    assert all(isinstance(species_id, int) for species_id in next_population.species_assignments)


def test_species_assignments_stay_aligned_across_generations(
    small_neat_config: NEATConfig, rng: np.random.Generator
) -> None:
    algorithm = NEATAlgorithm.from_config(small_neat_config)
    population = algorithm.create_initial_population(rng)
    for _generation_index in range(3):
        fitnesses = [float(index) for index in range(population.size())]
        population, _statistics = algorithm.advance_one_generation(population, fitnesses, rng)
        assert population.species_assignments is not None
        assert len(population.species_assignments) == population.size()
