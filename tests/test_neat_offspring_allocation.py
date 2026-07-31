from __future__ import annotations

import collections

import numpy as np

from polyneat.configs.neat.neat_config import NEATConfig
from polyneat.core.neat.initial_population import build_fully_connected_initial_population
from polyneat.core.neat.neat_algorithm import NEATAlgorithm
from polyneat.core.population import Population
from polyneat.core.type_aliases import SpeciesId


class _FixedSpeciator:
    """Speciator returning a scripted assignment, so allocation can be tested exactly."""

    def __init__(self, species_id_per_genome: list[SpeciesId]) -> None:
        self._species_id_per_genome = species_id_per_genome

    def assign_genomes_to_species(
        self, genomes: list, rng: np.random.Generator
    ) -> list[SpeciesId]:
        return list(self._species_id_per_genome)


def _algorithm_with_scripted_species(
    population_size: int, species_id_per_genome: list[SpeciesId]
) -> tuple[NEATAlgorithm, Population]:
    config = NEATConfig(
        population_size=population_size,
        number_of_input_nodes=2,
        number_of_output_nodes=1,
        random_seed=0,
    )
    algorithm = NEATAlgorithm.from_config(config)
    algorithm.speciator = _FixedSpeciator(species_id_per_genome)
    population = build_fully_connected_initial_population(
        config=config,
        innovation_tracker=algorithm.innovation_tracker,
        rng=np.random.default_rng(0),
    )
    return algorithm, population


def test_species_allocated_no_slots_does_not_emit_an_elite() -> None:
    """A species the allocator gave 0 offspring to must not sneak a champion through.

    Species sizes 6 / 7 / 7 with the weak one first in population order, all
    species knobs at library defaults (elitism count 1, minimum size 5).
    """
    species_id_per_genome = [2] * 6 + [0] * 7 + [1] * 7
    algorithm, population = _algorithm_with_scripted_species(20, species_id_per_genome)
    fitnesses = [0.0] * 6 + [1.0] * 14

    next_population, _ = algorithm.advance_one_generation(
        population, fitnesses, np.random.default_rng(1)
    )

    emitted_per_species = collections.Counter(next_population.species_assignments)
    assert emitted_per_species[2] == 0


def test_no_species_loses_offspring_to_truncation() -> None:
    """Each species must receive exactly the slots the allocator gave it."""
    species_id_per_genome = [2] * 6 + [0] * 7 + [1] * 7
    algorithm, population = _algorithm_with_scripted_species(20, species_id_per_genome)
    fitnesses = [0.0] * 6 + [1.0] * 14

    next_population, _ = algorithm.advance_one_generation(
        population, fitnesses, np.random.default_rng(1)
    )

    emitted_per_species = collections.Counter(next_population.species_assignments)
    assert emitted_per_species[0] == 10
    assert emitted_per_species[1] == 10


def test_next_generation_keeps_the_configured_population_size() -> None:
    species_id_per_genome = [2] * 6 + [0] * 7 + [1] * 7
    algorithm, population = _algorithm_with_scripted_species(20, species_id_per_genome)
    fitnesses = [0.0] * 6 + [1.0] * 14

    next_population, _ = algorithm.advance_one_generation(
        population, fitnesses, np.random.default_rng(1)
    )

    assert len(next_population.genomes) == 20


def test_species_with_slots_still_carries_its_champion_over() -> None:
    """Elitism must keep working where the species actually has room for it."""
    species_id_per_genome = [0] * 10 + [1] * 10
    algorithm, population = _algorithm_with_scripted_species(20, species_id_per_genome)
    champion_index = 3
    fitnesses = [1.0] * 20
    fitnesses[champion_index] = 99.0

    next_population, _ = algorithm.advance_one_generation(
        population, fitnesses, np.random.default_rng(1)
    )

    champion_genome = population.genomes[champion_index]
    assert any(genome == champion_genome for genome in next_population.genomes)


def test_species_of_exactly_five_gets_no_elite_by_default() -> None:
    """Paper, section 4.1: the champion of each species with *more than* five networks."""
    algorithm, population = _algorithm_with_scripted_species(20, [0] * 20)
    elites = algorithm._pick_elite_genomes_from_species(
        member_genomes_in_species=list(population.genomes[:5]),
        member_fitnesses_in_species=[1.0, 2.0, 3.0, 4.0, 5.0],
    )
    assert elites == []


def test_species_of_six_gets_an_elite_by_default() -> None:
    algorithm, population = _algorithm_with_scripted_species(20, [0] * 20)
    elites = algorithm._pick_elite_genomes_from_species(
        member_genomes_in_species=list(population.genomes[:6]),
        member_fitnesses_in_species=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    )
    assert len(elites) == 1
