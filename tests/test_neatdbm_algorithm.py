from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from polyneat.algorithms.neatdbm.difference_based_weight_mutation import (
    DifferenceBasedWeightMutation,
)
from polyneat.algorithms.neatdbm.neatdbm_algorithm import NEATDBMAlgorithm
from polyneat.configs.neatdbm.neatdbm_config import NEATDBMConfig
from polyneat.core.neat.neat_algorithm import NEATAlgorithm
from polyneat.core.neat.neat_genome import NEATGenome


@pytest.fixture
def small_neatdbm_config() -> NEATDBMConfig:
    """Tiny config for fast structural tests (8 genomes, 2 inputs, 1 output)."""
    return NEATDBMConfig(
        population_size=8,
        number_of_input_nodes=2,
        number_of_output_nodes=1,
        random_seed=42,
    )


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


class _RecordingDifferenceBasedMutation:
    """Test double recording every donor triple passed to the operator."""

    def __init__(self) -> None:
        self.recorded_donor_triples: list[tuple[NEATGenome, NEATGenome, NEATGenome]] = []

    def apply_to_genome_with_donors(
        self,
        genome: NEATGenome,
        donor_base_genome: NEATGenome,
        donor_difference_first_genome: NEATGenome,
        donor_difference_second_genome: NEATGenome,
    ) -> NEATGenome:
        self.recorded_donor_triples.append(
            (
                donor_base_genome,
                donor_difference_first_genome,
                donor_difference_second_genome,
            )
        )
        return genome


def test_neatdbm_is_a_neat_algorithm() -> None:
    assert issubclass(NEATDBMAlgorithm, NEATAlgorithm)


def test_from_config_builds_difference_mutation_operator(
    small_neatdbm_config: NEATDBMConfig,
) -> None:
    algorithm = NEATDBMAlgorithm.from_config(small_neatdbm_config)
    assert isinstance(algorithm, NEATDBMAlgorithm)
    assert isinstance(
        algorithm.difference_based_weight_mutation, DifferenceBasedWeightMutation
    )


def test_every_child_undergoes_dbm_when_probability_is_one(
    small_neatdbm_config: NEATDBMConfig, rng: np.random.Generator
) -> None:
    config = dataclasses.replace(
        small_neatdbm_config, probability_of_difference_based_mutation=1.0
    )
    algorithm = NEATDBMAlgorithm.from_config(config)
    recording_mutation = _RecordingDifferenceBasedMutation()
    algorithm.difference_based_weight_mutation = recording_mutation  # type: ignore[assignment]

    population = algorithm.create_initial_population(rng)
    fitnesses = [float(index) for index in range(population.size())]
    algorithm.advance_one_generation(population, fitnesses, rng)

    assert recording_mutation.recorded_donor_triples
    population_genome_identities = {id(genome) for genome in population.genomes}
    for donor_triple in recording_mutation.recorded_donor_triples:
        donor_identities = {id(donor_genome) for donor_genome in donor_triple}
        # three mutually distinct donors, all members of the current population
        assert len(donor_identities) == 3
        assert donor_identities <= population_genome_identities


def test_no_dbm_when_probability_is_zero(
    small_neatdbm_config: NEATDBMConfig, rng: np.random.Generator
) -> None:
    config = dataclasses.replace(
        small_neatdbm_config, probability_of_difference_based_mutation=0.0
    )
    algorithm = NEATDBMAlgorithm.from_config(config)
    recording_mutation = _RecordingDifferenceBasedMutation()
    algorithm.difference_based_weight_mutation = recording_mutation  # type: ignore[assignment]

    population = algorithm.create_initial_population(rng)
    fitnesses = [float(index) for index in range(population.size())]
    algorithm.advance_one_generation(population, fitnesses, rng)

    assert recording_mutation.recorded_donor_triples == []


def test_dbm_is_skipped_when_population_has_fewer_than_three_genomes(
    rng: np.random.Generator,
) -> None:
    config = NEATDBMConfig(
        population_size=2,
        number_of_input_nodes=2,
        number_of_output_nodes=1,
        random_seed=42,
        probability_of_difference_based_mutation=1.0,
    )
    algorithm = NEATDBMAlgorithm.from_config(config)
    recording_mutation = _RecordingDifferenceBasedMutation()
    algorithm.difference_based_weight_mutation = recording_mutation  # type: ignore[assignment]

    population = algorithm.create_initial_population(rng)
    fitnesses = [float(index) for index in range(population.size())]
    next_population, _ = algorithm.advance_one_generation(population, fitnesses, rng)

    assert recording_mutation.recorded_donor_triples == []
    assert next_population.size() == 2


def test_one_full_generation_runs_end_to_end_with_real_operator(
    small_neatdbm_config: NEATDBMConfig, rng: np.random.Generator
) -> None:
    config = dataclasses.replace(
        small_neatdbm_config, probability_of_difference_based_mutation=1.0
    )
    algorithm = NEATDBMAlgorithm.from_config(config)
    population = algorithm.create_initial_population(rng)
    fitnesses = [float(index) for index in range(population.size())]
    next_population, statistics = algorithm.advance_one_generation(population, fitnesses, rng)
    assert next_population.size() == config.population_size
    assert statistics.generation_number == 0
    for genome in next_population.genomes:
        assert isinstance(genome, NEATGenome)
