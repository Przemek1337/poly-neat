from __future__ import annotations

import dataclasses

import numpy as np
from numpy.random import Generator

from polyneat.algorithms.neat.compatibility_distance_speciator import (
    CompatibilityDistanceSpeciator,
)
from polyneat.algorithms.neat.global_innovation_tracker import GlobalInnovationTracker
from polyneat.algorithms.neat.mutations.composite_neat_mutation import CompositeNEATMutation
from polyneat.algorithms.neat.neat_algorithm import NEATAlgorithm
from polyneat.algorithms.neat.neat_crossover import NEATCrossover
from polyneat.algorithms.neat.neat_genome import NEATGenome
from polyneat.algorithms.neat.neat_phenotype_decoder import NEATPhenotypeDecoder
from polyneat.algorithms.neat.tournament_parent_selection import TournamentParentSelection
from polyneat.config.neat_config import NEATConfig
from polyneat.core.component_protocols import InnovationTracker
from polyneat.core.population import Population


class _IdentityMutation:
    def apply_to_genome(
        self, genome: NEATGenome, rng: Generator, innovation_tracker: InnovationTracker
    ) -> NEATGenome:
        return genome


class _FirstParentCrossover:
    def apply_to_parents(
        self, fitter_parent: NEATGenome, less_fit_parent: NEATGenome, rng: Generator
    ) -> NEATGenome:
        return fitter_parent


class _SingleSpeciesSpeciator:
    def assign_genomes_to_species(self, genomes, rng):
        return [0] * len(genomes)


class _PassthroughPhenotypeDecoder:
    def build_phenotype_from_genome(self, genome):
        raise NotImplementedError("structural stand-in, never called in this test")


def _sentinel_initial_population_factory(config, innovation_tracker, rng) -> Population:
    return Population(genomes=[], species_assignments=None, generation_number=0)


def test_from_config_defaults_build_standard_components(
    small_neat_config: NEATConfig,
) -> None:
    algorithm = NEATAlgorithm.from_config(small_neat_config)
    assert isinstance(algorithm.mutation, CompositeNEATMutation)
    assert isinstance(algorithm.crossover, NEATCrossover)
    assert isinstance(algorithm.parent_selection, TournamentParentSelection)
    assert isinstance(algorithm.speciator, CompatibilityDistanceSpeciator)
    assert isinstance(algorithm.innovation_tracker, GlobalInnovationTracker)
    assert isinstance(algorithm.phenotype_decoder, NEATPhenotypeDecoder)
    assert algorithm.initial_population_factory is None


def test_replace_swaps_selected_components_and_keeps_the_rest(
    small_neat_config: NEATConfig,
) -> None:
    custom_mutation = _IdentityMutation()
    custom_decoder = _PassthroughPhenotypeDecoder()

    algorithm = dataclasses.replace(
        NEATAlgorithm.from_config(small_neat_config),
        mutation=custom_mutation,
        _phenotype_decoder=custom_decoder,
        initial_population_factory=_sentinel_initial_population_factory,
    )
    # Swapped components are the ones we passed.
    assert algorithm.mutation is custom_mutation
    assert algorithm.phenotype_decoder is custom_decoder
    assert algorithm.initial_population_factory is _sentinel_initial_population_factory
    # Untouched components carry over from the vanilla-NEAT build.
    assert isinstance(algorithm.crossover, NEATCrossover)
    assert isinstance(algorithm.parent_selection, TournamentParentSelection)
    assert isinstance(algorithm.speciator, CompatibilityDistanceSpeciator)


def test_swapped_components_are_actually_used(
    small_neat_config: NEATConfig, rng: np.random.Generator
) -> None:
    """One full generation with identity mutation + first-parent crossover must not crash
    and must return a population of the configured size."""
    algorithm = dataclasses.replace(
        NEATAlgorithm.from_config(small_neat_config),
        mutation=_IdentityMutation(),
        crossover=_FirstParentCrossover(),
        speciator=_SingleSpeciesSpeciator(),
    )
    population = algorithm.create_initial_population(rng)
    fitnesses = [float(index) for index in range(population.size())]
    next_population, statistics = algorithm.advance_one_generation(population, fitnesses, rng)
    assert next_population.size() == small_neat_config.population_size
    assert statistics.number_of_species == 1
