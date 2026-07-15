from __future__ import annotations

import numpy as np

from polyneat.config.neat_config import NEATConfig
from polyneat.core.neat.global_innovation_tracker import GlobalInnovationTracker
from polyneat.core.neat.initial_population import (
    build_fully_connected_initial_population,
)
from polyneat.core.neat.neat_algorithm import NEATAlgorithm
from polyneat.core.neat.neat_genome import NEATGenome
from polyneat.core.population import Population


def test_factory_produces_fully_connected_population(
    small_neat_config: NEATConfig, rng: np.random.Generator
) -> None:
    population = build_fully_connected_initial_population(
        config=small_neat_config,
        innovation_tracker=GlobalInnovationTracker(),
        rng=rng,
    )
    assert population.size() == small_neat_config.population_size
    assert population.generation_number == 0
    assert population.species_assignments is None

    expected_node_count = (
        small_neat_config.number_of_input_nodes + 1 + small_neat_config.number_of_output_nodes
    )
    expected_connection_count = (
        small_neat_config.number_of_input_nodes + 1
    ) * small_neat_config.number_of_output_nodes
    for genome in population.genomes:
        assert isinstance(genome, NEATGenome)
        assert len(genome.node_genes) == expected_node_count
        assert len(genome.connection_genes) == expected_connection_count
        assert all(connection.is_enabled for connection in genome.connection_genes)


def test_factory_assigns_identical_innovation_ids_across_genomes(
    small_neat_config: NEATConfig, rng: np.random.Generator
) -> None:
    population = build_fully_connected_initial_population(
        config=small_neat_config,
        innovation_tracker=GlobalInnovationTracker(),
        rng=rng,
    )
    first_genome = population.genomes[0]
    assert isinstance(first_genome, NEATGenome)
    reference_innovation_ids = {
        connection.innovation_id for connection in first_genome.connection_genes
    }
    for genome in population.genomes[1:]:
        assert isinstance(genome, NEATGenome)
        genome_innovation_ids = {connection.innovation_id for connection in genome.connection_genes}
        assert genome_innovation_ids == reference_innovation_ids


def test_algorithm_delegates_to_factory(
    small_neat_config: NEATConfig, rng: np.random.Generator
) -> None:
    algorithm = NEATAlgorithm.from_config(small_neat_config)
    population = algorithm.create_initial_population(rng)
    assert population.size() == small_neat_config.population_size
    assert population.generation_number == 0


def test_algorithm_uses_custom_initial_population_factory(
    small_neat_config: NEATConfig, rng: np.random.Generator
) -> None:
    sentinel_population = Population(genomes=[], species_assignments=None, generation_number=0)
    received_arguments = {}

    def sentinel_factory(config, innovation_tracker, factory_rng):
        received_arguments["config"] = config
        received_arguments["innovation_tracker"] = innovation_tracker
        received_arguments["rng"] = factory_rng
        return sentinel_population

    algorithm = NEATAlgorithm.from_config(small_neat_config)
    algorithm.initial_population_factory = sentinel_factory
    assert algorithm.create_initial_population(rng) is sentinel_population
    # The factory contract: it receives the algorithm's own config and
    # innovation tracker plus the caller's RNG.
    assert received_arguments["config"] is algorithm.config
    assert received_arguments["innovation_tracker"] is algorithm.innovation_tracker
    assert received_arguments["rng"] is rng
