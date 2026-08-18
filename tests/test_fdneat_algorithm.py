from __future__ import annotations

import numpy as np

from polyneat.algorithms.fdneat.fdneat_algorithm import FDNEATAlgorithm
from polyneat.algorithms.fdneat.mutations.delete_input_connection_mutation import (
    DeleteInputConnectionMutation,
)
from polyneat.configs.fdneat.fdneat_config import FDNEATConfig
from polyneat.configs.neat.neat_config import NEATConfig
from polyneat.core.neat.mutations.add_connection_mutation import AddConnectionMutation
from polyneat.core.neat.mutations.add_node_mutation import AddNodeMutation
from polyneat.core.neat.mutations.toggle_connection_enabled_mutation import (
    ToggleConnectionEnabledMutation,
)
from polyneat.core.neat.mutations.weight_modification_mutation import (
    WeightModificationMutation,
)
from polyneat.core.neat.neat_algorithm import NEATAlgorithm


def _config(**overrides) -> FDNEATConfig:
    base = {
        "population_size": 8,
        "number_of_input_nodes": 4,
        "number_of_output_nodes": 1,
        "random_seed": 42,
    }
    base.update(overrides)
    return FDNEATConfig(**base)


def test_fdneat_is_a_neat_algorithm() -> None:
    assert issubclass(FDNEATAlgorithm, NEATAlgorithm)


def test_generational_loop_is_inherited_unchanged() -> None:
    # FD-NEAT differs from NEAT by one mutation operator and nothing else.
    # Plain methods compare by identity; the _build_* factories are classmethods,
    # whose attribute access builds a fresh bound object each time, so those are
    # compared through __func__.
    assert FDNEATAlgorithm.advance_one_generation is NEATAlgorithm.advance_one_generation
    assert FDNEATAlgorithm.create_initial_population is NEATAlgorithm.create_initial_population
    for inherited_factory_name in (
        "_build_crossover",
        "_build_speciator",
        "_build_parent_selection",
        "_build_innovation_tracker",
        "_build_phenotype_decoder",
    ):
        assert (
            getattr(FDNEATAlgorithm, inherited_factory_name).__func__
            is getattr(NEATAlgorithm, inherited_factory_name).__func__
        ), f"{inherited_factory_name} must stay inherited"


def test_only_the_mutation_factory_is_overridden() -> None:
    assert (
        FDNEATAlgorithm._build_mutation.__func__ is not NEATAlgorithm._build_mutation.__func__
    )


def test_mutation_pipeline_has_five_operators_in_the_documented_order() -> None:
    algorithm = FDNEATAlgorithm.from_config(_config())
    operators = algorithm.mutation._ordered_individual_mutations
    assert [type(operator) for operator in operators] == [
        WeightModificationMutation,
        AddConnectionMutation,
        AddNodeMutation,
        ToggleConnectionEnabledMutation,
        DeleteInputConnectionMutation,
    ]


def test_deletion_operator_gets_the_configured_probability() -> None:
    algorithm = FDNEATAlgorithm.from_config(
        _config(probability_of_deleting_input_connection=0.33)
    )
    deletion_operator = algorithm.mutation._ordered_individual_mutations[-1]
    assert deletion_operator._probability_of_application == 0.33


def test_plain_neat_config_yields_an_inert_deletion_operator() -> None:
    # from_config must not explode when handed a bare NEATConfig; the operator
    # simply never fires, so the algorithm behaves as vanilla NEAT.
    algorithm = FDNEATAlgorithm.from_config(
        NEATConfig(population_size=8, number_of_input_nodes=4, number_of_output_nodes=1)
    )
    deletion_operator = algorithm.mutation._ordered_individual_mutations[-1]
    assert isinstance(deletion_operator, DeleteInputConnectionMutation)
    assert deletion_operator._probability_of_application == 0.0


def test_initial_population_is_fully_connected_like_vanilla_neat() -> None:
    config = _config()
    algorithm = FDNEATAlgorithm.from_config(config)
    population = algorithm.create_initial_population(np.random.default_rng(0))
    expected_connections = (config.number_of_input_nodes + 1) * config.number_of_output_nodes
    for genome in population.genomes:
        assert len(genome.connection_genes) == expected_connections
        assert all(gene.is_enabled for gene in genome.connection_genes)


def test_evolution_runs_and_can_shrink_connection_counts() -> None:
    config = _config(
        population_size=30,
        probability_of_deleting_input_connection=0.9,
        probability_of_add_node_mutation=0.0,
        probability_of_add_connection_mutation=0.0,
    )
    algorithm = FDNEATAlgorithm.from_config(config)
    rng = np.random.default_rng(0)
    population = algorithm.create_initial_population(rng)
    connections_at_start = max(len(genome.connection_genes) for genome in population.genomes)

    for _generation in range(10):
        fitnesses = list(rng.uniform(0.0, 1.0, size=len(population.genomes)))
        population, _statistics = algorithm.advance_one_generation(
            current_population=population,
            fitnesses_of_current_population=fitnesses,
            rng=rng,
        )

    connections_at_end = min(len(genome.connection_genes) for genome in population.genomes)
    assert connections_at_end < connections_at_start
    assert all(len(genome.connection_genes) >= 1 for genome in population.genomes)
    assert len(population.genomes) == config.population_size


def test_evolution_without_deletion_keeps_every_connection() -> None:
    # Control for the test above: with the operator off, FD-NEAT is NEAT.
    config = _config(
        population_size=20,
        probability_of_deleting_input_connection=0.0,
        probability_of_add_node_mutation=0.0,
        probability_of_add_connection_mutation=0.0,
        probability_of_toggle_connection_enabled=0.0,
    )
    algorithm = FDNEATAlgorithm.from_config(config)
    rng = np.random.default_rng(0)
    population = algorithm.create_initial_population(rng)
    expected_connections = (config.number_of_input_nodes + 1) * config.number_of_output_nodes

    for _generation in range(5):
        fitnesses = list(rng.uniform(0.0, 1.0, size=len(population.genomes)))
        population, _statistics = algorithm.advance_one_generation(
            current_population=population,
            fitnesses_of_current_population=fitnesses,
            rng=rng,
        )

    assert all(
        len(genome.connection_genes) == expected_connections for genome in population.genomes
    )
