from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from polyneat.config.configuration_errors import ConfigurationError
from polyneat.config.neat_config import NEATConfig
from polyneat.core.neat.initial_population import (
    INITIAL_POPULATION_STRATEGY_NAME_TO_CALLABLE,
    build_fully_connected_initial_population,
    register_initial_population_strategy,
    resolve_initial_population_strategy_by_name,
)
from polyneat.core.neat.neat_algorithm import NEATAlgorithm
from polyneat.core.population import Population


@pytest.fixture(autouse=True)
def restore_strategy_registry():
    """Keep the module-global registry hermetic across tests."""
    snapshot = dict(INITIAL_POPULATION_STRATEGY_NAME_TO_CALLABLE)
    yield
    INITIAL_POPULATION_STRATEGY_NAME_TO_CALLABLE.clear()
    INITIAL_POPULATION_STRATEGY_NAME_TO_CALLABLE.update(snapshot)


def _sentinel_strategy(config, innovation_tracker, rng) -> Population:
    return Population(genomes=[], species_assignments=None, generation_number=0)


def test_resolve_returns_builtin_fully_connected_strategy() -> None:
    resolved = resolve_initial_population_strategy_by_name("fully_connected")
    assert resolved is build_fully_connected_initial_population


def test_resolve_unknown_name_raises_configuration_error_listing_known_names() -> None:
    with pytest.raises(ConfigurationError) as raised:
        resolve_initial_population_strategy_by_name("no_such_strategy")
    message = str(raised.value)
    assert "no_such_strategy" in message
    assert "fully_connected" in message


def test_register_makes_custom_strategy_resolvable() -> None:
    register_initial_population_strategy("custom_sentinel", _sentinel_strategy)
    assert resolve_initial_population_strategy_by_name("custom_sentinel") is _sentinel_strategy


def test_register_duplicate_name_raises_configuration_error() -> None:
    with pytest.raises(ConfigurationError) as raised:
        register_initial_population_strategy("fully_connected", _sentinel_strategy)
    assert "fully_connected" in str(raised.value)


def test_builtin_strategy_satisfies_initial_population_strategy_protocol() -> None:
    from polyneat.core.component_protocols import InitialPopulationStrategy

    assert isinstance(build_fully_connected_initial_population, InitialPopulationStrategy)


def test_neat_config_defaults_to_fully_connected_strategy(
    small_neat_config: NEATConfig,
) -> None:
    assert small_neat_config.initial_population_strategy == "fully_connected"


def test_from_config_resolves_default_strategy_into_factory(
    small_neat_config: NEATConfig,
) -> None:
    algorithm = NEATAlgorithm.from_config(small_neat_config)
    assert algorithm.initial_population_factory is build_fully_connected_initial_population


def test_from_config_resolves_named_strategy_into_factory(
    small_neat_config: NEATConfig, rng: np.random.Generator
) -> None:
    register_initial_population_strategy("custom_sentinel", _sentinel_strategy)
    config = dataclasses.replace(small_neat_config, initial_population_strategy="custom_sentinel")
    algorithm = NEATAlgorithm.from_config(config)
    population = algorithm.create_initial_population(rng)
    assert algorithm.initial_population_factory is _sentinel_strategy
    assert population.genomes == []


def test_from_config_unknown_strategy_raises_configuration_error(
    small_neat_config: NEATConfig,
) -> None:
    config = dataclasses.replace(small_neat_config, initial_population_strategy="no_such_strategy")
    with pytest.raises(ConfigurationError) as raised:
        NEATAlgorithm.from_config(config)
    assert "no_such_strategy" in str(raised.value)


def test_explicit_factory_assignment_overrides_config_strategy(
    small_neat_config: NEATConfig, rng: np.random.Generator
) -> None:
    algorithm = NEATAlgorithm.from_config(small_neat_config)
    algorithm.initial_population_factory = _sentinel_strategy
    population = algorithm.create_initial_population(rng)
    assert population.genomes == []


# --- FS-NEAT ---------------------------------------------------------------


@pytest.fixture
def fs_neat_config(small_neat_config: NEATConfig) -> NEATConfig:
    """3 inputs x 2 outputs so dangling outputs and source variety are exercised."""
    return dataclasses.replace(small_neat_config, number_of_input_nodes=3, number_of_output_nodes=2)


def test_fs_neat_is_registered_under_its_name() -> None:
    from polyneat.core.neat.initial_population import (
        build_fs_neat_initial_population,
    )

    assert (
        resolve_initial_population_strategy_by_name("fs_neat") is build_fs_neat_initial_population
    )


def test_fs_neat_genomes_have_full_node_template_and_single_enabled_connection(
    fs_neat_config: NEATConfig, rng: np.random.Generator
) -> None:
    from polyneat.core.neat.global_innovation_tracker import (
        GlobalInnovationTracker,
    )
    from polyneat.core.neat.initial_population import (
        build_fs_neat_initial_population,
    )
    from polyneat.core.neat.neat_genome import NEATGenome

    population = build_fs_neat_initial_population(fs_neat_config, GlobalInnovationTracker(), rng)
    assert population.size() == fs_neat_config.population_size
    assert population.generation_number == 0

    expected_node_count = (
        fs_neat_config.number_of_input_nodes + 1 + fs_neat_config.number_of_output_nodes
    )
    input_node_ids = set(range(fs_neat_config.number_of_input_nodes))
    output_node_ids = set(
        range(
            fs_neat_config.number_of_input_nodes + 1,
            fs_neat_config.number_of_input_nodes + 1 + fs_neat_config.number_of_output_nodes,
        )
    )
    for genome in population.genomes:
        assert isinstance(genome, NEATGenome)
        assert len(genome.node_genes) == expected_node_count
        assert len(genome.connection_genes) == 1
        only_connection = genome.connection_genes[0]
        assert only_connection.is_enabled
        assert only_connection.source_node_id in input_node_ids  # not bias
        assert only_connection.target_node_id in output_node_ids


def test_fs_neat_draws_varied_connections_but_shares_innovation_numbering(
    fs_neat_config: NEATConfig, rng: np.random.Generator
) -> None:
    from polyneat.core.neat.global_innovation_tracker import (
        GlobalInnovationTracker,
    )
    from polyneat.core.neat.initial_population import (
        build_fs_neat_initial_population,
    )

    population = build_fs_neat_initial_population(fs_neat_config, GlobalInnovationTracker(), rng)
    pair_to_innovation_id: dict[tuple[int, int], int] = {}
    for genome in population.genomes:
        connection = genome.connection_genes[0]
        pair = (connection.source_node_id, connection.target_node_id)
        if pair in pair_to_innovation_id:
            assert connection.innovation_id == pair_to_innovation_id[pair]
        else:
            pair_to_innovation_id[pair] = connection.innovation_id
    assert len(pair_to_innovation_id) > 1  # random per genome, not one fixed link


def test_fs_neat_genome_decodes_and_forward_passes_despite_dangling_outputs(
    fs_neat_config: NEATConfig, rng: np.random.Generator
) -> None:
    import torch

    from polyneat.core.neat.global_innovation_tracker import (
        GlobalInnovationTracker,
    )
    from polyneat.core.neat.initial_population import (
        build_fs_neat_initial_population,
    )
    from polyneat.core.neat.neat_phenotype_decoder import NEATPhenotypeDecoder

    population = build_fs_neat_initial_population(fs_neat_config, GlobalInnovationTracker(), rng)
    decoder = NEATPhenotypeDecoder()
    batch = torch.zeros((4, fs_neat_config.number_of_input_nodes))
    for genome in population.genomes:
        phenotype = decoder.build_phenotype_from_genome(genome)
        outputs = phenotype.forward_pass(batch)
        assert outputs.shape == (4, fs_neat_config.number_of_output_nodes)
        assert torch.isfinite(outputs).all()
