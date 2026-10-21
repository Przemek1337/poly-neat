from __future__ import annotations

import numpy as np

from polyneat.algorithms.exact.exact_genome import EXACTGenome
from polyneat.algorithms.exact.exact_initial_population import (
    build_minimal_cnn_initial_population,
)
from polyneat.algorithms.exact.single_species_speciator import SingleSpeciesSpeciator
from polyneat.configs.exact.exact_config import EXACTConfig
from polyneat.core.neat.global_innovation_tracker import GlobalInnovationTracker
from polyneat.core.neat.initial_population import (
    resolve_initial_population_strategy_by_name,
)


def test_single_species_speciator_puts_every_genome_in_species_zero() -> None:
    genomes = [object(), object(), object()]
    species_ids = SingleSpeciesSpeciator().assign_genomes_to_species(
        genomes, np.random.default_rng(0)
    )
    assert species_ids == [0, 0, 0]


def test_minimal_cnn_population_matches_the_paper() -> None:
    config = EXACTConfig(
        population_size=5,
        number_of_output_nodes=10,
        input_image_height=28,
        input_image_width=28,
    )
    population = build_minimal_cnn_initial_population(
        config, GlobalInnovationTracker(), np.random.default_rng(0)
    )
    assert population.size() == 5
    assert population.generation_number == 0
    assert population.species_assignments is None
    genome = population.genomes[0]
    assert isinstance(genome, EXACTGenome)
    # Section 2.1: solely the input node plus one 1x1 output node per class.
    input_nodes = [n for n in genome.node_genes if n.node_type == "input"]
    output_nodes = [n for n in genome.node_genes if n.node_type == "output"]
    assert len(input_nodes) == 1
    assert (input_nodes[0].filter_height, input_nodes[0].filter_width) == (28, 28)
    assert input_nodes[0].depth == 0.0
    assert len(output_nodes) == 10
    assert all((n.filter_height, n.filter_width) == (1, 1) for n in output_nodes)
    assert all(n.depth == 1.0 for n in output_nodes)
    assert len(genome.edge_genes) == 10  # input wired to every output
    assert all(e.is_enabled and e.kernel_weights is None for e in genome.edge_genes)
    assert genome.is_trained is False
    assert genome.are_all_output_nodes_reachable() is True


def test_strategy_is_registered_under_the_config_default_name() -> None:
    strategy = resolve_initial_population_strategy_by_name("exact_minimal_cnn")
    assert strategy is build_minimal_cnn_initial_population
