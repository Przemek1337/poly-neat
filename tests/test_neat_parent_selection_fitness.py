from __future__ import annotations

import numpy as np

from polyneat.configs.neat.neat_config import NEATConfig
from polyneat.core.neat.neat_algorithm import NEATAlgorithm
from polyneat.core.neat.neat_genome import ConnectionGene, NEATGenome, NodeGene
from polyneat.core.neat.tournament_parent_selection import TournamentParentSelection


def _one_connection_genome(weight: float) -> NEATGenome:
    return NEATGenome(
        node_genes=(
            NodeGene(node_id=0, node_type="input", activation_function_name="identity"),
            NodeGene(node_id=1, node_type="output", activation_function_name="sigmoid"),
        ),
        connection_genes=(
            ConnectionGene(
                innovation_id=0, source_node_id=0, target_node_id=1, weight=weight, is_enabled=True
            ),
        ),
    )


def _algorithm_with_decisive_tournament() -> NEATAlgorithm:
    algorithm = NEATAlgorithm.from_config(
        NEATConfig(
            population_size=10,
            number_of_input_nodes=1,
            number_of_output_nodes=1,
            random_seed=0,
        )
    )
    # A 20-way tournament over two candidates picks the fitter one with
    # probability 1 - 0.5**20, i.e. effectively always.
    algorithm.parent_selection = TournamentParentSelection(tournament_size=20)
    return algorithm


def test_selected_parent_fitness_belongs_to_the_selected_candidate() -> None:
    """Equal-valued genomes must not make the lookup return a stand-in's fitness.

    ``NEATGenome`` is a frozen dataclass, so two structurally identical genomes
    compare equal. Recovering the winner's index by value therefore reports the
    first equal genome's fitness, which then decides which parent counts as the
    fitter one in crossover.
    """
    algorithm = _algorithm_with_decisive_tournament()
    candidate_genomes = [_one_connection_genome(0.5), _one_connection_genome(0.5)]
    candidate_fitnesses = [0.1, 0.9]

    _, selected_fitness = algorithm._select_single_parent_with_fitness(
        candidate_genomes=candidate_genomes,
        candidate_fitnesses=candidate_fitnesses,
        rng=np.random.default_rng(0),
    )

    assert selected_fitness == 0.9


def test_selected_parent_fitness_is_correct_for_distinguishable_genomes() -> None:
    algorithm = _algorithm_with_decisive_tournament()
    candidate_genomes = [_one_connection_genome(0.5), _one_connection_genome(-1.25)]
    candidate_fitnesses = [0.1, 0.9]

    selected_genome, selected_fitness = algorithm._select_single_parent_with_fitness(
        candidate_genomes=candidate_genomes,
        candidate_fitnesses=candidate_fitnesses,
        rng=np.random.default_rng(0),
    )

    assert selected_genome == candidate_genomes[1]
    assert selected_fitness == 0.9


def test_select_parent_indices_returns_positions_in_the_candidate_pool() -> None:
    selection: TournamentParentSelection[NEATGenome] = TournamentParentSelection(tournament_size=20)
    candidate_genomes = [_one_connection_genome(0.5), _one_connection_genome(0.5)]

    selected_indices = selection.select_parent_indices(
        candidate_genomes=candidate_genomes,
        candidate_fitnesses=[0.1, 0.9],
        number_of_parents_to_select=5,
        rng=np.random.default_rng(0),
    )

    assert selected_indices == [1, 1, 1, 1, 1]


def test_select_parents_agrees_with_select_parent_indices() -> None:
    selection: TournamentParentSelection[NEATGenome] = TournamentParentSelection(tournament_size=3)
    candidate_genomes = [_one_connection_genome(w) for w in (0.5, -1.25, 2.0)]
    candidate_fitnesses = [0.1, 0.9, 0.4]

    selected_genomes = selection.select_parents(
        candidate_genomes=candidate_genomes,
        candidate_fitnesses=candidate_fitnesses,
        number_of_parents_to_select=6,
        rng=np.random.default_rng(7),
    )
    selected_indices = selection.select_parent_indices(
        candidate_genomes=candidate_genomes,
        candidate_fitnesses=candidate_fitnesses,
        number_of_parents_to_select=6,
        rng=np.random.default_rng(7),
    )

    assert selected_genomes == [candidate_genomes[index] for index in selected_indices]
