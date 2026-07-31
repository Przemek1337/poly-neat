import numpy as np

from polyneat.configs.neat.neat_config import NEATConfig
from polyneat.core.neat.mutations.composite_neat_mutation import CompositeNEATMutation
from polyneat.core.neat.neat_algorithm import NEATAlgorithm
from polyneat.core.neat.tournament_parent_selection import TournamentParentSelection


class _MarkerParentSelection(TournamentParentSelection):
    """Marker subclass proving that an overridden factory was used."""


class _CustomizedNEATAlgorithm(NEATAlgorithm):
    @classmethod
    def _build_parent_selection(cls, config: NEATConfig) -> _MarkerParentSelection:
        return _MarkerParentSelection(tournament_size=config.tournament_size_for_parent_selection)


def test_vanilla_from_config_uses_default_components() -> None:
    algorithm = NEATAlgorithm.from_config(NEATConfig(population_size=10))
    assert isinstance(algorithm.mutation, CompositeNEATMutation)
    assert isinstance(algorithm.parent_selection, TournamentParentSelection)
    assert not isinstance(algorithm.parent_selection, _MarkerParentSelection)


def test_subclass_overriding_factory_gets_custom_component() -> None:
    algorithm = _CustomizedNEATAlgorithm.from_config(NEATConfig(population_size=10))
    assert isinstance(algorithm, _CustomizedNEATAlgorithm)
    assert isinstance(algorithm.parent_selection, _MarkerParentSelection)


def test_advance_one_generation_still_works_after_factory_split() -> None:
    algorithm = NEATAlgorithm.from_config(
        NEATConfig(
            population_size=12,
            number_of_input_nodes=2,
            number_of_output_nodes=1,
            random_seed=5,
        )
    )
    rng = np.random.default_rng(5)
    population = algorithm.create_initial_population(rng)
    fitnesses = [float(index) for index in range(len(population.genomes))]
    next_population, statistics = algorithm.advance_one_generation(population, fitnesses, rng)
    assert len(next_population.genomes) == 12
    assert statistics.best_fitness == 11.0
