from __future__ import annotations

import numpy as np

from polyneat.algorithms.cneat.cneat_algorithm import CNEATAlgorithm
from polyneat.config.cneat_config import CNEATConfig
from polyneat.core.component_protocols import NeuroevolutionAlgorithm
from polyneat.core.neat.neat_algorithm import NEATAlgorithm
from polyneat.evaluators.class_indexed_evaluator_base import ClassIndexedFitnessEvaluator


def _make_small_config() -> CNEATConfig:
    return CNEATConfig(
        population_size=9,
        number_of_input_nodes=2,
        number_of_output_nodes=1,
        number_of_class_labels=3,
        random_seed=42,
    )


def test_cneat_is_a_neat_algorithm() -> None:
    assert issubclass(CNEATAlgorithm, NEATAlgorithm)


def test_from_config_returns_cneat_algorithm_conforming_to_protocol() -> None:
    algorithm = CNEATAlgorithm.from_config(_make_small_config())
    assert isinstance(algorithm, CNEATAlgorithm)
    assert isinstance(algorithm, NeuroevolutionAlgorithm)
    assert algorithm.config.number_of_class_labels == 3


def test_advances_one_generation_with_class_indexed_fitness() -> None:
    class _ConstantFitnessEvaluator(ClassIndexedFitnessEvaluator):
        def evaluate_phenotype_for_class(self, phenotype, class_label_index: int) -> float:
            return 0.5

    config = _make_small_config()
    algorithm = CNEATAlgorithm.from_config(config)
    rng = np.random.default_rng(config.random_seed)

    population = algorithm.create_initial_population(rng)
    assert population.size() == config.population_size

    phenotypes = [
        algorithm.phenotype_decoder.build_phenotype_from_genome(genome)
        for genome in population.genomes
    ]
    evaluator = _ConstantFitnessEvaluator(number_of_class_labels=3)
    fitnesses = evaluator.evaluate_batch_of_phenotypes(phenotypes)

    next_population, statistics = algorithm.advance_one_generation(
        current_population=population,
        fitnesses_of_current_population=fitnesses,
        rng=rng,
    )
    assert next_population.size() == config.population_size
    assert next_population.generation_number == 1
    assert statistics.number_of_genomes_evaluated == config.population_size
