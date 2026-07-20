from __future__ import annotations

import numpy as np
import torch
from numpy.random import Generator

from polyneat.algorithms.lneat.backpropagation_weight_trainer import (
    BackpropagationWeightTrainer,
)
from polyneat.algorithms.lneat.lneat_algorithm import LNEATAlgorithm
from polyneat.config.lneat_config import LNEATConfig
from polyneat.core.generation_statistics import GenerationStatistics
from polyneat.core.neat.neat_algorithm import NEATAlgorithm
from polyneat.core.population import Population

_CPU = torch.device("cpu")


def _make_config(learning_interval_generations: int = 2) -> LNEATConfig:
    return LNEATConfig(
        population_size=12,
        number_of_input_nodes=1,
        number_of_output_nodes=1,
        number_of_class_labels=2,
        learning_interval_generations=learning_interval_generations,
        number_of_learning_samples=4,
        backpropagation_iterations_per_session=3,
        random_seed=7,
    )


def _make_trainer(config: LNEATConfig) -> BackpropagationWeightTrainer:
    learning_features = torch.tensor([[0.0], [0.2], [0.8], [1.0]])
    learning_targets = torch.tensor([[0.0], [0.0], [1.0], [1.0]])
    return BackpropagationWeightTrainer(
        classification_features=torch.tensor([[0.0], [0.2], [0.45], [0.55], [0.8], [1.0]]),
        classification_binary_targets=torch.tensor(
            [[0.0], [0.0], [0.0], [1.0], [1.0], [1.0]]
        ),
        learning_sample_features=learning_features,
        learning_sample_binary_targets=learning_targets,
        learning_rate=config.backpropagation_learning_rate,
        number_of_iterations=config.backpropagation_iterations_per_session,
        training_indicator=config.training_indicator,
        classification_threshold=config.classification_threshold,
        device_for_computation=_CPU,
    )


def _advance(
    algorithm: LNEATAlgorithm, population: Population, rng: Generator
) -> tuple[Population, GenerationStatistics]:
    fitnesses = [float(index) for index in range(population.size())]
    return algorithm.advance_one_generation(population, fitnesses, rng)


def test_lneat_is_a_neat_algorithm() -> None:
    assert issubclass(LNEATAlgorithm, NEATAlgorithm)


def test_inherits_neat_genetics_and_carries_lneat_config() -> None:
    config = _make_config()
    algorithm = LNEATAlgorithm.from_config(config)
    assert isinstance(algorithm, LNEATAlgorithm)
    assert algorithm.config is config
    assert algorithm.backpropagation_trainer is None


def test_learning_fires_only_on_interval_generations() -> None:
    config = _make_config(learning_interval_generations=2)
    algorithm = LNEATAlgorithm.from_config(config)
    algorithm.backpropagation_trainer = _make_trainer(config)
    rng = np.random.default_rng(7)

    population = algorithm.create_initial_population(rng)  # generation 0
    population, statistics = _advance(algorithm, population, rng)  # -> generation 1
    assert "number_of_genomes_backpropagated" not in statistics.extra_metrics

    population, statistics = _advance(algorithm, population, rng)  # -> generation 2
    assert "number_of_genomes_backpropagated" in statistics.extra_metrics
    assert statistics.extra_metrics["number_of_genomes_backpropagated"] >= 0.0


def test_learning_preserves_population_size_and_species_assignments() -> None:
    config = _make_config(learning_interval_generations=1)
    algorithm = LNEATAlgorithm.from_config(config)
    algorithm.backpropagation_trainer = _make_trainer(config)
    rng = np.random.default_rng(7)

    population = algorithm.create_initial_population(rng)
    next_population, _statistics = _advance(algorithm, population, rng)
    assert next_population.size() == config.population_size
    assert next_population.species_assignments is not None
    assert len(next_population.species_assignments) == config.population_size
    assert next_population.generation_number == 1


def test_without_trainer_behaves_like_plain_neat() -> None:
    config = _make_config(learning_interval_generations=1)
    algorithm = LNEATAlgorithm.from_config(config)
    rng = np.random.default_rng(7)

    population = algorithm.create_initial_population(rng)
    _next_population, statistics = _advance(algorithm, population, rng)
    assert "number_of_genomes_backpropagated" not in statistics.extra_metrics


def test_learning_generation_reports_trained_genome_count() -> None:
    # fresh random genomes cannot all be Type 1 on this separable task, so on a
    # learning generation the counter must be positive and the trained genomes
    # must differ from an identical run without a trainer (same seed).
    config = _make_config(learning_interval_generations=1)
    rng_with_learning = np.random.default_rng(7)
    algorithm_with_learning = LNEATAlgorithm.from_config(config)
    algorithm_with_learning.backpropagation_trainer = _make_trainer(config)
    population = algorithm_with_learning.create_initial_population(rng_with_learning)
    trained_population, statistics = _advance(
        algorithm_with_learning, population, rng_with_learning
    )
    assert statistics.extra_metrics["number_of_genomes_backpropagated"] > 0

    rng_without_learning = np.random.default_rng(7)
    algorithm_without_learning = LNEATAlgorithm.from_config(_make_config(1))
    population = algorithm_without_learning.create_initial_population(rng_without_learning)
    untrained_population, _statistics = _advance(
        algorithm_without_learning, population, rng_without_learning
    )
    trained_weights = [
        gene.weight for genome in trained_population.genomes
        for gene in genome.connection_genes  # type: ignore[attr-defined]
    ]
    untrained_weights = [
        gene.weight for genome in untrained_population.genomes
        for gene in genome.connection_genes  # type: ignore[attr-defined]
    ]
    assert trained_weights != untrained_weights
