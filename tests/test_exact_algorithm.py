from __future__ import annotations

import numpy as np
import torch

from polyneat.algorithms.exact.exact_algorithm import EXACTAlgorithm
from polyneat.algorithms.exact.exact_backpropagation_trainer import (
    EXACTBackpropagationTrainer,
)
from polyneat.algorithms.exact.exact_crossover import EXACTCrossover
from polyneat.algorithms.exact.exact_genome import EXACTGenome
from polyneat.algorithms.exact.exact_phenotype_decoder import EXACTPhenotypeDecoder
from polyneat.algorithms.exact.mutations.exact_composite_mutation import (
    EXACTCompositeMutation,
)
from polyneat.algorithms.exact.single_species_speciator import SingleSpeciesSpeciator
from polyneat.configs.exact.exact_config import EXACTConfig

_CPU = torch.device("cpu")


def _small_config() -> EXACTConfig:
    return EXACTConfig(
        population_size=6,
        number_of_output_nodes=2,
        input_image_height=2,
        input_image_width=2,
        number_of_training_epochs_per_genome=1,
        training_batch_size=4,
        random_seed=0,
    )


def _trainer(config: EXACTConfig) -> EXACTBackpropagationTrainer:
    torch.manual_seed(0)
    features = torch.randn(12, 4)
    labels = torch.arange(12, dtype=torch.long) % 2
    return EXACTBackpropagationTrainer.from_config(
        config, features, labels, device_for_computation=_CPU
    )


def test_from_config_wires_exact_components() -> None:
    algorithm = EXACTAlgorithm.from_config(_small_config())
    assert isinstance(algorithm.mutation, EXACTCompositeMutation)
    assert isinstance(algorithm.crossover, EXACTCrossover)
    assert isinstance(algorithm.speciator, SingleSpeciesSpeciator)
    assert isinstance(algorithm.phenotype_decoder, EXACTPhenotypeDecoder)
    assert algorithm.backpropagation_trainer is None


def test_initial_population_is_minimal_and_trained_when_trainer_attached() -> None:
    config = _small_config()
    algorithm = EXACTAlgorithm.from_config(config)
    algorithm.backpropagation_trainer = _trainer(config)
    population = algorithm.create_initial_population(np.random.default_rng(0))
    assert population.size() == config.population_size
    genomes = population.genomes
    assert all(isinstance(genome, EXACTGenome) for genome in genomes)
    assert all(genome.is_trained for genome in genomes)
    # Identical minimal genomes are trained once and shared, not retrained.
    assert all(genome is genomes[0] for genome in genomes)


def test_advance_one_generation_trains_every_untrained_offspring() -> None:
    config = _small_config()
    algorithm = EXACTAlgorithm.from_config(config)
    algorithm.backpropagation_trainer = _trainer(config)
    rng = np.random.default_rng(0)
    population = algorithm.create_initial_population(rng)
    fitnesses = [0.5] * population.size()
    next_population, statistics = algorithm.advance_one_generation(
        population, fitnesses, rng
    )
    assert next_population.size() == config.population_size
    assert all(
        isinstance(genome, EXACTGenome) and genome.is_trained
        for genome in next_population.genomes
    )
    assert "number_of_genomes_trained" in statistics.extra_metrics


def test_without_trainer_offspring_stay_untrained() -> None:
    config = _small_config()
    algorithm = EXACTAlgorithm.from_config(config)
    rng = np.random.default_rng(0)
    population = algorithm.create_initial_population(rng)
    fitnesses = [0.5] * population.size()
    next_population, statistics = algorithm.advance_one_generation(
        population, fitnesses, rng
    )
    assert "number_of_genomes_trained" not in statistics.extra_metrics
    assert any(not genome.is_trained for genome in next_population.genomes)


def test_sho_assigns_hyperparameters_to_offspring() -> None:
    config = EXACTConfig(
        population_size=6,
        number_of_output_nodes=2,
        input_image_height=4,
        input_image_width=4,
        number_of_training_epochs_per_genome=1,
        training_batch_size=4,
        random_seed=0,
        use_simplex_hyperparameter_optimization=True,
    )
    algorithm = EXACTAlgorithm.from_config(config)
    rng = np.random.default_rng(0)
    population = algorithm.create_initial_population(rng)
    assert all(
        genome.training_hyperparameters is not None for genome in population.genomes
    )
    fitnesses = [float(index) for index in range(len(population.genomes))]
    next_population, _statistics = algorithm.advance_one_generation(
        population, fitnesses, rng
    )
    assert all(
        genome.training_hyperparameters is not None
        for genome in next_population.genomes
    )
