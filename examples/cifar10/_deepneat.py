"""Shared CIFAR-10 DeepNEAT experiment runner."""

from __future__ import annotations

import time
from pathlib import Path

import torch

import polyneat as pn
from examples._datasets import (
    split_indices_into_train_and_test,
)
from examples._experiment import ExperimentReport
from examples.cifar10.dataset import load_cifar10

_SUBSET_SAMPLING_SEED = 12345
_CHANNELS = 3
_IMAGE_SIDE = 32
_PAPER_TEST_ERROR = 0.089


def run_cifar10_deepneat_experiment(
    *,
    config_file_path: Path,
    tensorboard_run_label: str,
    device: torch.device | None = None,
    random_seed: int | None = None,
    artifacts_directory: Path | None = None,
) -> ExperimentReport:
    """Run one configured DeepNEAT search on the official CIFAR-10 split."""
    experiment_started_at = time.perf_counter()
    config = pn.DeepNEATConfig.load_from_yaml_file(config_file_path)
    dataset = load_cifar10(
        random_seed=_SUBSET_SAMPLING_SEED,
        max_train_samples=config.maximum_training_samples,
        max_test_samples=config.maximum_test_samples,
        standardize=False,
    )
    print(
        f"CIFAR-10 loaded: {dataset.train_features.shape[0]} official train / "
        f"{dataset.test_features.shape[0]} official test samples."
    )
    print(
        f"Evolution budget: {config.population_size} candidates x "
        f"{config.number_of_generations} generations = "
        f"{config.population_size * config.number_of_generations} fitness evaluations."
    )

    config.number_of_classes = dataset.number_of_classes
    config.number_of_output_nodes = dataset.number_of_classes
    algorithm = pn.DeepNEATAlgorithm.from_config(
        config, device_for_phenotype_computation=device
    )
    resolved_device = device or torch.device(config.device_for_phenotype_evaluation)
    resolved_random_seed = config.random_seed if random_seed is None else random_seed
    torch.manual_seed(resolved_random_seed)

    train_indices, validation_indices = split_indices_into_train_and_test(
        number_of_samples=dataset.train_features.shape[0],
        train_fraction=1.0 - config.validation_fraction,
        random_seed=_SUBSET_SAMPLING_SEED,
    )
    official_train_images = dataset.train_features.reshape(
        -1, _CHANNELS, _IMAGE_SIDE, _IMAGE_SIDE
    )
    train_features = official_train_images[train_indices]
    train_labels = dataset.train_labels[train_indices]
    validation_features = official_train_images[validation_indices]
    validation_labels = dataset.train_labels[validation_indices]
    test_features = dataset.test_features.reshape(-1, _CHANNELS, _IMAGE_SIDE, _IMAGE_SIDE)
    test_labels = dataset.test_labels
    fitness_evaluator = pn.TrainedNetworkAccuracyEvaluator(
        train_features=train_features,
        train_labels=train_labels,
        validation_features=validation_features,
        validation_labels=validation_labels,
        number_of_epochs=config.training_epochs_per_evaluation,
        learning_rate=config.training_learning_rate,
        batch_size=config.training_batch_size,
        device_for_computation=resolved_device,
        base_random_seed=resolved_random_seed,
        use_deterministic_algorithms=config.use_deterministic_training_algorithms,
    )
    callbacks: list = [pn.ConsoleStatisticsLogger()]
    if artifacts_directory is not None:
        callbacks.extend(
            [
                pn.BestGenomePersister(output_directory=artifacts_directory),
                pn.TensorBoardLogger(
                    log_directory=artifacts_directory / "tensorboard",
                    run_label=tensorboard_run_label,
                ),
            ]
        )

    result = pn.EvolutionRunner(
        algorithm=algorithm,
        fitness_evaluator=fitness_evaluator,
        termination_criterion=pn.MaxGenerationsTermination(
            max_generations=config.number_of_generations - 1
        ),
        callbacks=callbacks,
        random_seed=resolved_random_seed,
    ).run_evolution()

    best_genome = result.best_genome_ever_found
    final_phenotype = algorithm.phenotype_decoder.build_phenotype_from_genome(best_genome)
    final_training_started_at = time.perf_counter()
    final_evaluator = pn.TrainedNetworkAccuracyEvaluator(
        train_features=torch.cat([train_features, validation_features]),
        train_labels=torch.cat([train_labels, validation_labels]),
        validation_features=test_features,
        validation_labels=test_labels,
        number_of_epochs=config.final_training_epochs,
        learning_rate=config.training_learning_rate,
        batch_size=config.training_batch_size,
        device_for_computation=resolved_device,
        base_random_seed=resolved_random_seed,
        use_deterministic_algorithms=config.use_deterministic_training_algorithms,
    )
    (test_accuracy,) = final_evaluator.evaluate_batch_of_phenotypes([final_phenotype])
    final_training_runtime_seconds = time.perf_counter() - final_training_started_at
    test_error = 1.0 - float(test_accuracy)
    return ExperimentReport(
        metric_values={
            "validation_accuracy": float(result.best_fitness_ever_achieved),
            "test_accuracy": float(test_accuracy),
            "test_error": test_error,
            "paper_reference_test_error": _PAPER_TEST_ERROR,
            "test_error_gap_to_paper_percentage_points": 100.0
            * (test_error - _PAPER_TEST_ERROR),
            "evolution_runtime_seconds": float(result.total_runtime_seconds),
            "final_training_runtime_seconds": final_training_runtime_seconds,
            "number_of_layers": float(final_phenotype.number_of_layer_modules),
            "total_parameter_count": float(final_phenotype.total_parameter_count),
        },
        number_of_generations=len(result.full_generation_history),
        runtime_seconds=time.perf_counter() - experiment_started_at,
    )
