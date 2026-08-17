"""Runner scaffolding and per-algorithm runs, all on CPU with tiny budgets."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from gpu_sweep.algorithm_runners import (
    CellOutcome,
    move_dataset_to_device,
    phenotype_output_device_name,
    run_algorithm_on_dataset,
    summarize_generation_history,
)
from gpu_sweep.dataset_catalog import TabularDataset
from polyneat.core.generation_statistics import GenerationStatistics


def build_tiny_dataset(number_of_classes: int = 2, number_of_features: int = 4) -> TabularDataset:
    """Deterministic, linearly separable toy data - enough to drive a run."""
    generator = np.random.default_rng(0)
    rows_per_class = 12
    feature_rows = []
    label_rows = []
    for class_index in range(number_of_classes):
        block = generator.normal(
            loc=float(class_index), scale=0.2, size=(rows_per_class, number_of_features)
        )
        feature_rows.append(block)
        label_rows.extend([class_index] * rows_per_class)
    features = torch.tensor(np.vstack(feature_rows), dtype=torch.float32)
    labels = torch.tensor(label_rows, dtype=torch.long)
    train_positions = torch.arange(0, len(labels), 2)
    test_positions = torch.arange(1, len(labels), 2)
    return TabularDataset(
        dataset_key="tiny",
        train_features=features[train_positions],
        train_labels=labels[train_positions],
        test_features=features[test_positions],
        test_labels=labels[test_positions],
        number_of_classes=number_of_classes,
    )


def _statistics(generation_number: int, best_fitness: float) -> GenerationStatistics:
    return GenerationStatistics(
        generation_number=generation_number,
        best_fitness=best_fitness,
        mean_fitness=best_fitness / 2,
        median_fitness=best_fitness / 2,
        number_of_species=1,
        number_of_genomes_evaluated=10,
        elapsed_seconds=0.1,
    )


def test_summarize_generation_history_reports_first_and_last_best_fitness() -> None:
    history = [_statistics(0, 0.30), _statistics(1, 0.42), _statistics(2, 0.55)]

    outcome = summarize_generation_history(
        history,
        runtime_seconds=1.5,
        metric_values={"test_accuracy": 0.8},
        phenotype_output_device="cpu",
    )

    assert outcome.generations_completed == 3
    assert outcome.first_generation_best_fitness == pytest.approx(0.30)
    assert outcome.last_generation_best_fitness == pytest.approx(0.55)
    assert outcome.metric_values["test_accuracy"] == pytest.approx(0.8)


def test_summarize_generation_history_tolerates_an_empty_history() -> None:
    outcome = summarize_generation_history(
        [], runtime_seconds=0.0, metric_values={}, phenotype_output_device="cpu"
    )

    assert outcome.generations_completed == 0
    assert np.isnan(outcome.first_generation_best_fitness)


def test_move_dataset_to_device_keeps_shapes_and_classes() -> None:
    dataset = build_tiny_dataset()

    moved = move_dataset_to_device(dataset, torch.device("cpu"))

    assert moved.train_features.shape == dataset.train_features.shape
    assert moved.number_of_classes == dataset.number_of_classes


def test_phenotype_output_device_name_reports_the_forward_pass_device() -> None:
    class ConstantPhenotype:
        def forward_pass(self, input_tensor: torch.Tensor) -> torch.Tensor:
            return torch.zeros((input_tensor.shape[0], 2))

    name = phenotype_output_device_name(ConstantPhenotype(), torch.zeros((3, 4)))

    assert name == "cpu"


def test_run_algorithm_on_dataset_rejects_an_unknown_algorithm() -> None:
    with pytest.raises(ValueError, match="unknown algorithm"):
        run_algorithm_on_dataset(
            "not_an_algorithm",
            build_tiny_dataset(),
            device=torch.device("cpu"),
            population_size=6,
            number_of_generations=1,
            random_seed=0,
        )


def assert_outcome_is_well_formed(outcome: CellOutcome, expected_metrics: tuple[str, ...]) -> None:
    """Shared assertions every per-algorithm test in this file reuses."""
    assert outcome.generations_completed >= 1
    assert outcome.runtime_seconds >= 0.0
    assert outcome.phenotype_output_device.startswith("cpu")
    for metric_name in expected_metrics:
        assert 0.0 <= outcome.metric_values[metric_name] <= 1.0


@pytest.mark.parametrize("algorithm_name", ["neat", "fsneat", "neatdbm"])
def test_softmax_family_runners_produce_a_well_formed_outcome(algorithm_name: str) -> None:
    outcome = run_algorithm_on_dataset(
        algorithm_name,
        build_tiny_dataset(number_of_classes=3),
        device=torch.device("cpu"),
        population_size=8,
        number_of_generations=2,
        random_seed=0,
    )

    assert_outcome_is_well_formed(outcome, ("train_accuracy", "test_accuracy"))
    assert not np.isnan(outcome.first_generation_best_fitness)


@pytest.mark.parametrize("algorithm_name", ["cneat", "lneat"])
def test_recognizer_family_runners_produce_a_well_formed_outcome(
    algorithm_name: str,
) -> None:
    outcome = run_algorithm_on_dataset(
        algorithm_name,
        build_tiny_dataset(number_of_classes=3),
        device=torch.device("cpu"),
        population_size=8,
        number_of_generations=2,
        random_seed=0,
    )

    assert_outcome_is_well_formed(outcome, ("train_accuracy", "test_accuracy"))


def test_lneat_learning_subset_never_asks_for_more_samples_than_a_class_has() -> None:
    from gpu_sweep.algorithm_runners import draw_stratified_learning_positions

    train_labels = torch.tensor([0, 0, 0, 0, 1], dtype=torch.long)

    positions = draw_stratified_learning_positions(
        train_labels,
        number_of_class_labels=2,
        number_of_learning_samples=8,
        rng=np.random.default_rng(0),
    )

    assert len(positions) == 5
    assert sorted(positions.tolist()) == [0, 1, 2, 3, 4]


def test_exact_runner_produces_a_well_formed_outcome() -> None:
    outcome = run_algorithm_on_dataset(
        "exact",
        build_tiny_dataset(number_of_classes=2, number_of_features=8),
        device=torch.device("cpu"),
        population_size=4,
        number_of_generations=1,
        random_seed=0,
    )

    assert_outcome_is_well_formed(
        outcome, ("train_accuracy", "generalizability_accuracy", "test_accuracy")
    )


def test_hyperneat_runner_produces_a_well_formed_outcome() -> None:
    outcome = run_algorithm_on_dataset(
        "hyperneat",
        build_tiny_dataset(number_of_classes=3),
        device=torch.device("cpu"),
        population_size=8,
        number_of_generations=2,
        random_seed=0,
    )

    assert_outcome_is_well_formed(outcome, ("train_accuracy", "test_accuracy"))
