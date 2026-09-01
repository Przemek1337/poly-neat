"""The one-hot MSE fitness the single-network algorithms are scored by."""

from __future__ import annotations

import pytest
import torch

from gpu_sweep.fitness import OneHotMeanSquaredErrorFitnessEvaluator


class ConstantPhenotype:
    """Returns the same score row for every sample."""

    def __init__(self, output_row: list[float]) -> None:
        self._output_row = output_row

    def forward_pass(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return torch.tensor([self._output_row] * input_tensor.shape[0])


def build_evaluator() -> OneHotMeanSquaredErrorFitnessEvaluator:
    return OneHotMeanSquaredErrorFitnessEvaluator(
        input_features=torch.zeros((2, 3)),
        target_labels=torch.tensor([0, 1]),
        number_of_classes=2,
    )


def test_a_perfect_recognizer_scores_one() -> None:
    class PerfectPhenotype:
        def forward_pass(self, input_tensor: torch.Tensor) -> torch.Tensor:
            return torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    assert build_evaluator().evaluate_single_phenotype(PerfectPhenotype()) == pytest.approx(1.0)


def test_the_worst_possible_recognizer_scores_zero() -> None:
    class InvertedPhenotype:
        def forward_pass(self, input_tensor: torch.Tensor) -> torch.Tensor:
            return torch.tensor([[0.0, 1.0], [1.0, 0.0]])

    assert build_evaluator().evaluate_single_phenotype(InvertedPhenotype()) == pytest.approx(0.0)


def test_an_all_half_output_scores_one_minus_a_quarter() -> None:
    # every one of the four entries is off by 0.5, so MSE is 0.25.
    fitness = build_evaluator().evaluate_single_phenotype(ConstantPhenotype([0.5, 0.5]))

    assert fitness == pytest.approx(0.75)


def test_fitness_stays_inside_the_unit_interval() -> None:
    for output_row in ([0.0, 0.0], [1.0, 1.0], [0.3, 0.9]):
        fitness = build_evaluator().evaluate_single_phenotype(ConstantPhenotype(output_row))
        assert 0.0 <= fitness <= 1.0


def test_batch_evaluation_preserves_order() -> None:
    # A constant [0.5, 0.5] is the best any constant output can do, so the
    # better phenotype has to vary with the sample.
    class PerfectPhenotype:
        def forward_pass(self, input_tensor: torch.Tensor) -> torch.Tensor:
            return torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    evaluator = build_evaluator()

    fitnesses = evaluator.evaluate_batch_of_phenotypes(
        [ConstantPhenotype([0.5, 0.5]), PerfectPhenotype()]
    )

    assert fitnesses[0] == pytest.approx(0.75)
    assert fitnesses[1] > fitnesses[0]


def test_rejects_features_and_labels_of_different_lengths() -> None:
    with pytest.raises(ValueError, match="samples"):
        OneHotMeanSquaredErrorFitnessEvaluator(
            input_features=torch.zeros((3, 2)),
            target_labels=torch.tensor([0, 1]),
            number_of_classes=2,
        )
