from __future__ import annotations

import pytest
import torch

from polyneat.evaluators.binary_recognizer_evaluator import BinaryRecognizerFitnessEvaluator


class _ConstantOutputPhenotype:
    """Phenotype stub returning a fixed column of outputs."""

    def __init__(self, outputs_per_sample: list[float]) -> None:
        self._outputs_per_sample = outputs_per_sample

    def forward_pass(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return torch.tensor(self._outputs_per_sample).unsqueeze(1)

    def reset_recurrent_state(self) -> None:
        return None


_FEATURES = torch.zeros((4, 2))
_LABELS = torch.tensor([0, 0, 1, 2])  # class 0 has samples 0 and 1


def test_perfect_recognizer_scores_one() -> None:
    evaluator = BinaryRecognizerFitnessEvaluator(
        input_features=_FEATURES,
        class_label_indices=_LABELS,
        target_class_label_index=0,
    )
    perfect = _ConstantOutputPhenotype([1.0, 1.0, 0.0, 0.0])
    assert evaluator.evaluate_single_phenotype(perfect) == pytest.approx(1.0)


def test_worst_recognizer_scores_zero() -> None:
    evaluator = BinaryRecognizerFitnessEvaluator(
        input_features=_FEATURES,
        class_label_indices=_LABELS,
        target_class_label_index=0,
    )
    inverted = _ConstantOutputPhenotype([0.0, 0.0, 1.0, 1.0])
    assert evaluator.evaluate_single_phenotype(inverted) == pytest.approx(0.0)


def test_fitness_is_paper_equation_one() -> None:
    # F = (P / (P + dis))**2 * (c / P): outputs [0.9, 0.4, 0.2, 0.2],
    # targets [1, 1, 0, 0], threshold 0.5 -> predictions [1, 0, 0, 0], c = 3;
    # distances [0.1, 0.6, 0.2, 0.2] -> dis = 1.1;
    # F = (4 / 5.1)**2 * (3 / 4)
    evaluator = BinaryRecognizerFitnessEvaluator(
        input_features=_FEATURES,
        class_label_indices=_LABELS,
        target_class_label_index=0,
    )
    phenotype = _ConstantOutputPhenotype([0.9, 0.4, 0.2, 0.2])
    assert evaluator.evaluate_single_phenotype(phenotype) == pytest.approx(
        (4.0 / 5.1) ** 2 * 0.75
    )


def test_batch_evaluation_preserves_order() -> None:
    evaluator = BinaryRecognizerFitnessEvaluator(
        input_features=_FEATURES,
        class_label_indices=_LABELS,
        target_class_label_index=0,
    )
    perfect = _ConstantOutputPhenotype([1.0, 1.0, 0.0, 0.0])
    inverted = _ConstantOutputPhenotype([0.0, 0.0, 1.0, 1.0])
    fitnesses = evaluator.evaluate_batch_of_phenotypes([perfect, inverted])
    assert fitnesses[0] > fitnesses[1]


def test_rejects_target_class_without_samples() -> None:
    with pytest.raises(ValueError):
        BinaryRecognizerFitnessEvaluator(
            input_features=_FEATURES,
            class_label_indices=_LABELS,
            target_class_label_index=5,
        )


def test_rejects_mismatched_sample_counts() -> None:
    with pytest.raises(ValueError):
        BinaryRecognizerFitnessEvaluator(
            input_features=torch.zeros((3, 2)),
            class_label_indices=_LABELS,
            target_class_label_index=0,
        )


def test_rejects_non_two_dimensional_features() -> None:
    with pytest.raises(ValueError):
        BinaryRecognizerFitnessEvaluator(
            input_features=torch.zeros(4),
            class_label_indices=_LABELS,
            target_class_label_index=0,
        )
