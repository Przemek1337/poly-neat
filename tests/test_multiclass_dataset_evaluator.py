from __future__ import annotations

import pytest
import torch

from polyneat.evaluators.multiclass_dataset_evaluator import MulticlassDatasetFitnessEvaluator

_FEATURES = torch.tensor([[0.0], [0.25], [0.5], [1.0]])
_LABELS = torch.tensor([0, 1, 1, 2])


class _ConstantOutputPhenotype:
    def __init__(self, constant_value: float) -> None:
        self._constant_value = constant_value

    def forward_pass(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return torch.full((input_tensor.shape[0], 1), self._constant_value)

    def reset_recurrent_state(self) -> None:
        pass


class _PerfectClassZeroRecognizer:
    def forward_pass(self, input_tensor: torch.Tensor) -> torch.Tensor:
        # outputs 1.0 exactly for the class-0 sample (feature 0.0), else 0.0
        return (input_tensor[:, :1] == 0.0).float()

    def reset_recurrent_state(self) -> None:
        pass


def test_perfect_recognizer_scores_one() -> None:
    evaluator = MulticlassDatasetFitnessEvaluator(
        input_features=_FEATURES,
        class_label_indices=_LABELS,
        number_of_class_labels=3,
    )
    fitness = evaluator.evaluate_phenotype_for_class(_PerfectClassZeroRecognizer(), 0)
    assert fitness == pytest.approx(1.0)


def test_constant_half_output_scores_point_seven_five() -> None:
    evaluator = MulticlassDatasetFitnessEvaluator(
        input_features=_FEATURES,
        class_label_indices=_LABELS,
        number_of_class_labels=3,
    )
    # every squared error is 0.25 regardless of the indicator, so MSE = 0.25
    fitness = evaluator.evaluate_phenotype_for_class(_ConstantOutputPhenotype(0.5), 1)
    assert fitness == pytest.approx(0.75)


def test_rejects_non_two_dimensional_features() -> None:
    with pytest.raises(ValueError):
        MulticlassDatasetFitnessEvaluator(
            input_features=torch.zeros(4),
            class_label_indices=_LABELS,
            number_of_class_labels=3,
        )


def test_rejects_mismatched_feature_and_label_lengths() -> None:
    with pytest.raises(ValueError):
        MulticlassDatasetFitnessEvaluator(
            input_features=_FEATURES,
            class_label_indices=torch.tensor([0, 1]),
            number_of_class_labels=3,
        )


def test_rejects_labels_outside_class_range() -> None:
    with pytest.raises(ValueError):
        MulticlassDatasetFitnessEvaluator(
            input_features=_FEATURES,
            class_label_indices=torch.tensor([0, 1, 1, 3]),
            number_of_class_labels=3,
        )


def test_rejects_class_with_no_samples() -> None:
    # class 2 is absent: its indicator row would be all zeros, so a degenerate
    # constant-zero network would score a perfect 1.0 and permanently occupy
    # the container cell for a class it can never win the argmax for
    with pytest.raises(ValueError):
        MulticlassDatasetFitnessEvaluator(
            input_features=_FEATURES,
            class_label_indices=torch.tensor([0, 1, 1, 0]),
            number_of_class_labels=3,
        )


def test_rejects_empty_dataset() -> None:
    with pytest.raises(ValueError):
        MulticlassDatasetFitnessEvaluator(
            input_features=torch.zeros((0, 1)),
            class_label_indices=torch.zeros((0,), dtype=torch.long),
            number_of_class_labels=3,
        )
