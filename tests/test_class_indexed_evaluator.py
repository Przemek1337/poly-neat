from __future__ import annotations

import pytest
import torch

from polyneat.evaluators.class_indexed_evaluator_base import ClassIndexedFitnessEvaluator


class _StubPhenotype:
    def forward_pass(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return torch.zeros((input_tensor.shape[0], 1))

    def reset_recurrent_state(self) -> None:
        pass


class _RecordingEvaluator(ClassIndexedFitnessEvaluator):
    def __init__(self, number_of_class_labels: int) -> None:
        super().__init__(number_of_class_labels)
        self.observed_class_assignments: list[int] = []

    def evaluate_phenotype_for_class(self, phenotype, class_label_index: int) -> float:
        self.observed_class_assignments.append(class_label_index)
        return float(class_label_index)


def test_rejects_fewer_than_two_class_labels() -> None:
    with pytest.raises(ValueError):
        _RecordingEvaluator(number_of_class_labels=1)


def test_base_class_requires_override() -> None:
    evaluator = ClassIndexedFitnessEvaluator(number_of_class_labels=2)
    with pytest.raises(NotImplementedError):
        evaluator.evaluate_phenotype_for_class(_StubPhenotype(), 0)


def test_batch_assigns_classes_by_index_modulo() -> None:
    evaluator = _RecordingEvaluator(number_of_class_labels=3)
    phenotypes = [_StubPhenotype() for _ in range(7)]

    fitnesses = evaluator.evaluate_batch_of_phenotypes(phenotypes)

    assert evaluator.observed_class_assignments == [0, 1, 2, 0, 1, 2, 0]
    assert fitnesses == [0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0]
