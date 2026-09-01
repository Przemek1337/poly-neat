"""Accuracy and macro-F1, checked against hand-computed confusion matrices."""

from __future__ import annotations

import pytest
import torch

from gpu_sweep.metrics import (
    classification_metrics,
    evaluate_phenotype_metrics,
    per_class_f1_scores,
    predict_class_indices,
)


def test_predict_class_indices_takes_the_argmax_of_the_forward_pass() -> None:
    class TwoColumnPhenotype:
        def forward_pass(self, input_tensor: torch.Tensor) -> torch.Tensor:
            return torch.tensor([[0.1, 0.9], [0.8, 0.2]])

    predicted = predict_class_indices(TwoColumnPhenotype(), torch.zeros((2, 3)))

    assert predicted.tolist() == [1, 0]


def test_per_class_f1_scores_match_a_hand_computed_binary_confusion() -> None:
    # class 0: tp=1 fp=0 fn=1 -> precision 1.0, recall 0.5, f1 2/3
    # class 1: tp=2 fp=1 fn=0 -> precision 2/3, recall 1.0, f1 0.8
    target_labels = torch.tensor([0, 0, 1, 1])
    predicted_labels = torch.tensor([0, 1, 1, 1])

    scores = per_class_f1_scores(predicted_labels, target_labels, number_of_classes=2)

    assert scores[0] == pytest.approx(2 / 3)
    assert scores[1] == pytest.approx(0.8)


def test_per_class_f1_is_one_for_a_perfect_prediction() -> None:
    labels = torch.tensor([0, 1, 2, 2])

    scores = per_class_f1_scores(labels, labels, number_of_classes=3)

    assert scores == [1.0, 1.0, 1.0]


def test_per_class_f1_is_zero_for_a_class_the_model_never_predicts() -> None:
    target_labels = torch.tensor([0, 0, 1])
    predicted_labels = torch.tensor([0, 0, 0])

    scores = per_class_f1_scores(predicted_labels, target_labels, number_of_classes=2)

    assert scores[1] == 0.0


def test_classification_metrics_reports_accuracy_and_the_macro_average() -> None:
    target_labels = torch.tensor([0, 0, 1, 1])
    predicted_labels = torch.tensor([0, 1, 1, 1])

    metrics = classification_metrics(predicted_labels, target_labels, number_of_classes=2)

    assert metrics["accuracy"] == pytest.approx(0.75)
    assert metrics["macro_f1"] == pytest.approx((2 / 3 + 0.8) / 2)
    assert metrics["per_class_f1"] == pytest.approx([2 / 3, 0.8])


def test_evaluate_phenotype_metrics_runs_the_phenotype_then_scores_it() -> None:
    class AlwaysClassZeroPhenotype:
        def forward_pass(self, input_tensor: torch.Tensor) -> torch.Tensor:
            return torch.tensor([[1.0, 0.0]] * input_tensor.shape[0])

    metrics = evaluate_phenotype_metrics(
        AlwaysClassZeroPhenotype(),
        torch.zeros((4, 3)),
        torch.tensor([0, 0, 1, 1]),
        number_of_classes=2,
    )

    assert metrics["accuracy"] == pytest.approx(0.5)
    assert metrics["per_class_f1"][1] == 0.0
