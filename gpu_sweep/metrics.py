"""Classification metrics for the sweep: accuracy and macro-averaged F1.

Accuracy alone is a poor summary on these datasets - several are strongly
skewed, so a model that answers with the majority class scores well while
learning nothing. Macro-F1 averages the per-class F1 without weighting by
class frequency, which makes ignoring a small class expensive.

Written here rather than taken from a library because the sweep adds no
dependencies, and because ``polyneat``'s own evaluators report accuracy only.
"""

from __future__ import annotations

import torch


def predict_class_indices(phenotype: object, features: torch.Tensor) -> torch.Tensor:
    """Return the predicted class index of every row.

    Every phenotype in this sweep - single networks and the C-NEAT and L-NEAT
    argmax ensembles alike - emits one score column per class, so the argmax is
    the prediction in all cases. This mirrors what
    ``ClassificationAccuracyEvaluator`` does inside ``polyneat``.

    Args:
        phenotype: Anything exposing ``forward_pass``.
        features: ``[n_samples, n_features]`` float tensor.

    Returns:
        ``[n_samples]`` long tensor of class indices, on the CPU.
    """
    with torch.no_grad():
        output_scores = phenotype.forward_pass(features)
    return output_scores.argmax(dim=1).cpu()


def per_class_f1_scores(
    predicted_labels: torch.Tensor,
    target_labels: torch.Tensor,
    number_of_classes: int,
) -> list[float]:
    """F1 of every class, in class-index order.

    Degenerate cases are resolved to zero rather than left undefined: a class
    the model never predicts has precision 0, a class absent from the targets
    has recall 0, and a class with neither has F1 0. The stratified split
    guarantees every class appears in both halves, so only the first case
    arises in practice.

    Args:
        predicted_labels: ``[n_samples]`` predicted class indices.
        target_labels: ``[n_samples]`` true class indices.
        number_of_classes: Number of classes to score.

    Returns:
        One F1 value per class, each in ``[0, 1]``.
    """
    predicted_labels = predicted_labels.cpu()
    target_labels = target_labels.cpu()
    f1_scores: list[float] = []
    for class_index in range(number_of_classes):
        predicted_positive = predicted_labels == class_index
        actually_positive = target_labels == class_index
        true_positive_count = int((predicted_positive & actually_positive).sum().item())
        false_positive_count = int((predicted_positive & ~actually_positive).sum().item())
        false_negative_count = int((~predicted_positive & actually_positive).sum().item())

        predicted_positive_count = true_positive_count + false_positive_count
        actual_positive_count = true_positive_count + false_negative_count
        precision = (
            true_positive_count / predicted_positive_count if predicted_positive_count else 0.0
        )
        recall = true_positive_count / actual_positive_count if actual_positive_count else 0.0
        if precision + recall == 0.0:
            f1_scores.append(0.0)
        else:
            f1_scores.append(2.0 * precision * recall / (precision + recall))
    return f1_scores


def classification_metrics(
    predicted_labels: torch.Tensor,
    target_labels: torch.Tensor,
    number_of_classes: int,
) -> dict[str, object]:
    """Bundle accuracy, macro-F1 and the per-class F1 vector.

    Args:
        predicted_labels: ``[n_samples]`` predicted class indices.
        target_labels: ``[n_samples]`` true class indices.
        number_of_classes: Number of classes to score.

    Returns:
        ``{"accuracy": float, "macro_f1": float, "per_class_f1": list[float]}``.
    """
    predicted_labels = predicted_labels.cpu()
    target_labels = target_labels.cpu()
    number_of_samples = int(target_labels.shape[0])
    number_correct = int((predicted_labels == target_labels).sum().item())
    f1_scores = per_class_f1_scores(predicted_labels, target_labels, number_of_classes)
    return {
        "accuracy": number_correct / number_of_samples if number_of_samples else 0.0,
        "macro_f1": sum(f1_scores) / len(f1_scores) if f1_scores else 0.0,
        "per_class_f1": f1_scores,
    }


def evaluate_phenotype_metrics(
    phenotype: object,
    features: torch.Tensor,
    target_labels: torch.Tensor,
    number_of_classes: int,
) -> dict[str, object]:
    """Run one phenotype over ``features`` and score its predictions.

    Args:
        phenotype: Anything exposing ``forward_pass``.
        features: ``[n_samples, n_features]`` float tensor.
        target_labels: ``[n_samples]`` true class indices.
        number_of_classes: Number of classes to score.

    Returns:
        The dict described by :func:`classification_metrics`.
    """
    predicted_labels = predict_class_indices(phenotype, features)
    return classification_metrics(predicted_labels, target_labels, number_of_classes)
