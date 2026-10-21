from __future__ import annotations

import torch

from polyneat.core.component_protocols import Phenotype
from polyneat.core.type_aliases import FitnessValue
from polyneat.evaluators.sequential_evaluator_base import SequentialFitnessEvaluator


class ClassificationAccuracyEvaluator(SequentialFitnessEvaluator):
    """Scores phenotypes by multi-class classification accuracy.

    Each phenotype is run once over a fixed feature matrix; the predicted class
    for a sample is the index of its largest output activation. Fitness is the
    fraction of samples classified correctly, ranging from 0.0 to 1.0.

    The evaluator is dataset-agnostic: supply any ``(num_samples, num_features)``
    feature matrix and matching integer labels. The number of output nodes in
    the evolved networks must equal the number of classes.
    """

    def __init__(self, input_features: torch.Tensor, target_labels: torch.Tensor) -> None:
        """Store the features and labels used to score every phenotype.

        Args:
            input_features: Float tensor of shape ``(num_samples, num_features)``.
            target_labels: Integer tensor of shape ``(num_samples,)`` holding the
                class index of each sample.

        Raises:
            ValueError: If ``input_features`` is not 2-D or its sample count does
                not match ``target_labels``.
        """
        if input_features.dim() != 2:
            raise ValueError(
                "ClassificationAccuracyEvaluator: input_features must be 2-D "
                f"(num_samples, num_features), got shape {tuple(input_features.shape)}"
            )
        if input_features.shape[0] != target_labels.shape[0]:
            raise ValueError(
                "ClassificationAccuracyEvaluator: input_features has "
                f"{input_features.shape[0]} samples but target_labels has "
                f"{target_labels.shape[0]}"
            )
        self._input_features = input_features.to(torch.float32)
        self._target_labels = target_labels.to(torch.long).cpu()

    def evaluate_single_phenotype(self, phenotype: Phenotype) -> FitnessValue:
        """Return the classification accuracy of a phenotype over the stored data.

        Args:
            phenotype: Phenotype to evaluate.

        Returns:
            The fraction of samples whose argmax output matches the label, in
            ``[0.0, 1.0]``.
        """
        with torch.no_grad():
            output_scores = phenotype.forward_pass(self._input_features)
        predicted_labels = output_scores.argmax(dim=1).cpu()
        number_correct = int((predicted_labels == self._target_labels).sum().item())
        return number_correct / self._target_labels.shape[0]
