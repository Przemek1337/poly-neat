from __future__ import annotations

import torch

from polyneat.core.component_protocols import Phenotype
from polyneat.core.type_aliases import FitnessValue
from polyneat.evaluators.sequential_evaluator_base import SequentialFitnessEvaluator


class SoftmaxLikelihoodFitnessEvaluator(SequentialFitnessEvaluator):
    """Smooth multi-class fitness: mean softmax probability of the correct class.

    Phenotype outputs are treated as class logits and passed through a softmax;
    fitness is the average probability assigned to the true class, in ``(0, 1)``.

    Unlike raw accuracy, this signal is continuous: a network is rewarded for
    increasing the correct class's probability even before its argmax flips.
    That denser gradient gives selection something to climb every generation,
    which matters for hard, many-class problems where accuracy alone leaves most
    genomes tied. For best results the output nodes should use a linear
    (``identity``) activation so the softmax receives true logits.
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
                "SoftmaxLikelihoodFitnessEvaluator: input_features must be 2-D "
                f"(num_samples, num_features), got shape {tuple(input_features.shape)}"
            )
        if input_features.shape[0] != target_labels.shape[0]:
            raise ValueError(
                "SoftmaxLikelihoodFitnessEvaluator: input_features has "
                f"{input_features.shape[0]} samples but target_labels has "
                f"{target_labels.shape[0]}"
            )
        self._input_features = input_features.to(torch.float32)
        self._target_labels = target_labels.to(torch.long).cpu()
        self._sample_indices = torch.arange(self._target_labels.shape[0])

    def evaluate_single_phenotype(self, phenotype: Phenotype) -> FitnessValue:
        """Return the mean correct-class softmax probability of a phenotype.

        Args:
            phenotype: Phenotype to evaluate.

        Returns:
            The average probability assigned to the true class, in ``(0, 1)``.
        """
        with torch.no_grad():
            output_logits = phenotype.forward_pass(self._input_features).cpu()
            class_probabilities = torch.softmax(output_logits, dim=1)
            correct_class_probabilities = class_probabilities[
                self._sample_indices, self._target_labels
            ]
        return float(correct_class_probabilities.mean().item())
