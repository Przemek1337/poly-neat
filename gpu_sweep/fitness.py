"""The fitness the single-network algorithms are scored by.

Paper-default NEAT uses ``steepened_sigmoid`` on its output nodes, so a
network's outputs live in ``[0, 1]``. The natural fitness for that is the mean
squared error against a one-hot target, shifted to ``1 - MSE`` so higher is
better and the value stays non-negative - NEAT's offspring allocation clamps
negative adjusted fitness to zero.

This mirrors what the library already does elsewhere: C-NEAT scores a
single-output recognizer as ``1 - mean((indicator - output) ** 2)``
(``polyneat/evaluators/multiclass_dataset_evaluator.py:17``). That evaluator
cannot be reused here because it scores one output against one assigned class,
whereas these algorithms evolve one network with an output per class - but the
scale matches deliberately, so fitness means the same thing across all six
algorithms in the sweep.
"""

from __future__ import annotations

import torch

from polyneat.core.component_protocols import Phenotype
from polyneat.core.type_aliases import FitnessValue
from polyneat.evaluators.sequential_evaluator_base import SequentialFitnessEvaluator


class OneHotMeanSquaredErrorFitnessEvaluator(SequentialFitnessEvaluator):
    """Score a multi-output network as ``1 - MSE`` against a one-hot target.

    A network that outputs 1.0 on the correct class and 0.0 everywhere else
    scores 1.0; one that does the exact opposite scores 0.0.
    """

    def __init__(
        self,
        input_features: torch.Tensor,
        target_labels: torch.Tensor,
        number_of_classes: int,
    ) -> None:
        """Store the features and precompute the one-hot target matrix.

        Args:
            input_features: ``[n_samples, n_features]`` float tensor.
            target_labels: ``[n_samples]`` long tensor of class indices.
            number_of_classes: Width of the network's output layer.

        Raises:
            ValueError: If ``input_features`` is not 2-D, or its sample count
                does not match ``target_labels``.
        """
        if input_features.dim() != 2:
            raise ValueError(
                "OneHotMeanSquaredErrorFitnessEvaluator: input_features must be 2-D "
                f"(num_samples, num_features), got shape {tuple(input_features.shape)}"
            )
        if input_features.shape[0] != target_labels.shape[0]:
            raise ValueError(
                "OneHotMeanSquaredErrorFitnessEvaluator: input_features has "
                f"{input_features.shape[0]} samples but target_labels has "
                f"{target_labels.shape[0]}"
            )
        self._input_features = input_features.to(torch.float32)
        self._number_of_classes = number_of_classes
        self._one_hot_targets = torch.nn.functional.one_hot(
            target_labels.to(torch.long).cpu(), num_classes=number_of_classes
        ).to(torch.float32)

    def evaluate_single_phenotype(self, phenotype: Phenotype) -> FitnessValue:
        """Return ``1 - MSE`` between the phenotype's outputs and the one-hot targets.

        Args:
            phenotype: Phenotype whose output width is the class count.

        Returns:
            Fitness in ``[0, 1]`` for outputs bounded in ``[0, 1]``; an
            unbounded output activation can push it below zero, which is why
            the callers pair this evaluator with a bounded one.
        """
        with torch.no_grad():
            output_scores = phenotype.forward_pass(self._input_features).cpu()
            mean_squared_error = torch.mean((self._one_hot_targets - output_scores) ** 2)
        return float(1.0 - mean_squared_error.item())
