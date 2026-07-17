from __future__ import annotations

import torch

from polyneat.core.component_protocols import Phenotype
from polyneat.core.type_aliases import FitnessValue
from polyneat.evaluators.class_indexed_evaluator_base import ClassIndexedFitnessEvaluator


class MulticlassDatasetFitnessEvaluator(ClassIndexedFitnessEvaluator):
    """Per-class fitness on a labelled multi-class dataset for C-NEAT.

    An organism assigned to class ``c`` is scored on how well its single
    output reproduces the binary indicator "this sample belongs to class c"
    over the whole dataset::

        fitness = 1.0 - mean((indicator - output) ** 2)

    A perfect recognizer scores 1.0. The paper (Alfaham et al., 2024)
    specifies the negative of a loss value without naming the loss; here the
    loss is the MSE against the binary indicator, shifted to ``1 - MSE`` so
    fitness stays non-negative for NEAT's offspring allocation (negative
    adjusted fitness is clamped to zero). ``CNEATConfig`` enforces the
    matching ``[0, 1]``-bounded output activation.
    """

    def __init__(
        self,
        input_features: torch.Tensor,
        class_label_indices: torch.Tensor,
        number_of_class_labels: int,
    ) -> None:
        """Store the features and precompute the per-class binary targets.

        Args:
            input_features: Float tensor of shape ``(num_samples, num_features)``.
            class_label_indices: Integer tensor of shape ``(num_samples,)``
                holding the class index of each sample.
            number_of_class_labels: Number of classes in the task.

        Raises:
            ValueError: If ``input_features`` is not 2-D, its sample count does
                not match ``class_label_indices``, a label falls outside
                ``[0, number_of_class_labels)``, or some class has no samples.
        """
        super().__init__(number_of_class_labels)
        if input_features.dim() != 2:
            raise ValueError(
                "MulticlassDatasetFitnessEvaluator: input_features must be 2-D "
                f"(num_samples, num_features), got shape {tuple(input_features.shape)}"
            )
        if input_features.shape[0] != class_label_indices.shape[0]:
            raise ValueError(
                "MulticlassDatasetFitnessEvaluator: input_features has "
                f"{input_features.shape[0]} samples but class_label_indices has "
                f"{class_label_indices.shape[0]}"
            )
        if class_label_indices.numel() > 0 and (
            int(class_label_indices.min()) < 0
            or int(class_label_indices.max()) >= number_of_class_labels
        ):
            raise ValueError(
                "MulticlassDatasetFitnessEvaluator: class_label_indices must lie in "
                f"[0, {number_of_class_labels}), got range "
                f"[{int(class_label_indices.min())}, {int(class_label_indices.max())}]"
            )
        labels_on_cpu = class_label_indices.to(torch.long).cpu()
        sample_count_per_class = torch.bincount(labels_on_cpu, minlength=number_of_class_labels)
        classes_without_any_sample = [
            class_label_index
            for class_label_index in range(number_of_class_labels)
            if int(sample_count_per_class[class_label_index]) == 0
        ]
        if classes_without_any_sample:
            raise ValueError(
                "MulticlassDatasetFitnessEvaluator: every class in "
                f"[0, {number_of_class_labels}) must have at least one sample, but classes "
                f"{classes_without_any_sample} have none"
            )
        self._input_features = input_features.to(torch.float32)
        self._binary_targets_per_class = torch.stack(
            [
                (labels_on_cpu == class_label_index).float()
                for class_label_index in range(number_of_class_labels)
            ]
        )

    def evaluate_phenotype_for_class(
        self, phenotype: Phenotype, class_label_index: int
    ) -> FitnessValue:
        """Return ``1 - MSE`` of the phenotype as a recognizer for one class.

        Args:
            phenotype: Single-output phenotype to evaluate.
            class_label_index: Class label the phenotype should recognize.

        Returns:
            ``1.0`` for a perfect recognizer, lower otherwise.
        """
        with torch.no_grad():
            outputs = phenotype.forward_pass(self._input_features)[:, 0].cpu()
        binary_targets = self._binary_targets_per_class[class_label_index]
        mean_squared_error = torch.mean((binary_targets - outputs) ** 2)
        return float(1.0 - mean_squared_error.item())
