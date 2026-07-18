from __future__ import annotations

import torch

from polyneat.core.component_protocols import Phenotype
from polyneat.core.type_aliases import FitnessValue
from polyneat.evaluators.sequential_evaluator_base import SequentialFitnessEvaluator


class BinaryRecognizerFitnessEvaluator(SequentialFitnessEvaluator):
    """Per-subtask fitness on a labelled multi-class dataset for L-NEAT.

    L-NEAT (Chen & Alahakoon, 2006, section IV.A) divides a K-class problem
    into K binary subtasks; each subtask evolves a single-output network that
    outputs 1 for samples of its own class and 0 for every other class. An
    organism is scored by the paper's equation 1::

        fitness = (P / (P + dis))**2 * (c / P)

    where ``dis = sum_i |output_i - target_i|`` is the total output distance
    (equation 2's per-sample Euclidean distance, summed), ``c`` counts samples
    whose thresholded output matches the binary target, and ``P`` is the
    number of samples. Fitness falls as the output distance grows and rises
    with the correct count, the two properties section IV.B.1 requires. A
    perfect recognizer scores 1.0; an inverted one (``dis = P``, ``c = 0``)
    scores 0.0. Both factors lie in ``[0, 1]`` for any non-negative ``dis``,
    so fitness is bounded and never negative regardless of the output range —
    NEAT's offspring allocation, which clamps negative adjusted fitness to
    zero, is satisfied intrinsically. ``LNEATConfig`` still restricts the
    output activation to ``[0, 1]``-bounded functions so that distances to
    the binary targets and the ``classification_threshold`` comparison remain
    meaningful.

    References:
        Chen, L., & Alahakoon, D. (2006). NeuroEvolution of Augmenting
        Topologies with Learning for Data Classification. *ICIA 2006*.
    """

    def __init__(
        self,
        input_features: torch.Tensor,
        class_label_indices: torch.Tensor,
        target_class_label_index: int,
        classification_threshold: float = 0.5,
    ) -> None:
        """Store the features and precompute the binary targets for one class.

        Args:
            input_features: Float tensor of shape ``(num_samples, num_features)``.
            class_label_indices: Integer tensor of shape ``(num_samples,)``
                holding the class index of each sample.
            target_class_label_index: The class this recognizer must detect.
            classification_threshold: Output level at or above which a sample
                counts as classified into the target class.

        Raises:
            ValueError: If ``input_features`` is not 2-D, its sample count
                does not match ``class_label_indices``, the threshold leaves
                ``(0, 1)``, or the target class has no samples.
        """
        if input_features.dim() != 2:
            raise ValueError(
                "BinaryRecognizerFitnessEvaluator: input_features must be 2-D "
                f"(num_samples, num_features), got shape {tuple(input_features.shape)}"
            )
        if input_features.shape[0] != class_label_indices.shape[0]:
            raise ValueError(
                "BinaryRecognizerFitnessEvaluator: input_features has "
                f"{input_features.shape[0]} samples but class_label_indices has "
                f"{class_label_indices.shape[0]}"
            )
        if not (0.0 < classification_threshold < 1.0):
            raise ValueError(
                "BinaryRecognizerFitnessEvaluator: classification_threshold must be in "
                f"(0.0, 1.0), got {classification_threshold}"
            )
        binary_targets = class_label_indices.to(torch.long).cpu() == target_class_label_index
        if int(binary_targets.sum()) == 0:
            raise ValueError(
                "BinaryRecognizerFitnessEvaluator: no sample carries "
                f"target_class_label_index={target_class_label_index}"
            )
        self._input_features = input_features.to(torch.float32)
        self._binary_targets = binary_targets.float()
        self._classification_threshold = classification_threshold

    def evaluate_single_phenotype(self, phenotype: Phenotype) -> FitnessValue:
        """Return the equation-1 fitness of one recognizer phenotype.

        Args:
            phenotype: Single-output phenotype to evaluate.

        Returns:
            ``1.0`` for a perfect recognizer, down to ``0.0`` for the worst.
        """
        with torch.no_grad():
            outputs = phenotype.forward_pass(self._input_features)[:, 0].cpu()
        total_output_distance = float(torch.abs(outputs - self._binary_targets).sum())
        predicted_memberships = (outputs >= self._classification_threshold).float()
        number_of_correct_classifications = float(
            (predicted_memberships == self._binary_targets).sum()
        )
        number_of_samples = self._binary_targets.numel()
        distance_factor = (
            number_of_samples / (number_of_samples + total_output_distance)
        ) ** 2
        return float(
            distance_factor * (number_of_correct_classifications / number_of_samples)
        )
