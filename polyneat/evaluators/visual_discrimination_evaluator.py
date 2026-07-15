"""Fitness for the visual discrimination task.

References:
    Stanley, K. O., D'Ambrosio, D. B., & Gauci, J. (2009). A hypercube-based
    encoding for evolving large-scale neural networks. Artificial Life, 15(2),
    185-212. (Visual discrimination task, Section 4.)
"""
from __future__ import annotations

import math

import numpy as np
import torch

from polyneat.core.component_protocols import Phenotype
from polyneat.core.type_aliases import FitnessValue
from polyneat.evaluators.sequential_evaluator_base import SequentialFitnessEvaluator


def _paint_square(field: np.ndarray, top_row: int, left_col: int, side: int) -> None:
    field[top_row : top_row + side, left_col : left_col + side] = 1.0


def generate_visual_discrimination_trials(
    field_side: int,
    big_object_side: int,
    small_object_side: int,
    number_of_trials: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build fixed trials for the visual discrimination task (Stanley et al. 2009, Section 4).

    Each trial places a big and a small square in a ``field_side x field_side``
    visual field without overlap.

    Args:
        field_side: Side length of the square visual field.
        big_object_side: Side length of the larger square (the target).
        small_object_side: Side length of the smaller square (the distractor).
        number_of_trials: How many non-overlapping trials to generate.
        seed: Seed for the deterministic placement of objects.

    Returns:
        A tuple ``(trial_fields, true_center_rows, true_center_cols)``.
        ``trial_fields`` has shape ``(number_of_trials, field_side*field_side)``
        (row-major flattened, so a column index equals a row-major node id).
        ``true_center_rows`` and ``true_center_cols`` give the center pixel of
        the big object per trial -- the node the target field should select.
    """
    rng = np.random.default_rng(seed)
    max_big_top = field_side - big_object_side
    big_center_offset = big_object_side // 2

    trial_field_rows: list[np.ndarray] = []
    true_rows: list[int] = []
    true_cols: list[int] = []
    while len(trial_field_rows) < number_of_trials:
        big_top = int(rng.integers(0, max_big_top + 1))
        big_left = int(rng.integers(0, max_big_top + 1))
        small_row = int(rng.integers(0, field_side))
        small_col = int(rng.integers(0, field_side))

        inside_big = (
            big_top <= small_row < big_top + big_object_side
            and big_left <= small_col < big_left + big_object_side
        )
        if inside_big:
            continue

        field = np.zeros((field_side, field_side), dtype=np.float32)
        _paint_square(field, big_top, big_left, big_object_side)
        _paint_square(field, small_row, small_col, small_object_side)
        trial_field_rows.append(field.reshape(-1))
        true_rows.append(big_top + big_center_offset)
        true_cols.append(big_left + big_center_offset)

    trial_fields = torch.from_numpy(np.stack(trial_field_rows)).to(torch.float32)
    return (
        trial_fields,
        torch.tensor(true_rows, dtype=torch.long),
        torch.tensor(true_cols, dtype=torch.long),
    )


class VisualDiscriminationFitnessEvaluator(SequentialFitnessEvaluator):
    """Score a phenotype on the visual discrimination task.

    The substrate's selection is the argmax output node, interpreted as a
    location in the ``field_side x field_side`` target field. Where the paper
    (Stanley, D'Ambrosio & Gauci, 2009, Section 4.2) uses the sum of squared
    distances between the target and the point of highest activation, this uses
    the same distance signal normalized to a smooth per-trial reward in
    ``[0, 1]`` (1.0 = the exact center), keeping fitness positive for the
    reproduction machinery.
    """

    def __init__(
        self,
        trial_fields: torch.Tensor,
        true_center_rows: torch.Tensor,
        true_center_cols: torch.Tensor,
        field_side: int,
    ) -> None:
        """Store the trials and their correct big-object centers.

        Args:
            trial_fields: Float tensor of shape ``(num_trials,
                field_side*field_side)`` -- one row-major flattened visual field
                per trial.
            true_center_rows: Long tensor of the correct big-object center row
                per trial.
            true_center_cols: Long tensor of the correct big-object center
                column per trial.
            field_side: Side length of the square field (used to convert an
                output-node index back into a ``(row, column)``).
        """
        self._trial_fields = trial_fields.to(torch.float32)
        self._true_center_rows = true_center_rows.to(torch.long)
        self._true_center_cols = true_center_cols.to(torch.long)
        self._field_side = field_side
        self._max_distance = math.hypot(field_side - 1, field_side - 1)

    def evaluate_single_phenotype(self, phenotype: Phenotype) -> FitnessValue:
        """Return the mean per-trial reward of a phenotype.

        Args:
            phenotype: Phenotype to evaluate; its argmax output node per trial is
                interpreted as the predicted big-object center.

        Returns:
            The mean of ``1 - distance / max_distance`` over all trials, in
            ``[0, 1]`` (1.0 = always picks the exact center).
        """
        with torch.no_grad():
            target_field_outputs = phenotype.forward_pass(self._trial_fields)
        selected_node_index = target_field_outputs.argmax(dim=1).cpu()
        predicted_rows = torch.div(
            selected_node_index, self._field_side, rounding_mode="floor"
        )
        predicted_cols = selected_node_index % self._field_side
        distances = torch.sqrt(
            (predicted_rows - self._true_center_rows).to(torch.float32) ** 2
            + (predicted_cols - self._true_center_cols).to(torch.float32) ** 2
        )
        per_trial_reward = 1.0 - distances / self._max_distance
        return float(per_trial_reward.mean().item())
