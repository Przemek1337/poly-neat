"""Visual-discrimination trials for the visual_discrimination example.

The trials are synthesised, not downloaded: each places a big and a small square
in a visual field without overlap (Stanley, D'Ambrosio & Gauci, 2009, Section 4).
The generator lives in :mod:`polyneat.evaluators.visual_discrimination_evaluator`,
which the library's evaluator also uses; this module wraps its bare 3-tuple into a
named bundle so the example reads like the others -- a ``dataset.py`` with a
top-level ``load_*`` returning a named type.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from polyneat.evaluators.visual_discrimination_evaluator import (
    generate_visual_discrimination_trials,
)


@dataclass(frozen=True)
class VisualDiscriminationTrials:
    """A fixed set of visual-discrimination trials as one named bundle.

    Field names replace the positional 3-tuple the generator returns, so the
    example does not have to remember the order of the fields and targets.
    """

    trial_fields: torch.Tensor  # [n_trials, field_side ** 2]
    true_center_rows: torch.Tensor  # [n_trials]
    true_center_cols: torch.Tensor  # [n_trials]


def load_visual_discrimination_trials(
    *,
    field_side: int,
    big_object_side: int,
    small_object_side: int,
    number_of_trials: int,
    seed: int,
) -> VisualDiscriminationTrials:
    """Generate visual-discrimination trials as a named bundle.

    Args:
        field_side: Side length of the square visual field.
        big_object_side: Side length of the larger square (the target).
        small_object_side: Side length of the smaller square (the distractor).
        number_of_trials: How many non-overlapping trials to generate.
        seed: Seed for the deterministic placement of objects.

    Returns:
        A :class:`VisualDiscriminationTrials` bundle. ``trial_fields`` has shape
        ``(number_of_trials, field_side ** 2)`` (row-major flattened); the two
        target tensors give the center pixel of the big object per trial.
    """
    trial_fields, true_center_rows, true_center_cols = (
        generate_visual_discrimination_trials(
            field_side=field_side,
            big_object_side=big_object_side,
            small_object_side=small_object_side,
            number_of_trials=number_of_trials,
            seed=seed,
        )
    )
    return VisualDiscriminationTrials(
        trial_fields=trial_fields,
        true_center_rows=true_center_rows,
        true_center_cols=true_center_cols,
    )
