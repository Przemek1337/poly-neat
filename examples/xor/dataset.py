"""XOR data for the xor examples.

XOR is four fixed patterns, not a downloadable dataset, so there is nothing to
cache or split. This module exists for structural symmetry with the other
examples -- a ``dataset.py`` with a top-level ``load_*`` returning a named pair --
and so a user can read and inspect the exact tensors the task scores against.

The patterns live in :mod:`polyneat.evaluators.xor_evaluator`, which stays the
single source of truth; this loader re-exposes them rather than duplicating them.
"""

from __future__ import annotations

import torch

from polyneat.evaluators.xor_evaluator import (
    _XOR_EXPECTED_OUTPUTS,
    _XOR_INPUT_PATTERNS,
)


def load_xor() -> tuple[torch.Tensor, torch.Tensor]:
    """Return the four XOR patterns and their expected outputs.

    Returns:
        ``(input_patterns, expected_outputs)`` where input_patterns is a
        ``[4, 2]`` float tensor of the binary input pairs and expected_outputs
        is a ``[4]`` float tensor of the target XOR values. XOR has no held-out
        set, so there is no split.
    """
    return _XOR_INPUT_PATTERNS.clone(), _XOR_EXPECTED_OUTPUTS.clone()
