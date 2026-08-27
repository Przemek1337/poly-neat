"""The retina problem's pattern set, exposed with the usual dataset contract.

The task is fully synthetic and exhaustively enumerable - all 256 patterns, no
train/test split - so this module is a thin bundle over the evaluator's builders,
kept for symmetry with the other example packages.

References:
    Kashtan, N., & Alon, U. (2005). Spontaneous evolution of modularity and network
        motifs. *Proceedings of the National Academy of Sciences*, 102(39), 13773-13778.
        DOI: 10.1073/pnas.0503610102
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from polyneat.evaluators.retina_evaluator import (
    build_all_retina_patterns,
    compute_expected_outputs,
)


@dataclass(frozen=True)
class RetinaPatterns:
    """Every retina pattern with its expected two-output answer.

    Attributes:
        patterns: ``[256, 8]`` tensor of 0/1 pixels.
        expected_outputs: ``[256, 2]`` tensor of 0/1 answers, left then right.
    """

    patterns: torch.Tensor
    expected_outputs: torch.Tensor

    @property
    def number_of_inputs(self) -> int:
        """Number of retina pixels, i.e. substrate input nodes."""
        return int(self.patterns.shape[1])

    @property
    def number_of_outputs(self) -> int:
        """Number of answers per pattern, i.e. substrate output nodes."""
        return int(self.expected_outputs.shape[1])


def load_retina_patterns() -> RetinaPatterns:
    """Return all 256 retina patterns with their expected answers.

    Returns:
        The complete pattern bundle. There is no train/test split because the
        task is exhaustively enumerable, so every network is scored on the whole
        problem.
    """
    patterns = build_all_retina_patterns()
    return RetinaPatterns(
        patterns=patterns,
        expected_outputs=compute_expected_outputs(patterns),
    )
