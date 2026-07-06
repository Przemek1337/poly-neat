from __future__ import annotations

import torch

from polyneat.core.component_protocols import Phenotype
from polyneat.core.type_aliases import FitnessValue
from polyneat.evaluators.sequential_evaluator_base import SequentialFitnessEvaluator

_XOR_INPUT_PATTERNS: torch.Tensor = torch.tensor(
    [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
)
_XOR_EXPECTED_OUTPUTS: torch.Tensor = torch.tensor([0.0, 1.0, 1.0, 0.0])


class XORFitnessEvaluator(SequentialFitnessEvaluator):
    """Evaluates a phenotype on all four XOR input patterns.

    Fitness is the sum of (1 - (expected - actual)^2) across the four patterns,
    giving a maximum of 4.0. A genome that perfectly solves XOR scores exactly
    4.0; the commonly used solved threshold is 3.95.

    Note: this threshold is much stricter than the original paper's success
    criterion, which only requires every output to land on the correct side
    of 0.5 (see ``classifies_all_patterns_correctly``). Generation counts
    measured against the two criteria are not comparable.
    """

    def evaluate_single_phenotype(self, phenotype: Phenotype) -> FitnessValue:
        with torch.no_grad():
            output_tensor = phenotype.forward_pass(_XOR_INPUT_PATTERNS)
        predicted = output_tensor[:, 0].cpu()
        squared_errors = (_XOR_EXPECTED_OUTPUTS - predicted) ** 2
        fitness = float(torch.sum(1.0 - squared_errors).item())
        return max(0.0, fitness)

    @staticmethod
    def classifies_all_patterns_correctly(phenotype: Phenotype) -> bool:
        """Paper success criterion: all four outputs on the correct side of 0.5."""
        with torch.no_grad():
            output_tensor = phenotype.forward_pass(_XOR_INPUT_PATTERNS)
        predicted = output_tensor[:, 0].cpu()
        return bool(torch.all((predicted > 0.5) == (_XOR_EXPECTED_OUTPUTS > 0.5)).item())
