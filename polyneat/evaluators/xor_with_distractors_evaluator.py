from __future__ import annotations

import numpy as np
import torch

from polyneat.core.component_protocols import Phenotype
from polyneat.core.type_aliases import FitnessValue
from polyneat.evaluators.sequential_evaluator_base import SequentialFitnessEvaluator

_XOR_INPUT_PATTERNS: torch.Tensor = torch.tensor(
    [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
)
_XOR_EXPECTED_OUTPUTS: torch.Tensor = torch.tensor([0.0, 1.0, 1.0, 0.0])


class XORWithDistractorsEvaluator(SequentialFitnessEvaluator):
    """XOR fitness over noise-padded inputs, averaged per pattern (max 4.0).

    The feature-selection scenario of Whiteson et al. (2005): the two real XOR
    inputs are padded with ``number_of_distractor_inputs`` pure-noise columns,
    so a network that keys on a distractor cannot score well. Each of the four
    XOR patterns is replicated ``number_of_noise_replicas_per_pattern`` times
    with *different* noise, so only the two true inputs give a consistent
    signal across a pattern's replicas.

    Fitness is ``sum over the 4 patterns of (1 - mean squared error across that
    pattern's noise replicas)``, so the scale matches the plain XOR benchmark:
    4.0 is perfect, >= 3.95 counts as solved. The noise is drawn once from a
    fixed seed, keeping fitness deterministic across the whole run.

    References:
        Whiteson, S., Stone, P., Stanley, K. O., Miikkulainen, R., & Kohl, N. (2005). Automatic
            Feature Selection in Neuroevolution. *GECCO 2005: Proceedings of the Genetic and
            Evolutionary Computation Conference*, pp. 1225-1232.
    """

    def __init__(
        self,
        number_of_distractor_inputs: int = 6,
        number_of_noise_replicas_per_pattern: int = 8,
        distractor_noise_seed: int = 7,
    ) -> None:
        """Build the noise-padded XOR dataset once.

        Args:
            number_of_distractor_inputs: How many pure-noise columns to append
                after the two real XOR inputs. The total input count the
                network must have is ``2 + number_of_distractor_inputs``.
            number_of_noise_replicas_per_pattern: How many differently-noised
                copies of each XOR pattern to evaluate. More replicas make it
                harder to score by exploiting the noise.
            distractor_noise_seed: Seed for the fixed distractor noise, so
                fitness is deterministic.
        """
        self._number_of_noise_replicas_per_pattern = number_of_noise_replicas_per_pattern
        replicated_xor_inputs = _XOR_INPUT_PATTERNS.repeat_interleave(
            number_of_noise_replicas_per_pattern, dim=0
        )
        noise_generator = np.random.default_rng(distractor_noise_seed)
        distractor_columns = torch.from_numpy(
            noise_generator.uniform(
                0.0,
                1.0,
                size=(replicated_xor_inputs.shape[0], number_of_distractor_inputs),
            )
        ).to(torch.float32)
        self._input_patterns = torch.cat([replicated_xor_inputs, distractor_columns], dim=1)
        self._expected_outputs = _XOR_EXPECTED_OUTPUTS

    def evaluate_single_phenotype(self, phenotype: Phenotype) -> FitnessValue:
        """Return the per-pattern-averaged XOR fitness, clamped to ``[0.0, 4.0]``."""
        with torch.no_grad():
            output_tensor = phenotype.forward_pass(self._input_patterns)
        predicted = output_tensor[:, 0].cpu()
        squared_errors_per_pattern = (
            (
                predicted.reshape(
                    _XOR_EXPECTED_OUTPUTS.shape[0],
                    self._number_of_noise_replicas_per_pattern,
                )
                - self._expected_outputs.unsqueeze(1)
            )
            ** 2
        ).mean(dim=1)
        return max(0.0, float(torch.sum(1.0 - squared_errors_per_pattern).item()))
