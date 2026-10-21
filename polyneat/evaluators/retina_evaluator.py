"""The left-and-right retina problem: a task built from two independent sub-problems.

Used to compare plain HyperNEAT with HyperNEAT-LEO. The task is modular by
construction - the left answer does not depend on any right-hand pixel and vice
versa - so a network keeping its two halves separate loses nothing, while one
wiring them together pays for connections it cannot use.

Pixel layout. The retina is four pixels wide and two high; the left four form one
2x2 block and the right four the other. Within each block the pixels are ordered
column by column, outer column first, so that a block's two columns occupy
adjacent positions and the substrate's x coordinate carries the column structure:

    index   0        1         2         3    |  4        5         6         7
    block   left                              |  right
    column  outer    outer     inner   inner  |  inner    inner     outer  outer
    row     top      bottom    top     bottom |  top      bottom    top    bottom

The ordering is mirror symmetric, so reversing all eight pixels maps one
hemisphere onto the other and swaps the two answers.

Deviation from the source, deliberate: Kashtan & Alon's networks have a **single**
output computing ``L AND R``. Here ``L`` and ``R`` are exposed as two outputs, so
the modular decomposition is observable at the output layer and each hemisphere
has an output of its own for HyperNEAT-LEO's locality seed to bind to. The two
sub-problems, and therefore the modular structure the benchmark is about, are
unchanged.

References:
    Kashtan, N., & Alon, U. (2005). Spontaneous evolution of modularity and network
        motifs. *Proceedings of the National Academy of Sciences*, 102(39), 13773-13778.
        DOI: 10.1073/pnas.0503610102
    Verbancsics, P., & Stanley, K. O. (2011). Constraining Connectivity to Encourage
        Modularity in HyperNEAT. *GECCO '11*, pp. 1483-1490. DOI: 10.1145/2001576.2001776
"""

from __future__ import annotations

import torch

from polyneat.core.component_protocols import Phenotype
from polyneat.core.type_aliases import FitnessValue
from polyneat.evaluators.sequential_evaluator_base import SequentialFitnessEvaluator

NUMBER_OF_RETINA_PIXELS: int = 8
NUMBER_OF_RETINA_PATTERNS: int = 256
MAXIMUM_RETINA_FITNESS: float = 256.0


def build_all_retina_patterns() -> torch.Tensor:
    """Build every one of the 256 eight-pixel retina patterns.

    Pixel values are 0.0 and 1.0, as in the source paper. See the module
    docstring for the index-to-position mapping.

    Returns:
        A ``[256, 8]`` float tensor holding each pattern exactly once.
    """
    pattern_indices = torch.arange(NUMBER_OF_RETINA_PATTERNS, dtype=torch.int64)
    bit_positions = torch.arange(NUMBER_OF_RETINA_PIXELS, dtype=torch.int64)
    bits = (pattern_indices.unsqueeze(1) >> bit_positions.unsqueeze(0)) & 1
    return bits.to(torch.float32)


def _half_contains_object(
    half_pixels: torch.Tensor, outer_column_indices: tuple[int, int]
) -> torch.Tensor:
    """Return 1.0 where a 2x2 half holds an object, per Kashtan & Alon's rule.

    An object exists when the half has three or more black pixels, or when it has
    one or two black pixels confined to its **outer** column.

    Args:
        half_pixels: ``[batch, 4]`` tensor for one hemisphere.
        outer_column_indices: Positions within the half that form its outer
            column - ``(0, 1)`` for the left hemisphere, ``(2, 3)`` for the right.

    Returns:
        A ``[batch]`` float tensor of 0.0/1.0 answers.
    """
    black_pixel_count = half_pixels.sum(dim=1)
    outer_column_count = half_pixels[:, list(outer_column_indices)].sum(dim=1)

    has_three_or_more = black_pixel_count >= 3.0
    has_one_or_two_in_outer_column_only = (
        (black_pixel_count >= 1.0)
        & (black_pixel_count <= 2.0)
        & (outer_column_count == black_pixel_count)
    )
    return (has_three_or_more | has_one_or_two_in_outer_column_only).to(torch.float32)


def compute_expected_outputs(patterns: torch.Tensor) -> torch.Tensor:
    """Return the target answers for a batch of retina patterns.

    Args:
        patterns: ``[batch, 8]`` tensor of 0/1 pixels.

    Returns:
        A ``[batch, 2]`` tensor: column 0 is "object on the left", column 1 is
        "object on the right".
    """
    left_answer = _half_contains_object(patterns[:, :4], outer_column_indices=(0, 1))
    right_answer = _half_contains_object(patterns[:, 4:], outer_column_indices=(2, 3))
    return torch.stack([left_answer, right_answer], dim=1)


class RetinaProblemEvaluator(SequentialFitnessEvaluator):
    """Scores a two-output network on all 256 retina patterns at once.

    Fitness is ``sum over patterns of (1 - mean squared error over the two
    outputs)``, so a perfect network scores ``256.0``. Squared rather than
    absolute error, for the same reason as ``XORFitnessEvaluator``: absolute
    error leaves a flat gradient around partial solutions.

    The whole task fits in one forward pass, which matters - a NEAT phenotype's
    per-call cost is dominated by Python overhead, so evaluating all 256 patterns
    in one batch is essentially free compared with looping over them.
    """

    def __init__(self, device_for_computation: torch.device | None = None) -> None:
        """Precompute the pattern set and its answers once.

        Args:
            device_for_computation: Device to hold the pattern tensor on;
                ``None`` keeps it on the CPU.
        """
        patterns = build_all_retina_patterns()
        self._expected_outputs = compute_expected_outputs(patterns)
        if device_for_computation is not None:
            patterns = patterns.to(device_for_computation)
        self._input_patterns = patterns

    def evaluate_single_phenotype(self, phenotype: Phenotype) -> FitnessValue:
        """Return the retina fitness of one network, clamped to ``[0, 256]``.

        Args:
            phenotype: A network with 8 inputs and 2 outputs.

        Returns:
            Fitness in ``[0.0, 256.0]``; 256.0 means every pattern answered
            exactly.
        """
        with torch.no_grad():
            output_tensor = phenotype.forward_pass(self._input_patterns)
        predicted = output_tensor[:, :2].cpu()
        mean_squared_error_per_pattern = (
            (predicted - self._expected_outputs) ** 2
        ).mean(dim=1)
        total_fitness = float(torch.sum(1.0 - mean_squared_error_per_pattern).item())
        return max(0.0, min(MAXIMUM_RETINA_FITNESS, total_fitness))
