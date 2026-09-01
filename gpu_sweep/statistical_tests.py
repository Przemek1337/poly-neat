"""Friedman, Iman-Davenport and Holm, per Derrac et al. (2011).

Reference:
    Derrac, J., Garcia, S., Molina, D., & Herrera, F. (2011). A practical
    tutorial on the use of nonparametric statistical tests as a methodology for
    comparing evolutionary and swarm intelligence algorithms. *Swarm and
    Evolutionary Computation*, 1(1), 3-18.

The paper's own worked examples are this module's test fixtures, so the
implementation is checked against published numbers rather than against itself.
Only Holm is implemented as a post-hoc procedure: section 6.2 states that it
"can always be considered better than Bonferroni-Dunn's procedure", and section
6.3 rejects Nemenyi for all-pairs work as "a very conservative procedure".
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

from gpu_sweep.special_functions import (
    chi_squared_survival,
    f_distribution_survival,
    standard_normal_two_tailed_p_value,
)

DEFAULT_ALPHA = 0.05


def rank_row(values: list[float], *, higher_is_better: bool) -> list[float]:
    """Rank one problem's results, 1 for the best, averaging ties.

    Section 4.2, step 2: "For each problem i, rank values from 1 (best result)
    to k (worst result)", and "in case of ties, we recommend computing average
    ranks".

    Args:
        values: One value per algorithm.
        higher_is_better: True for F1 and accuracy, False for error rates.

    Returns:
        One rank per algorithm, in the input order.
    """
    number_of_algorithms = len(values)
    ordering_key = (lambda index: -values[index]) if higher_is_better else (
        lambda index: values[index]
    )
    positions_best_first = sorted(range(number_of_algorithms), key=ordering_key)

    ranks = [0.0] * number_of_algorithms
    position = 0
    while position < number_of_algorithms:
        tie_end = position + 1
        while (
            tie_end < number_of_algorithms
            and values[positions_best_first[tie_end]]
            == values[positions_best_first[position]]
        ):
            tie_end += 1
        average_rank = (position + 1 + tie_end) / 2.0
        for tied_position in range(position, tie_end):
            ranks[positions_best_first[tied_position]] = average_rank
        position = tie_end
    return ranks


def average_ranks(
    matrix_values: list[list[float]], *, higher_is_better: bool
) -> list[float]:
    """Average each algorithm's rank across every problem.

    Section 4.2, step 3: ``R_j = (1/n) * sum_i r_ij``.

    Args:
        matrix_values: ``values[problem][algorithm]``.
        higher_is_better: Direction of the metric being ranked.

    Returns:
        One average rank per algorithm.
    """
    if not matrix_values:
        return []
    number_of_algorithms = len(matrix_values[0])
    rank_totals = [0.0] * number_of_algorithms
    for problem_values in matrix_values:
        for algorithm_index, rank in enumerate(
            rank_row(problem_values, higher_is_better=higher_is_better)
        ):
            rank_totals[algorithm_index] += rank
    return [total / len(matrix_values) for total in rank_totals]


@dataclass(frozen=True)
class FriedmanResult:
    """The omnibus test's ranks, statistics and p-values.

    Attributes:
        number_of_algorithms: ``k`` in the paper's notation.
        number_of_problems: ``n`` in the paper's notation.
        average_ranks: ``R_j`` for each algorithm, in algorithm order.
        chi_squared_statistic: Eq. 2.
        chi_squared_p_value: Upper tail with ``k - 1`` degrees of freedom.
        iman_davenport_statistic: Eq. 3.
        iman_davenport_p_value: Upper tail of ``F(k - 1, (k - 1)(n - 1))``.
    """

    number_of_algorithms: int
    number_of_problems: int
    average_ranks: list[float]
    chi_squared_statistic: float
    chi_squared_p_value: float
    iman_davenport_statistic: float
    iman_davenport_p_value: float


def friedman_statistics_from_average_ranks(
    average_rank_values: list[float], number_of_problems: int
) -> FriedmanResult:
    """Compute both omnibus statistics from average ranks alone.

    Eq. 2 for the Friedman statistic and Eq. 3 for the Iman-Davenport
    derivation, which the paper introduces because the Friedman statistic
    "often produces a conservative effect not desired".

    Args:
        average_rank_values: ``R_j`` per algorithm.
        number_of_problems: ``n``, the number of problems ranked.

    Returns:
        The assembled :class:`FriedmanResult`.
    """
    number_of_algorithms = len(average_rank_values)
    sum_of_squared_ranks = sum(rank * rank for rank in average_rank_values)
    correction_term = (
        number_of_algorithms * (number_of_algorithms + 1) ** 2
    ) / 4.0
    chi_squared_statistic = (
        12.0
        * number_of_problems
        / (number_of_algorithms * (number_of_algorithms + 1))
    ) * (sum_of_squared_ranks - correction_term)

    chi_squared_degrees_of_freedom = number_of_algorithms - 1
    denominator = (
        number_of_problems * chi_squared_degrees_of_freedom - chi_squared_statistic
    )
    if denominator <= 0.0:
        iman_davenport_statistic = math.inf
        iman_davenport_p_value = 0.0
    else:
        iman_davenport_statistic = (
            (number_of_problems - 1) * chi_squared_statistic
        ) / denominator
        iman_davenport_p_value = f_distribution_survival(
            iman_davenport_statistic,
            chi_squared_degrees_of_freedom,
            chi_squared_degrees_of_freedom * (number_of_problems - 1),
        )

    return FriedmanResult(
        number_of_algorithms=number_of_algorithms,
        number_of_problems=number_of_problems,
        average_ranks=list(average_rank_values),
        chi_squared_statistic=chi_squared_statistic,
        chi_squared_p_value=chi_squared_survival(
            chi_squared_statistic, chi_squared_degrees_of_freedom
        ),
        iman_davenport_statistic=iman_davenport_statistic,
        iman_davenport_p_value=iman_davenport_p_value,
    )


def friedman_test(
    matrix_values: list[list[float]], *, higher_is_better: bool
) -> FriedmanResult:
    """Rank a problem-by-algorithm matrix and run the omnibus test on it.

    Args:
        matrix_values: ``values[problem][algorithm]``.
        higher_is_better: Direction of the metric being ranked.

    Returns:
        The assembled :class:`FriedmanResult`.
    """
    return friedman_statistics_from_average_ranks(
        average_ranks(matrix_values, higher_is_better=higher_is_better),
        number_of_problems=len(matrix_values),
    )


def friedman_post_hoc_z(
    rank_i: float,
    rank_j: float,
    *,
    number_of_algorithms: int,
    number_of_problems: int,
) -> float:
    """The post-hoc z statistic for two average ranks (Eq. 12).

    ``z = (R_i - R_j) / sqrt(k(k + 1) / (6n))``.

    Args:
        rank_i: Average rank of the first algorithm.
        rank_j: Average rank of the second algorithm.
        number_of_algorithms: ``k``.
        number_of_problems: ``n``.

    Returns:
        The z statistic; positive when the first algorithm ranks worse.
    """
    standard_error = math.sqrt(
        number_of_algorithms
        * (number_of_algorithms + 1)
        / (6.0 * number_of_problems)
    )
    return (rank_i - rank_j) / standard_error


def holm_adjusted_p_values(unadjusted_p_values: list[float]) -> list[float]:
    """Holm's step-down adjusted p-values (Section 4.3).

    ``APV_i = min{1, max{(m - j + 1) * p_j : 1 <= j <= i}}`` over p-values
    sorted ascending. The running maximum is what makes the sequence monotone,
    and it is also what makes comparing an adjusted p-value against alpha
    exactly equivalent to the paper's step-down stopping rule - once one
    hypothesis is retained, every later one is retained too.

    Args:
        unadjusted_p_values: One p-value per hypothesis, in any order.

    Returns:
        Adjusted p-values aligned with the input positions.
    """
    family_size = len(unadjusted_p_values)
    if family_size == 0:
        return []
    positions_ascending = sorted(
        range(family_size), key=lambda index: unadjusted_p_values[index]
    )
    adjusted = [0.0] * family_size
    running_maximum = 0.0
    for sorted_position, original_index in enumerate(positions_ascending):
        scaled = (family_size - sorted_position) * unadjusted_p_values[original_index]
        running_maximum = max(running_maximum, scaled)
        adjusted[original_index] = min(running_maximum, 1.0)
    return adjusted


def control_method_index(average_rank_values: list[float]) -> int:
    """Index of the best-ranked algorithm, which becomes the control method.

    The paper takes the best performer as the control in every 1-to-N example,
    since the question being asked is whether it beats the rest.

    Args:
        average_rank_values: ``R_j`` per algorithm.

    Returns:
        Index of the lowest average rank; ties go to the earlier index.
    """
    return min(range(len(average_rank_values)), key=lambda index: average_rank_values[index])


@dataclass(frozen=True)
class Comparison:
    """One post-hoc hypothesis and its verdict.

    Attributes:
        left_name: Algorithm on the left of the comparison.
        right_name: Algorithm on the right; the control, in a 1-to-N family.
        z_statistic: Eq. 12 applied to the two average ranks.
        unadjusted_p_value: Two-tailed normal p-value of that statistic.
        holm_adjusted_p_value: Holm APV within this family.
        is_rejected: Whether the adjusted p-value clears alpha.
    """

    left_name: str
    right_name: str
    z_statistic: float
    unadjusted_p_value: float
    holm_adjusted_p_value: float
    is_rejected: bool


def _build_comparisons(
    index_pairs: list[tuple[int, int]],
    friedman_result: FriedmanResult,
    algorithm_names: list[str],
    *,
    alpha: float,
) -> list[Comparison]:
    """Score a family of hypotheses and apply Holm across all of them."""
    z_statistics = [
        friedman_post_hoc_z(
            friedman_result.average_ranks[left_index],
            friedman_result.average_ranks[right_index],
            number_of_algorithms=friedman_result.number_of_algorithms,
            number_of_problems=friedman_result.number_of_problems,
        )
        for left_index, right_index in index_pairs
    ]
    unadjusted = [
        standard_normal_two_tailed_p_value(z_statistic) for z_statistic in z_statistics
    ]
    adjusted = holm_adjusted_p_values(unadjusted)

    comparisons = [
        Comparison(
            left_name=algorithm_names[left_index],
            right_name=algorithm_names[right_index],
            z_statistic=z_statistics[position],
            unadjusted_p_value=unadjusted[position],
            holm_adjusted_p_value=adjusted[position],
            is_rejected=adjusted[position] <= alpha,
        )
        for position, (left_index, right_index) in enumerate(index_pairs)
    ]
    return sorted(comparisons, key=lambda comparison: comparison.unadjusted_p_value)


def post_hoc_control_comparisons(
    friedman_result: FriedmanResult,
    algorithm_names: list[str],
    *,
    alpha: float = DEFAULT_ALPHA,
) -> list[Comparison]:
    """Compare every algorithm against the best-ranked one (Section 4.3).

    The family holds ``k - 1`` hypotheses.

    Args:
        friedman_result: Output of the omnibus test.
        algorithm_names: Names aligned with ``friedman_result.average_ranks``.
        alpha: Family-wise significance level.

    Returns:
        Comparisons sorted by unadjusted p-value, ascending.
    """
    control_index = control_method_index(friedman_result.average_ranks)
    index_pairs = [
        (algorithm_index, control_index)
        for algorithm_index in range(len(algorithm_names))
        if algorithm_index != control_index
    ]
    return _build_comparisons(index_pairs, friedman_result, algorithm_names, alpha=alpha)


def post_hoc_all_pairs_comparisons(
    friedman_result: FriedmanResult,
    algorithm_names: list[str],
    *,
    alpha: float = DEFAULT_ALPHA,
) -> list[Comparison]:
    """Compare every algorithm against every other (Section 5).

    The family holds ``k(k - 1)/2`` hypotheses, so Holm divides alpha far more
    aggressively here than in the control family. The paper's alternative,
    Nemenyi, is deliberately not implemented - section 6.3 calls it "a very
    conservative procedure, and many of the obvious differences may not be
    detected".

    Args:
        friedman_result: Output of the omnibus test.
        algorithm_names: Names aligned with ``friedman_result.average_ranks``.
        alpha: Family-wise significance level.

    Returns:
        Comparisons sorted by unadjusted p-value, ascending.
    """
    index_pairs = list(itertools.combinations(range(len(algorithm_names)), 2))
    return _build_comparisons(index_pairs, friedman_result, algorithm_names, alpha=alpha)
