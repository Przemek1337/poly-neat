"""Friedman, Iman-Davenport and Holm, checked against Derrac et al. (2011).

The fixtures are the paper's own worked examples, so a passing suite means the
implementation reproduces published numbers rather than merely being
self-consistent.
"""

from __future__ import annotations

import pytest

from gpu_sweep.statistical_tests import (
    average_ranks,
    control_method_index,
    friedman_post_hoc_z,
    friedman_statistics_from_average_ranks,
    friedman_test,
    holm_adjusted_p_values,
    post_hoc_all_pairs_comparisons,
    post_hoc_control_comparisons,
    rank_row,
)

# Derrac et al. (2011), Table 7: error rates of four algorithms on four problems.
DERRAC_TABLE_7_ERROR_RATES = [
    [2.711, 3.147, 2.515, 2.612],
    [7.832, 9.828, 7.832, 7.921],
    [0.012, 0.532, 0.122, 0.005],
    [3.431, 4.111, 3.401, 3.401],
]
# Derrac et al. (2011), Table 8: the average ranks those error rates produce.
#
# NOTE: the paper prints 1.250 as the average for algorithm C, which contradicts
# its own rank column in the same table - C is ranked 1, 1.5, 3, 1.5 on the four
# problems, which averages to 1.75, not 1.250. The other three averages (2.375,
# 4.0, 1.875) match their rank columns exactly. The printed 1.250 is a typo in
# the publication, so the fixture below uses the value the paper's own ranks
# produce. Do not "fix" this back to 1.250.
DERRAC_TABLE_8_AVERAGE_RANKS = [2.375, 4.0, 1.75, 1.875]

# Derrac et al. (2011), Table 11: nine algorithms over the 25 CEC'2005 functions.
DERRAC_TABLE_11_AVERAGE_RANKS = [7.0, 4.84, 6.28, 5.5, 4.64, 5.4, 4.0, 3.5, 3.84]

# Derrac et al. (2011), Table 12: four algorithms, SaDE last and best.
DERRAC_TABLE_12_AVERAGE_RANKS = [2.48, 3.12, 2.44, 1.96]


def test_rank_row_gives_rank_one_to_the_best_value() -> None:
    assert rank_row([2.711, 3.147, 2.515, 2.612], higher_is_better=False) == [3.0, 4.0, 1.0, 2.0]


def test_rank_row_averages_tied_ranks() -> None:
    # Two values tie for the best rank, so both take (1 + 2) / 2.
    assert rank_row([7.832, 9.828, 7.832, 7.921], higher_is_better=False) == [1.5, 4.0, 1.5, 3.0]


def test_rank_row_flips_direction_for_a_higher_is_better_metric() -> None:
    assert rank_row([0.9, 0.5, 0.7], higher_is_better=True) == [1.0, 3.0, 2.0]


def test_average_ranks_reproduce_derrac_table_8() -> None:
    ranks = average_ranks(DERRAC_TABLE_7_ERROR_RATES, higher_is_better=False)

    assert ranks == pytest.approx(DERRAC_TABLE_8_AVERAGE_RANKS)


def test_friedman_statistics_reproduce_derrac_table_11() -> None:
    """The paper reports chi-squared 35.99733 and Iman-Davenport 5.267817."""
    result = friedman_statistics_from_average_ranks(
        DERRAC_TABLE_11_AVERAGE_RANKS, number_of_problems=25
    )

    assert result.number_of_algorithms == 9
    assert result.number_of_problems == 25
    assert result.chi_squared_statistic == pytest.approx(35.99733, abs=1e-5)
    assert result.iman_davenport_statistic == pytest.approx(5.267817, abs=1e-6)


def test_friedman_p_values_reproduce_derrac_table_11() -> None:
    """The paper reports p = 0.000018 for Friedman and 0.000006 for Iman-Davenport."""
    result = friedman_statistics_from_average_ranks(
        DERRAC_TABLE_11_AVERAGE_RANKS, number_of_problems=25
    )

    assert result.chi_squared_p_value == pytest.approx(0.000018, abs=5e-6)
    assert result.iman_davenport_p_value == pytest.approx(0.000006, abs=5e-7)


def test_friedman_test_ranks_then_computes_the_statistics() -> None:
    result = friedman_test(DERRAC_TABLE_7_ERROR_RATES, higher_is_better=False)

    assert result.average_ranks == pytest.approx(DERRAC_TABLE_8_AVERAGE_RANKS)
    assert result.number_of_algorithms == 4
    assert result.number_of_problems == 4


def test_friedman_test_finds_no_difference_when_every_algorithm_ties() -> None:
    identical = [[0.5, 0.5, 0.5], [0.4, 0.4, 0.4]]

    result = friedman_test(identical, higher_is_better=True)

    assert result.chi_squared_statistic == pytest.approx(0.0)
    assert result.chi_squared_p_value == pytest.approx(1.0)


def test_friedman_post_hoc_z_reproduces_derrac_table_13() -> None:
    """Example 6: control SaDE (rank 1.96), k = 4, n = 25."""
    z_for_chc = friedman_post_hoc_z(
        3.12, 1.96, number_of_algorithms=4, number_of_problems=25
    )
    z_for_ipop = friedman_post_hoc_z(
        2.48, 1.96, number_of_algorithms=4, number_of_problems=25
    )

    assert z_for_chc == pytest.approx(3.176791, abs=1e-6)
    assert z_for_ipop == pytest.approx(1.424079, abs=1e-6)


def test_control_method_index_picks_the_lowest_average_rank() -> None:
    assert control_method_index(DERRAC_TABLE_12_AVERAGE_RANKS) == 3


def test_holm_adjusted_p_values_multiply_by_the_descending_family_size() -> None:
    # m = 3: the smallest p is multiplied by 3, the next by 2, the last by 1.
    adjusted = holm_adjusted_p_values([0.001, 0.02, 0.04])

    assert adjusted == pytest.approx([0.003, 0.04, 0.04])


def test_holm_adjusted_p_values_are_monotone_non_decreasing() -> None:
    # 1 * 0.04 = 0.04 is smaller than the running maximum 2 * 0.02 = 0.04,
    # so the last value is raised to preserve the step-down ordering.
    adjusted = holm_adjusted_p_values([0.03, 0.04])

    assert adjusted[1] >= adjusted[0]
    assert adjusted == pytest.approx([0.06, 0.06])


def test_holm_adjusted_p_values_are_capped_at_one() -> None:
    adjusted = holm_adjusted_p_values([0.4, 0.5, 0.9])

    assert all(value <= 1.0 for value in adjusted)


def test_holm_adjusted_p_values_respect_the_original_order() -> None:
    # The input is unsorted; the output must line up with the input positions.
    adjusted = holm_adjusted_p_values([0.04, 0.001])

    assert adjusted[1] < adjusted[0]


def test_holm_adjusted_p_values_of_nothing_is_empty() -> None:
    assert holm_adjusted_p_values([]) == []


def test_post_hoc_control_compares_every_algorithm_against_the_best_ranked() -> None:
    result = friedman_statistics_from_average_ranks(
        DERRAC_TABLE_12_AVERAGE_RANKS, number_of_problems=25
    )

    comparisons = post_hoc_control_comparisons(
        result, ["IPOP-CMA-ES", "CHC", "SS-BLX", "SaDE"], alpha=0.05
    )

    assert len(comparisons) == 3
    assert {comparison.right_name for comparison in comparisons} == {"SaDE"}
    by_left = {comparison.left_name: comparison for comparison in comparisons}
    assert by_left["CHC"].z_statistic == pytest.approx(3.176791, abs=1e-6)
    assert by_left["CHC"].unadjusted_p_value == pytest.approx(0.001489, abs=5e-7)
    assert by_left["SS-BLX"].unadjusted_p_value == pytest.approx(0.188667, abs=5e-6)


def test_post_hoc_control_rejects_only_where_holm_allows() -> None:
    result = friedman_statistics_from_average_ranks(
        DERRAC_TABLE_12_AVERAGE_RANKS, number_of_problems=25
    )

    comparisons = post_hoc_control_comparisons(
        result, ["IPOP-CMA-ES", "CHC", "SS-BLX", "SaDE"], alpha=0.05
    )

    by_left = {comparison.left_name: comparison for comparison in comparisons}
    # 3 * 0.001489 = 0.004467, comfortably below 0.05.
    assert by_left["CHC"].is_rejected is True
    assert by_left["SS-BLX"].is_rejected is False


def test_post_hoc_all_pairs_covers_every_unordered_pair() -> None:
    result = friedman_statistics_from_average_ranks([1.0, 2.0, 3.0], number_of_problems=10)

    comparisons = post_hoc_all_pairs_comparisons(result, ["a", "b", "c"], alpha=0.05)

    assert len(comparisons) == 3
    assert {(c.left_name, c.right_name) for c in comparisons} == {
        ("a", "b"),
        ("a", "c"),
        ("b", "c"),
    }


def test_post_hoc_comparisons_are_sorted_by_unadjusted_p_value() -> None:
    result = friedman_statistics_from_average_ranks([1.0, 2.0, 3.4], number_of_problems=10)

    comparisons = post_hoc_all_pairs_comparisons(result, ["a", "b", "c"], alpha=0.05)

    p_values = [comparison.unadjusted_p_value for comparison in comparisons]
    assert p_values == sorted(p_values)
