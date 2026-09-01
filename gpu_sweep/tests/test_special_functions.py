"""Tail probabilities, checked against textbook and published values."""

from __future__ import annotations

import math

import pytest

from gpu_sweep.special_functions import (
    chi_squared_survival,
    f_distribution_survival,
    regularized_incomplete_beta,
    standard_normal_two_tailed_p_value,
)


def test_regularized_incomplete_beta_is_zero_and_one_at_the_ends() -> None:
    assert regularized_incomplete_beta(2.0, 3.0, 0.0) == 0.0
    assert regularized_incomplete_beta(2.0, 3.0, 1.0) == 1.0


def test_regularized_incomplete_beta_is_the_identity_for_a_equals_b_equals_one() -> None:
    # I_x(1, 1) == x, because Beta(1, 1) is the uniform distribution.
    assert regularized_incomplete_beta(1.0, 1.0, 0.25) == pytest.approx(0.25)
    assert regularized_incomplete_beta(1.0, 1.0, 0.75) == pytest.approx(0.75)


def test_regularized_incomplete_beta_is_one_half_at_the_symmetric_midpoint() -> None:
    assert regularized_incomplete_beta(3.0, 3.0, 0.5) == pytest.approx(0.5)


def test_f_distribution_survival_matches_a_textbook_critical_value() -> None:
    # F(3, 10) 95th percentile is 3.708, so the upper tail there is 0.05.
    assert f_distribution_survival(3.708, 3, 10) == pytest.approx(0.05, abs=1e-4)


def test_f_distribution_survival_is_one_half_at_one_for_f_one_one() -> None:
    assert f_distribution_survival(1.0, 1, 1) == pytest.approx(0.5)


def test_f_distribution_survival_reproduces_the_derrac_iman_davenport_p_value() -> None:
    """Derrac et al. (2011) Example 5: F_ID = 5.267817 with k = 9, n = 25.

    The paper reports the p-value as 0.000006.
    """
    p_value = f_distribution_survival(5.267817, 8, 192)

    assert p_value == pytest.approx(0.000006, abs=5e-7)


def test_f_distribution_survival_stays_positive_deep_in_the_tail() -> None:
    """Computed as 1 - lower_tail this returns exactly 0.0; the true value is 3.58e-18."""
    assert f_distribution_survival(30.0, 5, 95) > 0.0
    assert f_distribution_survival(30.0, 5, 95) == pytest.approx(3.578e-18, rel=1e-2)


def test_f_distribution_survival_is_one_at_or_below_zero() -> None:
    assert f_distribution_survival(0.0, 3, 10) == 1.0
    assert f_distribution_survival(-1.0, 3, 10) == 1.0


def test_chi_squared_survival_matches_a_textbook_critical_value() -> None:
    # chi-squared with 3 degrees of freedom has its 95th percentile at 7.815.
    assert chi_squared_survival(7.815, 3) == pytest.approx(0.05, abs=1e-4)


def test_chi_squared_survival_reproduces_the_derrac_friedman_p_value() -> None:
    """Derrac et al. (2011) Example 5: chi-squared = 35.99733 with k - 1 = 8.

    The paper reports the p-value as 0.000018.
    """
    p_value = chi_squared_survival(35.99733, 8)

    assert p_value == pytest.approx(0.000018, abs=5e-6)


def test_chi_squared_survival_stays_positive_deep_in_the_tail() -> None:
    """The regime this sweep actually reaches.

    With 20 problems and 6 algorithms a perfectly consistent ordering gives
    chi-squared = 100.0 exactly. Computed as 1 - lower_tail that returns 0.0;
    the true upper tail is 4.27e-18.
    """
    assert chi_squared_survival(100.0, 8) > 0.0
    assert chi_squared_survival(100.0, 8) == pytest.approx(4.269e-18, rel=1e-2)


def test_standard_normal_two_tailed_p_value_is_one_at_zero() -> None:
    assert standard_normal_two_tailed_p_value(0.0) == pytest.approx(1.0)


def test_standard_normal_two_tailed_p_value_ignores_the_sign() -> None:
    assert standard_normal_two_tailed_p_value(-1.96) == pytest.approx(
        standard_normal_two_tailed_p_value(1.96)
    )


def test_standard_normal_two_tailed_p_value_matches_the_familiar_1_96() -> None:
    assert standard_normal_two_tailed_p_value(1.959964) == pytest.approx(0.05, abs=1e-6)


def test_standard_normal_two_tailed_p_value_reproduces_a_derrac_post_hoc_p_value() -> None:
    """Derrac et al. (2011) Example 6, Table 13: z = 3.176791 gives p = 0.001489."""
    assert standard_normal_two_tailed_p_value(3.176791) == pytest.approx(0.001489, abs=5e-7)


def test_standard_normal_two_tailed_p_value_agrees_with_erfc() -> None:
    for z_value in (0.5, 1.0, 2.5, 4.0):
        assert standard_normal_two_tailed_p_value(z_value) == pytest.approx(
            math.erfc(z_value / math.sqrt(2.0))
        )
