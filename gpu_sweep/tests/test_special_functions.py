"""Tail probabilities, checked against textbook and published values."""

from __future__ import annotations

import math

import pytest

from gpu_sweep.special_functions import (
    chi_squared_survival,
    standard_normal_two_tailed_p_value,
)


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
