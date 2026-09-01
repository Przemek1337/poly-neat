"""Tail probabilities for the Friedman and Iman-Davenport statistics.

The chi-squared tail comes from ``torch.special.gammainc``, which is exact
enough to reproduce published critical values. The F tail needs a regularized
incomplete beta, which torch does not provide - ``torch.special.betainc`` does
not exist and ``torch.distributions.FisherSnedecor.cdf`` raises
``NotImplementedError`` - so it is written here as a Lentz continued fraction
rather than by adding scipy to a throwaway harness.
"""

from __future__ import annotations

import math

import torch

_CONTINUED_FRACTION_ITERATIONS = 300
_CONTINUED_FRACTION_EPSILON = 3e-16
_CONTINUED_FRACTION_TINY = 1e-30


def _beta_continued_fraction(a: float, b: float, x: float) -> float:
    """Evaluate the continued fraction of the incomplete beta by Lentz's method.

    Args:
        a: First shape parameter, positive.
        b: Second shape parameter, positive.
        x: Point in ``(0, 1)``.

    Returns:
        The continued fraction value used by
        :func:`regularized_incomplete_beta`.
    """
    a_plus_b = a + b
    a_plus_one = a + 1.0
    a_minus_one = a - 1.0

    numerator_ratio = 1.0
    denominator = 1.0 - a_plus_b * x / a_plus_one
    if abs(denominator) < _CONTINUED_FRACTION_TINY:
        denominator = _CONTINUED_FRACTION_TINY
    denominator = 1.0 / denominator
    fraction_value = denominator

    for iteration in range(1, _CONTINUED_FRACTION_ITERATIONS + 1):
        double_iteration = 2 * iteration

        even_term = (
            iteration
            * (b - iteration)
            * x
            / ((a_minus_one + double_iteration) * (a + double_iteration))
        )
        denominator = 1.0 + even_term * denominator
        if abs(denominator) < _CONTINUED_FRACTION_TINY:
            denominator = _CONTINUED_FRACTION_TINY
        numerator_ratio = 1.0 + even_term / numerator_ratio
        if abs(numerator_ratio) < _CONTINUED_FRACTION_TINY:
            numerator_ratio = _CONTINUED_FRACTION_TINY
        denominator = 1.0 / denominator
        fraction_value *= denominator * numerator_ratio

        odd_term = (
            -(a + iteration)
            * (a_plus_b + iteration)
            * x
            / ((a + double_iteration) * (a_plus_one + double_iteration))
        )
        denominator = 1.0 + odd_term * denominator
        if abs(denominator) < _CONTINUED_FRACTION_TINY:
            denominator = _CONTINUED_FRACTION_TINY
        numerator_ratio = 1.0 + odd_term / numerator_ratio
        if abs(numerator_ratio) < _CONTINUED_FRACTION_TINY:
            numerator_ratio = _CONTINUED_FRACTION_TINY
        denominator = 1.0 / denominator

        step = denominator * numerator_ratio
        fraction_value *= step
        if abs(step - 1.0) < _CONTINUED_FRACTION_EPSILON:
            break
    return fraction_value


def regularized_incomplete_beta(a: float, b: float, x: float) -> float:
    """Return ``I_x(a, b)``, the regularized incomplete beta function.

    The continued fraction converges quickly only on one side of
    ``(a + 1) / (a + b + 2)``, so the other side is evaluated through the
    symmetry ``I_x(a, b) = 1 - I_{1-x}(b, a)``.

    Args:
        a: First shape parameter, positive.
        b: Second shape parameter, positive.
        x: Point in ``[0, 1]``.

    Returns:
        The value in ``[0, 1]``.
    """
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0

    log_front = (
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log1p(-x)
    )
    front = math.exp(log_front)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _beta_continued_fraction(a, b, x) / a
    return 1.0 - front * _beta_continued_fraction(b, a, 1.0 - x) / b


def f_distribution_survival(
    f_value: float, degrees_of_freedom_1: int, degrees_of_freedom_2: int
) -> float:
    """Upper tail probability of the F distribution.

    Args:
        f_value: Observed statistic.
        degrees_of_freedom_1: Numerator degrees of freedom.
        degrees_of_freedom_2: Denominator degrees of freedom.

    Returns:
        ``P(F > f_value)``, which is 1.0 for a non-positive statistic.
    """
    if f_value <= 0.0:
        return 1.0
    x = (degrees_of_freedom_1 * f_value) / (
        degrees_of_freedom_1 * f_value + degrees_of_freedom_2
    )
    # Use the beta symmetry I_x(a, b) = 1 - I_{1-x}(b, a) to obtain the upper
    # tail *directly*. Writing ``1.0 - regularized_incomplete_beta(a, b, x)``
    # cancels to exactly 0.0 once the lower tail rounds to 1.0 - verified at
    # F = 30 with (5, 95) degrees of freedom, where the true tail is 3.58e-18.
    # torch has no ``betaincc``, so the symmetry is the fix.
    return regularized_incomplete_beta(
        degrees_of_freedom_2 / 2.0, degrees_of_freedom_1 / 2.0, 1.0 - x
    )


def chi_squared_survival(chi_squared_value: float, degrees_of_freedom: int) -> float:
    """Upper tail probability of the chi-squared distribution.

    Uses ``torch.special.gammainc`` in double precision - the chi-squared CDF
    is the regularized lower incomplete gamma ``P(df/2, x/2)``.

    Args:
        chi_squared_value: Observed statistic.
        degrees_of_freedom: Degrees of freedom.

    Returns:
        ``P(X > chi_squared_value)``, which is 1.0 for a non-positive statistic.
    """
    if chi_squared_value <= 0.0:
        return 1.0
    # torch.special.gammaincc is the *upper* regularized incomplete gamma.
    # Computing 1 - gammainc(...) instead loses every digit once the lower tail
    # rounds to 1.0: at chi-squared 100 with 8 degrees of freedom the subtraction
    # returns exactly 0.0 where the true value is 4.27e-18. That is not an edge
    # case here - with 20 problems and 6 algorithms a perfectly consistent
    # ordering produces chi-squared = 100.0 exactly, so the strongest possible
    # result is precisely the one the subtraction destroys.
    upper_tail = torch.special.gammaincc(
        torch.tensor(degrees_of_freedom / 2.0, dtype=torch.float64),
        torch.tensor(chi_squared_value / 2.0, dtype=torch.float64),
    )
    return float(upper_tail.item())


def standard_normal_two_tailed_p_value(z_value: float) -> float:
    """Two-tailed p-value of a standard normal statistic.

    Args:
        z_value: Observed z; the sign is irrelevant.

    Returns:
        ``P(|Z| > |z_value|)`` in ``[0, 1]``.
    """
    return math.erfc(abs(z_value) / math.sqrt(2.0))
