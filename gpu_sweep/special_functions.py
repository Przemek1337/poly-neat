"""Tail probabilities for the Friedman statistic and the Holm post-hoc z.

The chi-squared tail comes from ``torch.special.gammaincc``, which is exact
enough to reproduce published critical values; the normal tail comes from
``math.erfc``. Neither needs scipy.
"""

from __future__ import annotations

import math

import torch


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
