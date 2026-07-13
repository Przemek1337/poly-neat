from __future__ import annotations

import math

import torch

from polyneat.nn.activation_functions import (
    absolute_value_activation,
    gaussian_activation,
    resolve_activation_function_by_name,
    sine_activation,
)


def test_gaussian_peaks_at_zero_and_decays():
    values = gaussian_activation(torch.tensor([0.0, 1.0, -1.0]))
    assert math.isclose(values[0].item(), 1.0, abs_tol=1e-6)
    assert values[1].item() < 1.0
    # even function: g(1) == g(-1)
    assert math.isclose(values[1].item(), values[2].item(), abs_tol=1e-6)


def test_sine_matches_torch_sin():
    x = torch.tensor([0.0, math.pi / 2.0, math.pi])
    expected = torch.sin(x)
    assert torch.allclose(sine_activation(x), expected, atol=1e-6)


def test_absolute_value_is_nonnegative_and_even():
    values = absolute_value_activation(torch.tensor([-2.0, 2.0, -0.5]))
    assert torch.all(values >= 0.0)
    assert math.isclose(values[0].item(), values[1].item(), abs_tol=1e-6)


def test_new_functions_are_registered_by_name():
    for name in ("gaussian", "sine", "absolute_value"):
        resolved = resolve_activation_function_by_name(name)
        assert callable(resolved)
