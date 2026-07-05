from __future__ import annotations

from collections.abc import Callable

import torch

from polyneat.config.configuration_errors import ConfigurationError

ActivationFunction = Callable[[torch.Tensor], torch.Tensor]


def sigmoid_activation(pre_activation_tensor: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(pre_activation_tensor)


def tanh_activation(pre_activation_tensor: torch.Tensor) -> torch.Tensor:
    return torch.tanh(pre_activation_tensor)


def relu_activation(pre_activation_tensor: torch.Tensor) -> torch.Tensor:
    return torch.relu(pre_activation_tensor)


def leaky_relu_activation(pre_activation_tensor: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.leaky_relu(pre_activation_tensor, negative_slope=0.01)


def identity_activation(pre_activation_tensor: torch.Tensor) -> torch.Tensor:
    return pre_activation_tensor


ACTIVATION_FUNCTION_NAME_TO_CALLABLE: dict[str, ActivationFunction] = {
    "sigmoid": sigmoid_activation,
    "tanh": tanh_activation,
    "relu": relu_activation,
    "leaky_relu": leaky_relu_activation,
    "identity": identity_activation,
}


def resolve_activation_function_by_name(activation_function_name: str) -> ActivationFunction:
    """Return the callable registered under ``activation_function_name``.

    Raises ``ConfigurationError`` when the name is not registered so genome
    validation surfaces the typo at construction time rather than at the first
    forward pass.
    """
    if activation_function_name not in ACTIVATION_FUNCTION_NAME_TO_CALLABLE:
        raise ConfigurationError(
            f"Unknown activation function name '{activation_function_name}'. "
            f"Known: {sorted(ACTIVATION_FUNCTION_NAME_TO_CALLABLE.keys())}"
        )
    return ACTIVATION_FUNCTION_NAME_TO_CALLABLE[activation_function_name]
