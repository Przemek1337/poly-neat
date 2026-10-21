"""One shared, recorded parameter initialization for track B.

Track B asks a different question from track A: how useful is the *topology*
that a search found, once every model is trained the same way. Answering it
requires taking the evolved learning recipe out of the picture, and the
initialization is part of that recipe - EXACT inherits kernels, DeepNEAT
evolves a weight-scaling gene. So track B overwrites both with one scheme,
applied by layer kind and driven by an explicit generator.

The scheme is the usual one for rectifier networks: Kaiming normal on
convolution and linear weights, zero biases, and normalization layers reset to
an identity transform with cleared running statistics. Because the draws come
from a passed-in generator rather than the global torch state, two models
initialized with the same stream start from the same numbers regardless of what
else the process did in between.

References:
    He, K., Zhang, X., Ren, S., & Sun, J. (2015). Delving Deep into Rectifiers:
        Surpassing Human-Level Performance on ImageNet Classification.
        *ICCV 2015*. arXiv:1502.01852.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from polyneat.logging_utils.custom_logger import get_logger

logger = get_logger(__name__)

# Gain of the rectifier nonlinearity assumed by the shared scheme; see He et al.
# (2015), section 2.2. A leaky variant would use a slightly smaller gain, but
# track B fixes one scheme rather than tracking each algorithm's activations.
_RECTIFIER_GAIN = math.sqrt(2.0)


def _kaiming_normal_(weight: torch.Tensor, generator: torch.Generator) -> None:
    """Fill ``weight`` in place with He-normal values drawn from ``generator``."""
    fan_in = weight[0].numel() if weight.ndim > 1 else weight.numel()
    standard_deviation = _RECTIFIER_GAIN / math.sqrt(max(fan_in, 1))
    drawn = torch.randn(weight.shape, generator=generator, dtype=torch.float32)
    with torch.no_grad():
        weight.copy_((drawn * standard_deviation).to(dtype=weight.dtype, device=weight.device))


def initialize_module_parameters(module: nn.Module, generator: torch.Generator) -> int:
    """Reinitialize every layer of ``module`` under the shared track B scheme.

    Args:
        module: Model to reinitialize in place.
        generator: CPU generator owning the initialization stream. Draws are
            taken here and moved to the parameter's device, so the values do
            not depend on which device the run got.

    Returns:
        How many layers were reinitialized, so a caller can assert that a
        reset actually touched something.
    """
    number_of_reinitialized_layers = 0
    for submodule in module.modules():
        if isinstance(submodule, nn.Conv1d | nn.Conv2d | nn.Conv3d | nn.Linear):
            _kaiming_normal_(submodule.weight, generator)
            if submodule.bias is not None:
                with torch.no_grad():
                    submodule.bias.zero_()
            number_of_reinitialized_layers += 1
        elif isinstance(submodule, nn.modules.batchnorm._NormBase | nn.LayerNorm | nn.GroupNorm):
            with torch.no_grad():
                if getattr(submodule, "weight", None) is not None:
                    submodule.weight.fill_(1.0)
                if getattr(submodule, "bias", None) is not None:
                    submodule.bias.zero_()
                running_mean = getattr(submodule, "running_mean", None)
                running_variance = getattr(submodule, "running_var", None)
                batches_tracked = getattr(submodule, "num_batches_tracked", None)
                if running_mean is not None:
                    running_mean.zero_()
                if running_variance is not None:
                    running_variance.fill_(1.0)
                if batches_tracked is not None:
                    batches_tracked.zero_()
            number_of_reinitialized_layers += 1
    logger.debug(
        "Shared initialization touched %d layers", number_of_reinitialized_layers
    )
    return number_of_reinitialized_layers
