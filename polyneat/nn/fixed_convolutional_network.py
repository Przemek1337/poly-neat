"""A small, fixed-architecture CNN used as a hand-designed baseline.

A neuroevolution result only means something against something. This is that
something: a conventional convolutional stack whose architecture is chosen by
hand, frozen before the result series, and trained through exactly the same
path as every evolved candidate - same preprocessing, same class weights, same
recipe, same split permissions.

The architecture is a parameter, not a constant, because the choice belongs to
whichever benchmark uses it. What lives here is only the assembly: a sequence
of convolution-normalization-activation blocks, each followed by pooling, then
global average pooling and a linear classifier. If several hand-designed
configurations are compared, the cost of trying them counts as baseline tuning
and has to be reported as such.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn


@dataclass(frozen=True)
class FixedConvolutionalNetworkConfig:
    """The frozen shape of the hand-designed baseline.

    Attributes:
        input_channels: Channels of the input images.
        number_of_classes: Output width of the classifier.
        convolution_channels: Output channels of each convolution block, in
            order. Its length is the depth of the stack.
        kernel_size: Square kernel side used by every block.
        uses_batch_normalization: Whether each block normalizes before its
            activation.
        dropout_probability: Dropout applied before the classifier.
    """

    input_channels: int = 1
    number_of_classes: int = 2
    convolution_channels: tuple[int, ...] = field(default=(32, 64, 128))
    kernel_size: int = 3
    uses_batch_normalization: bool = True
    dropout_probability: float = 0.25

    def __post_init__(self) -> None:
        if not self.convolution_channels:
            raise ValueError("convolution_channels must name at least one block")
        if self.kernel_size < 1 or self.kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be a positive odd number, got {self.kernel_size}")
        if not 0.0 <= self.dropout_probability < 1.0:
            raise ValueError(
                f"dropout_probability must be in [0, 1), got {self.dropout_probability}"
            )

    def describe(self) -> str:
        """One-line description recorded in the protocol lock and artifacts."""
        blocks = "-".join(f"conv{channels}" for channels in self.convolution_channels)
        normalization = "bn" if self.uses_batch_normalization else "nobn"
        return f"{blocks}-{normalization}-gap-fc{self.number_of_classes}"


class FixedConvolutionalNetwork(nn.Module):
    """The baseline network, satisfying the shared trainable-model contract.

    Global average pooling before the classifier is what makes one frozen
    architecture usable at more than one input resolution: the smoke profile
    runs at 64 pixels and the full protocol at 128, and the classifier width
    does not change with either.
    """

    def __init__(self, config: FixedConvolutionalNetworkConfig | None = None) -> None:
        """Build the stack described by ``config``."""
        super().__init__()
        self._config = config or FixedConvolutionalNetworkConfig()

        blocks: list[nn.Module] = []
        in_channels = self._config.input_channels
        for out_channels in self._config.convolution_channels:
            blocks.append(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=self._config.kernel_size,
                    padding=self._config.kernel_size // 2,
                    bias=not self._config.uses_batch_normalization,
                )
            )
            if self._config.uses_batch_normalization:
                blocks.append(nn.BatchNorm2d(out_channels))
            blocks.append(nn.ReLU(inplace=True))
            blocks.append(nn.MaxPool2d(kernel_size=2, ceil_mode=True))
            in_channels = out_channels

        self._feature_extractor = nn.Sequential(*blocks)
        self._global_pool = nn.AdaptiveAvgPool2d(1)
        self._dropout = nn.Dropout(self._config.dropout_probability)
        self._classifier = nn.Linear(in_channels, self._config.number_of_classes)

    @property
    def config(self) -> FixedConvolutionalNetworkConfig:
        """The frozen architecture this instance was built from."""
        return self._config

    @property
    def total_parameter_count(self) -> int:
        """Trainable parameters, reported alongside the evolved models."""
        return sum(int(parameter.numel()) for parameter in self.parameters())

    def forward_pass(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Return raw class logits for an ``NCHW`` batch."""
        features = self._feature_extractor(input_tensor)
        pooled = self._global_pool(features).flatten(start_dim=1)
        return self._classifier(self._dropout(pooled))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``nn.Module`` compatibility shim - delegates to :meth:`forward_pass`."""
        return self.forward_pass(x)

    def reset_recurrent_state(self) -> None:
        """No recurrent state to reset; present for the phenotype contract."""

    def reinitialize_parameters(self) -> None:
        """Reinitialize every layer under the shared track B scheme.

        Delegates to :func:`~polyneat.training.parameter_initialization
        .initialize_module_parameters` so the baseline and the evolved models
        start from the same scheme rather than from two different defaults.
        """
        from polyneat.training.parameter_initialization import initialize_module_parameters

        generator = torch.Generator()
        generator.manual_seed(int(torch.randint(0, 2**62, (1,)).item()))
        initialize_module_parameters(self, generator)
