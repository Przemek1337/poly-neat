"""Bridging the shared image path to EXACT's flat phenotype interface.

An EXACT phenotype takes one flat row per image and reshapes it internally to
the height and width its genome was built for. The shared preprocessing works
in ``NCHW``, because padding, resizing, affine transforms and contrast are
spatial operations that a flat vector cannot express.

This adapter is the seam between the two. It accepts a spatial batch, flattens
it in row-major order and hands it to the phenotype - so the augmentation
happens on images, and the pixel ordering the phenotype receives is exactly
what it received before this adapter existed.

Everything else is delegation. The adapter owns no parameters of its own, so
the checkpoint it produces is the phenotype's checkpoint.
"""

from __future__ import annotations

from collections.abc import Iterator

import torch
from torch import nn

from polyneat.algorithms.exact.torch_convolutional_phenotype import (
    TorchConvolutionalPhenotype,
)


class FlattenedImageInputAdapter(nn.Module):
    """Presents an EXACT phenotype as a model that takes ``NCHW`` batches."""

    def __init__(self, phenotype: TorchConvolutionalPhenotype) -> None:
        """Wrap ``phenotype`` without copying or rebuilding it."""
        super().__init__()
        self._phenotype = phenotype

    @property
    def phenotype(self) -> TorchConvolutionalPhenotype:
        """The wrapped phenotype, for callers that need its EXACT-specific API."""
        return self._phenotype

    @property
    def is_degenerate(self) -> bool:
        """Whether the wrapped phenotype reports itself unusable."""
        return bool(getattr(self._phenotype, "is_degenerate", False))

    def forward_pass(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Flatten a spatial batch and run the phenotype on it.

        Args:
            input_tensor: ``(batch, channels, height, width)`` batch. A batch
                that is already flat is passed through unchanged, so the
                adapter is safe to place in front of either caller.

        Returns:
            The phenotype's raw logits.
        """
        if input_tensor.ndim == 4:
            input_tensor = input_tensor.reshape(input_tensor.shape[0], -1)
        return self._phenotype.forward_pass(input_tensor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``nn.Module`` compatibility shim - delegates to :meth:`forward_pass`."""
        return self.forward_pass(x)

    def parameters(self, recurse: bool = True) -> Iterator[nn.Parameter]:
        """Yield the wrapped phenotype's parameters; the adapter adds none."""
        return self._phenotype.parameters(recurse=recurse)

    def reset_recurrent_state(self) -> None:
        """Delegate to the wrapped phenotype."""
        self._phenotype.reset_recurrent_state()
