"""What the shared trainer needs from a model, and nothing more.

The base :class:`~polyneat.core.component_protocols.Phenotype` deliberately
promises only a forward pass and a recurrent-state reset. Most algorithms in
this library never train by backpropagation, and requiring every phenotype to
own optimizer state, checkpointing and a PyTorch dependency would push one
family's needs onto all of them.

This protocol is the narrower contract, declared next to the component that
requires it. A phenotype that happens to be an ``nn.Module`` satisfies it
already; anything else can satisfy it without inheriting from anything.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class TrainableModel(Protocol):
    """A model the shared trainer can run a supervised session against."""

    def forward_pass(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Return raw logits for one batch."""
        ...

    def parameters(self) -> Iterator[torch.nn.Parameter]:
        """Yield the tensors an optimizer should update."""
        ...

    def train(self, mode: bool = True) -> TrainableModel:
        """Switch to training mode, so dropout and normalization behave as such."""
        ...

    def eval(self) -> TrainableModel:
        """Switch to evaluation mode."""
        ...

    def state_dict(self) -> dict:
        """Return parameters and buffers, for a checkpoint snapshot."""
        ...

    def load_state_dict(self, state: dict) -> object:
        """Restore parameters and buffers saved by :meth:`state_dict`."""
        ...


def move_model_to_device(model: TrainableModel, device: torch.device) -> TrainableModel:
    """Put ``model`` on ``device`` when it is a module that can be moved.

    The protocol above deliberately does not promise ``.to()``: a PolyNEAT
    phenotype is built for a device by its decoder and owns that placement
    already. A plain ``nn.Module`` handed in by a baseline or rebuilt for a
    retraining track has no such decoder, so it arrives wherever it was
    constructed - the CPU - and would meet batches on another device.

    Moving here rather than at every construction site keeps the rule in one
    place: whatever runs a forward pass puts the model where the batches are.

    Args:
        model: Model about to be trained or scored.
        device: Device its batches will live on.

    Returns:
        The same model, on ``device`` when it was movable. Models that manage
        their own placement are returned untouched.
    """
    if isinstance(model, torch.nn.Module):
        return model.to(device)
    return model
