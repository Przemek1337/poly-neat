from __future__ import annotations

import torch

from polyneat.algorithms.neat.neat_genome import NEATGenome
from polyneat.algorithms.neat.torch_feedforward_phenotype import TorchFeedForwardPhenotype


class NEATPhenotypeBuilder:
    """Builds a ``TorchFeedForwardPhenotype`` from a ``NEATGenome``."""

    def __init__(self, device_for_computation: torch.device = torch.device("cpu")) -> None:
        self._device_for_computation = device_for_computation

    def build_phenotype_from_genome(self, genome: NEATGenome) -> TorchFeedForwardPhenotype:
        return TorchFeedForwardPhenotype(
            neat_genome=genome,
            device_for_computation=self._device_for_computation,
        )
