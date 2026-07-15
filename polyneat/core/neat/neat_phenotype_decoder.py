from __future__ import annotations

import torch

from polyneat.core.neat.neat_genome import NEATGenome
from polyneat.core.neat.torch_feedforward_phenotype import TorchFeedForwardPhenotype


class NEATPhenotypeDecoder:
    """Decodes a ``NEATGenome`` into an executable ``TorchFeedForwardPhenotype``.

    The genotype-to-phenotype mapping of the paper (section 3.1, Figure 2):
    only enabled connection genes are expressed in the network.
    """

    def __init__(self, device_for_computation: torch.device | None = None) -> None:
        """Bind the decoder to the device all built phenotypes will run on.

        Args:
            device_for_computation: Torch device for phenotype evaluation;
                ``None`` means CPU.
        """
        self._device_for_computation = device_for_computation or torch.device("cpu")

    def build_phenotype_from_genome(self, genome: NEATGenome) -> TorchFeedForwardPhenotype:
        """Build an executable feed-forward network from ``genome``.

        Args:
            genome: Genome to decode.

        Returns:
            A phenotype ready for ``forward_pass`` on the configured device.
        """
        return TorchFeedForwardPhenotype(
            neat_genome=genome,
            device_for_computation=self._device_for_computation,
        )
