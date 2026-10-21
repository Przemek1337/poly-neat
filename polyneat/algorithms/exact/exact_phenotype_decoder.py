"""Decoder mapping EXACT genomes to convolutional torch phenotypes."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from polyneat.algorithms.exact.exact_genome import EXACTGenome
from polyneat.algorithms.exact.torch_convolutional_phenotype import (
    TorchConvolutionalPhenotype,
)
from polyneat.core.component_protocols import Phenotype


@dataclass
class EXACTPhenotypeDecoder:
    """Builds a :class:`TorchConvolutionalPhenotype` from an EXACT genome.

    Attributes:
        input_image_height: Height flat input rows are reshaped to.
        input_image_width: Width flat input rows are reshaped to.
        leaky_relu_negative_slope: Hidden activation leak (paper: 0.1).
        activation_clamp_maximum: Hidden activation cap (paper: 5.5).
        device_for_computation: Device the phenotypes run on.
        use_batch_normalization: Whether hidden nodes are batch normalized
            (section VII). Evaluation phenotypes run in eval mode, so the
            stored running statistics are used as-is and dropout is inert.
    """

    input_image_height: int
    input_image_width: int
    leaky_relu_negative_slope: float
    activation_clamp_maximum: float
    device_for_computation: torch.device
    use_batch_normalization: bool = False

    def build_phenotype_from_genome(self, genome: EXACTGenome) -> Phenotype:
        """Express ``genome`` as an executable, trainable CNN.

        Args:
            genome: EXACT genome to express.

        Returns:
            The convolutional phenotype on the configured device.
        """
        return TorchConvolutionalPhenotype(
            exact_genome=genome,
            input_image_height=self.input_image_height,
            input_image_width=self.input_image_width,
            leaky_relu_negative_slope=self.leaky_relu_negative_slope,
            activation_clamp_maximum=self.activation_clamp_maximum,
            device_for_computation=self.device_for_computation,
            use_batch_normalization=self.use_batch_normalization,
        )
