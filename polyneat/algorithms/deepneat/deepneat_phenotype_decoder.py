"""Decoder mapping DeepNEAT genomes to executable torch layer-stack phenotypes."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from polyneat.algorithms.deepneat.deepneat_genome import DeepNEATGenome
from polyneat.algorithms.deepneat.layer_shape_propagation import TensorShape
from polyneat.algorithms.deepneat.torch_layer_stack_phenotype import (
    TorchLayerStackPhenotype,
)


@dataclass
class DeepNEATPhenotypeDecoder:
    """Builds a :class:`TorchLayerStackPhenotype` from a DeepNEAT genome.

    A thin wrapper: every genome-independent parameter the phenotype needs —
    the input shape, the number of output classes, the parameter budget and
    the target device — is fixed once here, at decoder construction, so that
    building a phenotype for any genome only requires the genome itself.

    Attributes:
        input_shape: Shape of the tensor a built phenotype's ``forward_pass``
            expects, excluding the batch dimension.
        number_of_classes: Width of every built phenotype's output layer.
        maximum_total_parameter_count: Parameter budget passed through to
            every built phenotype; see
            :class:`~polyneat.algorithms.deepneat.torch_layer_stack_phenotype
            .TorchLayerStackPhenotype` for what exceeding it does.
        device_for_computation: Device every built phenotype is moved to.

    References:
        Miikkulainen, R., Liang, J., Meyerson, E., Rawal, A., Fink, D., Francon, O.,
            Raju, B., Shahrzad, H., Navruzyan, A., Duffy, N., & Hodjat, B. (2017).
            Evolving Deep Neural Networks. *arXiv:1703.00548*. Published in
            *Artificial Intelligence in the Age of Neural Networks and Brain
            Computing* (2019), pp. 293-312. DOI: 10.1016/B978-0-12-815480-9.00015-3
    """

    input_shape: TensorShape
    number_of_classes: int
    maximum_total_parameter_count: int
    device_for_computation: torch.device

    def build_phenotype_from_genome(self, genome: DeepNEATGenome) -> TorchLayerStackPhenotype:
        """Express ``genome`` as an executable, trainable network.

        Args:
            genome: DeepNEAT genome to express.

        Returns:
            The layer-stack phenotype, moved to ``device_for_computation``.
        """
        phenotype = TorchLayerStackPhenotype(
            genome=genome,
            input_shape=self.input_shape,
            number_of_classes=self.number_of_classes,
            maximum_total_parameter_count=self.maximum_total_parameter_count,
            device_for_computation=self.device_for_computation,
        )
        return phenotype.to(self.device_for_computation)
