"""Decode a substrate ANN from a CPPN whose second output governs link expression.

References:
    Verbancsics, P., & Stanley, K. O. (2011). Constraining Connectivity to Encourage
        Modularity in HyperNEAT. *GECCO '11: Proceedings of the 13th Annual Conference on
        Genetic and Evolutionary Computation*, pp. 1483-1490. DOI: 10.1145/2001576.2001776
"""

from __future__ import annotations

import torch

from polyneat.algorithms.hyperneat.substrate import Substrate, SubstrateNode
from polyneat.core.neat.neat_genome import ConnectionGene, NEATGenome, NodeGene
from polyneat.core.neat.neat_phenotype_decoder import NEATPhenotypeDecoder
from polyneat.core.neat.torch_feedforward_phenotype import TorchFeedForwardPhenotype

_WEIGHT_OUTPUT_COLUMN_INDEX = 0
_LINK_EXPRESSION_OUTPUT_COLUMN_INDEX = 1


def scale_leo_output_to_substrate_weight(
    raw_cppn_output: float,
    max_magnitude: float,
) -> float:
    """Map the CPPN's weight output to a substrate weight, linearly.

    Unlike classic HyperNEAT there is **no** cutoff: whether the connection
    exists is decided by the link expression output, so the weight function is
    free to return values near zero. The magnitude is clamped to 1.0 first, then
    scaled, so the substrate weight stays within ``+/- max_magnitude``.

    Args:
        raw_cppn_output: The CPPN's weight output for one queried connection.
        max_magnitude: Maximum substrate connection magnitude.

    Returns:
        The scaled substrate weight.
    """
    clamped_output = max(-1.0, min(1.0, raw_cppn_output))
    return clamped_output * max_magnitude


class HyperNEATLEOPhenotypeDecoder:
    """Decode a substrate ANN from a CPPN with a link expression output.

    For every candidate substrate connection the CPPN is queried once with the two
    endpoints' coordinates ``(x1, y1, x2, y2)`` - the same four inputs plain
    HyperNEAT uses. A coordinate *difference* needs no extra input: the locality
    seed forms it inside the CPPN, by feeding an axis' two coordinates into a
    Gaussian node through equal and opposite weights.

    The CPPN answers with two numbers. The first becomes the connection weight
    (linearly scaled, no cutoff); the second, the link expression output, decides
    whether the connection exists. Separating the two is the whole idea: classic
    HyperNEAT cannot express a weak-but-present connection, because there
    ``|weight| > threshold`` *is* the existence test.
    """

    def __init__(
        self,
        substrate: Substrate,
        cppn_phenotype_decoder: NEATPhenotypeDecoder,
        link_expression_threshold: float,
        max_substrate_connection_weight_magnitude: float,
        substrate_node_activation_function_name: str,
        device_for_computation: torch.device,
    ) -> None:
        """Store the substrate layout and the expression settings.

        Args:
            substrate: The fixed substrate whose connections are painted.
            cppn_phenotype_decoder: Decoder turning a CPPN genome into an
                executable CPPN used to answer the coordinate queries.
            link_expression_threshold: A connection exists when the CPPN's link
                expression output reaches this value. The source's rule is
                ``LEO >= 0``, so the comparison is inclusive.
            max_substrate_connection_weight_magnitude: Maximum substrate
                connection magnitude.
            substrate_node_activation_function_name: Activation applied to hidden
                and output substrate nodes.
            device_for_computation: Torch device the substrate phenotype runs on.
        """
        self._substrate = substrate
        self._cppn_phenotype_decoder = cppn_phenotype_decoder
        self._link_expression_threshold = link_expression_threshold
        self._max_substrate_connection_weight_magnitude = (
            max_substrate_connection_weight_magnitude
        )
        self._substrate_node_activation_function_name = substrate_node_activation_function_name
        self._device_for_computation = device_for_computation

    @property
    def substrate(self) -> Substrate:
        """The substrate this decoder paints, exposed for geometry-based metrics."""
        return self._substrate

    def build_phenotype_from_genome(self, genome: NEATGenome) -> TorchFeedForwardPhenotype:
        """Decode a CPPN genome into an executable substrate phenotype.

        Args:
            genome: The CPPN genome (6 inputs, 2 outputs) to query.

        Returns:
            A `TorchFeedForwardPhenotype` running the decoded substrate.
        """
        return TorchFeedForwardPhenotype(
            neat_genome=self.decode_substrate_genome(genome),
            device_for_computation=self._device_for_computation,
        )

    def decode_substrate_genome(self, genome: NEATGenome) -> NEATGenome:
        """Query the CPPN and return the substrate as a genome.

        Kept separate from :meth:`build_phenotype_from_genome` because
        ``TorchFeedForwardPhenotype`` does not retain the genome it was built
        from - it decomposes it into a topological order and connection maps -
        so anything needing to inspect the produced structure, such as the
        modularity metrics, has to obtain it here.

        Args:
            genome: The CPPN genome (6 inputs, 2 outputs) to query.

        Returns:
            A synthetic `NEATGenome` holding one node gene per substrate node and
            only the connections whose link expression output cleared the
            threshold.
        """
        cppn_phenotype = self._cppn_phenotype_decoder.build_phenotype_from_genome(genome)
        candidate_pairs = self._enumerate_candidate_connections()
        substrate_node_genes = self._build_substrate_node_genes()

        if not candidate_pairs:
            return NEATGenome(node_genes=substrate_node_genes, connection_genes=())

        coordinate_query_tensor = torch.tensor(
            [
                [
                    source_node.x_coordinate,
                    source_node.y_coordinate,
                    target_node.x_coordinate,
                    target_node.y_coordinate,
                ]
                for source_node, target_node in candidate_pairs
            ],
            dtype=torch.float32,
            device=self._device_for_computation,
        )
        with torch.no_grad():
            raw_cppn_outputs = cppn_phenotype.forward_pass(coordinate_query_tensor)

        raw_weight_values = raw_cppn_outputs[:, _WEIGHT_OUTPUT_COLUMN_INDEX].cpu().tolist()
        raw_expression_values = (
            raw_cppn_outputs[:, _LINK_EXPRESSION_OUTPUT_COLUMN_INDEX].cpu().tolist()
        )

        substrate_connection_genes: list[ConnectionGene] = []
        next_innovation_id = 0
        for (source_node, target_node), raw_weight, raw_expression in zip(
            candidate_pairs, raw_weight_values, raw_expression_values, strict=True
        ):
            if raw_expression < self._link_expression_threshold:
                continue
            substrate_connection_genes.append(
                ConnectionGene(
                    innovation_id=next_innovation_id,
                    source_node_id=source_node.node_id,
                    target_node_id=target_node.node_id,
                    weight=scale_leo_output_to_substrate_weight(
                        raw_cppn_output=raw_weight,
                        max_magnitude=self._max_substrate_connection_weight_magnitude,
                    ),
                    is_enabled=True,
                )
            )
            next_innovation_id += 1

        return NEATGenome(
            node_genes=substrate_node_genes,
            connection_genes=tuple(substrate_connection_genes),
        )

    def _build_substrate_node_genes(self) -> tuple[NodeGene, ...]:
        """Build one node gene per substrate node.

        Returns:
            Input and bias nodes use the identity activation; hidden and output
            nodes use the configured substrate activation.
        """
        node_genes: list[NodeGene] = []
        for substrate_node in self._substrate.all_nodes():
            activation_function_name = (
                "identity"
                if substrate_node.role in ("input", "bias")
                else self._substrate_node_activation_function_name
            )
            node_genes.append(
                NodeGene(
                    node_id=substrate_node.node_id,
                    node_type=substrate_node.role,
                    activation_function_name=activation_function_name,
                )
            )
        return tuple(node_genes)

    def _enumerate_candidate_connections(self) -> list[tuple[SubstrateNode, SubstrateNode]]:
        """Enumerate every connection the CPPN should be queried for.

        Returns:
            ``(source, target)`` pairs between adjacent layers, plus the bias
            node to every hidden and output node.
        """
        candidate_pairs: list[tuple[SubstrateNode, SubstrateNode]] = []
        for source_layer, target_layer in self._substrate.feed_forward_layer_adjacent_pairs():
            for source_node in source_layer.nodes:
                for target_node in target_layer.nodes:
                    candidate_pairs.append((source_node, target_node))

        bias_node = self._substrate.bias_node
        if bias_node is not None:
            for layer in self._substrate.hidden_layers + (self._substrate.output_layer,):
                for target_node in layer.nodes:
                    candidate_pairs.append((bias_node, target_node))

        return candidate_pairs
