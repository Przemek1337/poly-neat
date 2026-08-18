"""Decode a substrate ANN from a CPPN genome (the HyperNEAT mapping).

The decoder queries a CPPN for the weight of every candidate substrate
connection, applies the paper's expression threshold and magnitude scaling, and
assembles the surviving connections into a synthetic ``NEATGenome`` executed by
the existing feed-forward phenotype.

References:
    Stanley, K. O., D'Ambrosio, D. B., & Gauci, J. (2009). A Hypercube-Based Encoding for
        Evolving Large-Scale Neural Networks. *Artificial Life*, 15(2), 185-212.
        DOI: 10.1162/artl.2009.15.2.15202
"""

from __future__ import annotations

import torch

from polyneat.algorithms.hyperneat.substrate import Substrate, SubstrateNode
from polyneat.core.neat.neat_genome import ConnectionGene, NEATGenome, NodeGene
from polyneat.core.neat.neat_phenotype_decoder import NEATPhenotypeDecoder
from polyneat.core.neat.torch_feedforward_phenotype import TorchFeedForwardPhenotype
from polyneat.logging_utils.custom_logger import get_logger

logger = get_logger(__name__)


def scale_cppn_output_to_substrate_weight(
    raw_cppn_output: float,
    threshold: float,
    max_magnitude: float,
) -> float | None:
    """Map a raw CPPN output to a substrate connection weight (or ``None``).

    Implements the paper's expression rule (Stanley et al. 2009, Table 1:
    outputs of magnitude <= 0.2 are unexpressed; larger outputs are scaled to a
    magnitude between zero and three).

    Args:
        raw_cppn_output: The CPPN's raw output for one queried connection.
        threshold: Expression threshold ``w_min``; connections at or below this
            magnitude are not expressed.
        max_magnitude: Maximum substrate connection magnitude an expressed
            connection is scaled toward.

    Returns:
        ``None`` when ``|raw_cppn_output| <= threshold`` (the connection is not
        expressed). Otherwise the magnitude is clamped to 1.0 and linearly
        rescaled from ``[threshold, 1]`` to ``[0, max_magnitude]``, preserving
        sign.
    """
    output_magnitude = abs(raw_cppn_output)
    if output_magnitude <= threshold:
        return None
    clamped_magnitude = min(output_magnitude, 1.0)
    scaled_magnitude = (clamped_magnitude - threshold) / (1.0 - threshold) * max_magnitude
    sign = 1.0 if raw_cppn_output >= 0.0 else -1.0
    return sign * scaled_magnitude


class HyperNEATPhenotypeDecoder:
    """Decode a substrate ANN from a CPPN genome by querying the CPPN.

    For every candidate substrate connection (between adjacent layers, plus the
    bias node to every non-input node), the CPPN is queried once with the two
    endpoints' coordinates ``(x1, y1, x2, y2)``. Outputs above the expression
    threshold become weighted connections in a synthetic ``NEATGenome`` that is
    executed by the existing ``TorchFeedForwardPhenotype``.
    """

    def __init__(
        self,
        substrate: Substrate,
        cppn_phenotype_decoder: NEATPhenotypeDecoder,
        weight_expression_threshold: float,
        max_substrate_connection_weight_magnitude: float,
        substrate_node_activation_function_name: str,
        device_for_computation: torch.device,
    ) -> None:
        """Store the substrate layout and CPPN-to-weight interpretation settings.

        Args:
            substrate: The fixed substrate whose connections are painted.
            cppn_phenotype_decoder: Decoder that turns a CPPN genome into an
                executable CPPN phenotype used to answer the coordinate queries.
            weight_expression_threshold: Magnitude ``w_min`` below which a
                queried connection is not expressed.
            max_substrate_connection_weight_magnitude: Maximum substrate
                connection magnitude expressed connections are scaled toward.
            substrate_node_activation_function_name: Activation applied to hidden
                and output substrate nodes (input and bias nodes use identity).
            device_for_computation: Torch device the substrate phenotype runs on.
        """
        self._substrate = substrate
        self._cppn_phenotype_decoder = cppn_phenotype_decoder
        self._weight_expression_threshold = weight_expression_threshold
        self._max_substrate_connection_weight_magnitude = max_substrate_connection_weight_magnitude
        self._substrate_node_activation_function_name = substrate_node_activation_function_name
        self._device_for_computation = device_for_computation

    @property
    def substrate(self) -> Substrate:
        """The substrate this decoder paints, exposed for geometry-based metrics."""
        return self._substrate

    def build_phenotype_from_genome(self, genome: NEATGenome) -> TorchFeedForwardPhenotype:
        """Decode a CPPN genome into an executable substrate phenotype.

        Args:
            genome: The CPPN genome (a `NEATGenome` with 4 coordinate inputs and
                1 weight output) to query.

        Returns:
            A `TorchFeedForwardPhenotype` running the decoded substrate.
        """
        return TorchFeedForwardPhenotype(
            neat_genome=self.decode_substrate_genome(genome),
            device_for_computation=self._device_for_computation,
        )

    def decode_substrate_genome(self, genome: NEATGenome) -> NEATGenome:
        """Query the CPPN and return the substrate as a genome.

        Separate from :meth:`build_phenotype_from_genome` because
        ``TorchFeedForwardPhenotype`` does not retain the genome it was built
        from - it decomposes it into a topological order and connection maps - so
        structural measurements, such as the modularity metrics used to compare
        this decoder with the HyperNEAT-LEO one, have to obtain it here.

        Args:
            genome: The CPPN genome to query.

        Returns:
            A synthetic `NEATGenome` with one node gene per substrate node, and
            only the connections whose queried weight cleared the expression
            threshold.
        """
        cppn_phenotype = self._cppn_phenotype_decoder.build_phenotype_from_genome(genome)

        candidate_source_target_node_pairs = self._enumerate_candidate_connections()

        substrate_node_genes = self._build_substrate_node_genes()

        if not candidate_source_target_node_pairs:
            return NEATGenome(node_genes=substrate_node_genes, connection_genes=())

        coordinate_query_tensor = torch.tensor(
            [
                [
                    source_node.x_coordinate,
                    source_node.y_coordinate,
                    target_node.x_coordinate,
                    target_node.y_coordinate,
                ]
                for source_node, target_node in candidate_source_target_node_pairs
            ],
            dtype=torch.float32,
            device=self._device_for_computation,
        )
        with torch.no_grad():
            raw_cppn_outputs = cppn_phenotype.forward_pass(coordinate_query_tensor)
        raw_output_values = raw_cppn_outputs[:, 0].cpu().tolist()

        substrate_connection_genes: list[ConnectionGene] = []
        next_innovation_id = 0
        for (source_node, target_node), raw_output_value in zip(
            candidate_source_target_node_pairs, raw_output_values, strict=True
        ):
            scaled_weight = scale_cppn_output_to_substrate_weight(
                raw_cppn_output=raw_output_value,
                threshold=self._weight_expression_threshold,
                max_magnitude=self._max_substrate_connection_weight_magnitude,
            )
            if scaled_weight is None:
                continue
            substrate_connection_genes.append(
                ConnectionGene(
                    innovation_id=next_innovation_id,
                    source_node_id=source_node.node_id,
                    target_node_id=target_node.node_id,
                    weight=scaled_weight,
                    is_enabled=True,
                )
            )
            next_innovation_id += 1

        logger.debug(
            "HyperNEAT substrate decoded: %d of %d candidate connections expressed over %d nodes",
            len(substrate_connection_genes),
            len(candidate_source_target_node_pairs),
            len(substrate_node_genes),
        )
        return NEATGenome(
            node_genes=substrate_node_genes,
            connection_genes=tuple(substrate_connection_genes),
        )

    def _build_substrate_node_genes(self) -> tuple[NodeGene, ...]:
        """Build the synthetic node genes for the substrate.

        Returns:
            One `NodeGene` per substrate node. Input and bias nodes use the
            identity activation; hidden and output nodes use the configured
            substrate activation function.
        """
        node_genes: list[NodeGene] = []
        for substrate_node in self._substrate.all_nodes():
            if substrate_node.role in ("input", "bias"):
                activation_function_name = "identity"
            else:
                activation_function_name = self._substrate_node_activation_function_name
            node_genes.append(
                NodeGene(
                    node_id=substrate_node.node_id,
                    node_type=substrate_node.role,
                    activation_function_name=activation_function_name,
                )
            )
        return tuple(node_genes)

    def _enumerate_candidate_connections(
        self,
    ) -> list[tuple[SubstrateNode, SubstrateNode]]:
        """Enumerate every substrate connection the CPPN should be queried for.

        Returns:
            ``(source_node, target_node)`` pairs for every connection between
            adjacent layers, plus the bias node to every hidden and output node.
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
