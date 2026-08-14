from __future__ import annotations

import torch

from polyneat.algorithms.hyperneat.substrate import (
    build_substrate_from_explicit_layer_coordinates,
)
from polyneat.algorithms.hyperneatleo.leo_phenotype_decoder import (
    HyperNEATLEOPhenotypeDecoder,
    scale_leo_output_to_substrate_weight,
)


class _StubCPPNPhenotype:
    """Returns a fixed (weight, leo) pair for every queried connection."""

    def __init__(self, weight_output: float, leo_output: float) -> None:
        self._weight_output = weight_output
        self._leo_output = leo_output
        self.last_query_tensor: torch.Tensor | None = None

    def forward_pass(self, input_tensor: torch.Tensor) -> torch.Tensor:
        self.last_query_tensor = input_tensor
        return torch.tensor(
            [[self._weight_output, self._leo_output]] * input_tensor.shape[0],
            dtype=torch.float32,
        )

    def reset_recurrent_state(self) -> None:
        return None


class _StubCPPNDecoder:
    def __init__(self, phenotype: _StubCPPNPhenotype) -> None:
        self._phenotype = phenotype

    def build_phenotype_from_genome(self, genome):  # noqa: ANN001, ANN201
        return self._phenotype


def _decoder(weight_output: float, leo_output: float):
    substrate = build_substrate_from_explicit_layer_coordinates(
        layer_x_coordinates=((-1.0, 1.0), (-1.0, 1.0)),
        coordinate_range_min=-1.0,
        coordinate_range_max=1.0,
        bias_x_coordinate=0.0,
    )
    stub_phenotype = _StubCPPNPhenotype(weight_output, leo_output)
    decoder = HyperNEATLEOPhenotypeDecoder(
        substrate=substrate,
        cppn_phenotype_decoder=_StubCPPNDecoder(stub_phenotype),
        link_expression_threshold=0.0,
        max_substrate_connection_weight_magnitude=3.0,
        substrate_node_activation_function_name="steepened_sigmoid",
        device_for_computation=torch.device("cpu"),
    )
    return decoder, stub_phenotype


def test_scaling_is_linear_with_no_cutoff() -> None:
    assert scale_leo_output_to_substrate_weight(0.0, max_magnitude=3.0) == 0.0
    assert scale_leo_output_to_substrate_weight(1.0, max_magnitude=3.0) == 3.0
    assert scale_leo_output_to_substrate_weight(-1.0, max_magnitude=3.0) == -3.0
    assert scale_leo_output_to_substrate_weight(0.5, max_magnitude=3.0) == 1.5


def test_scaling_clamps_magnitudes_above_one() -> None:
    assert scale_leo_output_to_substrate_weight(7.0, max_magnitude=3.0) == 3.0
    assert scale_leo_output_to_substrate_weight(-7.0, max_magnitude=3.0) == -3.0


def test_tiny_weight_is_kept_when_leo_is_positive() -> None:
    # The whole point of LEO: classic HyperNEAT would cut |w| <= 0.2.
    decoder, _stub = _decoder(weight_output=0.01, leo_output=1.0)
    decoded = decoder.decode_substrate_genome(genome=None)
    assert len(decoded.connection_genes) > 0
    assert all(abs(gene.weight) < 0.1 for gene in decoded.connection_genes)


def test_large_weight_is_dropped_when_leo_is_negative() -> None:
    decoder, _stub = _decoder(weight_output=1.0, leo_output=-1.0)
    assert decoder.decode_substrate_genome(genome=None).connection_genes == ()


def test_expression_threshold_is_inclusive() -> None:
    # The source's rule is LEO >= 0, so a link sitting exactly on the threshold
    # counts as expressed.
    decoder, _stub = _decoder(weight_output=1.0, leo_output=0.0)
    assert len(decoder.decode_substrate_genome(genome=None).connection_genes) > 0


def test_expression_rejects_values_below_the_threshold() -> None:
    decoder, _stub = _decoder(weight_output=1.0, leo_output=-1e-6)
    assert decoder.decode_substrate_genome(genome=None).connection_genes == ()


def test_cppn_is_queried_with_hyperneats_four_coordinate_columns() -> None:
    # No delta columns: the locality seed builds the difference inside the CPPN,
    # from x1 and x2 through opposite-signed weights.
    decoder, stub = _decoder(weight_output=1.0, leo_output=1.0)
    decoder.decode_substrate_genome(genome=None)
    assert stub.last_query_tensor is not None
    assert stub.last_query_tensor.shape[1] == 4


def test_every_substrate_node_becomes_a_node_gene() -> None:
    decoder, _stub = _decoder(weight_output=1.0, leo_output=1.0)
    decoded = decoder.decode_substrate_genome(genome=None)
    assert len(decoded.node_genes) == 5  # 2 in + 2 out + bias


def test_input_and_bias_nodes_use_identity_activation() -> None:
    decoder, _stub = _decoder(weight_output=1.0, leo_output=1.0)
    decoded = decoder.decode_substrate_genome(genome=None)
    for node in decoded.node_genes:
        if node.node_type in ("input", "bias"):
            assert node.activation_function_name == "identity"
        else:
            assert node.activation_function_name == "steepened_sigmoid"


def test_build_phenotype_runs_the_decoded_genome() -> None:
    decoder, _stub = _decoder(weight_output=1.0, leo_output=1.0)
    phenotype = decoder.build_phenotype_from_genome(genome=None)
    outputs = phenotype.forward_pass(torch.zeros(3, 2))
    assert outputs.shape == (3, 2)


def test_substrate_is_exposed_as_a_property() -> None:
    decoder, _stub = _decoder(weight_output=1.0, leo_output=1.0)
    assert decoder.substrate.input_layer.nodes[0].x_coordinate == -1.0


def test_expressed_connections_get_contiguous_innovation_ids() -> None:
    decoder, _stub = _decoder(weight_output=1.0, leo_output=1.0)
    decoded = decoder.decode_substrate_genome(genome=None)
    assert [gene.innovation_id for gene in decoded.connection_genes] == list(
        range(len(decoded.connection_genes))
    )
