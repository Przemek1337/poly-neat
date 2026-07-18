from __future__ import annotations

import math

import torch

from polyneat.algorithms.hyperneat.hyperneat_phenotype_decoder import (
    HyperNEATPhenotypeDecoder,
    scale_cppn_output_to_substrate_weight,
)
from polyneat.algorithms.hyperneat.substrate import build_layered_substrate
from polyneat.core.neat.neat_genome import ConnectionGene, NEATGenome, NodeGene
from polyneat.core.neat.neat_phenotype_decoder import NEATPhenotypeDecoder


def _cppn_genome_with_constant_output(output_weight: float) -> NEATGenome:
    """A CPPN whose single output equals ``output_weight`` from the bias alone.

    Inputs 0..3 are the coordinates (x1, y1, x2, y2); node 4 is the bias; node 5
    is the identity-activated output. Only the bias->output connection is
    enabled, so the CPPN outputs ``output_weight`` for every query.
    """
    node_genes = (
        NodeGene(node_id=0, node_type="input", activation_function_name="identity"),
        NodeGene(node_id=1, node_type="input", activation_function_name="identity"),
        NodeGene(node_id=2, node_type="input", activation_function_name="identity"),
        NodeGene(node_id=3, node_type="input", activation_function_name="identity"),
        NodeGene(node_id=4, node_type="bias", activation_function_name="identity"),
        NodeGene(node_id=5, node_type="output", activation_function_name="identity"),
    )
    connection_genes = (
        ConnectionGene(
            innovation_id=0,
            source_node_id=4,
            target_node_id=5,
            weight=output_weight,
            is_enabled=True,
        ),
    )
    return NEATGenome(node_genes=node_genes, connection_genes=connection_genes)


def test_scaling_drops_connections_below_threshold():
    assert scale_cppn_output_to_substrate_weight(0.1, threshold=0.2, max_magnitude=3.0) is None
    assert scale_cppn_output_to_substrate_weight(-0.2, threshold=0.2, max_magnitude=3.0) is None


def test_scaling_maps_unit_output_to_max_magnitude_preserving_sign():
    scaled_positive = scale_cppn_output_to_substrate_weight(1.0, threshold=0.2, max_magnitude=3.0)
    scaled_negative = scale_cppn_output_to_substrate_weight(-1.0, threshold=0.2, max_magnitude=3.0)
    assert scaled_positive is not None and math.isclose(scaled_positive, 3.0, abs_tol=1e-6)
    assert scaled_negative is not None and math.isclose(scaled_negative, -3.0, abs_tol=1e-6)


def test_decoder_produces_phenotype_whose_output_width_matches_substrate_outputs():
    substrate = build_layered_substrate(
        input_layer_size=2,
        hidden_layer_sizes=(3,),
        output_layer_size=1,
        coordinate_range_min=-1.0,
        coordinate_range_max=1.0,
        include_bias_node=True,
    )
    decoder = HyperNEATPhenotypeDecoder(
        substrate=substrate,
        cppn_phenotype_decoder=NEATPhenotypeDecoder(device_for_computation=torch.device("cpu")),
        weight_expression_threshold=0.2,
        max_substrate_connection_weight_magnitude=3.0,
        substrate_node_activation_function_name="steepened_sigmoid",
        device_for_computation=torch.device("cpu"),
    )
    # constant output 1.0 -> every candidate connection is expressed at full magnitude
    phenotype = decoder.build_phenotype_from_genome(_cppn_genome_with_constant_output(1.0))
    output_tensor = phenotype.forward_pass(torch.tensor([[0.0, 1.0]]))
    # substrate has one output node
    assert output_tensor.shape == (1, 1)


def test_below_threshold_cppn_produces_a_disconnected_substrate():
    substrate = build_layered_substrate(
        input_layer_size=2,
        hidden_layer_sizes=(2,),
        output_layer_size=1,
        coordinate_range_min=-1.0,
        coordinate_range_max=1.0,
        include_bias_node=True,
    )
    decoder = HyperNEATPhenotypeDecoder(
        substrate=substrate,
        cppn_phenotype_decoder=NEATPhenotypeDecoder(device_for_computation=torch.device("cpu")),
        weight_expression_threshold=0.2,
        max_substrate_connection_weight_magnitude=3.0,
        substrate_node_activation_function_name="steepened_sigmoid",
        device_for_computation=torch.device("cpu"),
    )
    # constant output 0.05 (< threshold) -> no connection expressed
    phenotype = decoder.build_phenotype_from_genome(_cppn_genome_with_constant_output(0.05))
    # forward still runs; with no incoming connections the output node sees a
    # zero pre-activation and returns a finite value
    output_tensor = phenotype.forward_pass(torch.tensor([[0.0, 1.0]]))
    assert torch.isfinite(output_tensor).all()
