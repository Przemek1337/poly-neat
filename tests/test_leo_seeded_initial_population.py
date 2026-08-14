from __future__ import annotations

import numpy as np
import torch

from polyneat.algorithms.hyperneat.substrate import (
    build_substrate_from_explicit_layer_coordinates,
)
from polyneat.algorithms.hyperneatleo.leo_phenotype_decoder import (
    HyperNEATLEOPhenotypeDecoder,
)
from polyneat.algorithms.hyperneatleo.leo_seeded_initial_population import (
    build_leo_seeded_initial_population,
)
from polyneat.configs.hyperneatleo.hyperneatleo_config import HyperNEATLEOConfig
from polyneat.core.neat.global_innovation_tracker import GlobalInnovationTracker
from polyneat.core.neat.initial_population import (
    resolve_initial_population_strategy_by_name,
)
from polyneat.core.neat.neat_phenotype_decoder import NEATPhenotypeDecoder

# inputs 0..3 = x1, y1, x2, y2 | bias 4 | outputs 5 = weight, 6 = LEO | gaussians from 7
_X1_INPUT_NODE_ID = 0
_X2_INPUT_NODE_ID = 2
_BIAS_NODE_ID = 4
_WEIGHT_OUTPUT_NODE_ID = 5
_LEO_OUTPUT_NODE_ID = 6

_RETINA_X = (-1.0, -0.9167, -0.8333, -0.75, 0.75, 0.8333, 0.9167, 1.0)


def _population(population_size: int = 6, **config_overrides):
    config = HyperNEATLEOConfig(
        population_size=population_size, random_seed=0, **config_overrides
    )
    return build_leo_seeded_initial_population(
        config, GlobalInnovationTracker(), np.random.default_rng(0)
    )


def test_strategy_is_registered_under_its_name() -> None:
    assert (
        resolve_initial_population_strategy_by_name("leo_seeded")
        is build_leo_seeded_initial_population
    )


def test_population_has_the_configured_size() -> None:
    assert len(_population(population_size=11).genomes) == 11


def test_generation_number_is_zero() -> None:
    assert _population().generation_number == 0


def test_gaussian_node_receives_both_x_coordinates_with_opposite_weights() -> None:
    # This is what makes a separate delta input unnecessary: the weighted sum
    # entering the Gaussian already is w * (x2 - x1).
    for genome in _population().genomes:
        gaussian_node_ids = {
            node.node_id
            for node in genome.node_genes
            if node.activation_function_name == "gaussian"
        }
        assert len(gaussian_node_ids) == 1
        gaussian_node_id = next(iter(gaussian_node_ids))

        into_gaussian = {
            connection.source_node_id: connection.weight
            for connection in genome.connection_genes
            if connection.target_node_id == gaussian_node_id and connection.is_enabled
        }
        assert set(into_gaussian) == {_X1_INPUT_NODE_ID, _X2_INPUT_NODE_ID}
        assert into_gaussian[_X1_INPUT_NODE_ID] == -0.6
        assert into_gaussian[_X2_INPUT_NODE_ID] == 0.6
        assert into_gaussian[_X1_INPUT_NODE_ID] == -into_gaussian[_X2_INPUT_NODE_ID]


def test_gaussian_node_and_bias_both_feed_the_leo_output() -> None:
    for genome in _population().genomes:
        gaussian_node_ids = {
            node.node_id
            for node in genome.node_genes
            if node.activation_function_name == "gaussian"
        }
        into_leo = {
            connection.source_node_id: connection.weight
            for connection in genome.connection_genes
            if connection.target_node_id == _LEO_OUTPUT_NODE_ID and connection.is_enabled
        }
        assert gaussian_node_ids <= set(into_leo)
        assert into_leo[_BIAS_NODE_ID] == -1.0
        for gaussian_node_id in gaussian_node_ids:
            assert into_leo[gaussian_node_id] == 2.0


def test_leo_output_uses_a_hyperbolic_tangent() -> None:
    # The smooth stand-in for the original's step function; "expressed" is then a
    # sign test on the pre-activation.
    for genome in _population().genomes:
        leo_node = next(
            node for node in genome.node_genes if node.node_id == _LEO_OUTPUT_NODE_ID
        )
        assert leo_node.activation_function_name == "tanh"


def test_weight_output_is_wired_like_a_plain_hyperneat_cppn() -> None:
    for genome in _population().genomes:
        sources_into_weight = {
            connection.source_node_id
            for connection in genome.connection_genes
            if connection.target_node_id == _WEIGHT_OUTPUT_NODE_ID
        }
        assert sources_into_weight == {0, 1, 2, 3, _BIAS_NODE_ID}


def test_seed_constants_are_configurable() -> None:
    population = _population(
        population_size=3,
        locality_seed_delta_weight=0.9,
        locality_seed_gaussian_to_leo_weight=3.0,
        locality_seed_bias_weight=-1.5,
    )
    for genome in population.genomes:
        weight_by_edge = {
            (c.source_node_id, c.target_node_id): c.weight for c in genome.connection_genes
        }
        gaussian_node_id = next(
            node.node_id
            for node in genome.node_genes
            if node.activation_function_name == "gaussian"
        )
        assert weight_by_edge[(_X1_INPUT_NODE_ID, gaussian_node_id)] == -0.9
        assert weight_by_edge[(_X2_INPUT_NODE_ID, gaussian_node_id)] == 0.9
        assert weight_by_edge[(gaussian_node_id, _LEO_OUTPUT_NODE_ID)] == 3.0
        assert weight_by_edge[(_BIAS_NODE_ID, _LEO_OUTPUT_NODE_ID)] == -1.5


def test_seeding_both_axes_adds_two_gaussian_nodes() -> None:
    population = _population(population_size=2, locality_seed_coordinate_axes=("x", "y"))
    for genome in population.genomes:
        gaussian_nodes = [
            node for node in genome.node_genes if node.activation_function_name == "gaussian"
        ]
        assert len(gaussian_nodes) == 2


def test_all_genomes_share_the_seed_innovation_ids() -> None:
    genomes = _population(population_size=5).genomes
    seed_edges_per_genome = [
        {
            (c.source_node_id, c.target_node_id): c.innovation_id
            for c in genome.connection_genes
            if c.target_node_id == _LEO_OUTPUT_NODE_ID
        }
        for genome in genomes
    ]
    assert all(edges == seed_edges_per_genome[0] for edges in seed_edges_per_genome)


def test_genomes_differ_in_their_weight_output_wiring_weights() -> None:
    genomes = _population(population_size=5).genomes
    weight_output_weights = [
        tuple(
            c.weight for c in genome.connection_genes if c.target_node_id == _WEIGHT_OUTPUT_NODE_ID
        )
        for genome in genomes
    ]
    assert len(set(weight_output_weights)) > 1


def _retina_decoder(config: HyperNEATLEOConfig) -> HyperNEATLEOPhenotypeDecoder:
    substrate = build_substrate_from_explicit_layer_coordinates(
        layer_x_coordinates=(_RETINA_X, _RETINA_X, (-0.875, 0.875)),
        coordinate_range_min=-1.0,
        coordinate_range_max=1.0,
        bias_x_coordinate=0.0,
    )
    return HyperNEATLEOPhenotypeDecoder(
        substrate=substrate,
        cppn_phenotype_decoder=NEATPhenotypeDecoder(device_for_computation=torch.device("cpu")),
        link_expression_threshold=config.link_expression_threshold,
        max_substrate_connection_weight_magnitude=(
            config.max_substrate_connection_weight_magnitude
        ),
        substrate_node_activation_function_name=config.substrate_node_activation_function,
        device_for_computation=torch.device("cpu"),
    )


def test_decoded_generation_zero_has_no_cross_hemisphere_connections() -> None:
    # The property the whole experiment rests on.
    config = HyperNEATLEOConfig(population_size=8, random_seed=0)
    population = build_leo_seeded_initial_population(
        config, GlobalInnovationTracker(), np.random.default_rng(0)
    )
    decoder = _retina_decoder(config)
    x_by_node_id = {node.node_id: node.x_coordinate for node in decoder.substrate.all_nodes()}

    for genome in population.genomes:
        decoded = decoder.decode_substrate_genome(genome)
        crossing = [
            gene
            for gene in decoded.connection_genes
            if x_by_node_id[gene.source_node_id] * x_by_node_id[gene.target_node_id] < 0
        ]
        assert crossing == [], f"seed expressed {len(crossing)} cross-hemisphere links"
        assert decoded.connection_genes, "seed expressed nothing at all"


def test_decoded_generation_zero_still_connects_each_hemisphere_to_its_output() -> None:
    config = HyperNEATLEOConfig(population_size=4, random_seed=0)
    population = build_leo_seeded_initial_population(
        config, GlobalInnovationTracker(), np.random.default_rng(0)
    )
    decoder = _retina_decoder(config)
    x_by_node_id = {node.node_id: node.x_coordinate for node in decoder.substrate.all_nodes()}
    output_node_ids = {node.node_id for node in decoder.substrate.output_layer.nodes}

    for genome in population.genomes:
        decoded = decoder.decode_substrate_genome(genome)
        for output_node_id in output_node_ids:
            incoming = [
                gene for gene in decoded.connection_genes if gene.target_node_id == output_node_id
            ]
            assert incoming, "an output ended up with no incoming connection at all"
            assert all(
                x_by_node_id[gene.source_node_id] * x_by_node_id[output_node_id] >= 0
                for gene in incoming
            )
