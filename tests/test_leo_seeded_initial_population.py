from __future__ import annotations

import math

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

_WEIGHT_OUTPUT_NODE_ID = 7
_LEO_OUTPUT_NODE_ID = 8
_DELTA_X_INPUT_NODE_ID = 4
_BIAS_NODE_ID = 6

_RETINA_X = (-1.00, -0.85, -0.70, -0.55, 0.55, 0.70, 0.85, 1.00)


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


def test_every_genome_has_a_gaussian_node_fed_by_delta_x() -> None:
    for genome in _population().genomes:
        gaussian_node_ids = {
            node.node_id
            for node in genome.node_genes
            if node.activation_function_name == "gaussian"
        }
        assert gaussian_node_ids, "no gaussian node in the seed"
        fed_by_delta_x = {
            connection.target_node_id
            for connection in genome.connection_genes
            if connection.source_node_id == _DELTA_X_INPUT_NODE_ID and connection.is_enabled
        }
        assert gaussian_node_ids & fed_by_delta_x


def test_gaussian_node_and_bias_both_feed_the_leo_output() -> None:
    for genome in _population().genomes:
        gaussian_node_ids = {
            node.node_id
            for node in genome.node_genes
            if node.activation_function_name == "gaussian"
        }
        sources_into_leo = {
            connection.source_node_id
            for connection in genome.connection_genes
            if connection.target_node_id == _LEO_OUTPUT_NODE_ID and connection.is_enabled
        }
        assert gaussian_node_ids <= sources_into_leo
        assert _BIAS_NODE_ID in sources_into_leo


def test_delta_inputs_do_not_feed_the_weight_output() -> None:
    # The weight function starts identical to plain HyperNEAT's; wiring the
    # deltas into it is left to evolution.
    for genome in _population().genomes:
        sources_into_weight = {
            connection.source_node_id
            for connection in genome.connection_genes
            if connection.target_node_id == _WEIGHT_OUTPUT_NODE_ID
        }
        assert sources_into_weight == {0, 1, 2, 3, _BIAS_NODE_ID}


def test_bias_into_leo_carries_the_configured_negative_weight() -> None:
    population = _population(population_size=3, locality_seed_bias_weight=-0.25)
    for genome in population.genomes:
        bias_into_leo = [
            connection
            for connection in genome.connection_genes
            if connection.source_node_id == _BIAS_NODE_ID
            and connection.target_node_id == _LEO_OUTPUT_NODE_ID
        ]
        assert len(bias_into_leo) == 1
        assert bias_into_leo[0].weight == -0.25


def test_delta_into_gaussian_carries_the_configured_delta_weight() -> None:
    population = _population(population_size=3, locality_seed_delta_weight=2.5)
    for genome in population.genomes:
        delta_edges = [
            connection
            for connection in genome.connection_genes
            if connection.source_node_id == _DELTA_X_INPUT_NODE_ID
        ]
        assert len(delta_edges) == 1
        assert delta_edges[0].weight == 2.5


def test_seeding_both_axes_adds_two_gaussian_nodes() -> None:
    population = _population(population_size=2, locality_seed_coordinate_axes=("x", "y"))
    for genome in population.genomes:
        gaussian_nodes = [
            node for node in genome.node_genes if node.activation_function_name == "gaussian"
        ]
        assert len(gaussian_nodes) == 2


def test_all_genomes_share_the_seed_innovation_ids() -> None:
    # Historical markings must align across generation 0, or crossover cannot
    # match the identical seed structure.
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


def test_seed_threshold_matches_the_documented_formula() -> None:
    config = HyperNEATLEOConfig(locality_seed_delta_weight=1.0, locality_seed_bias_weight=-0.5)
    expected_cutoff = math.sqrt(
        math.log(-1.0 / config.locality_seed_bias_weight)
    ) / config.locality_seed_delta_weight
    assert abs(expected_cutoff - 0.8326) < 1e-3


def _retina_decoder(config: HyperNEATLEOConfig) -> HyperNEATLEOPhenotypeDecoder:
    substrate = build_substrate_from_explicit_layer_coordinates(
        layer_x_coordinates=(_RETINA_X, _RETINA_X, (-0.775, 0.775)),
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
                gene
                for gene in decoded.connection_genes
                if gene.target_node_id == output_node_id
            ]
            assert incoming, "an output ended up with no incoming connection at all"
            assert all(
                x_by_node_id[gene.source_node_id] * x_by_node_id[output_node_id] >= 0
                for gene in incoming
            )
