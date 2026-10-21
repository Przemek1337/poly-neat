from __future__ import annotations

from polyneat.algorithms.hyperneat.substrate import (
    build_substrate_from_explicit_layer_coordinates,
)
from polyneat.algorithms.hyperneat.substrate_modularity import (
    count_cross_hemisphere_connections,
    count_expressed_connections,
)
from polyneat.core.neat.neat_genome import ConnectionGene, NEATGenome, NodeGene

# ids: 0 (x=-1), 1 (x=+1) inputs | 2 (x=-1), 3 (x=+1) outputs | 4 bias (x=0)
_SUBSTRATE = build_substrate_from_explicit_layer_coordinates(
    layer_x_coordinates=((-1.0, 1.0), (-1.0, 1.0)),
    coordinate_range_min=-1.0,
    coordinate_range_max=1.0,
    bias_x_coordinate=0.0,
)

_NODE_GENES = tuple(
    NodeGene(
        node_id=node.node_id,
        node_type=node.role,
        activation_function_name="identity",
    )
    for node in _SUBSTRATE.all_nodes()
)


def _genome(edges: list[tuple[int, int]]) -> NEATGenome:
    return NEATGenome(
        node_genes=_NODE_GENES,
        connection_genes=tuple(
            ConnectionGene(
                innovation_id=index,
                source_node_id=source,
                target_node_id=target,
                weight=1.0,
                is_enabled=True,
            )
            for index, (source, target) in enumerate(edges)
        ),
    )


def test_same_hemisphere_connections_do_not_count() -> None:
    assert count_cross_hemisphere_connections(_SUBSTRATE, _genome([(0, 2), (1, 3)])) == 0


def test_opposite_hemisphere_connections_count() -> None:
    assert count_cross_hemisphere_connections(_SUBSTRATE, _genome([(0, 3), (1, 2)])) == 2


def test_mixed_genome_counts_only_the_crossing_ones() -> None:
    assert (
        count_cross_hemisphere_connections(_SUBSTRATE, _genome([(0, 2), (0, 3), (1, 3)])) == 1
    )


def test_bias_at_x_zero_belongs_to_neither_hemisphere() -> None:
    assert count_cross_hemisphere_connections(_SUBSTRATE, _genome([(4, 2), (4, 3)])) == 0


def test_disabled_connections_do_not_count() -> None:
    genome = NEATGenome(
        node_genes=_NODE_GENES,
        connection_genes=(
            ConnectionGene(
                innovation_id=0, source_node_id=0, target_node_id=3, weight=1.0, is_enabled=False
            ),
        ),
    )
    assert count_cross_hemisphere_connections(_SUBSTRATE, genome) == 0
    assert count_expressed_connections(genome) == 0


def test_expressed_count_is_the_number_of_enabled_genes() -> None:
    assert count_expressed_connections(_genome([(0, 2), (1, 3), (0, 3)])) == 3


def test_empty_genome_counts_zero_of_both() -> None:
    empty = NEATGenome(node_genes=_NODE_GENES, connection_genes=())
    assert count_cross_hemisphere_connections(_SUBSTRATE, empty) == 0
    assert count_expressed_connections(empty) == 0


def test_hyperneat_decoder_exposes_the_decoded_genome_too() -> None:
    # The baseline example must measure modularity with exactly the same code as
    # the LEO one, so the plain decoder needs the same accessor.
    import torch

    from polyneat.algorithms.hyperneat.hyperneat_phenotype_decoder import (
        HyperNEATPhenotypeDecoder,
    )
    from polyneat.core.neat.neat_genome import NEATGenome as CPPNGenome
    from polyneat.core.neat.neat_phenotype_decoder import NEATPhenotypeDecoder

    cppn_genome = CPPNGenome(
        node_genes=(
            NodeGene(node_id=0, node_type="input", activation_function_name="identity"),
            NodeGene(node_id=1, node_type="input", activation_function_name="identity"),
            NodeGene(node_id=2, node_type="input", activation_function_name="identity"),
            NodeGene(node_id=3, node_type="input", activation_function_name="identity"),
            NodeGene(node_id=4, node_type="bias", activation_function_name="identity"),
            NodeGene(node_id=5, node_type="output", activation_function_name="identity"),
        ),
        connection_genes=(
            ConnectionGene(
                innovation_id=0, source_node_id=4, target_node_id=5, weight=1.0, is_enabled=True
            ),
        ),
    )
    decoder = HyperNEATPhenotypeDecoder(
        substrate=_SUBSTRATE,
        cppn_phenotype_decoder=NEATPhenotypeDecoder(device_for_computation=torch.device("cpu")),
        weight_expression_threshold=0.2,
        max_substrate_connection_weight_magnitude=3.0,
        substrate_node_activation_function_name="steepened_sigmoid",
        device_for_computation=torch.device("cpu"),
    )
    decoded = decoder.decode_substrate_genome(cppn_genome)
    assert isinstance(decoded, CPPNGenome)
    assert decoder.substrate is _SUBSTRATE
    # A constant bias of 1.0 clears the 0.2 threshold, so every candidate is expressed.
    assert count_expressed_connections(decoded) > 0
    # And the phenotype built from the same decoder runs.
    assert decoder.build_phenotype_from_genome(cppn_genome) is not None


def test_functional_modularity_ignores_which_side_a_hidden_node_sits_on() -> None:
    # The distinction the coordinate-based count cannot make: routing the left
    # inputs through a RIGHT-side hidden node into the left output never mixes
    # left and right information, so it is perfectly modular - even though every
    # one of its connections crosses x = 0.
    from polyneat.algorithms.hyperneat.substrate_modularity import (
        count_cross_hemisphere_input_dependencies,
    )

    substrate = build_substrate_from_explicit_layer_coordinates(
        layer_x_coordinates=((-1.0, 1.0), (-0.5, 0.5), (-1.0, 1.0)),
        coordinate_range_min=-1.0,
        coordinate_range_max=1.0,
        bias_x_coordinate=0.0,
    )
    # ids: 0,1 inputs | 2,3 hidden | 4,5 outputs | 6 bias
    node_genes = tuple(
        NodeGene(
            node_id=node.node_id, node_type=node.role, activation_function_name="identity"
        )
        for node in substrate.all_nodes()
    )

    def genome(edges: list[tuple[int, int]]) -> NEATGenome:
        return NEATGenome(
            node_genes=node_genes,
            connection_genes=tuple(
                ConnectionGene(
                    innovation_id=index,
                    source_node_id=source,
                    target_node_id=target,
                    weight=1.0,
                    is_enabled=True,
                )
                for index, (source, target) in enumerate(edges)
            ),
        )

    # left input 0 -> right hidden 3 -> left output 4, mirrored for the right.
    fully_crossed_wiring = genome([(0, 3), (3, 4), (1, 2), (2, 5)])
    assert count_cross_hemisphere_connections(substrate, fully_crossed_wiring) == 4
    assert count_cross_hemisphere_input_dependencies(substrate, fully_crossed_wiring) == 0


def test_functional_modularity_counts_a_genuine_information_leak() -> None:
    from polyneat.algorithms.hyperneat.substrate_modularity import (
        count_cross_hemisphere_input_dependencies,
    )

    substrate = build_substrate_from_explicit_layer_coordinates(
        layer_x_coordinates=((-1.0, 1.0), (-0.5, 0.5), (-1.0, 1.0)),
        coordinate_range_min=-1.0,
        coordinate_range_max=1.0,
        bias_x_coordinate=0.0,
    )
    node_genes = tuple(
        NodeGene(
            node_id=node.node_id, node_type=node.role, activation_function_name="identity"
        )
        for node in substrate.all_nodes()
    )
    # right input 1 also reaches the LEFT output 4: a real leak.
    leaking = NEATGenome(
        node_genes=node_genes,
        connection_genes=tuple(
            ConnectionGene(
                innovation_id=index,
                source_node_id=source,
                target_node_id=target,
                weight=1.0,
                is_enabled=True,
            )
            for index, (source, target) in enumerate([(0, 2), (1, 2), (2, 4)])
        ),
    )
    assert count_cross_hemisphere_input_dependencies(substrate, leaking) == 1


def test_functional_modularity_is_zero_for_a_disconnected_substrate() -> None:
    from polyneat.algorithms.hyperneat.substrate_modularity import (
        count_cross_hemisphere_input_dependencies,
    )

    empty = NEATGenome(node_genes=_NODE_GENES, connection_genes=())
    assert count_cross_hemisphere_input_dependencies(_SUBSTRATE, empty) == 0
