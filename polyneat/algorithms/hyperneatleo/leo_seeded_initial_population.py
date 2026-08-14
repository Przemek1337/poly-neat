"""Generation 0 for HyperNEAT-LEO: CPPNs seeded with a bias toward local links.

References:
    Verbancsics, P., & Stanley, K. O. (2011). Constraining Connectivity to Encourage
        Modularity in HyperNEAT. *GECCO '11: Proceedings of the 13th Annual Conference on
        Genetic and Evolutionary Computation*, pp. 1483-1490. DOI: 10.1145/2001576.2001776
"""

from __future__ import annotations

from typing import cast

from numpy.random import Generator

from polyneat.configs.hyperneatleo.hyperneatleo_config import (
    SEED_AXIS_NAME_TO_DELTA_INPUT_NODE_ID,
    HyperNEATLEOConfig,
)
from polyneat.configs.neat.neat_config import NEATConfig
from polyneat.core.component_protocols import InnovationTracker
from polyneat.core.neat.initial_population import register_initial_population_strategy
from polyneat.core.neat.neat_genome import ConnectionGene, NEATGenome, NodeGene
from polyneat.core.population import Population

_COORDINATE_INPUT_NODE_IDS = (0, 1, 2, 3)
_WEIGHT_OUTPUT_NODE_ID = 7
_LINK_EXPRESSION_OUTPUT_NODE_ID = 8
_FIRST_SEED_HIDDEN_NODE_ID = 9


def build_leo_seeded_initial_population(
    config: NEATConfig,
    innovation_tracker: InnovationTracker,
    rng: Generator,
) -> Population:
    """Build generation 0 with the link expression output seeded for locality.

    Every genome gets the same fixed structure on the link expression output: a
    Gaussian hidden node per configured axis, fed by that axis' coordinate delta,
    plus a negative bias. Because ``gaussian(v) = exp(-v^2)`` peaks at ``v = 0``
    and the CPPN's output activation is the identity, the seeded rule reads

        ``LEO = exp(-(w * delta)^2) + b``

    which is positive - the connection is expressed - exactly while
    ``|delta| < sqrt(ln(-1/b)) / w``. At generation 0 only nearby substrate nodes
    are wired together; loosening that costs evolution an actual change to the
    CPPN, which is the mechanism the paper credits for modularity.

    The weight output is wired to the four coordinate inputs and the bias with
    random weights, exactly as a minimal HyperNEAT CPPN would be. The delta
    inputs deliberately do *not* feed it, so the weight function starts out
    identical to plain HyperNEAT's and evolution decides whether to involve the
    deltas.

    All genomes share one node numbering and one innovation numbering (the
    tracker deduplicates per source/target pair within a generation), so
    crossover can align the identical seed structure across the population.

    Node layout, fixed for the whole algorithm: inputs ``0..5`` are
    ``x1, y1, x2, y2, dx, dy``; ``6`` is the bias; ``7`` is the weight output and
    ``8`` the link expression output; seeded Gaussian nodes start at ``9``.

    Args:
        config: A :class:`HyperNEATLEOConfig`; the ``NEATConfig`` type keeps the
            signature compatible with the strategy registry.
        innovation_tracker: Tracker issuing the shared innovation ids.
        rng: Source of randomness for the weight-output wiring.

    Returns:
        Generation-0 population of locality-seeded CPPN genomes.
    """
    leo_config = cast("HyperNEATLEOConfig", config)

    seed_axis_names = tuple(leo_config.locality_seed_coordinate_axes)
    gaussian_node_ids = tuple(
        _FIRST_SEED_HIDDEN_NODE_ID + offset for offset in range(len(seed_axis_names))
    )
    bias_node_id = leo_config.number_of_input_nodes

    node_genes: list[NodeGene] = [
        NodeGene(node_id=node_id, node_type="input", activation_function_name="identity")
        for node_id in range(leo_config.number_of_input_nodes)
    ]
    node_genes.append(
        NodeGene(
            node_id=bias_node_id,
            node_type="bias",
            activation_function_name="identity",
        )
    )
    node_genes.extend(
        NodeGene(
            node_id=node_id,
            node_type="output",
            activation_function_name=leo_config.default_activation_function_for_output_nodes,
        )
        for node_id in (_WEIGHT_OUTPUT_NODE_ID, _LINK_EXPRESSION_OUTPUT_NODE_ID)
    )
    node_genes.extend(
        NodeGene(node_id=node_id, node_type="hidden", activation_function_name="gaussian")
        for node_id in gaussian_node_ids
    )
    template_node_genes = tuple(node_genes)

    def _connection(
        source_node_id: int, target_node_id: int, weight: float
    ) -> ConnectionGene:
        return ConnectionGene(
            innovation_id=innovation_tracker.get_or_assign_innovation_id_for_connection(
                source_node_id=source_node_id,
                target_node_id=target_node_id,
            ),
            source_node_id=source_node_id,
            target_node_id=target_node_id,
            weight=weight,
            is_enabled=True,
        )

    initial_genomes: list[NEATGenome] = []
    for _individual_index in range(leo_config.population_size):
        connection_genes: list[ConnectionGene] = []

        for source_node_id in (*_COORDINATE_INPUT_NODE_IDS, bias_node_id):
            connection_genes.append(
                _connection(
                    source_node_id,
                    _WEIGHT_OUTPUT_NODE_ID,
                    float(
                        rng.uniform(
                            leo_config.initial_weight_range_min,
                            leo_config.initial_weight_range_max,
                        )
                    ),
                )
            )

        for axis_name, gaussian_node_id in zip(
            seed_axis_names, gaussian_node_ids, strict=True
        ):
            delta_input_node_id = SEED_AXIS_NAME_TO_DELTA_INPUT_NODE_ID[axis_name]
            connection_genes.append(
                _connection(
                    delta_input_node_id,
                    gaussian_node_id,
                    leo_config.locality_seed_delta_weight,
                )
            )
            connection_genes.append(
                _connection(gaussian_node_id, _LINK_EXPRESSION_OUTPUT_NODE_ID, 1.0)
            )

        connection_genes.append(
            _connection(
                bias_node_id,
                _LINK_EXPRESSION_OUTPUT_NODE_ID,
                leo_config.locality_seed_bias_weight,
            )
        )

        initial_genomes.append(
            NEATGenome(
                node_genes=template_node_genes,
                connection_genes=tuple(connection_genes),
            )
        )

    innovation_tracker.reset_for_new_generation()

    return Population(
        genomes=initial_genomes,
        species_assignments=None,
        generation_number=0,
    )


register_initial_population_strategy("leo_seeded", build_leo_seeded_initial_population)
