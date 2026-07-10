from __future__ import annotations

from collections.abc import Callable

from numpy.random import Generator

from polyneat.algorithms.neat.neat_genome import ConnectionGene, NEATGenome, NodeGene
from polyneat.config.neat_config import NEATConfig
from polyneat.core.component_protocols import InnovationTracker
from polyneat.core.population import Population

# Signature every initial-population factory must match. Set it on
# ``NEATAlgorithm.initial_population_factory`` (via ``dataclasses.replace`` or direct
# assignment) to change how generation 0 is built without subclassing. A plain
# callable, so the default is a free function; a factory needing state is a class
# with ``__call__``.
InitialPopulationCallable = Callable[[NEATConfig, InnovationTracker, Generator], Population]


def build_fully_connected_initial_population(
    config: NEATConfig,
    innovation_tracker: InnovationTracker,
    rng: Generator,
) -> Population:
    """Vanilla-NEAT generation 0: every input and the bias node connected to every output.

    All genomes share one node template and one innovation-id numbering (the
    tracker deduplicates per source/target pair), differing only in their
    random connection weights drawn uniformly from the configured range.
    """
    input_node_id_range = range(config.number_of_input_nodes)
    bias_node_id = config.number_of_input_nodes
    output_node_id_range = range(
        config.number_of_input_nodes + 1,
        config.number_of_input_nodes + 1 + config.number_of_output_nodes,
    )

    input_node_genes = tuple(
        NodeGene(
            node_id=node_id,
            node_type="input",
            activation_function_name="identity",
        )
        for node_id in input_node_id_range
    )
    bias_node_gene = NodeGene(
        node_id=bias_node_id,
        node_type="bias",
        activation_function_name="identity",
    )
    output_node_genes = tuple(
        NodeGene(
            node_id=node_id,
            node_type="output",
            activation_function_name=config.default_activation_function_for_output_nodes,
        )
        for node_id in output_node_id_range
    )

    template_node_genes: tuple[NodeGene, ...] = (
        input_node_genes + (bias_node_gene,) + output_node_genes
    )

    initial_genomes: list[NEATGenome] = []
    for _individual_index in range(config.population_size):
        initial_connection_genes: list[ConnectionGene] = []
        for input_or_bias_node_id in [*input_node_id_range, bias_node_id]:
            for output_node_id in output_node_id_range:
                innovation_id = innovation_tracker.get_or_assign_innovation_id_for_connection(
                    source_node_id=input_or_bias_node_id,
                    target_node_id=output_node_id,
                )
                initial_weight = float(
                    rng.uniform(
                        config.initial_weight_range_min,
                        config.initial_weight_range_max,
                    )
                )
                initial_connection_genes.append(
                    ConnectionGene(
                        innovation_id=innovation_id,
                        source_node_id=input_or_bias_node_id,
                        target_node_id=output_node_id,
                        weight=initial_weight,
                        is_enabled=True,
                    )
                )
        initial_genomes.append(
            NEATGenome(
                node_genes=template_node_genes,
                connection_genes=tuple(initial_connection_genes),
            )
        )

    innovation_tracker.reset_for_new_generation()

    return Population(
        genomes=initial_genomes,
        species_assignments=None,
        generation_number=0,
    )
