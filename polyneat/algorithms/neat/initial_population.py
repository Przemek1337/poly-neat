from __future__ import annotations

from numpy.random import Generator

from polyneat.algorithms.neat.neat_genome import ConnectionGene, NEATGenome, NodeGene
from polyneat.config.configuration_errors import ConfigurationError
from polyneat.config.neat_config import NEATConfig
from polyneat.core.component_protocols import (
    InitialPopulationStrategy,
    InnovationTracker,
)
from polyneat.core.population import Population

# Concrete strategy type for NEAT: see InitialPopulationStrategy in
# core/component_protocols.py for the contract.
NEATInitialPopulationStrategy = InitialPopulationStrategy[NEATConfig]


def _build_node_gene_template(config: NEATConfig) -> tuple[NodeGene, ...]:
    """Node genes shared by every generation-0 genome: inputs, bias, outputs."""
    input_node_genes = tuple(
        NodeGene(
            node_id=node_id,
            node_type="input",
            activation_function_name="identity",
        )
        for node_id in range(config.number_of_input_nodes)
    )
    bias_node_gene = NodeGene(
        node_id=config.number_of_input_nodes,
        node_type="bias",
        activation_function_name="identity",
    )
    output_node_genes = tuple(
        NodeGene(
            node_id=node_id,
            node_type="output",
            activation_function_name=config.default_activation_function_for_output_nodes,
        )
        for node_id in range(
            config.number_of_input_nodes + 1,
            config.number_of_input_nodes + 1 + config.number_of_output_nodes,
        )
    )
    return input_node_genes + (bias_node_gene,) + output_node_genes


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

    template_node_genes = _build_node_gene_template(config)

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


def build_fs_neat_initial_population(
    config: NEATConfig,
    innovation_tracker: InnovationTracker,
    rng: Generator,
) -> Population:
    """FS-NEAT generation 0 (Whiteson et al., 2005): one random input wired per genome.

    Each genome starts with a single enabled connection from a randomly drawn
    input node to a randomly drawn output node, so evolution itself selects
    which input features get connected at all - automatic feature selection.
    The bias node is present but starts unconnected: wiring it in is left to
    ``AddConnectionMutation``, the same as for any input deemed relevant.
    """
    template_node_genes = _build_node_gene_template(config)

    initial_genomes: list[NEATGenome] = []
    for _individual_index in range(config.population_size):
        source_node_id = int(rng.integers(0, config.number_of_input_nodes))
        target_node_id = int(
            config.number_of_input_nodes
            + 1
            + rng.integers(0, config.number_of_output_nodes)
        )
        innovation_id = innovation_tracker.get_or_assign_innovation_id_for_connection(
            source_node_id=source_node_id,
            target_node_id=target_node_id,
        )
        initial_weight = float(
            rng.uniform(
                config.initial_weight_range_min,
                config.initial_weight_range_max,
            )
        )
        only_connection = ConnectionGene(
            innovation_id=innovation_id,
            source_node_id=source_node_id,
            target_node_id=target_node_id,
            weight=initial_weight,
            is_enabled=True,
        )
        initial_genomes.append(
            NEATGenome(
                node_genes=template_node_genes,
                connection_genes=(only_connection,),
            )
        )

    innovation_tracker.reset_for_new_generation()

    return Population(
        genomes=initial_genomes,
        species_assignments=None,
        generation_number=0,
    )


# Name -> strategy registry, mirroring ACTIVATION_FUNCTION_NAME_TO_CALLABLE in
# polyneat.nn: built-in strategies are selectable from YAML by name, and users
# can register their own without touching library code.
INITIAL_POPULATION_STRATEGY_NAME_TO_CALLABLE: dict[str, NEATInitialPopulationStrategy] = {
    "fully_connected": build_fully_connected_initial_population,
    "fs_neat": build_fs_neat_initial_population,
}


def resolve_initial_population_strategy_by_name(
    strategy_name: str,
) -> NEATInitialPopulationStrategy:
    try:
        return INITIAL_POPULATION_STRATEGY_NAME_TO_CALLABLE[strategy_name]
    except KeyError:
        known_names = sorted(INITIAL_POPULATION_STRATEGY_NAME_TO_CALLABLE)
        raise ConfigurationError(
            f"Unknown initial population strategy: {strategy_name!r}. "
            f"Known strategies: {known_names}"
        ) from None


def register_initial_population_strategy(
    strategy_name: str, strategy: NEATInitialPopulationStrategy
) -> None:
    if strategy_name in INITIAL_POPULATION_STRATEGY_NAME_TO_CALLABLE:
        raise ConfigurationError(
            f"Initial population strategy {strategy_name!r} is already registered. "
            f"Pick a different name."
        )
    INITIAL_POPULATION_STRATEGY_NAME_TO_CALLABLE[strategy_name] = strategy
