"""Generation-0 strategies: NEAT's minimal start and FS-NEAT's sparse start.

The fully connected strategy implements the minimal initial structure of
Stanley & Miikkulainen (2002), sections 2.4 and 3.4; the FS-NEAT strategy
implements the single-random-connection start of Whiteson et al. (2005).
A name-to-strategy registry makes strategies selectable from YAML and
user-extensible without touching library code.
"""

from __future__ import annotations

from numpy.random import Generator

from polyneat.config.configuration_errors import ConfigurationError
from polyneat.config.neat_config import NEATConfig
from polyneat.core.component_protocols import (
    InitialPopulationStrategy,
    InnovationTracker,
)
from polyneat.core.neat.neat_genome import ConnectionGene, NEATGenome, NodeGene
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

    Implements the minimal initial structure of the paper (sections 2.4 and
    3.4): networks start with zero hidden nodes so the search begins in the
    lowest-dimensional weight space, and topology grows only when it helps.

    All genomes share one node template and one innovation-id numbering (the
    tracker deduplicates per source/target pair), differing only in their
    random connection weights drawn uniformly from the configured range.

    Args:
        config: Network sizing and weight-range hyperparameters.
        innovation_tracker: Tracker issuing the shared innovation ids.
        rng: Source of randomness for connection weights.

    Returns:
        Generation-0 population of ``config.population_size`` genomes.
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

# TODO: Move this to separate file. Population factory handling should be further discussed.
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

    Args:
        config: Network sizing and weight-range hyperparameters.
        innovation_tracker: Tracker issuing the shared innovation ids.
        rng: Source of randomness for connection endpoints and weights.

    Returns:
        Generation-0 population of single-connection genomes.

    References:
        Whiteson, S., Stone, P., Stanley, K. O., Miikkulainen, R., &
        Kohl, N. (2005). Automatic Feature Selection in Neuroevolution.
        *GECCO 2005*.
    """
    template_node_genes = _build_node_gene_template(config)

    initial_genomes: list[NEATGenome] = []
    for _individual_index in range(config.population_size):
        source_node_id = int(rng.integers(0, config.number_of_input_nodes))
        target_node_id = int(
            config.number_of_input_nodes + 1 + rng.integers(0, config.number_of_output_nodes)
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


INITIAL_POPULATION_STRATEGY_NAME_TO_CALLABLE: dict[str, NEATInitialPopulationStrategy] = {
    "fully_connected": build_fully_connected_initial_population,
    "fs_neat": build_fs_neat_initial_population,
}


def resolve_initial_population_strategy_by_name(
    strategy_name: str,
) -> NEATInitialPopulationStrategy:
    """Look up a strategy registered under ``strategy_name``.

    Args:
        strategy_name: Registry key, e.g. ``"fully_connected"`` or ``"fs_neat"``.

    Returns:
        The registered strategy callable.

    Raises:
        ConfigurationError: If no strategy is registered under the name.
    """
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
    """Register a user-defined generation-0 strategy under a new name.

    Args:
        strategy_name: Registry key to expose the strategy under (usable as
            ``initial_population_strategy`` in YAML configs).
        strategy: Callable satisfying ``InitialPopulationStrategy``.

    Raises:
        ConfigurationError: If the name is already registered.
    """
    if strategy_name in INITIAL_POPULATION_STRATEGY_NAME_TO_CALLABLE:
        raise ConfigurationError(
            f"Initial population strategy {strategy_name!r} is already registered. "
            f"Pick a different name."
        )
    INITIAL_POPULATION_STRATEGY_NAME_TO_CALLABLE[strategy_name] = strategy
