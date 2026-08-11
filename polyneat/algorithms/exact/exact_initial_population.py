"""EXACT generation 0: the minimal CNN genome (Desell, 2017, section III-A)."""

from __future__ import annotations

from numpy.random import Generator

from polyneat.algorithms.exact.exact_genome import (
    ConvolutionEdgeGene,
    EXACTGenome,
    FilterNodeGene,
)
from polyneat.configs.exact.exact_config import EXACTConfig
from polyneat.core.component_protocols import InnovationTracker
from polyneat.core.neat.initial_population import (
    INITIAL_POPULATION_STRATEGY_NAME_TO_CALLABLE,
    register_initial_population_strategy,
)
from polyneat.core.population import Population

EXACT_MINIMAL_CNN_STRATEGY_NAME = "exact_minimal_cnn"

_INPUT_NODE_DEPTH = 0.0
_OUTPUT_NODE_DEPTH = 1.0


def build_minimal_cnn_initial_population(
    config: EXACTConfig,
    innovation_tracker: InnovationTracker,
    rng: Generator,
) -> Population:
    """EXACT generation 0: input node wired to one 1x1 output node per class.

    The paper's minimal CNN genome (section III-A): solely the input node,
    sized to the training images, and one output node per training class for
    the softmax layer — which is also the paper's Benchmark Architecture I.
    All genomes are identical; diversity comes from mutation, and the first
    training session fills the ``None`` kernels via He initialization.

    Args:
        config: Supplies the image size, class count and population size.
        innovation_tracker: Issues the shared edge markings.
        rng: Unused (the minimal genome carries no random weights); present
            for the strategy protocol.

    Returns:
        Generation-0 population of ``config.population_size`` identical
        minimal CNN genomes.
    """
    input_node_gene = FilterNodeGene(
        node_id=0,
        node_type="input",
        filter_height=config.input_image_height,
        filter_width=config.input_image_width,
        depth=_INPUT_NODE_DEPTH,
    )
    output_node_genes = tuple(
        FilterNodeGene(
            node_id=output_index + 1,
            node_type="output",
            filter_height=1,
            filter_width=1,
            depth=_OUTPUT_NODE_DEPTH,
        )
        for output_index in range(config.number_of_output_nodes)
    )
    edge_genes = tuple(
        ConvolutionEdgeGene(
            innovation_id=innovation_tracker.get_or_assign_innovation_id_for_connection(
                source_node_id=input_node_gene.node_id,
                target_node_id=output_node_gene.node_id,
            ),
            source_node_id=input_node_gene.node_id,
            target_node_id=output_node_gene.node_id,
            is_enabled=True,
            kernel_weights=None,
        )
        for output_node_gene in output_node_genes
    )
    minimal_cnn_genome = EXACTGenome(
        node_genes=(input_node_gene, *output_node_genes),
        edge_genes=edge_genes,
        is_trained=False,
    )
    innovation_tracker.reset_for_new_generation()
    return Population(
        genomes=[minimal_cnn_genome] * config.population_size,
        species_assignments=None,
        generation_number=0,
    )


if EXACT_MINIMAL_CNN_STRATEGY_NAME not in INITIAL_POPULATION_STRATEGY_NAME_TO_CALLABLE:
    register_initial_population_strategy(
        EXACT_MINIMAL_CNN_STRATEGY_NAME, build_minimal_cnn_initial_population
    )
