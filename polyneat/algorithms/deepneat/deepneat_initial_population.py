"""DeepNEAT generation 0: the fixed input -> output linear classifier.

References:
    Miikkulainen, R., Liang, J., Meyerson, E., Rawal, A., Fink, D., Francon, O., Raju, B.,
        Shahrzad, H., Navruzyan, A., Duffy, N., & Hodjat, B. (2017). Evolving Deep Neural
        Networks. *arXiv:1703.00548*. Published in *Artificial Intelligence in the Age of
        Neural Networks and Brain Computing* (2019), pp. 293-312.
        DOI: 10.1016/B978-0-12-815480-9.00015-3
"""

from __future__ import annotations

from numpy.random import Generator

from polyneat.algorithms.deepneat.deepneat_genome import (
    DeepNEATGenome,
    LayerNodeGene,
    TensorEdgeGene,
)
from polyneat.algorithms.deepneat.mutations.global_hyperparameter_mutation import (
    draw_global_hyperparameters,
)
from polyneat.configs.deepneat.deepneat_config import DeepNEATConfig
from polyneat.core.component_protocols import InnovationTracker
from polyneat.core.population import Population

_INPUT_NODE_ID = 0
_OUTPUT_NODE_ID = 1


def build_deepneat_initial_population(
    config: DeepNEATConfig,
    innovation_tracker: InnovationTracker,
    rng: Generator,
) -> Population:
    """Build generation 0 with a shared minimal topology and global genes.

    Every genome is one input layer wired directly to one output layer through
    a single enabled edge - no hidden layers at all, following DeepNEAT's
    minimal-complexity initialization. Every chromosome independently draws
    the global training/preprocessing genes that are part of the source
    encoding. This is not the
    ``initial_population_strategy`` registry: DeepNEAT does not use it (see
    :class:`~polyneat.algorithms.deepneat.deepneat_algorithm.DeepNEATAlgorithm`).

    Args:
        config: Supplies ``population_size``. Image and class geometry are not
            needed here - the linear classifier carries no hyperparameters of
            its own; the phenotype decoder gives it a concrete shape later.
        innovation_tracker: Issues the shared marking for the single edge.
        rng: Source of randomness for chromosome-wide hyperparameters.

    Returns:
        Generation-0 population of ``config.population_size`` chromosomes
        sharing the input-to-output topology but carrying independent global
        genes.
    """
    input_node_gene = LayerNodeGene(node_id=_INPUT_NODE_ID, layer_type="input")
    output_node_gene = LayerNodeGene(node_id=_OUTPUT_NODE_ID, layer_type="output")
    edge_gene = TensorEdgeGene(
        innovation_id=innovation_tracker.get_or_assign_innovation_id_for_connection(
            source_node_id=_INPUT_NODE_ID,
            target_node_id=_OUTPUT_NODE_ID,
        ),
        source_node_id=_INPUT_NODE_ID,
        target_node_id=_OUTPUT_NODE_ID,
        is_enabled=True,
    )
    genomes = [
        DeepNEATGenome(
            node_genes=(input_node_gene, output_node_gene),
            edge_genes=(edge_gene,),
            global_hyperparameters=draw_global_hyperparameters(config, rng),
        )
        for _ in range(config.population_size)
    ]
    innovation_tracker.reset_for_new_generation()
    return Population(
        genomes=genomes,
        species_assignments=None,
        generation_number=0,
    )
