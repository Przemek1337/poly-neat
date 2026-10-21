"""Stripping every trace of EXACT's Lamarckian inheritance from a genome.

EXACT is Lamarckian: a trained kernel is written back into the genotype and
inherited by the next generation, and each hidden node also carries the batch
normalization state it ended training with. That is the algorithm working as
published, and track A depends on it.

Track B asks the opposite question - how good is this *topology* under one
shared recipe - so it has to start from parameters that carry nothing forward.
Clearing the kernels is not enough on its own for two separate reasons, both of
which this module handles:

* ``use_epigenetic_weight_initialization=False`` drops the kernels but leaves
  ``batch_normalization_state`` on every node, so the phenotype would still be
  seeded with the scale, shift and running statistics learned in track A;
* :meth:`EXACTBackpropagationTrainer.train_genome` returns immediately when
  ``is_trained`` is set, so a genome coming out of a search would skip training
  entirely and be reported as if it had been retrained.

The reset returns a new genome and never mutates the original, so the track A
checkpoint stays exactly as the search selected it.

References:
    Desell, T. (2017). Developing a Volunteer Computing Project to Evolve
        Convolutional Neural Networks and Their Hyperparameters. *IEEE
        13th International Conference on e-Science*.
"""

from __future__ import annotations

from dataclasses import replace

from polyneat.algorithms.exact.exact_genome import EXACTGenome
from polyneat.logging_utils.custom_logger import get_logger

logger = get_logger(__name__)


def reset_genome_for_fresh_training(genome: EXACTGenome) -> EXACTGenome:
    """Return a copy of ``genome`` with every learned parameter removed.

    The copy has ``is_trained`` cleared, ``kernel_weights`` cleared on every
    edge gene and ``batch_normalization_state`` cleared on every node gene. A
    phenotype decoded from it therefore draws fresh kernels and builds batch
    normalization at its identity defaults - scale one, shift zero, running
    mean zero, running variance one, update counters at zero.

    The evolved training hyperparameters are left on the genome. They are part
    of what EXACT searched over and belong in the record; track B simply does
    not read them, because it applies one shared recipe to every topology.

    Args:
        genome: The genome selected by a search. Not modified.

    Returns:
        A new genome carrying the same topology and no learned parameters.
    """
    reset_genome = replace(
        genome,
        node_genes=tuple(
            replace(node_gene, batch_normalization_state=None)
            for node_gene in genome.node_genes
        ),
        edge_genes=tuple(
            replace(edge_gene, kernel_weights=None) for edge_gene in genome.edge_genes
        ),
        is_trained=False,
    )
    logger.info(
        "EXACT genome reset for fresh training: %d nodes and %d edges cleared",
        len(reset_genome.node_genes),
        len(reset_genome.edge_genes),
    )
    return reset_genome


def genome_carries_learned_parameters(genome: EXACTGenome) -> bool:
    """Whether any learned parameter survives on ``genome``.

    Used by tests and by the track B stage as a precondition check: a topology
    about to be retrained from scratch must answer ``False`` here, and the
    track A checkpoint it came from must still answer ``True``.
    """
    return (
        genome.is_trained
        or any(edge_gene.kernel_weights is not None for edge_gene in genome.edge_genes)
        or any(
            node_gene.batch_normalization_state is not None for node_gene in genome.node_genes
        )
    )
