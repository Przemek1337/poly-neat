"""FD-NEAT's feature-deselecting mutation: drop one input connection.

References:
    Tan, M., Deklerck, R., Jansen, B., & Cornelis, J. (2012). Analysis of a Feature-Deselective
        Neuroevolution Classifier (FD-NEAT) in a Computer-Aided Lung Nodule Detection System
        for CT Images. *GECCO '12 Companion: Proceedings of the 14th Annual Conference Companion
        on Genetic and Evolutionary Computation*, pp. 539-546. DOI: 10.1145/2330784.2330869
"""

from __future__ import annotations

from numpy.random import Generator

from polyneat.core.component_protocols import InnovationTracker
from polyneat.core.neat.neat_genome import NEATGenome
from polyneat.logging_utils.custom_logger import get_logger

logger = get_logger(__name__)


class DeleteInputConnectionMutation:
    """Removes one enabled input-sourced connection gene with fixed probability.

    This is the single operator that separates FD-NEAT from vanilla NEAT. NEAT
    only ever adds structure; FD-NEAT starts fully connected and lets evolution
    *deselect* input features by dropping the connections that carry them.

    Three implementation choices are not settled by the source paper - the PDF
    could not be obtained (ACM 403, ResearchGate 403, and the open-access mirror
    at the authors' institution now returns 404) - and are made here
    deliberately:

    1. The gene is **removed**, not disabled. Disabling is already
       :class:`~polyneat.core.neat.mutations.toggle_connection_enabled_mutation.ToggleConnectionEnabledMutation`'s
       job, and a disabled gene stays in the genome where crossover can revive
       it, which would defeat deselection.
    2. Only connections whose source is an ``input`` node are candidates. The
       bias node is not a feature, so it is excluded.
    3. One connection per application, keeping the step size in line with the
       rest of NEAT's structural operators.

    Points 1 and 2 are supported by the secondary-source description of the
    method ("a mutation operator is implemented that enables dropping initial
    input connections"); point 3 is ours alone. See
    ``docs/superpowers/specs/2026-08-14-fdneat-design.md`` for the full record.

    A genome holding a single connection is never emptied, and a genome with no
    enabled input-sourced connection is returned untouched.

    References:
        Tan, M., Deklerck, R., Jansen, B., & Cornelis, J. (2012). Analysis of a
            Feature-Deselective Neuroevolution Classifier (FD-NEAT) in a Computer-Aided
            Lung Nodule Detection System for CT Images. *GECCO '12 Companion*, pp. 539-546.
            DOI: 10.1145/2330784.2330869
    """

    def __init__(self, probability_of_application: float) -> None:
        """Store the firing probability.

        Args:
            probability_of_application: Chance the operator fires on a genome.
        """
        self._probability_of_application = probability_of_application

    def apply_to_genome(
        self,
        genome: NEATGenome,
        rng: Generator,
        innovation_tracker: InnovationTracker,
    ) -> NEATGenome:
        """Return a copy of the genome with one input connection removed.

        Args:
            genome: Genome to mutate; never modified in place.
            rng: Source of randomness for choosing the victim connection.
            innovation_tracker: Unused - this operator creates no structure, so
                it has nothing to number. Present to satisfy ``MutationOperator``.

        Returns:
            A new genome without one enabled input-sourced connection, or the
            original genome when the operator does not fire or finds no
            candidate.
        """
        if rng.random() >= self._probability_of_application:
            return genome
        if len(genome.connection_genes) <= 1:
            logger.debug(
                "DeleteInputConnectionMutation skipped: genome has %d connection(s)",
                len(genome.connection_genes),
            )
            return genome

        input_node_ids = {
            node_gene.node_id
            for node_gene in genome.node_genes
            if node_gene.node_type == "input"
        }
        candidate_positions = [
            position
            for position, connection_gene in enumerate(genome.connection_genes)
            if connection_gene.is_enabled and connection_gene.source_node_id in input_node_ids
        ]
        if not candidate_positions:
            logger.debug(
                "DeleteInputConnectionMutation skipped: no enabled input-sourced connection"
            )
            return genome

        position_to_delete = candidate_positions[int(rng.integers(0, len(candidate_positions)))]
        deleted_gene = genome.connection_genes[position_to_delete]
        logger.debug(
            "DeleteInputConnectionMutation removing innov=%d (%d->%d)",
            deleted_gene.innovation_id,
            deleted_gene.source_node_id,
            deleted_gene.target_node_id,
        )

        remaining_connection_genes = tuple(
            connection_gene
            for position, connection_gene in enumerate(genome.connection_genes)
            if position != position_to_delete
        )
        return NEATGenome(
            node_genes=genome.node_genes,
            connection_genes=remaining_connection_genes,
        )
