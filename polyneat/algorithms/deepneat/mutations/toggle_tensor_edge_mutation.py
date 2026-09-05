"""Flip one tensor edge's enable bit, refusing flips that would close a cycle.

References:
    Miikkulainen, R., et al. (2017). Evolving Deep Neural Networks. *arXiv:1703.00548*.
        DOI: 10.1016/B978-0-12-815480-9.00015-3
"""

from __future__ import annotations

from numpy.random import Generator

from polyneat.algorithms.deepneat.deepneat_genome import DeepNEATGenome, TensorEdgeGene
from polyneat.core.component_protocols import InnovationTracker
from polyneat.logging_utils.custom_logger import get_logger
from polyneat.nn.topology_utilities import would_directed_edge_create_cycle

logger = get_logger(__name__)


class ToggleTensorEdgeMutation:
    """Flips one random edge's ``is_enabled`` flag.

    Re-enabling is skipped when it would introduce a cycle: ``AddTensorEdge``
    only checks the *currently enabled* topology, so a disabled edge may have
    become cycle-forming since it was switched off.
    """

    def __init__(self, probability_of_application: float) -> None:
        """Store the firing probability.

        Args:
            probability_of_application: Chance the operator fires on a genome.
        """
        self._probability_of_application = probability_of_application

    def apply_to_genome(
        self,
        genome: DeepNEATGenome,
        rng: Generator,
        innovation_tracker: InnovationTracker,
    ) -> DeepNEATGenome:
        """Return a copy with one edge's enable bit flipped.

        Args:
            genome: Genome to mutate; never modified in place.
            rng: Source of randomness for the edge choice.
            innovation_tracker: Unused - no structure is created.

        Returns:
            The mutated genome, or the original when the operator does not fire,
            the genome has no edges, or the flip would create a cycle.
        """
        if rng.random() >= self._probability_of_application:
            return genome
        if not genome.edge_genes:
            return genome

        position_to_toggle = int(rng.integers(0, len(genome.edge_genes)))
        target_edge = genome.edge_genes[position_to_toggle]

        if not target_edge.is_enabled:
            enabled_edges = [
                (edge.source_node_id, edge.target_node_id)
                for edge in genome.edge_genes
                if edge.is_enabled
            ]
            if would_directed_edge_create_cycle(
                candidate_source_node_id=target_edge.source_node_id,
                candidate_target_node_id=target_edge.target_node_id,
                existing_enabled_edges=enabled_edges,
            ):
                logger.debug(
                    "ToggleTensorEdgeMutation skipped: re-enabling innov=%d would cycle",
                    target_edge.innovation_id,
                )
                return genome

        return DeepNEATGenome(
            node_genes=genome.node_genes,
            edge_genes=tuple(
                TensorEdgeGene(
                    innovation_id=edge.innovation_id,
                    source_node_id=edge.source_node_id,
                    target_node_id=edge.target_node_id,
                    is_enabled=not edge.is_enabled,
                )
                if position == position_to_toggle
                else edge
                for position, edge in enumerate(genome.edge_genes)
            ),
            global_hyperparameters=genome.global_hyperparameters,
        )
