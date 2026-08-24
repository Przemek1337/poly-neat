"""Add a skip connection between two layers that are not yet connected.

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


class AddTensorEdgeMutation:
    """Adds one tensor edge between an unconnected, acyclic pair of layers.

    Candidates exclude edges into the input layer, out of the output layer,
    self-loops, duplicates of an existing edge (enabled or not), and any pair
    that would close a cycle among the enabled edges - the phenotype executes in
    topological order, so the graph has to stay acyclic.
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
        """Return a copy with one new tensor edge, when a legal one exists.

        Args:
            genome: Genome to mutate; never modified in place.
            rng: Source of randomness for the endpoint choice.
            innovation_tracker: Issues the new edge's marking.

        Returns:
            The mutated genome, or the original when the operator does not fire
            or no legal edge remains.
        """
        if rng.random() >= self._probability_of_application:
            return genome

        input_node_id = genome.input_node_id
        output_node_id = genome.output_node_id
        existing_pairs = {
            (edge.source_node_id, edge.target_node_id) for edge in genome.edge_genes
        }
        enabled_edges = [
            (edge.source_node_id, edge.target_node_id)
            for edge in genome.edge_genes
            if edge.is_enabled
        ]

        candidate_pairs = [
            (source.node_id, target.node_id)
            for source in genome.node_genes
            for target in genome.node_genes
            if source.node_id != target.node_id
            and target.node_id != input_node_id
            and source.node_id != output_node_id
            and (source.node_id, target.node_id) not in existing_pairs
            and not would_directed_edge_create_cycle(
                candidate_source_node_id=source.node_id,
                candidate_target_node_id=target.node_id,
                existing_enabled_edges=enabled_edges,
            )
        ]
        if not candidate_pairs:
            logger.debug("AddTensorEdgeMutation skipped: no legal edge available")
            return genome

        source_node_id, target_node_id = candidate_pairs[
            int(rng.integers(0, len(candidate_pairs)))
        ]
        new_edge = TensorEdgeGene(
            innovation_id=innovation_tracker.get_or_assign_innovation_id_for_connection(
                source_node_id=source_node_id,
                target_node_id=target_node_id,
            ),
            source_node_id=source_node_id,
            target_node_id=target_node_id,
            is_enabled=True,
        )
        logger.debug(
            "AddTensorEdgeMutation added %d->%d (innov=%d)",
            source_node_id,
            target_node_id,
            new_edge.innovation_id,
        )
        return DeepNEATGenome(
            node_genes=genome.node_genes,
            edge_genes=(*genome.edge_genes, new_edge),
        )
