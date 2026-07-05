from __future__ import annotations

from numpy.random import Generator

from polyneat.algorithms.neat.global_innovation_tracker import GlobalInnovationTracker
from polyneat.algorithms.neat.neat_genome import ConnectionGene, NEATGenome, NodeGene
from polyneat.logging_utils.custom_logger import get_logger

logger = get_logger(__name__)


class AddNodeMutation:
    """Splits an enabled connection by inserting a new hidden node.

    Stanley's rule: pick a random enabled connection, disable it, insert a
    hidden node, and add two new connections. The connection *into* the new
    node has weight 1.0, and the connection *out of* the new node has the
    original connection's weight. This preserves the network's behaviour at
    the moment of insertion.
    """

    def __init__(
        self,
        probability_of_application: float,
        activation_function_name_for_new_hidden_node: str,
    ) -> None:
        self._probability_of_application = probability_of_application
        self._activation_function_name_for_new_hidden_node = (
            activation_function_name_for_new_hidden_node
        )

    def apply_to_genome(
        self,
        genome: NEATGenome,
        rng: Generator,
        innovation_tracker: GlobalInnovationTracker,
    ) -> NEATGenome:
        if rng.random() >= self._probability_of_application:
            return genome

        enabled_connection_genes = [
            connection_gene
            for connection_gene in genome.connection_genes
            if connection_gene.is_enabled
        ]
        if not enabled_connection_genes:
            logger.debug("AddNodeMutation skipped: no enabled connections to split")
            return genome

        connection_to_split_index = int(rng.integers(0, len(enabled_connection_genes)))
        connection_to_split = enabled_connection_genes[connection_to_split_index]

        new_hidden_node_id = _pick_next_unused_node_id(genome)
        new_hidden_node_gene = NodeGene(
            node_id=new_hidden_node_id,
            node_type="hidden",
            activation_function_name=self._activation_function_name_for_new_hidden_node,
        )

        innovation_id_from_source_to_new_node = (
            innovation_tracker.get_or_assign_innovation_id_for_connection(
                source_node_id=connection_to_split.source_node_id,
                target_node_id=new_hidden_node_id,
            )
        )
        innovation_id_from_new_node_to_target = (
            innovation_tracker.get_or_assign_innovation_id_for_connection(
                source_node_id=new_hidden_node_id,
                target_node_id=connection_to_split.target_node_id,
            )
        )

        connection_source_to_new_node = ConnectionGene(
            innovation_id=innovation_id_from_source_to_new_node,
            source_node_id=connection_to_split.source_node_id,
            target_node_id=new_hidden_node_id,
            weight=1.0,
            is_enabled=True,
        )
        connection_new_node_to_target = ConnectionGene(
            innovation_id=innovation_id_from_new_node_to_target,
            source_node_id=new_hidden_node_id,
            target_node_id=connection_to_split.target_node_id,
            weight=connection_to_split.weight,
            is_enabled=True,
        )

        updated_connection_genes = tuple(
            _disabled_copy_of(connection_gene)
            if connection_gene.innovation_id == connection_to_split.innovation_id
            else connection_gene
            for connection_gene in genome.connection_genes
        ) + (connection_source_to_new_node, connection_new_node_to_target)

        updated_node_genes = genome.node_genes + (new_hidden_node_gene,)

        return NEATGenome(node_genes=updated_node_genes, connection_genes=updated_connection_genes)


def _pick_next_unused_node_id(genome: NEATGenome) -> int:
    highest_existing_node_id = max(node_gene.node_id for node_gene in genome.node_genes)
    return highest_existing_node_id + 1


def _disabled_copy_of(connection_gene: ConnectionGene) -> ConnectionGene:
    return ConnectionGene(
        innovation_id=connection_gene.innovation_id,
        source_node_id=connection_gene.source_node_id,
        target_node_id=connection_gene.target_node_id,
        weight=connection_gene.weight,
        is_enabled=False,
    )
