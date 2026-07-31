"""Add-node mutation that assigns a random CPPN activation function.

References:
    Stanley, K. O., & Miikkulainen, R. (2002). Evolving Neural Networks through Augmenting
        Topologies. *Evolutionary Computation*, 10(2), 99-127.
    Stanley, K. O., D'Ambrosio, D. B., & Gauci, J. (2009). A Hypercube-Based Encoding for
        Evolving Large-Scale Neural Networks. *Artificial Life*, 15(2), 185-212.
        DOI: 10.1162/artl.2009.15.2.15202
"""

from __future__ import annotations

from numpy.random import Generator

from polyneat.core.component_protocols import InnovationTracker
from polyneat.core.neat.mutations.add_node_mutation import (
    _disabled_copy_of,
    _pick_next_unused_node_id,
)
from polyneat.core.neat.neat_genome import ConnectionGene, NEATGenome, NodeGene
from polyneat.logging_utils.custom_logger import get_logger

logger = get_logger(__name__)


class AddNodeWithRandomActivationMutation:
    """Split an enabled connection, inserting a hidden node whose activation is
    chosen at random from the CPPN function set.

    Structurally identical to ``AddNodeMutation`` (disable the split connection;
    add a weight-1.0 connection into the new node and a weight-equal-to-original
    connection out of it), but the new node's activation function is sampled
    uniformly from ``available_activation_function_names``. Composing different
    functions over generations is exactly how a CPPN builds up geometric
    regularities (Stanley et al. 2009, Section 3.2).
    """

    def __init__(
        self,
        probability_of_application: float,
        available_activation_function_names: tuple[str, ...],
    ) -> None:
        """Store the mutation rate and the CPPN function set to sample from.

        Args:
            probability_of_application: Probability in ``[0, 1]`` that the
                mutation fires on any given call.
            available_activation_function_names: The activation function names a
                newly inserted node may be assigned, sampled uniformly.

        Raises:
            ValueError: If ``available_activation_function_names`` is empty.
        """
        if not available_activation_function_names:
            raise ValueError("available_activation_function_names must be non-empty")
        self._probability_of_application = probability_of_application
        self._available_activation_function_names = available_activation_function_names

    def apply_to_genome(
        self,
        genome: NEATGenome,
        rng: Generator,
        innovation_tracker: InnovationTracker,
    ) -> NEATGenome:
        """Return a genome with one connection split by a random-activation node.

        Args:
            genome: The genome to mutate (left unchanged; a new one is returned).
            rng: Random generator threaded through the run.
            innovation_tracker: Issues innovation ids for the two new
                connections.

        Returns:
            A new `NEATGenome` with the split applied, or the input genome
            unchanged when the mutation does not fire or there is no enabled
            connection to split.
        """
        if rng.random() >= self._probability_of_application:
            return genome

        enabled_connection_genes = [
            connection_gene
            for connection_gene in genome.connection_genes
            if connection_gene.is_enabled
        ]
        if not enabled_connection_genes:
            logger.debug("AddNodeWithRandomActivationMutation skipped: no enabled connections")
            return genome

        connection_to_split_index = int(rng.integers(0, len(enabled_connection_genes)))
        connection_to_split = enabled_connection_genes[connection_to_split_index]

        new_hidden_node_id = _pick_next_unused_node_id(genome)
        chosen_activation_index = int(
            rng.integers(0, len(self._available_activation_function_names))
        )
        chosen_activation_function_name = self._available_activation_function_names[
            chosen_activation_index
        ]
        new_hidden_node_gene = NodeGene(
            node_id=new_hidden_node_id,
            node_type="hidden",
            activation_function_name=chosen_activation_function_name,
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

        return NEATGenome(
            node_genes=updated_node_genes,
            connection_genes=updated_connection_genes,
        )
