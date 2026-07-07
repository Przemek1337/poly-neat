from __future__ import annotations

from numpy.random import Generator

from polyneat.algorithms.neat.neat_genome import ConnectionGene, NEATGenome
from polyneat.core.component_protocols import InnovationTracker
from polyneat.logging_utils.custom_logger import get_logger
from polyneat.nn.topology_utilities import would_directed_edge_create_cycle

logger = get_logger(__name__)


class ToggleConnectionEnabledMutation:
    """Flips the ``is_enabled`` flag on a random connection with fixed probability.

    Re-enabling a previously-disabled connection can revive useful structure
    that was disabled by ``AddNodeMutation``. When re-enabling would introduce
    a directed cycle (possible because ``AddConnectionMutation`` only checks
    the currently-enabled topology), the toggle is silently skipped.
    """

    def __init__(self, probability_of_application: float) -> None:
        self._probability_of_application = probability_of_application

    def apply_to_genome(
        self,
        genome: NEATGenome,
        rng: Generator,
        innovation_tracker: InnovationTracker,
    ) -> NEATGenome:
        if rng.random() >= self._probability_of_application:
            return genome
        if not genome.connection_genes:
            logger.debug("ToggleConnectionEnabledMutation skipped: genome has no connections")
            return genome

        connection_gene_index_to_toggle = int(rng.integers(0, len(genome.connection_genes)))
        target_gene = genome.connection_genes[connection_gene_index_to_toggle]

        if not target_gene.is_enabled:
            existing_enabled_edges = [
                (c.source_node_id, c.target_node_id)
                for c in genome.connection_genes
                if c.is_enabled
            ]
            if would_directed_edge_create_cycle(
                candidate_source_node_id=target_gene.source_node_id,
                candidate_target_node_id=target_gene.target_node_id,
                existing_enabled_edges=existing_enabled_edges,
            ):
                logger.debug(
                    "ToggleConnectionEnabledMutation skipped: re-enabling innov=%d "
                    "(%d->%d) would create a cycle",
                    target_gene.innovation_id,
                    target_gene.source_node_id,
                    target_gene.target_node_id,
                )
                return genome

        toggled_connection_genes = tuple(
            ConnectionGene(
                innovation_id=existing_connection_gene.innovation_id,
                source_node_id=existing_connection_gene.source_node_id,
                target_node_id=existing_connection_gene.target_node_id,
                weight=existing_connection_gene.weight,
                is_enabled=not existing_connection_gene.is_enabled,
            )
            if position_index == connection_gene_index_to_toggle
            else existing_connection_gene
            for position_index, existing_connection_gene in enumerate(genome.connection_genes)
        )
        return NEATGenome(
            node_genes=genome.node_genes,
            connection_genes=toggled_connection_genes,
        )
