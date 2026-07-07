from __future__ import annotations

from numpy.random import Generator

from polyneat.algorithms.neat.neat_genome import ConnectionGene, NEATGenome
from polyneat.core.component_protocols import InnovationTracker
from polyneat.logging_utils.custom_logger import get_logger
from polyneat.nn.topology_utilities import would_directed_edge_create_cycle

logger = get_logger(__name__)


class AddConnectionMutation:
    """Adds an enabled connection between two currently unconnected nodes.

    Constraints:
      * The source may not be an ``output``.
      * The target may not be an ``input`` or ``bias``.
      * No self-loops.
      * No duplicate connections (matching ``(source, target)``).
      * No cycles — this is a feed-forward NEAT.

    When no candidate pair satisfies the constraints, the mutation returns the
    genome unchanged (normal runtime condition, not an error).
    """

    def __init__(
        self,
        probability_of_application: float,
        initial_weight_range_min: float,
        initial_weight_range_max: float,
        maximum_number_of_random_attempts_per_call: int = 50,
    ) -> None:
        self._probability_of_application = probability_of_application
        self._initial_weight_range_min = initial_weight_range_min
        self._initial_weight_range_max = initial_weight_range_max
        self._maximum_number_of_random_attempts_per_call = (
            maximum_number_of_random_attempts_per_call
        )

    def apply_to_genome(
        self,
        genome: NEATGenome,
        rng: Generator,
        innovation_tracker: InnovationTracker,
    ) -> NEATGenome:
        if rng.random() >= self._probability_of_application:
            return genome

        eligible_source_node_ids = [
            node_gene.node_id
            for node_gene in genome.node_genes
            if node_gene.node_type != "output"
        ]
        eligible_target_node_ids = [
            node_gene.node_id
            for node_gene in genome.node_genes
            if node_gene.node_type not in ("input", "bias")
        ]

        if not eligible_source_node_ids or not eligible_target_node_ids:
            logger.debug("AddConnectionMutation skipped: no eligible source/target nodes")
            return genome

        existing_connection_source_target_pairs: set[tuple[int, int]] = {
            (connection_gene.source_node_id, connection_gene.target_node_id)
            for connection_gene in genome.connection_genes
        }
        existing_enabled_edges = [
            (connection_gene.source_node_id, connection_gene.target_node_id)
            for connection_gene in genome.connection_genes
            if connection_gene.is_enabled
        ]

        for _attempt_index in range(self._maximum_number_of_random_attempts_per_call):
            candidate_source_node_id = int(
                rng.choice(eligible_source_node_ids)  # type: ignore[arg-type]
            )
            candidate_target_node_id = int(
                rng.choice(eligible_target_node_ids)  # type: ignore[arg-type]
            )

            if candidate_source_node_id == candidate_target_node_id:
                continue
            if (candidate_source_node_id, candidate_target_node_id) in existing_connection_source_target_pairs:
                continue
            if would_directed_edge_create_cycle(
                candidate_source_node_id=candidate_source_node_id,
                candidate_target_node_id=candidate_target_node_id,
                existing_enabled_edges=existing_enabled_edges,
            ):
                continue

            innovation_id_for_new_connection = (
                innovation_tracker.get_or_assign_innovation_id_for_connection(
                    source_node_id=candidate_source_node_id,
                    target_node_id=candidate_target_node_id,
                )
            )
            initial_weight_value = float(
                rng.uniform(self._initial_weight_range_min, self._initial_weight_range_max)
            )
            new_connection_gene = ConnectionGene(
                innovation_id=innovation_id_for_new_connection,
                source_node_id=candidate_source_node_id,
                target_node_id=candidate_target_node_id,
                weight=initial_weight_value,
                is_enabled=True,
            )
            return NEATGenome(
                node_genes=genome.node_genes,
                connection_genes=genome.connection_genes + (new_connection_gene,),
            )

        logger.debug(
            "AddConnectionMutation skipped: no valid candidate found in %d attempts",
            self._maximum_number_of_random_attempts_per_call,
        )
        return genome
