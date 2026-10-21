"""Insert a new layer by splitting an existing tensor edge.

References:
    Miikkulainen, R., et al. (2017). Evolving Deep Neural Networks. *arXiv:1703.00548*.
        DOI: 10.1016/B978-0-12-815480-9.00015-3
    Stanley, K. O., & Miikkulainen, R. (2002). Evolving Neural Networks through Augmenting
        Topologies. *Evolutionary Computation*, 10(2), 99-127. (Add-node mutation and
        historical markings: section 3.1-3.2.)
"""

from __future__ import annotations

from numpy.random import Generator

from polyneat.algorithms.deepneat.deepneat_genome import DeepNEATGenome, TensorEdgeGene
from polyneat.algorithms.deepneat.mutations.layer_hyperparameter_mutation import (
    draw_conv_layer_hyperparameters,
    draw_dense_layer_hyperparameters,
)
from polyneat.core.component_protocols import InnovationTracker
from polyneat.logging_utils.custom_logger import get_logger

logger = get_logger(__name__)


class AddLayerNodeMutation:
    """Splits an enabled edge and puts a freshly sampled layer in the middle.

    Structurally identical to NEAT's add-node mutation: the split edge is
    disabled rather than deleted, so crossover can still align it, and the new
    node plus its two edges take their markings from the innovation tracker,
    keyed on the split edge - two genomes splitting the same edge must come out
    with the same ids or crossover would align unrelated layers.

    A repeated split of the same edge within one generation reuses the same
    record, as NEAT historical markings require. If a genome already carries
    that node, this operator excludes the edge and tries another candidate.
    """

    def __init__(
        self,
        probability_of_application: float,
        available_filter_counts: tuple[int, ...],
        available_kernel_sizes: tuple[int, ...],
        available_dense_unit_counts: tuple[int, ...],
        dropout_rate_min: float,
        dropout_rate_max: float,
        probability_of_new_conv_layer: float,
        initial_weight_scaling_min: float = 0.0,
        initial_weight_scaling_max: float = 2.0,
        available_batch_normalization_options: tuple[bool, ...] = (False,),
        number_of_filters_min: int | None = None,
        number_of_filters_max: int | None = None,
    ) -> None:
        """Store the firing probability and the search space.

        Args:
            probability_of_application: Chance the operator fires on a genome.
            available_filter_counts: Allowed conv output-channel counts.
            available_kernel_sizes: Allowed conv kernel sides.
            available_dense_unit_counts: Allowed dense widths.
            dropout_rate_min: Lower end of the dropout range.
            dropout_rate_max: Upper end of the dropout range.
            probability_of_new_conv_layer: Chance the inserted layer is a conv
                rather than a dense layer.
        """
        self._probability_of_application = probability_of_application
        self._available_filter_counts = available_filter_counts
        self._available_kernel_sizes = available_kernel_sizes
        self._available_dense_unit_counts = available_dense_unit_counts
        self._dropout_rate_min = dropout_rate_min
        self._dropout_rate_max = dropout_rate_max
        self._probability_of_new_conv_layer = probability_of_new_conv_layer
        self._initial_weight_scaling_min = initial_weight_scaling_min
        self._initial_weight_scaling_max = initial_weight_scaling_max
        self._available_batch_normalization_options = (
            available_batch_normalization_options
        )
        self._number_of_filters_min = number_of_filters_min
        self._number_of_filters_max = number_of_filters_max

    def apply_to_genome(
        self,
        genome: DeepNEATGenome,
        rng: Generator,
        innovation_tracker: InnovationTracker,
    ) -> DeepNEATGenome:
        """Return a copy with one edge split by a new layer.

        Args:
            genome: Genome to mutate; never modified in place.
            rng: Source of randomness for the edge choice and the layer's
                hyperparameters.
            innovation_tracker: Issues the new node id and the two edge
                markings.

        Returns:
            The mutated genome, or the original when the operator does not
            fire, there is no enabled edge to split, or every enabled edge's
            split would replay a node id this genome already carries.
        """
        if rng.random() >= self._probability_of_application:
            return genome

        enabled_positions = [
            position for position, edge in enumerate(genome.edge_genes) if edge.is_enabled
        ]
        if not enabled_positions:
            logger.debug("AddLayerNodeMutation skipped: no enabled edge to split")
            return genome

        highest_existing_node_id = max(node.node_id for node in genome.node_genes)

        # Splitting the same edge twice within one generation replays the
        # cached NodeSplitRecord. If
        # this genome still carries the node from the earlier split (the
        # common case: splitting disables the edge but never removes the
        # node), inserting it again would duplicate a node_id. A fresh split
        # record can never collide (its node id always clears every id the
        # genome already has), so only a replayed record ever triggers this;
        # exclude that one edge and re-draw among the rest rather than
        # forfeiting the whole mutation event, since the other enabled edges
        # are usually still perfectly legal splits.
        remaining_candidate_positions = list(enabled_positions)
        while remaining_candidate_positions:
            draw_index = int(rng.integers(0, len(remaining_candidate_positions)))
            position_to_split = remaining_candidate_positions.pop(draw_index)
            edge_to_split = genome.edge_genes[position_to_split]
            split_record = innovation_tracker.get_or_assign_node_split(
                split_connection_innovation_id=edge_to_split.innovation_id,
                minimum_new_node_id=highest_existing_node_id + 1,
            )
            if any(node.node_id == split_record.new_node_id for node in genome.node_genes):
                logger.debug(
                    "AddLayerNodeMutation: split of edge innov=%d would replay node %d, "
                    "which this genome already carries; excluding it and re-drawing",
                    edge_to_split.innovation_id,
                    split_record.new_node_id,
                )
                continue
            break
        else:
            logger.debug(
                "AddLayerNodeMutation skipped: every enabled edge's split would replay a "
                "node id this genome already carries"
            )
            return genome

        if rng.random() < self._probability_of_new_conv_layer:
            inserted_node = draw_conv_layer_hyperparameters(
                node_id=split_record.new_node_id,
                rng=rng,
                available_filter_counts=self._available_filter_counts,
                available_kernel_sizes=self._available_kernel_sizes,
                dropout_rate_min=self._dropout_rate_min,
                dropout_rate_max=self._dropout_rate_max,
                initial_weight_scaling_min=self._initial_weight_scaling_min,
                initial_weight_scaling_max=self._initial_weight_scaling_max,
                available_batch_normalization_options=(
                    self._available_batch_normalization_options
                ),
                number_of_filters_min=self._number_of_filters_min,
                number_of_filters_max=self._number_of_filters_max,
            )
        else:
            inserted_node = draw_dense_layer_hyperparameters(
                node_id=split_record.new_node_id,
                rng=rng,
                available_dense_unit_counts=self._available_dense_unit_counts,
                dropout_rate_min=self._dropout_rate_min,
                dropout_rate_max=self._dropout_rate_max,
                initial_weight_scaling_min=self._initial_weight_scaling_min,
                initial_weight_scaling_max=self._initial_weight_scaling_max,
                available_batch_normalization_options=(
                    self._available_batch_normalization_options
                ),
            )

        disabled_original_edge = TensorEdgeGene(
            innovation_id=edge_to_split.innovation_id,
            source_node_id=edge_to_split.source_node_id,
            target_node_id=edge_to_split.target_node_id,
            is_enabled=False,
        )
        edge_into_new_node = TensorEdgeGene(
            innovation_id=split_record.innovation_id_into_new_node,
            source_node_id=edge_to_split.source_node_id,
            target_node_id=split_record.new_node_id,
            is_enabled=True,
        )
        edge_out_of_new_node = TensorEdgeGene(
            innovation_id=split_record.innovation_id_out_of_new_node,
            source_node_id=split_record.new_node_id,
            target_node_id=edge_to_split.target_node_id,
            is_enabled=True,
        )

        logger.debug(
            "AddLayerNodeMutation inserted %s node %d into edge innov=%d",
            inserted_node.layer_type,
            inserted_node.node_id,
            edge_to_split.innovation_id,
        )
        return DeepNEATGenome(
            node_genes=(*genome.node_genes, inserted_node),
            edge_genes=(
                *(
                    disabled_original_edge if position == position_to_split else edge
                    for position, edge in enumerate(genome.edge_genes)
                ),
                edge_into_new_node,
                edge_out_of_new_node,
            ),
            global_hyperparameters=genome.global_hyperparameters,
        )
