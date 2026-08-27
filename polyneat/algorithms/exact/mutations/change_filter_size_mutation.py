"""Change-node-size mutations of EXACT (Desell, 2017, section III-B)."""

from __future__ import annotations

from dataclasses import dataclass, replace

from numpy.random import Generator

from polyneat.algorithms.exact.exact_genome import EXACTGenome
from polyneat.core.component_protocols import InnovationTracker
from polyneat.logging_utils.custom_logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class ChangeFilterSizeMutation:
    """Randomly grows or shrinks one hidden node's filter size.

    One class covers the paper's three operators: *change node size*
    (both dimensions), *change node size x*, and *change node size y* —
    the composite instantiates it three times with different dimension
    flags. Only hidden nodes are eligible: the input node's size is the
    image and outputs are fixed 1x1 softmax nodes. Kernels of every edge
    incident to the resized node are invalidated (``kernel_weights=None``)
    because their ``|out_d - in_d| + 1`` shape changed.

    Attributes:
        change_height: Whether the y dimension is modified.
        change_width: Whether the x dimension is modified.
        filter_size_change_options: Size deltas drawn from uniformly
            (the paper used ``[-2, -1, +1, +2]``).
        minimum_filter_size: Lower clamp for the resulting size.
    """

    change_height: bool
    change_width: bool
    filter_size_change_options: tuple[int, ...]
    minimum_filter_size: int

    def apply_to_genome(
        self,
        genome: EXACTGenome,
        rng: Generator,
        innovation_tracker: InnovationTracker,
    ) -> EXACTGenome:
        """Return a genome with one hidden node resized, or ``genome`` when none exist.

        Args:
            genome: Genome to mutate.
            rng: Source of randomness for the node and delta draws.
            innovation_tracker: Unused; present for the operator protocol.

        Returns:
            The mutated genome, marked untrained.
        """
        hidden_node_indices = [
            node_index
            for node_index, node_gene in enumerate(genome.node_genes)
            if node_gene.node_type == "hidden"
        ]
        if not hidden_node_indices:
            logger.debug("ChangeFilterSizeMutation skipped: genome has no hidden nodes")
            return genome
        selected_node_index = hidden_node_indices[int(rng.integers(len(hidden_node_indices)))]
        selected_node = genome.node_genes[selected_node_index]

        new_filter_height = selected_node.filter_height
        new_filter_width = selected_node.filter_width
        if self.change_height:
            height_delta = self.filter_size_change_options[
                int(rng.integers(len(self.filter_size_change_options)))
            ]
            new_filter_height = max(self.minimum_filter_size, new_filter_height + height_delta)
        if self.change_width:
            width_delta = self.filter_size_change_options[
                int(rng.integers(len(self.filter_size_change_options)))
            ]
            new_filter_width = max(self.minimum_filter_size, new_filter_width + width_delta)
        if (new_filter_height, new_filter_width) == (
            selected_node.filter_height,
            selected_node.filter_width,
        ):
            logger.debug("ChangeFilterSizeMutation skipped: size delta clamped to no change")
            return genome

        resized_node = replace(
            selected_node,
            filter_height=new_filter_height,
            filter_width=new_filter_width,
        )
        mutated_node_genes = tuple(
            resized_node if node_index == selected_node_index else node_gene
            for node_index, node_gene in enumerate(genome.node_genes)
        )
        mutated_edge_genes = tuple(
            replace(edge_gene, kernel_weights=None)
            if selected_node.node_id
            in (edge_gene.source_node_id, edge_gene.target_node_id)
            else edge_gene
            for edge_gene in genome.edge_genes
        )
        return EXACTGenome(
            node_genes=mutated_node_genes,
            edge_genes=mutated_edge_genes,
            is_trained=False,
        )
