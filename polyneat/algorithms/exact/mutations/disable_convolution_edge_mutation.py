"""Disable-edge mutation of EXACT (Desell, 2017, section III-B)."""

from __future__ import annotations

from dataclasses import dataclass, replace

from numpy.random import Generator

from polyneat.algorithms.exact.exact_genome import EXACTGenome
from polyneat.core.component_protocols import InnovationTracker
from polyneat.logging_utils.custom_logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class DisableConvolutionEdgeMutation:
    """Disables one randomly selected enabled edge; the gene stays in the genome.

    Applied unconditionally when drawn by the composite — reachability of the
    output nodes is checked once per child, after all draws (section III-B).
    """

    def apply_to_genome(
        self,
        genome: EXACTGenome,
        rng: Generator,
        innovation_tracker: InnovationTracker,
    ) -> EXACTGenome:
        """Return a genome with one enabled edge disabled, or ``genome`` when none exist.

        Args:
            genome: Genome to mutate.
            rng: Source of randomness for the edge draw.
            innovation_tracker: Unused; present for the operator protocol.

        Returns:
            The mutated genome, marked untrained.
        """
        enabled_edge_indices = [
            edge_index
            for edge_index, edge_gene in enumerate(genome.edge_genes)
            if edge_gene.is_enabled
        ]
        if not enabled_edge_indices:
            logger.debug("DisableConvolutionEdgeMutation skipped: no enabled edges")
            return genome
        selected_edge_index = enabled_edge_indices[
            int(rng.integers(len(enabled_edge_indices)))
        ]
        mutated_edge_genes = tuple(
            replace(edge_gene, is_enabled=False)
            if edge_index == selected_edge_index
            else edge_gene
            for edge_index, edge_gene in enumerate(genome.edge_genes)
        )
        return replace(genome, edge_genes=mutated_edge_genes, is_trained=False)
