"""Enable-edge mutation of EXACT (Desell, 2017, section III-B)."""

from __future__ import annotations

from dataclasses import dataclass, replace

from numpy.random import Generator

from polyneat.algorithms.exact.exact_genome import EXACTGenome
from polyneat.core.component_protocols import InnovationTracker


@dataclass(frozen=True)
class EnableConvolutionEdgeMutation:
    """Enables one randomly selected disabled edge."""

    def apply_to_genome(
        self,
        genome: EXACTGenome,
        rng: Generator,
        innovation_tracker: InnovationTracker,
    ) -> EXACTGenome:
        """Return a genome with one disabled edge enabled, or ``genome`` when none exist.

        Args:
            genome: Genome to mutate.
            rng: Source of randomness for the edge draw.
            innovation_tracker: Unused; present for the operator protocol.

        Returns:
            The mutated genome, marked untrained.
        """
        disabled_edge_indices = [
            edge_index
            for edge_index, edge_gene in enumerate(genome.edge_genes)
            if not edge_gene.is_enabled
        ]
        if not disabled_edge_indices:
            return genome
        selected_edge_index = disabled_edge_indices[
            int(rng.integers(len(disabled_edge_indices)))
        ]
        mutated_edge_genes = tuple(
            replace(edge_gene, is_enabled=True)
            if edge_index == selected_edge_index
            else edge_gene
            for edge_index, edge_gene in enumerate(genome.edge_genes)
        )
        return replace(genome, edge_genes=mutated_edge_genes, is_trained=False)
