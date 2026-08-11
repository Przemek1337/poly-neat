"""EXACT's non-speciation: the whole population is one species."""

from __future__ import annotations

from dataclasses import dataclass

from numpy.random import Generator

from polyneat.core.type_aliases import SpeciesId


@dataclass(frozen=True)
class SingleSpeciesSpeciator:
    """Assigns every genome to species 0.

    EXACT (Desell, 2017) has no speciation. Feeding a single species into
    the inherited NEAT loop degenerates fitness sharing and offspring
    allocation into plain whole-population selection, and the stagnation
    rule never fires because the species containing the best genome is
    always preserved.
    """

    def assign_genomes_to_species(self, genomes: list, rng: Generator) -> list[SpeciesId]:
        """Return species id 0 for every genome.

        Args:
            genomes: Genomes to assign (contents are not inspected).
            rng: Unused; present for the speciator protocol.

        Returns:
            ``[0] * len(genomes)``.
        """
        return [0] * len(genomes)
