from __future__ import annotations

from dataclasses import dataclass

from polyneat.core.component_protocols import Genome
from polyneat.core.type_aliases import SpeciesId


@dataclass(frozen=True)
class Population:
    """One generation of genomes.

    Attributes:
        genomes: The genomes of this generation.
        species_assignments: ``species_assignments[i]`` is the id of the
            species that produced ``genomes[i]`` in the reproduction step
            that built this population. ``None`` for populations that precede
            any reproduction (the initial population) and for algorithms that
            do not use speciation.
        generation_number: 0 for the initial population, then incremented.
    """

    genomes: list[Genome]
    species_assignments: list[SpeciesId] | None
    generation_number: int

    def size(self) -> int:
        """Return the number of genomes in this generation."""
        return len(self.genomes)
