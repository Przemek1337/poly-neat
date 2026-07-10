from __future__ import annotations

from dataclasses import dataclass

from polyneat.core.component_protocols import Genome
from polyneat.core.type_aliases import SpeciesId


@dataclass(frozen=True)
class Population:
    """One generation of genomes.

    ``species_assignments[i]`` is the id of the species that *produced*.
    It is ``None`` for populations that precede any reproduction (the initial
    population) and for algorithms that do not use speciation
    ``genomes[i]`` in the reproduction step that built this population.
    """

    genomes: list[Genome]
    species_assignments: list[SpeciesId] | None
    generation_number: int

    def size(self) -> int:
        return len(self.genomes)
