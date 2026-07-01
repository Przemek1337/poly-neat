from __future__ import annotations

from dataclasses import dataclass

from polyneat.core.component_protocols import Genome
from polyneat.core.type_aliases import SpeciesId


@dataclass(frozen=True)
class Population:
    genomes: list[Genome]
    species_assignments: list[SpeciesId] | None  # None when algorithm doesn't use speciation
    generation_number: int

    def size(self) -> int:
        return len(self.genomes)
