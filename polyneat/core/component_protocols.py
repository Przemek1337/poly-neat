from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import torch
from numpy.random import Generator

from polyneat.core.type_aliases import FitnessValue, SpeciesId

if TYPE_CHECKING:
    from polyneat.core.population import Population
    from polyneat.core.generation_statistics import GenerationStatistics


@runtime_checkable
class Genome(Protocol):
    """Immutable genotype. All operators return a new Genome instead of mutating in place."""

    def clone_genome(self) -> "Genome": ...
    def to_serializable_dict(self) -> dict: ...

    @classmethod
    def from_serializable_dict(cls, payload: dict) -> "Genome": ...


@runtime_checkable
class Phenotype(Protocol):
    """Executable neural network built from a Genome."""

    def forward_pass(self, input_tensor: torch.Tensor) -> torch.Tensor: ...
    def reset_recurrent_state(self) -> None: ...


@runtime_checkable
class PhenotypeBuilder(Protocol):
    """Maps Genome â†’ Phenotype. Different builders for different encodings."""

    def build_phenotype_from_genome(self, genome: Genome) -> Phenotype: ...


@runtime_checkable
class InnovationTracker(Protocol):
    """Issues globally-unique InnovationIds for new structural mutations."""

    def get_or_assign_innovation_id_for_connection(
        self, source_node_id: int, target_node_id: int
    ) -> int: ...

    def reset_for_new_generation(self) -> None: ...


@runtime_checkable
class MutationOperator(Protocol):
    """A single mutation, as a pure function (deterministic given RNG)."""

    def apply_to_genome(
        self,
        genome: Genome,
        rng: Generator,
        innovation_tracker: InnovationTracker,
    ) -> Genome: ...


@runtime_checkable
class CrossoverOperator(Protocol):
    def apply_to_parents(
        self,
        fitter_parent: Genome,
        less_fit_parent: Genome,
        rng: Generator,
    ) -> Genome: ...


@runtime_checkable
class ParentSelection(Protocol):
    def select_parents(
        self,
        candidate_genomes: list[Genome],
        candidate_fitnesses: list[FitnessValue],
        number_of_parents_to_select: int,
        rng: Generator,
    ) -> list[Genome]: ...


@runtime_checkable
class Speciator(Protocol):
    """Assigns each genome to a species."""

    def assign_genomes_to_species(self, genomes: list[Genome]) -> list[SpeciesId]: ...


@runtime_checkable
class FitnessEvaluator(Protocol):
    """Measures fitness. Batch is the primary interface."""

    def evaluate_batch_of_phenotypes(
        self, phenotypes: list[Phenotype]
    ) -> list[FitnessValue]: ...


@runtime_checkable
class NeuroevolutionAlgorithm(Protocol):
    """The algorithm â€” composes the protocols above."""

    def create_initial_population(self, rng: Generator) -> "Population": ...

    def advance_one_generation(
        self,
        current_population: "Population",
        fitnesses_of_current_population: list[FitnessValue],
        rng: Generator,
    ) -> tuple["Population", "GenerationStatistics"]: ...

    @property
    def phenotype_builder(self) -> PhenotypeBuilder: ...
