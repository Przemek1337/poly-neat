"""Component protocols shared by every NEAT-family algorithm in PolyNEAT.

Scope note: these interfaces are deliberately NEAT-family-wide, not
algorithm-agnostic in general. Concepts that every NEAT variant shares —
innovation tracking (historical markings), speciation, genome/phenotype
separation — belong here. Anything specific to a single variant belongs in
that variant's package under ``polyneat/algorithms/``.

Section numbers quoted on the individual protocols below refer to the NEAT
paper. Where a protocol has no counterpart there (``ParentSelection``,
``FitnessEvaluator``), the docstring says so instead of citing one.

References:
    Stanley, K. O., & Miikkulainen, R. (2002). Evolving Neural Networks
        through Augmenting Topologies. *Evolutionary Computation*, 10(2), 99-127.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, TypeVar, runtime_checkable

import torch
from numpy.random import Generator

from polyneat.core.type_aliases import FitnessValue, SpeciesId

if TYPE_CHECKING:
    from polyneat.configs.algorithm_config import AlgorithmConfig
    from polyneat.core.generation_statistics import GenerationStatistics
    from polyneat.core.population import Population


@runtime_checkable
class Genome(Protocol):
    """Immutable genotype. All operators return a new Genome instead of mutating in place.

    The genotype half of the paper's genotype/phenotype split (section 3.1).
    """

    def clone_genome(self) -> Genome: ...
    def to_serializable_dict(self) -> dict: ...

    @classmethod
    def from_serializable_dict(cls, payload: dict) -> Genome: ...


@runtime_checkable
class Phenotype(Protocol):
    """Executable neural network built from a Genome.

    The phenotype half of the paper's genotype/phenotype split (section 3.1).
    """

    def forward_pass(self, input_tensor: torch.Tensor) -> torch.Tensor: ...
    def reset_recurrent_state(self) -> None: ...


# Operator protocols below are generic over the concrete genome type they act on.
# Each NEAT-family algorithm shares the same operator interfaces but plugs in its
# own genome encoding (NEATGenome, a CPPN genome for HyperNEAT, ...). Binding to a
# TypeVar lets a concrete operator declare ``MutationOperator[NEATGenome]`` and get
# real static checking instead of erasing everything to the ``Genome`` base.
GenomeType = TypeVar("GenomeType", bound=Genome)


@runtime_checkable
class InnovationTracker(Protocol):
    """Issues globally-unique InnovationIds for new structural mutations.

    The historical markings of Stanley & Miikkulainen (2002), section 3.2 —
    shared by every NEAT-family algorithm so crossover can align genes.
    """

    def get_or_assign_innovation_id_for_connection(
        self, source_node_id: int, target_node_id: int
    ) -> int: ...

    def get_or_assign_node_split(
        self, split_connection_innovation_id: int, minimum_new_node_id: int
    ) -> tuple[int, int, int]:
        ...

    def reset_for_new_generation(self) -> None: ...


@runtime_checkable
class PhenotypeDecoder(Protocol[GenomeType]):
    """Maps Genome -> Phenotype. Different decoders for different encodings."""

    def build_phenotype_from_genome(self, genome: GenomeType) -> Phenotype: ...


# Contravariant because the config only appears in parameter position: a strategy
# accepting a broader config type may stand in where a narrower one is expected.
ConfigType = TypeVar("ConfigType", bound="AlgorithmConfig", contravariant=True)


@runtime_checkable
class InitialPopulationStrategy(Protocol[ConfigType]):
    """Builds generation 0 of a population.

    The paper starts from a uniform minimal topology with no hidden nodes
    (section 3.4, "Minimizing Dimensionality through Incremental Growth"); this
    protocol keeps that starting point swappable, so FS-NEAT can substitute its
    single-connection start without touching the generational loop.

    Selected by name from config via a per-algorithm strategy registry, or
    assigned directly to the algorithm's ``initial_population_factory``. Plain
    functions satisfy this protocol; a strategy needing state is a class with
    ``__call__``.
    """

    def __call__(
        self,
        config: ConfigType,
        innovation_tracker: InnovationTracker,
        rng: Generator,
    ) -> Population: ...


@runtime_checkable
class MutationOperator(Protocol[GenomeType]):
    """A single mutation, as a pure function (deterministic given RNG).

    Covers the paper's mutation types (section 3.1): weight mutation plus the
    two structural mutations, add-connection and add-node.
    """

    def apply_to_genome(
        self,
        genome: GenomeType,
        rng: Generator,
        innovation_tracker: InnovationTracker,
    ) -> GenomeType: ...


@runtime_checkable
class CrossoverOperator(Protocol[GenomeType]):
    """Mates two parents into one child; the fitter parent goes first.

    Gene alignment by historical marking (paper, section 3.2, Figure 4): the
    fitter-parent-first argument order encodes the paper's rule that disjoint
    and excess genes are inherited from the fitter parent.
    """

    def apply_to_parents(
        self,
        fitter_parent: GenomeType,
        less_fit_parent: GenomeType,
        rng: Generator,
    ) -> GenomeType: ...


@runtime_checkable
class ParentSelection(Protocol[GenomeType]):
    """Selects reproduction parents from a candidate pool.

    Not a concept of the source paper: Stanley & Miikkulainen (2002) prescribe
    fitness sharing and per-species offspring allocation but leave the draw of
    an individual parent unspecified. Implementations choose their own scheme.
    """

    def select_parent_indices(
        self,
        candidate_genomes: list[GenomeType],
        candidate_fitnesses: list[FitnessValue],
        number_of_parents_to_select: int,
        rng: Generator,
    ) -> list[int]:
        ...

    def select_parents(
        self,
        candidate_genomes: list[GenomeType],
        candidate_fitnesses: list[FitnessValue],
        number_of_parents_to_select: int,
        rng: Generator,
    ) -> list[GenomeType]: ...


@runtime_checkable
class Speciator(Protocol[GenomeType]):
    """Assigns each genome to a species (Stanley & Miikkulainen 2002, section 3.3)."""

    def assign_genomes_to_species(
        self, genomes: list[GenomeType], rng: Generator
    ) -> list[SpeciesId]: ...


@runtime_checkable
class FitnessEvaluator(Protocol):
    """Measures fitness. Batch is the primary interface."""

    def evaluate_batch_of_phenotypes(self, phenotypes: list[Phenotype]) -> list[FitnessValue]: ...


@runtime_checkable
class NeuroevolutionAlgorithm(Protocol):
    """The algorithm - composes the protocols above."""

    def create_initial_population(self, rng: Generator) -> Population: ...

    def advance_one_generation(
        self,
        current_population: Population,
        fitnesses_of_current_population: list[FitnessValue],
        rng: Generator,
    ) -> tuple[Population, GenerationStatistics]: ...

    @property
    def phenotype_decoder(self) -> PhenotypeDecoder: ...
