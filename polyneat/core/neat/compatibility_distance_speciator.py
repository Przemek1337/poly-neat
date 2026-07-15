"""Speciation by compatibility distance (paper, section 3.3).

Protects topological innovation by clustering structurally similar genomes
into species, so new structures compete within their own niche instead of
against the whole population.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from numpy.random import Generator

from polyneat.core.neat.neat_genome import NEATGenome
from polyneat.core.type_aliases import SpeciesId
from polyneat.logging_utils.custom_logger import get_logger

logger = get_logger(__name__)


@dataclass
class SpeciesRepresentative:
    """A species and the genome that represents it during assignment.

    Attributes:
        species_id: Stable id of the species across generations.
        representative_genome: Genome new candidates are compared against.
        member_genome_count_in_current_generation: Members assigned this pass.
        member_indices_in_current_generation: Population indices of those members.
    """

    species_id: SpeciesId
    representative_genome: NEATGenome
    member_genome_count_in_current_generation: int = 0
    member_indices_in_current_generation: list[int] = field(default_factory=list)


class CompatibilityDistanceSpeciator:
    """Stanley's NEAT speciation using the compatibility distance formula.

    Distance between two genomes (paper, section 3.3, eq. 1):

        δ = c1·E/N + c2·D/N + c3·W̄

    where ``E`` counts excess genes (past the max innovation id of the other),
    ``D`` counts disjoint genes (before that boundary but unmatched), ``W̄`` is
    the mean absolute weight difference over matching connections, and ``N`` is
    the size of the larger genome (clamped to 1 for tiny genomes).

    Each genome is assigned to the first species whose representative is within
    ``compatibility_threshold``. If none match, a new species is created and
    the current genome becomes its representative.

    After every assignment pass the representative of each surviving species is
    *resampled*: a random member of the just-assigned generation replaces the
    old representative. This matches the paper ("each existing species is
    represented by a random genome inside the species from the previous
    generation") and prevents species drifting away from a stale, frozen
    founder genome — which fragments the population into spurious new species.
    """

    def __init__(
        self,
        coefficient_excess_c1: float,
        coefficient_disjoint_c2: float,
        coefficient_weight_difference_c3: float,
        compatibility_threshold: float,
        small_genome_size_threshold_for_normalization: int = 20,
    ) -> None:
        self._coefficient_excess_c1 = coefficient_excess_c1
        self._coefficient_disjoint_c2 = coefficient_disjoint_c2
        self._coefficient_weight_difference_c3 = coefficient_weight_difference_c3
        self._compatibility_threshold = compatibility_threshold
        self._small_genome_size_threshold_for_normalization = (
            small_genome_size_threshold_for_normalization
        )
        self._species_representatives_from_previous_generation: list[SpeciesRepresentative] = []
        self._next_species_id: SpeciesId = 0

    def assign_genomes_to_species(
        self, genomes: list[NEATGenome], rng: Generator
    ) -> list[SpeciesId]:
        """Assign every genome to a species, creating new species as needed.

        Args:
            genomes: The population of the current generation, in order.
            rng: Source of randomness for representative resampling.

        Returns:
            The species id per genome, aligned with ``genomes``.
        """
        species_id_per_genome: list[SpeciesId] = [-1] * len(genomes)
        for representative in self._species_representatives_from_previous_generation:
            representative.member_genome_count_in_current_generation = 0
            representative.member_indices_in_current_generation = []

        for genome_index, genome in enumerate(genomes):
            assigned_species_id = self._find_or_create_species_for_genome(genome, genome_index)
            species_id_per_genome[genome_index] = assigned_species_id

        self._prune_empty_species()
        self._resample_species_representatives_from_current_members(genomes, rng)
        return species_id_per_genome

    def _resample_species_representatives_from_current_members(
        self,
        genomes: list[NEATGenome],
        rng: Generator,
    ) -> None:
        """Replace each species representative with a random current member.

        Runs after assignment, so from the perspective of the *next* generation
        the representative is a random genome of the previous generation, as
        prescribed by the paper.
        """
        for representative in self._species_representatives_from_previous_generation:
            member_indices = representative.member_indices_in_current_generation
            if not member_indices:
                continue
            sampled_member_index = member_indices[int(rng.integers(0, len(member_indices)))]
            representative.representative_genome = genomes[sampled_member_index]

    def _find_or_create_species_for_genome(
        self,
        genome: NEATGenome,
        genome_index_in_population: int,
    ) -> SpeciesId:
        """Place the genome in the first compatible species, or found a new one.

        Matches the paper's sequential placement: the genome joins the first
        species whose representative is within the compatibility threshold,
        so species never overlap.
        """
        for representative in self._species_representatives_from_previous_generation:
            distance_to_representative = self.compute_compatibility_distance(
                genome_a=genome,
                genome_b=representative.representative_genome,
            )
            if distance_to_representative < self._compatibility_threshold:
                representative.member_genome_count_in_current_generation += 1
                representative.member_indices_in_current_generation.append(
                    genome_index_in_population
                )
                return representative.species_id

        new_species_id = self._next_species_id
        self._next_species_id += 1
        new_representative = SpeciesRepresentative(
            species_id=new_species_id,
            representative_genome=genome,
            member_genome_count_in_current_generation=1,
            member_indices_in_current_generation=[genome_index_in_population],
        )
        self._species_representatives_from_previous_generation.append(new_representative)
        return new_species_id

    def _prune_empty_species(self) -> None:
        self._species_representatives_from_previous_generation = [
            representative
            for representative in self._species_representatives_from_previous_generation
            if representative.member_genome_count_in_current_generation > 0
        ]

    def compute_compatibility_distance(
        self,
        genome_a: NEATGenome,
        genome_b: NEATGenome,
    ) -> float:
        """Compute δ = c1·E/N + c2·D/N + c3·W̄ between two genomes (eq. 1).

        Args:
            genome_a: First genome.
            genome_b: Second genome.

        Returns:
            The compatibility distance; 0.0 for two empty genomes.
        """
        connections_by_innovation_id_a = {
            connection_gene.innovation_id: connection_gene
            for connection_gene in genome_a.connection_genes
        }
        connections_by_innovation_id_b = {
            connection_gene.innovation_id: connection_gene
            for connection_gene in genome_b.connection_genes
        }

        if not connections_by_innovation_id_a and not connections_by_innovation_id_b:
            return 0.0

        max_innovation_id_in_a = (
            max(connections_by_innovation_id_a) if connections_by_innovation_id_a else -1
        )
        max_innovation_id_in_b = (
            max(connections_by_innovation_id_b) if connections_by_innovation_id_b else -1
        )
        highest_shared_boundary_innovation_id = min(max_innovation_id_in_a, max_innovation_id_in_b)

        excess_gene_count = 0
        disjoint_gene_count = 0
        matching_gene_absolute_weight_differences: list[float] = []

        all_innovation_ids_across_both_genomes = set(connections_by_innovation_id_a) | set(
            connections_by_innovation_id_b
        )
        for innovation_id in all_innovation_ids_across_both_genomes:
            gene_a = connections_by_innovation_id_a.get(innovation_id)
            gene_b = connections_by_innovation_id_b.get(innovation_id)

            if gene_a is not None and gene_b is not None:
                matching_gene_absolute_weight_differences.append(abs(gene_a.weight - gene_b.weight))
            elif innovation_id > highest_shared_boundary_innovation_id:
                excess_gene_count += 1
            else:
                disjoint_gene_count += 1

        larger_genome_connection_count = max(
            len(connections_by_innovation_id_a), len(connections_by_innovation_id_b)
        )
        normalization_factor = (
            larger_genome_connection_count
            if larger_genome_connection_count >= self._small_genome_size_threshold_for_normalization
            else 1
        )

        mean_matching_weight_difference = (
            sum(matching_gene_absolute_weight_differences)
            / len(matching_gene_absolute_weight_differences)
            if matching_gene_absolute_weight_differences
            else 0.0
        )

        compatibility_distance_value = (
            self._coefficient_excess_c1 * excess_gene_count / normalization_factor
            + self._coefficient_disjoint_c2 * disjoint_gene_count / normalization_factor
            + self._coefficient_weight_difference_c3 * mean_matching_weight_difference
        )
        return compatibility_distance_value
