"""Speciation for DeepNEAT: compatibility distance plus a hyperparameter term.

DeepNEAT genomes are speciated the same way NEAT genomes are, but a node is a
whole layer rather than a single neuron, so the third term of the distance
formula compares layer hyperparameters instead of connection weights, and the
excess/disjoint counts are taken over edges (tensor flows) rather than over
connection genes.

The excess/disjoint structure of the distance formula follows the source paper's
NEAT-derived scheme directly. The hyperparameter term H̄ does not: the source paper
settles that a third term should compare layer hyperparameters, but not how. It gives
no field-wise comparison rule and no normalization scheme, because it describes
DeepNEAT at a level that does not go that far. The comparison and normalization used
here — position in the sorted allowed-value set for ``filters``/``kernel_size``/
``units``, min-max scaling for ``dropout_rate``, 0/1 for booleans, the mean over the
fields applicable to the layer's type, 1.0 for two layers of different types, and 1.0
when two genomes share no node — are this library's own reconstruction, filling a gap
the paper leaves open.

References:
    Miikkulainen, R., Liang, J., Meyerson, E., Rawal, A., Fink, D., Francon, O., Raju, B.,
        Shahrzad, H., Navruzyan, A., Duffy, N., & Hodjat, B. (2017). Evolving Deep Neural
        Networks. *arXiv:1703.00548*. Published in *Artificial Intelligence in the Age of
        Neural Networks and Brain Computing* (2019), pp. 293-312.
        DOI: 10.1016/B978-0-12-815480-9.00015-3
"""

from __future__ import annotations

from dataclasses import dataclass, field

from numpy.random import Generator

from polyneat.algorithms.deepneat.deepneat_genome import DeepNEATGenome, LayerNodeGene
from polyneat.core.type_aliases import SpeciesId
from polyneat.logging_utils.custom_logger import get_logger

logger = get_logger(__name__)


def _normalized_position_in_choices(value: int, choices: tuple[int, ...]) -> float:
    """Map a value to [0, 1] by its position in the sorted allowed set.

    Using the position rather than the raw value keeps the distance meaningful
    when the search space is geometric: 16 and 32 filters are adjacent choices,
    while 16 and 128 are three steps apart, which the raw difference would
    exaggerate.

    Args:
        value: The value to place.
        choices: The allowed values.

    Returns:
        0.0 for the smallest choice, 1.0 for the largest; 0.0 when there is only
        one choice or the value is not among them.
    """
    sorted_choices = sorted(choices)
    if len(sorted_choices) < 2 or value not in sorted_choices:
        return 0.0
    return sorted_choices.index(value) / (len(sorted_choices) - 1)


def compute_layer_hyperparameter_distance(
    first_node: LayerNodeGene,
    second_node: LayerNodeGene,
    available_filter_counts: tuple[int, ...],
    available_kernel_sizes: tuple[int, ...],
    available_dense_unit_counts: tuple[int, ...],
    dropout_rate_min: float,
    dropout_rate_max: float,
) -> float:
    """Return a normalized distance in [0, 1] between two layers.

    Layers of different types are maximally distant: a conv and a dense layer
    share no hyperparameters, so no field-wise comparison is meaningful.
    Otherwise the distance is the mean of the per-field absolute differences,
    each normalized to [0, 1]; numeric fields with a discrete allowed set are
    compared by position in that set rather than by raw value.

    This comparison and its normalization are a reconstruction by this library,
    not a rule taken from the source paper: the paper calls for a hyperparameter
    term in the distance formula without specifying how two layers' hyperparameters
    should be compared.

    Args:
        first_node: One layer.
        second_node: The other layer.
        available_filter_counts: Allowed conv output-channel counts.
        available_kernel_sizes: Allowed conv kernel sides.
        available_dense_unit_counts: Allowed dense widths.
        dropout_rate_min: Lower end of the dropout range.
        dropout_rate_max: Upper end of the dropout range.

    Returns:
        The distance; 0.0 for identical layers, 1.0 for different types.
    """
    if first_node.layer_type != second_node.layer_type:
        return 1.0
    if first_node.layer_type in ("input", "output"):
        return 0.0

    per_field_differences: list[float] = []

    dropout_span = dropout_rate_max - dropout_rate_min
    if dropout_span > 0.0:
        per_field_differences.append(
            min(1.0, abs(first_node.dropout_rate - second_node.dropout_rate) / dropout_span)
        )
    per_field_differences.append(
        0.0 if first_node.uses_batch_normalization == second_node.uses_batch_normalization
        else 1.0
    )

    if first_node.layer_type == "conv":
        per_field_differences.append(
            abs(
                _normalized_position_in_choices(
                    int(first_node.number_of_filters or 0), available_filter_counts
                )
                - _normalized_position_in_choices(
                    int(second_node.number_of_filters or 0), available_filter_counts
                )
            )
        )
        per_field_differences.append(
            abs(
                _normalized_position_in_choices(
                    int(first_node.kernel_size or 0), available_kernel_sizes
                )
                - _normalized_position_in_choices(
                    int(second_node.kernel_size or 0), available_kernel_sizes
                )
            )
        )
        per_field_differences.append(
            0.0
            if first_node.is_followed_by_max_pooling == second_node.is_followed_by_max_pooling
            else 1.0
        )
    else:
        per_field_differences.append(
            abs(
                _normalized_position_in_choices(
                    int(first_node.number_of_units or 0), available_dense_unit_counts
                )
                - _normalized_position_in_choices(
                    int(second_node.number_of_units or 0), available_dense_unit_counts
                )
            )
        )

    return sum(per_field_differences) / len(per_field_differences)


def compute_compatibility_distance(
    first_genome: DeepNEATGenome,
    second_genome: DeepNEATGenome,
    coefficient_excess_c1: float,
    coefficient_disjoint_c2: float,
    coefficient_hyperparameter_c3: float,
    available_filter_counts: tuple[int, ...],
    available_kernel_sizes: tuple[int, ...],
    available_dense_unit_counts: tuple[int, ...],
    dropout_rate_min: float,
    dropout_rate_max: float,
) -> float:
    """Return delta = c1*E/N + c2*D/N + c3*H, DeepNEAT's compatibility distance.

    ``E`` and ``D`` count excess and disjoint **edges** aligned by innovation id,
    ``N`` normalizes by the larger genome's edge count (set to 1 below 20 edges,
    as in the NEAT paper), and ``H`` is the mean hyperparameter distance over the
    layers the two genomes share by node id. Sharing no layer gives ``H = 1.0``.

    Args:
        first_genome: One genome.
        second_genome: The other genome.
        coefficient_excess_c1: Weight on excess edges.
        coefficient_disjoint_c2: Weight on disjoint edges.
        coefficient_hyperparameter_c3: Weight on the hyperparameter term.
        available_filter_counts: Allowed conv output-channel counts.
        available_kernel_sizes: Allowed conv kernel sides.
        available_dense_unit_counts: Allowed dense widths.
        dropout_rate_min: Lower end of the dropout range.
        dropout_rate_max: Upper end of the dropout range.

    Returns:
        The compatibility distance.
    """
    first_ids = {edge.innovation_id for edge in first_genome.edge_genes}
    second_ids = {edge.innovation_id for edge in second_genome.edge_genes}
    if not first_ids or not second_ids:
        excess_count = len(first_ids ^ second_ids)
        disjoint_count = 0
    else:
        highest_shared_innovation_id = min(max(first_ids), max(second_ids))
        symmetric_difference = first_ids ^ second_ids
        excess_count = sum(
            1 for innovation_id in symmetric_difference
            if innovation_id > highest_shared_innovation_id
        )
        disjoint_count = len(symmetric_difference) - excess_count

    larger_genome_edge_count = max(len(first_genome.edge_genes), len(second_genome.edge_genes))
    normalization = 1.0 if larger_genome_edge_count < 20 else float(larger_genome_edge_count)

    second_nodes_by_id = {node.node_id: node for node in second_genome.node_genes}
    shared_layer_distances = [
        compute_layer_hyperparameter_distance(
            first_node,
            second_nodes_by_id[first_node.node_id],
            available_filter_counts=available_filter_counts,
            available_kernel_sizes=available_kernel_sizes,
            available_dense_unit_counts=available_dense_unit_counts,
            dropout_rate_min=dropout_rate_min,
            dropout_rate_max=dropout_rate_max,
        )
        for first_node in first_genome.node_genes
        if first_node.node_id in second_nodes_by_id
    ]
    mean_hyperparameter_distance = (
        sum(shared_layer_distances) / len(shared_layer_distances)
        if shared_layer_distances
        else 1.0
    )

    return (
        coefficient_excess_c1 * excess_count / normalization
        + coefficient_disjoint_c2 * disjoint_count / normalization
        + coefficient_hyperparameter_c3 * mean_hyperparameter_distance
    )


@dataclass
class _SpeciesRepresentative:
    """A species and the genome that represents it during assignment.

    Attributes:
        species_id: Stable id of the species across generations.
        representative_genome: Genome new candidates are compared against.
        member_genome_count_in_current_generation: Members assigned this pass.
        member_indices_in_current_generation: Population indices of those members.
    """

    species_id: SpeciesId
    representative_genome: DeepNEATGenome
    member_genome_count_in_current_generation: int = 0
    member_indices_in_current_generation: list[int] = field(default_factory=list)


class DeepNEATSpeciator:
    """Speciates DeepNEAT genomes with a layer-hyperparameter distance term.

    Distance between two genomes:

        delta = c1*E/N + c2*D/N + c3*H

    where ``E`` counts excess edges (past the max innovation id of the other),
    ``D`` counts disjoint edges (before that boundary but unmatched), ``H`` is
    the mean layer-hyperparameter distance over nodes the two genomes share by
    ``node_id`` (see :func:`compute_layer_hyperparameter_distance`), and ``N``
    is the size of the larger genome's edge set (clamped to 1 for tiny genomes).

    Each genome is assigned to the first species whose representative is within
    ``compatibility_distance_threshold``. If none match, a new species is
    created and the current genome becomes its representative.

    After every assignment pass the representative of each surviving species is
    *resampled*: a random member of the just-assigned generation replaces the
    old representative. This matches the paper ("each existing species is
    represented by a random genome inside the species from the previous
    generation") and prevents species drifting away from a stale, frozen
    founder genome, which would fragment the population into spurious new
    species.

    This mirrors ``CompatibilityDistanceSpeciator`` in
    ``polyneat.core.neat.compatibility_distance_speciator`` almost exactly; the
    representative-tracking machinery is duplicated rather than shared so that
    the existing, test-covered speciator is left untouched.
    """

    def __init__(
        self,
        coefficient_excess_c1: float,
        coefficient_disjoint_c2: float,
        coefficient_hyperparameter_c3: float,
        compatibility_distance_threshold: float,
        available_filter_counts: tuple[int, ...],
        available_kernel_sizes: tuple[int, ...],
        available_dense_unit_counts: tuple[int, ...],
        dropout_rate_min: float,
        dropout_rate_max: float,
    ) -> None:
        self._coefficient_excess_c1 = coefficient_excess_c1
        self._coefficient_disjoint_c2 = coefficient_disjoint_c2
        self._coefficient_hyperparameter_c3 = coefficient_hyperparameter_c3
        self._compatibility_distance_threshold = compatibility_distance_threshold
        self._available_filter_counts = available_filter_counts
        self._available_kernel_sizes = available_kernel_sizes
        self._available_dense_unit_counts = available_dense_unit_counts
        self._dropout_rate_min = dropout_rate_min
        self._dropout_rate_max = dropout_rate_max
        self._species_representatives_from_previous_generation: list[_SpeciesRepresentative] = []
        self._next_species_id: SpeciesId = 0

    def assign_genomes_to_species(
        self, genomes: list[DeepNEATGenome], rng: Generator
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
        logger.debug(
            "Speciation assigned %d genomes to %d species",
            len(genomes),
            len(self._species_representatives_from_previous_generation),
        )
        return species_id_per_genome

    def _resample_species_representatives_from_current_members(
        self,
        genomes: list[DeepNEATGenome],
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
        genome: DeepNEATGenome,
        genome_index_in_population: int,
    ) -> SpeciesId:
        """Place the genome in the first compatible species, or found a new one.

        Matches the paper's sequential placement: the genome joins the first
        species whose representative is within the compatibility threshold,
        so species never overlap.
        """
        for representative in self._species_representatives_from_previous_generation:
            distance_to_representative = compute_compatibility_distance(
                first_genome=genome,
                second_genome=representative.representative_genome,
                coefficient_excess_c1=self._coefficient_excess_c1,
                coefficient_disjoint_c2=self._coefficient_disjoint_c2,
                coefficient_hyperparameter_c3=self._coefficient_hyperparameter_c3,
                available_filter_counts=self._available_filter_counts,
                available_kernel_sizes=self._available_kernel_sizes,
                available_dense_unit_counts=self._available_dense_unit_counts,
                dropout_rate_min=self._dropout_rate_min,
                dropout_rate_max=self._dropout_rate_max,
            )
            if distance_to_representative < self._compatibility_distance_threshold:
                representative.member_genome_count_in_current_generation += 1
                representative.member_indices_in_current_generation.append(
                    genome_index_in_population
                )
                return representative.species_id

        new_species_id = self._next_species_id
        self._next_species_id += 1
        new_representative = _SpeciesRepresentative(
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
