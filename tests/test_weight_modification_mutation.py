from __future__ import annotations

import numpy as np

from polyneat.core.neat.global_innovation_tracker import GlobalInnovationTracker
from polyneat.core.neat.mutations.weight_modification_mutation import (
    WeightModificationMutation,
)
from polyneat.core.neat.neat_genome import ConnectionGene, NEATGenome, NodeGene

# Replacement draws land far away from the starting weight, so the three
# outcomes are told apart by value alone.
_REPLACEMENT_RANGE_MIN = 100.0
_REPLACEMENT_RANGE_MAX = 101.0
_STARTING_WEIGHT = 0.0
_CONNECTION_COUNT = 10


def _genome_with_uniform_weights() -> NEATGenome:
    node_genes = (
        NodeGene(node_id=0, node_type="input", activation_function_name="identity"),
        NodeGene(node_id=1, node_type="output", activation_function_name="sigmoid"),
    )
    connection_genes = tuple(
        ConnectionGene(
            innovation_id=index,
            source_node_id=0,
            target_node_id=1,
            weight=_STARTING_WEIGHT,
            is_enabled=True,
        )
        for index in range(_CONNECTION_COUNT)
    )
    # A single source/target pair repeated would be rejected as duplicate edges by
    # nothing in the genome validator, and the mutation never looks at endpoints.
    return NEATGenome(node_genes=node_genes, connection_genes=connection_genes)


_PERTURBATION_MAGNITUDE = 0.5


def _mutation(
    probability_of_genome_weight_mutation: float,
    probability_of_perturbation: float = 0.9,
    probability_of_replacement: float = 0.1,
) -> WeightModificationMutation:
    return WeightModificationMutation(
        probability_of_genome_weight_mutation=probability_of_genome_weight_mutation,
        probability_of_perturbation=probability_of_perturbation,
        probability_of_replacement=probability_of_replacement,
        weight_perturbation_magnitude=_PERTURBATION_MAGNITUDE,
        initial_weight_range_min=_REPLACEMENT_RANGE_MIN,
        initial_weight_range_max=_REPLACEMENT_RANGE_MAX,
    )


def _classify(weight: float) -> str:
    if weight == _STARTING_WEIGHT:
        return "unchanged"
    if _REPLACEMENT_RANGE_MIN <= weight <= _REPLACEMENT_RANGE_MAX:
        return "replaced"
    return "perturbed"


def test_genome_is_returned_unchanged_when_the_genome_level_draw_fails() -> None:
    """Section 4.1: only 80% of genomes have their connection weights mutated."""
    genome = _genome_with_uniform_weights()
    mutated = _mutation(probability_of_genome_weight_mutation=0.0).apply_to_genome(
        genome, np.random.default_rng(0), GlobalInnovationTracker()
    )
    assert mutated is genome


def test_every_weight_is_touched_once_the_genome_level_draw_passes() -> None:
    genome = _genome_with_uniform_weights()
    mutated = _mutation(
        probability_of_genome_weight_mutation=1.0, probability_of_perturbation=1.0
    ).apply_to_genome(genome, np.random.default_rng(0), GlobalInnovationTracker())
    assert all(_classify(gene.weight) == "perturbed" for gene in mutated.connection_genes)


def test_a_genome_is_spared_as_a_whole_rather_than_weight_by_weight() -> None:
    """The paper's gate is per *genome*, so untouched genomes come out whole.

    With per-connection independence a 10-connection genome would survive
    completely untouched with probability 0.1**10, i.e. never.
    """
    genome = _genome_with_uniform_weights()
    mutation = _mutation(probability_of_genome_weight_mutation=0.8)
    # One continuing stream: the genome, not the weight, is the sampling unit,
    # so the bounds below are three sigma on 3000 genome-level draws.
    rng = np.random.default_rng(12345)
    tracker = GlobalInnovationTracker()

    completely_untouched_count = 0
    sample_size = 3000
    for _ in range(sample_size):
        mutated = mutation.apply_to_genome(genome, rng, tracker)
        if all(gene.weight == _STARTING_WEIGHT for gene in mutated.connection_genes):
            completely_untouched_count += 1

    untouched_fraction = completely_untouched_count / sample_size
    assert 0.178 < untouched_fraction < 0.222, untouched_fraction


def test_per_weight_rates_match_the_paper() -> None:
    """80% genome gate x (90% perturb / 10% replace) -> 0.72 / 0.08 / 0.20."""
    genome = _genome_with_uniform_weights()
    mutation = _mutation(probability_of_genome_weight_mutation=0.8)
    rng = np.random.default_rng(12345)
    tracker = GlobalInnovationTracker()

    outcome_counts = {"perturbed": 0, "replaced": 0, "unchanged": 0}
    sample_size = 3000
    for _ in range(sample_size):
        mutated = mutation.apply_to_genome(genome, rng, tracker)
        for gene in mutated.connection_genes:
            outcome_counts[_classify(gene.weight)] += 1

    total = sample_size * _CONNECTION_COUNT
    # The genome-level gate correlates the ten weights of one genome, so the
    # tolerance is driven by 3000 genome draws, not 30000 weight draws.
    assert 0.695 < outcome_counts["perturbed"] / total < 0.745, outcome_counts
    assert 0.065 < outcome_counts["replaced"] / total < 0.095, outcome_counts
    assert 0.178 < outcome_counts["unchanged"] / total < 0.222, outcome_counts


def test_replacement_draws_from_the_configured_weight_range() -> None:
    genome = _genome_with_uniform_weights()
    mutated = _mutation(
        probability_of_genome_weight_mutation=1.0,
        probability_of_perturbation=0.0,
        probability_of_replacement=1.0,
    ).apply_to_genome(genome, np.random.default_rng(0), GlobalInnovationTracker())
    assert all(
        _REPLACEMENT_RANGE_MIN <= gene.weight <= _REPLACEMENT_RANGE_MAX
        for gene in mutated.connection_genes
    )


def test_expression_flags_and_endpoints_survive_weight_mutation() -> None:
    genome = _genome_with_uniform_weights()
    mutated = _mutation(probability_of_genome_weight_mutation=1.0).apply_to_genome(
        genome, np.random.default_rng(0), GlobalInnovationTracker()
    )
    assert [gene.innovation_id for gene in mutated.connection_genes] == [
        gene.innovation_id for gene in genome.connection_genes
    ]
    assert all(gene.is_enabled for gene in mutated.connection_genes)
    assert mutated.node_genes == genome.node_genes


def _perturbation_deltas(sample_size: int) -> list[float]:
    genome = _genome_with_uniform_weights()
    mutation = _mutation(
        probability_of_genome_weight_mutation=1.0, probability_of_perturbation=1.0
    )
    rng = np.random.default_rng(2024)
    tracker = GlobalInnovationTracker()
    deltas: list[float] = []
    for _ in range(sample_size):
        mutated = mutation.apply_to_genome(genome, rng, tracker)
        deltas.extend(gene.weight - _STARTING_WEIGHT for gene in mutated.connection_genes)
    return deltas


def test_perturbation_never_exceeds_the_configured_magnitude() -> None:
    """Section 4.1 perturbs weights *uniformly*, so the delta is bounded.

    Stanley's reference draws randposneg() * randfloat() * power, i.e. uniform
    on [-power, power]. A Gaussian of the same nominal strength would leave the
    bound roughly a third of the time.
    """
    deltas = _perturbation_deltas(sample_size=500)
    assert max(abs(delta) for delta in deltas) <= _PERTURBATION_MAGNITUDE


def test_perturbation_is_spread_evenly_across_its_range() -> None:
    """Four equal-width bins over [-magnitude, magnitude] should be equally likely."""
    deltas = _perturbation_deltas(sample_size=1000)
    bin_edges = np.linspace(-_PERTURBATION_MAGNITUDE, _PERTURBATION_MAGNITUDE, 5)
    counts, _ = np.histogram(deltas, bins=bin_edges)
    fractions = counts / len(deltas)
    assert all(0.22 < fraction < 0.28 for fraction in fractions), fractions
