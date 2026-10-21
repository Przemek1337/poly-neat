from __future__ import annotations

from dataclasses import dataclass, field

from numpy.random import Generator

from polyneat.algorithms.neatdbm.difference_based_weight_mutation import (
    DifferenceBasedWeightMutation,
)
from polyneat.configs.neatdbm.neatdbm_config import NEATDBMConfig
from polyneat.core.neat.neat_algorithm import NEATAlgorithm
from polyneat.core.neat.neat_genome import NEATGenome
from polyneat.core.type_aliases import FitnessValue


@dataclass
class NEATDBMAlgorithm(NEATAlgorithm):
    """NEAT-DBM (Stanovov, Akhmedova & Semenkin, 2021).

    NEAT-DBM extends NEAT with the difference-based mutation (paper,
    section 3), a differential-evolution-inspired operator that recombines
    connection weights across four genomes at the positions their innovation
    numbers share. This class keeps NEAT's genetics and generational loop
    unchanged — the paper's conclusion presents the operator as general and
    applicable to "other realizations of NEAT framework" — and overrides
    only the child-production hook: after standard crossover and mutation,
    each freshly produced child undergoes the difference-based mutation with
    probability ``config.probability_of_difference_based_mutation``, using
    three mutually distinct donors drawn uniformly from the current
    population. When the population holds fewer than three genomes the
    operator is skipped.

    Build it with the inherited ``NEATDBMAlgorithm.from_config(neatdbm_config)``.

    Attributes:
        config: Validated NEAT-DBM hyperparameters.
        difference_based_weight_mutation: The difference-based mutation
            operator, constructed from ``config`` in ``__post_init__``.

    References:
        Stanovov, V., Akhmedova, Sh., & Semenkin, E. (2021). Neuroevolution of augmented
            topologies with difference-based mutation. *IOP Conference Series: Materials Science
            and Engineering*, 1047, 012075.
            DOI: 10.1088/1757-899X/1047/1/012075
    """

    config: NEATDBMConfig
    difference_based_weight_mutation: DifferenceBasedWeightMutation = field(init=False)

    def __post_init__(self) -> None:
        self.difference_based_weight_mutation = DifferenceBasedWeightMutation(
            scaling_factor=self.config.difference_mutation_scaling_factor
        )

    def _produce_single_child_from_species(
        self,
        reproducing_member_genomes: list[NEATGenome],
        reproducing_member_fitnesses: list[FitnessValue],
        all_genomes_in_population: list[NEATGenome],
        all_fitnesses_in_population: list[FitnessValue],
        rng: Generator,
    ) -> NEATGenome:
        """Produce one child, then apply the difference-based mutation with configured probability.

        The standard NEAT reproduction (selection, crossover, composite
        mutation) is inherited unchanged; the difference-based mutation is
        applied afterwards, drawing three mutually distinct donors from the
        whole current population. The child itself is a fresh offspring and
        never a population member, so distinct donors satisfy the paper's
        ``i != r1 != r2 != r3`` requirement.

        Args:
            reproducing_member_genomes: Reproduction survivors of the species.
            reproducing_member_fitnesses: Their raw fitnesses, aligned.
            all_genomes_in_population: Every genome of the current
                generation; the donor pool of the difference-based mutation.
            all_fitnesses_in_population: Their raw fitnesses, aligned.
            rng: Source of randomness for reproduction, the probability
                gate, and donor selection.

        Returns:
            The produced child genome, difference-mutated or not.
        """
        child_genome = super()._produce_single_child_from_species(
            reproducing_member_genomes=reproducing_member_genomes,
            reproducing_member_fitnesses=reproducing_member_fitnesses,
            all_genomes_in_population=all_genomes_in_population,
            all_fitnesses_in_population=all_fitnesses_in_population,
            rng=rng,
        )
        if len(all_genomes_in_population) < 3:
            return child_genome
        if rng.random() >= self.config.probability_of_difference_based_mutation:
            return child_genome

        donor_indices = rng.choice(len(all_genomes_in_population), size=3, replace=False)
        return self.difference_based_weight_mutation.apply_to_genome_with_donors(
            genome=child_genome,
            donor_base_genome=all_genomes_in_population[int(donor_indices[0])],
            donor_difference_first_genome=all_genomes_in_population[int(donor_indices[1])],
            donor_difference_second_genome=all_genomes_in_population[int(donor_indices[2])],
        )
