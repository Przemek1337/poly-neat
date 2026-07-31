"""Vanilla NEAT generational loop and its overridable component factories.

Implements the full NEAT reproduction cycle from Stanley & Miikkulainen
(2002): speciation (section 3.3), explicit fitness sharing (eq. 2), species
stagnation (section 4.1), proportional offspring allocation, elitism,
survival-threshold truncation, and crossover/mutation reproduction.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from statistics import mean, median
from typing import cast

import torch
from numpy.random import Generator

from polyneat.configs.neat.neat_config import NEATConfig
from polyneat.core.component_protocols import (
    CrossoverOperator,
    Genome,
    MutationOperator,
    ParentSelection,
    PhenotypeDecoder,
    Speciator,
)
from polyneat.core.generation_statistics import GenerationStatistics
from polyneat.core.neat.compatibility_distance_speciator import (
    CompatibilityDistanceSpeciator,
)
from polyneat.core.neat.global_innovation_tracker import GlobalInnovationTracker
from polyneat.core.neat.initial_population import (
    NEATInitialPopulationStrategy,
    build_fully_connected_initial_population,
    resolve_initial_population_strategy_by_name,
)
from polyneat.core.neat.mutations.add_connection_mutation import AddConnectionMutation
from polyneat.core.neat.mutations.add_node_mutation import AddNodeMutation
from polyneat.core.neat.mutations.composite_neat_mutation import CompositeNEATMutation
from polyneat.core.neat.mutations.toggle_connection_enabled_mutation import (
    ToggleConnectionEnabledMutation,
)
from polyneat.core.neat.mutations.weight_modification_mutation import (
    WeightModificationMutation,
)
from polyneat.core.neat.neat_crossover import NEATCrossover
from polyneat.core.neat.neat_genome import NEATGenome
from polyneat.core.neat.neat_phenotype_decoder import NEATPhenotypeDecoder
from polyneat.core.neat.tournament_parent_selection import TournamentParentSelection
from polyneat.core.population import Population
from polyneat.core.type_aliases import FitnessValue, SpeciesId
from polyneat.logging_utils.custom_logger import get_logger

logger = get_logger(__name__)


@dataclass
class _SpeciesReproductionState:
    """Internal per-species bookkeeping used inside ``advance_one_generation``."""

    species_id: SpeciesId
    member_indices_in_current_population: list[int]
    raw_fitnesses: list[FitnessValue]
    adjusted_fitnesses: list[FitnessValue]
    mean_adjusted_fitness: float
    best_adjusted_fitness: float
    best_raw_fitness: float
    offspring_slot_count: int = 0
    generations_since_last_improvement: int = 0
    best_raw_fitness_ever: float = float("-inf")


@dataclass
class NEATAlgorithm:
    """Vanilla NEAT (Stanley & Miikkulainen, 2002) built from protocol-conforming components.

    This class is the base of PolyNEAT's algorithm hierarchy. Derived
    algorithms (e.g. :class:`~polyneat.algorithms.fsneat.FSNEATAlgorithm`)
    subclass it and override only the ``_build_*`` factory methods or
    ``create_initial_population``; the generational loop in
    ``advance_one_generation`` is shared and stays untouched.

    Attributes:
        config: Validated NEAT hyperparameters.
        mutation: Mutation operator applied to every non-elite child.
        crossover: Crossover operator aligning genes by innovation id
            (section 3.2 of the paper).
        parent_selection: Within-species parent selection operator.
        speciator: Assigns genomes to species by compatibility distance
            (section 3.3).
        innovation_tracker: Issues innovation ids for structural mutations
            (section 3.2, historical markings).
        initial_population_factory: Strategy building generation 0; ``None``
            falls back to the fully connected minimal start (section 3.4).
    """

    config: NEATConfig
    mutation: MutationOperator[NEATGenome]
    crossover: CrossoverOperator[NEATGenome]
    parent_selection: ParentSelection[NEATGenome]
    speciator: Speciator[NEATGenome]
    innovation_tracker: GlobalInnovationTracker
    _phenotype_decoder: PhenotypeDecoder[NEATGenome]
    initial_population_factory: NEATInitialPopulationStrategy | None = None
    _species_stagnation_bookkeeping: dict[SpeciesId, _SpeciesReproductionState] = field(
        default_factory=dict
    )

    @property
    def phenotype_decoder(self) -> PhenotypeDecoder:
        return self._phenotype_decoder

    @classmethod
    def from_config(
        cls,
        config: NEATConfig,
        device_for_phenotype_computation: torch.device | None = None,
    ) -> NEATAlgorithm:
        """Build the algorithm from a config via overridable ``_build_*`` factories.

        Template method: every component is produced by a dedicated factory
        method, so a derived algorithm overrides only the pieces it changes
        while the generational loop and the remaining components stay shared.

        Args:
            config: Validated NEAT hyperparameters.
            device_for_phenotype_computation: Device override for phenotype
                evaluation. ``None`` falls back to
                ``config.device_for_phenotype_evaluation``.

        Returns:
            A fully wired algorithm instance of ``cls``.
        """
        resolved_device = device_for_phenotype_computation or torch.device(
            config.device_for_phenotype_evaluation
        )

        return cls(
            config=config,
            mutation=cls._build_mutation(config),
            crossover=cls._build_crossover(config),
            parent_selection=cls._build_parent_selection(config),
            speciator=cls._build_speciator(config),
            innovation_tracker=GlobalInnovationTracker(),
            _phenotype_decoder=cls._build_phenotype_decoder(config, resolved_device),
            initial_population_factory=resolve_initial_population_strategy_by_name(
                config.initial_population_strategy
            ),
        )

    @classmethod
    def _build_mutation(cls, config: NEATConfig) -> MutationOperator[NEATGenome]:
        """Compose the four canonical NEAT mutations in their fixed order."""
        return CompositeNEATMutation(
            ordered_individual_mutations=[
                WeightModificationMutation(
                    probability_of_genome_weight_mutation=(
                        config.probability_of_genome_weight_mutation
                    ),
                    probability_of_perturbation=config.probability_of_weight_perturbation,
                    probability_of_replacement=config.probability_of_weight_replacement,
                    weight_perturbation_magnitude=config.weight_perturbation_magnitude,
                    initial_weight_range_min=config.initial_weight_range_min,
                    initial_weight_range_max=config.initial_weight_range_max,
                ),
                AddConnectionMutation(
                    probability_of_application=config.probability_of_add_connection_mutation,
                    initial_weight_range_min=config.initial_weight_range_min,
                    initial_weight_range_max=config.initial_weight_range_max,
                ),
                AddNodeMutation(
                    probability_of_application=config.probability_of_add_node_mutation,
                    activation_function_name_for_new_hidden_node=(
                        config.default_activation_function_for_hidden_nodes
                    ),
                ),
                ToggleConnectionEnabledMutation(
                    probability_of_application=config.probability_of_toggle_connection_enabled,
                ),
            ]
        )

    @classmethod
    def _build_crossover(cls, config: NEATConfig) -> CrossoverOperator[NEATGenome]:
        """Build the innovation-aligned NEAT crossover operator."""
        return NEATCrossover(
            probability_of_inheriting_from_fitter_parent_for_matching_genes=(
                config.probability_of_inheriting_from_fitter_parent_for_matching_genes
            ),
        )

    @classmethod
    def _build_parent_selection(cls, config: NEATConfig) -> ParentSelection[NEATGenome]:
        """Build the within-species parent selection operator."""
        return TournamentParentSelection(
            tournament_size=config.tournament_size_for_parent_selection
        )

    @classmethod
    def _build_speciator(cls, config: NEATConfig) -> Speciator[NEATGenome]:
        """Build the compatibility-distance speciator."""
        return CompatibilityDistanceSpeciator(
            coefficient_excess_c1=config.compatibility_distance_coefficient_excess_c1,
            coefficient_disjoint_c2=config.compatibility_distance_coefficient_disjoint_c2,
            coefficient_weight_difference_c3=(
                config.compatibility_distance_coefficient_weight_difference_c3
            ),
            compatibility_threshold=config.compatibility_distance_threshold,
        )

    @classmethod
    def _build_phenotype_decoder(
        cls, config: NEATConfig, device: torch.device
    ) -> PhenotypeDecoder[NEATGenome]:
        """Build the genome-to-phenotype decoder bound to ``device``."""
        return NEATPhenotypeDecoder(
            device_for_computation=device,
        )

    def create_initial_population(self, rng: Generator) -> Population:
        """Build generation 0, delegating to the configured strategy.

        Defaults to the fully connected minimal topology of section 3.4 of
        the paper (all inputs and the bias wired to all outputs, no hidden
        nodes) when no strategy is configured.

        Args:
            rng: Source of randomness for initial connection weights.

        Returns:
            Generation-0 population of ``config.population_size`` genomes.
        """
        if self.initial_population_factory is not None:
            return self.initial_population_factory(self.config, self.innovation_tracker, rng)
        return build_fully_connected_initial_population(
            config=self.config,
            innovation_tracker=self.innovation_tracker,
            rng=rng,
        )

    def advance_one_generation(
        self,
        current_population: Population,
        fitnesses_of_current_population: list[FitnessValue],
        rng: Generator,
    ) -> tuple[Population, GenerationStatistics]:
        """Run one full NEAT reproduction cycle and return the next generation.

        Pipeline (in paper order): speciation → adjusted fitness (explicit
        fitness sharing, eq. 2) → species stagnation (section 4.1) →
        offspring-slot allocation proportional to each species' summed
        adjusted fitness (section 3.3) → per-species elitism → survival
        threshold truncation → reproduction (crossover + mutation, with rare
        interspecies mating) → innovation-tracker reset.

        Args:
            current_population: The population to reproduce from.
            fitnesses_of_current_population: Raw fitness per genome, aligned
                with ``current_population.genomes``.
            rng: Source of randomness for the whole cycle.

        Returns:
            A tuple of the next-generation population (same size) and the
            statistics of the generation that was just evaluated.
        """
        generation_start_wall_time = time.perf_counter()

        neat_genomes_in_current_population = cast("list[NEATGenome]", current_population.genomes)

        species_id_per_genome_index = self.speciator.assign_genomes_to_species(
            neat_genomes_in_current_population,
            rng,
        )
        member_indices_by_species_id: dict[SpeciesId, list[int]] = {}
        for genome_index, assigned_species_id in enumerate(species_id_per_genome_index):
            member_indices_by_species_id.setdefault(assigned_species_id, []).append(genome_index)

        per_species_reproduction_states = self._build_per_species_reproduction_states(
            member_indices_by_species_id=member_indices_by_species_id,
            fitnesses_of_current_population=fitnesses_of_current_population,
        )
        self._update_species_stagnation_bookkeeping(per_species_reproduction_states)

        non_stagnated_species_states = self._drop_stagnated_species_preserving_overall_best(
            per_species_reproduction_states=per_species_reproduction_states,
            fitnesses_of_current_population=fitnesses_of_current_population,
        )
        if not non_stagnated_species_states:
            logger.warning(
                "All species stagnated at generation %d; reviving them for this step",
                current_population.generation_number,
            )
            non_stagnated_species_states = per_species_reproduction_states

        self._allocate_offspring_slots_across_species(
            non_stagnated_species_states=non_stagnated_species_states,
            total_offspring_slots_available=self.config.population_size,
        )

        offspring_genomes: list[NEATGenome] = []
        offspring_species_ids: list[SpeciesId] = []
        for species_state in non_stagnated_species_states:
            member_genomes_in_species = [
                neat_genomes_in_current_population[member_index]
                for member_index in species_state.member_indices_in_current_population
            ]
            member_fitnesses_in_species = species_state.raw_fitnesses

            # Elitism spends the species' own offspring slots, it does not add
            # to them: a species the allocator gave no slots to is dying out and
            # must not emit a champion anyway. Without the cap the emitted total
            # exceeds population_size and the trailing genomes - belonging to
            # whichever species happens to be last, not to the weakest one - are
            # cut by the truncation below.
            elite_genomes_from_species = self._pick_elite_genomes_from_species(
                member_genomes_in_species=member_genomes_in_species,
                member_fitnesses_in_species=member_fitnesses_in_species,
            )[: species_state.offspring_slot_count]

            offspring_genomes.extend(elite_genomes_from_species)
            offspring_species_ids.extend(
                [species_state.species_id] * len(elite_genomes_from_species)
            )

            (
                surviving_member_genomes,
                surviving_member_fitnesses,
            ) = self._truncate_species_to_reproduction_survivors(
                member_genomes_in_species=member_genomes_in_species,
                member_fitnesses_in_species=member_fitnesses_in_species,
            )

            remaining_offspring_slots_to_fill_for_species = (
                species_state.offspring_slot_count - len(elite_genomes_from_species)
            )
            for _slot_index in range(max(0, remaining_offspring_slots_to_fill_for_species)):
                child_genome = self._produce_single_child_from_species(
                    reproducing_member_genomes=surviving_member_genomes,
                    reproducing_member_fitnesses=surviving_member_fitnesses,
                    all_genomes_in_population=neat_genomes_in_current_population,
                    all_fitnesses_in_population=fitnesses_of_current_population,
                    rng=rng,
                )
                offspring_genomes.append(child_genome)
                offspring_species_ids.append(species_state.species_id)

        overall_best_genome_index = fitnesses_of_current_population.index(
            max(fitnesses_of_current_population)
        )

        overall_best_genome_in_current_population = neat_genomes_in_current_population[
            overall_best_genome_index
        ]

        overall_best_genome_species_id = species_id_per_genome_index[overall_best_genome_index]
        while len(offspring_genomes) < self.config.population_size:
            offspring_genomes.append(overall_best_genome_in_current_population)
            offspring_species_ids.append(overall_best_genome_species_id)
        offspring_genomes = offspring_genomes[: self.config.population_size]
        offspring_species_ids = offspring_species_ids[: self.config.population_size]

        next_generation_population = Population(
            genomes=cast("list[Genome]", offspring_genomes),
            species_assignments=offspring_species_ids,
            generation_number=current_population.generation_number + 1,
        )

        self.innovation_tracker.reset_for_new_generation()

        elapsed_seconds = time.perf_counter() - generation_start_wall_time
        generation_statistics = GenerationStatistics(
            generation_number=current_population.generation_number,
            best_fitness=max(fitnesses_of_current_population),
            mean_fitness=mean(fitnesses_of_current_population),
            median_fitness=median(fitnesses_of_current_population),
            number_of_species=len(non_stagnated_species_states),
            number_of_genomes_evaluated=len(fitnesses_of_current_population),
            elapsed_seconds=elapsed_seconds,
            extra_metrics={
                "innovation_id_high_water_mark": float(
                    self.innovation_tracker.next_innovation_id_snapshot
                ),
            },
        )
        return next_generation_population, generation_statistics

    def _build_per_species_reproduction_states(
        self,
        member_indices_by_species_id: dict[SpeciesId, list[int]],
        fitnesses_of_current_population: list[FitnessValue],
    ) -> list[_SpeciesReproductionState]:
        """Compute per-species adjusted fitnesses (explicit fitness sharing).

        Follows eq. 2 of the paper: with species already clustered by the
        compatibility threshold, the sharing denominator reduces to the
        species size, so ``adjusted = raw / |species|``.
        """
        per_species_reproduction_states: list[_SpeciesReproductionState] = []
        for species_id, member_indices_in_species in member_indices_by_species_id.items():
            raw_fitnesses_in_species = [
                fitnesses_of_current_population[member_index]
                for member_index in member_indices_in_species
            ]
            species_member_count = max(1, len(member_indices_in_species))
            adjusted_fitnesses_in_species = [
                raw_fitness / species_member_count for raw_fitness in raw_fitnesses_in_species
            ]
            per_species_reproduction_states.append(
                _SpeciesReproductionState(
                    species_id=species_id,
                    member_indices_in_current_population=list(member_indices_in_species),
                    raw_fitnesses=raw_fitnesses_in_species,
                    adjusted_fitnesses=adjusted_fitnesses_in_species,
                    mean_adjusted_fitness=mean(adjusted_fitnesses_in_species),
                    best_adjusted_fitness=max(adjusted_fitnesses_in_species),
                    best_raw_fitness=max(raw_fitnesses_in_species),
                )
            )
        return per_species_reproduction_states

    def _update_species_stagnation_bookkeeping(
        self,
        per_species_reproduction_states: list[_SpeciesReproductionState],
    ) -> None:
        """Track how long each species has gone without improving its best raw fitness.

        Stagnation deliberately tracks *raw* fitness: adjusted fitness shrinks
        as a species grows, so tracking it would make healthy growing species
        look stagnant.
        """
        current_generation_species_ids: set[SpeciesId] = set()
        for species_state in per_species_reproduction_states:
            current_generation_species_ids.add(species_state.species_id)
            existing_bookkeeping = self._species_stagnation_bookkeeping.get(
                species_state.species_id
            )
            if existing_bookkeeping is None:
                self._species_stagnation_bookkeeping[species_state.species_id] = species_state
                species_state.best_raw_fitness_ever = species_state.best_raw_fitness
                species_state.generations_since_last_improvement = 0
                continue

            if species_state.best_raw_fitness > existing_bookkeeping.best_raw_fitness_ever:
                species_state.best_raw_fitness_ever = species_state.best_raw_fitness
                species_state.generations_since_last_improvement = 0
            else:
                species_state.best_raw_fitness_ever = existing_bookkeeping.best_raw_fitness_ever
                species_state.generations_since_last_improvement = (
                    existing_bookkeeping.generations_since_last_improvement + 1
                )
            self._species_stagnation_bookkeeping[species_state.species_id] = species_state

        for stale_species_id in list(self._species_stagnation_bookkeeping.keys()):
            if stale_species_id not in current_generation_species_ids:
                del self._species_stagnation_bookkeeping[stale_species_id]

    def _drop_stagnated_species_preserving_overall_best(
        self,
        per_species_reproduction_states: list[_SpeciesReproductionState],
        fitnesses_of_current_population: list[FitnessValue],
    ) -> list[_SpeciesReproductionState]:
        """Remove species stagnated past the configured limit (paper, section 4.1).

        The species containing the population's best genome always survives,
        so the champion is never lost to stagnation removal.
        """
        overall_best_genome_index = fitnesses_of_current_population.index(
            max(fitnesses_of_current_population)
        )
        surviving_species_states: list[_SpeciesReproductionState] = []
        for species_state in per_species_reproduction_states:
            contains_overall_best_genome = (
                overall_best_genome_index in species_state.member_indices_in_current_population
            )
            has_stagnated = (
                species_state.generations_since_last_improvement
                >= self.config.species_stagnation_generations_limit
            )
            if has_stagnated and not contains_overall_best_genome:
                logger.debug(
                    "Dropping stagnated species_id=%d (no improvement for %d gens)",
                    species_state.species_id,
                    species_state.generations_since_last_improvement,
                )
                continue
            surviving_species_states.append(species_state)
        return surviving_species_states

    def _allocate_offspring_slots_across_species(
        self,
        non_stagnated_species_states: list[_SpeciesReproductionState],
        total_offspring_slots_available: int,
    ) -> None:
        """Distribute offspring slots proportionally to each species' summed adjusted fitness.

        The paper assigns each species a share proportional to the *sum* of its
        members' adjusted fitnesses (equivalently: the species' mean raw
        fitness). Using the mean adjusted fitness instead would divide by the
        species size twice and starve large, well-performing species.
        Largest-remainder rounding keeps the slot total exact.
        """
        total_adjusted_fitness_per_species = [
            max(0.0, sum(species_state.adjusted_fitnesses))
            for species_state in non_stagnated_species_states
        ]
        sum_of_total_adjusted_fitnesses = sum(total_adjusted_fitness_per_species)

        if sum_of_total_adjusted_fitnesses <= 0.0:
            equal_share = total_offspring_slots_available // len(non_stagnated_species_states)
            leftover_slots = total_offspring_slots_available - equal_share * len(
                non_stagnated_species_states
            )
            for species_index, species_state in enumerate(non_stagnated_species_states):
                species_state.offspring_slot_count = equal_share + (
                    1 if species_index < leftover_slots else 0
                )
            return

        exact_proportional_shares = [
            species_total / sum_of_total_adjusted_fitnesses * total_offspring_slots_available
            for species_total in total_adjusted_fitness_per_species
        ]
        for species_state, exact_share in zip(
            non_stagnated_species_states, exact_proportional_shares, strict=True
        ):
            species_state.offspring_slot_count = int(exact_share)

        remaining_slots_after_flooring = total_offspring_slots_available - sum(
            species_state.offspring_slot_count for species_state in non_stagnated_species_states
        )
        species_indices_by_descending_fractional_part = sorted(
            range(len(non_stagnated_species_states)),
            key=lambda index: (
                exact_proportional_shares[index] - int(exact_proportional_shares[index])
            ),
            reverse=True,
        )
        for species_index in species_indices_by_descending_fractional_part[
            :remaining_slots_after_flooring
        ]:
            non_stagnated_species_states[species_index].offspring_slot_count += 1

    def _pick_elite_genomes_from_species(
        self,
        member_genomes_in_species: list[NEATGenome],
        member_fitnesses_in_species: list[FitnessValue],
    ) -> list[NEATGenome]:
        """Return the genomes copied unchanged into the next generation.

        The paper (section 4.1) carries over the champion of each species with
        *more than* five networks. Both the count and the size threshold are
        configurable here; the threshold is applied inclusively
        (``len(members) >= minimum_species_size_for_elitism``), so the default of
        6 reproduces the paper's rule exactly.

        Stanley's reference implementation gates elitism on the species' *allocated
        offspring* instead (``expected_offspring > 5``); PolyNEAT follows the
        published text and gates on member count.
        """
        if len(member_genomes_in_species) < self.config.minimum_species_size_for_elitism:
            return []
        member_indices_sorted_by_fitness_descending = sorted(
            range(len(member_genomes_in_species)),
            key=lambda index: member_fitnesses_in_species[index],
            reverse=True,
        )
        return [
            member_genomes_in_species[index]
            for index in member_indices_sorted_by_fitness_descending[
                : self.config.species_elitism_count
            ]
        ]

    def _truncate_species_to_reproduction_survivors(
        self,
        member_genomes_in_species: list[NEATGenome],
        member_fitnesses_in_species: list[FitnessValue],
    ) -> tuple[list[NEATGenome], list[FitnessValue]]:
        """Keep only the top fraction of a species as reproduction candidates.

        The paper eliminates the lowest performing members before reproduction;
        Stanley's reference implementation keeps the top ``survival_thresh``
        fraction (0.2 by default). At least two members are kept when available
        so within-species crossover stays possible.
        """
        species_member_count = len(member_genomes_in_species)
        surviving_member_count = min(
            species_member_count,
            max(
                2,
                math.ceil(
                    self.config.species_survival_fraction_for_reproduction * species_member_count
                ),
            ),
        )
        member_indices_sorted_by_fitness_descending = sorted(
            range(species_member_count),
            key=lambda index: member_fitnesses_in_species[index],
            reverse=True,
        )
        surviving_member_indices = member_indices_sorted_by_fitness_descending[
            :surviving_member_count
        ]
        return (
            [member_genomes_in_species[index] for index in surviving_member_indices],
            [member_fitnesses_in_species[index] for index in surviving_member_indices],
        )

    def _select_single_parent_with_fitness(
        self,
        candidate_genomes: list[NEATGenome],
        candidate_fitnesses: list[FitnessValue],
        rng: Generator,
    ) -> tuple[NEATGenome, FitnessValue]:
        """Select one parent and return it together with its fitness.

        Goes through ``select_parent_indices`` so the fitness comes from the
        genome that was actually drawn. Looking the winner up by value would
        return the first *equal* genome's fitness instead — genomes are frozen
        dataclasses, and duplicates are routine (elites are copied by reference,
        mutations can be no-ops) — and that fitness decides which parent
        crossover treats as the fitter one.
        """
        selected_parent_index = self.parent_selection.select_parent_indices(
            candidate_genomes=candidate_genomes,
            candidate_fitnesses=candidate_fitnesses,
            number_of_parents_to_select=1,
            rng=rng,
        )[0]
        return candidate_genomes[selected_parent_index], candidate_fitnesses[selected_parent_index]

    def _produce_single_child_from_species(
        self,
        reproducing_member_genomes: list[NEATGenome],
        reproducing_member_fitnesses: list[FitnessValue],
        all_genomes_in_population: list[NEATGenome],
        all_fitnesses_in_population: list[FitnessValue],
        rng: Generator,
    ) -> NEATGenome:
        """Produce one child by crossover-then-mutation or mutation only.

        Matches the paper's reproduction mix (section 4.1): 25% of offspring
        come from mutation without crossover, and interspecies mating occurs
        at a small configured rate (0.001 in the paper), drawing the second
        parent from the whole population.
        """
        wants_crossover = rng.random() < self.config.probability_of_crossover_vs_mutation_only
        is_interspecies_mating = (
            wants_crossover
            and len(all_genomes_in_population) >= 2
            and rng.random() < self.config.probability_of_interspecies_mating
        )
        crossover_is_possible = is_interspecies_mating or len(reproducing_member_genomes) >= 2

        if wants_crossover and crossover_is_possible:
            first_parent_genome, first_parent_fitness = self._select_single_parent_with_fitness(
                candidate_genomes=reproducing_member_genomes,
                candidate_fitnesses=reproducing_member_fitnesses,
                rng=rng,
            )
            if is_interspecies_mating:
                # Interspecies mating (rate 0.001 in the paper): the second
                # parent is drawn from the entire population, ignoring species
                # boundaries.
                second_parent_genome, second_parent_fitness = (
                    self._select_single_parent_with_fitness(
                        candidate_genomes=all_genomes_in_population,
                        candidate_fitnesses=all_fitnesses_in_population,
                        rng=rng,
                    )
                )
            else:
                second_parent_genome, second_parent_fitness = (
                    self._select_single_parent_with_fitness(
                        candidate_genomes=reproducing_member_genomes,
                        candidate_fitnesses=reproducing_member_fitnesses,
                        rng=rng,
                    )
                )
            if first_parent_fitness >= second_parent_fitness:
                fitter_parent, less_fit_parent = first_parent_genome, second_parent_genome
            else:
                fitter_parent, less_fit_parent = second_parent_genome, first_parent_genome
            crossover_child_genome = self.crossover.apply_to_parents(
                fitter_parent=fitter_parent,
                less_fit_parent=less_fit_parent,
                rng=rng,
            )
        else:
            crossover_child_genome, _ = self._select_single_parent_with_fitness(
                candidate_genomes=reproducing_member_genomes,
                candidate_fitnesses=reproducing_member_fitnesses,
                rng=rng,
            )

        mutated_child_genome = self.mutation.apply_to_genome(
            genome=crossover_child_genome,
            rng=rng,
            innovation_tracker=self.innovation_tracker,
        )
        return mutated_child_genome
