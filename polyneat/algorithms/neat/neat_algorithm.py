from __future__ import annotations

import time
from dataclasses import dataclass, field
from statistics import mean, median

import torch
from numpy.random import Generator

from polyneat.algorithms.neat.compatibility_distance_speciator import (
    CompatibilityDistanceSpeciator,
    SpeciesRepresentative,
)
from polyneat.algorithms.neat.global_innovation_tracker import GlobalInnovationTracker
from polyneat.algorithms.neat.mutations.add_connection_mutation import AddConnectionMutation
from polyneat.algorithms.neat.mutations.add_node_mutation import AddNodeMutation
from polyneat.algorithms.neat.mutations.composite_neat_mutation import CompositeNEATMutation
from polyneat.algorithms.neat.mutations.toggle_connection_enabled_mutation import (
    ToggleConnectionEnabledMutation,
)
from polyneat.algorithms.neat.mutations.weight_modification_mutation import (
    WeightModificationMutation,
)
from polyneat.algorithms.neat.neat_crossover import NEATCrossover
from polyneat.algorithms.neat.neat_genome import ConnectionGene, NEATGenome, NodeGene
from polyneat.algorithms.neat.neat_phenotype_builder import NEATPhenotypeBuilder
from polyneat.algorithms.neat.tournament_parent_selection import TournamentParentSelection
from polyneat.config.neat_config import NEATConfig
from polyneat.core.component_protocols import (
    CrossoverOperator,
    MutationOperator,
    ParentSelection,
    PhenotypeBuilder,
    Speciator,
)
from polyneat.core.generation_statistics import GenerationStatistics
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
    """Vanilla NEAT (Stanley & Miikkulainen, 2002) as a composition of protocol-conforming components."""

    config: NEATConfig
    mutation: MutationOperator
    crossover: CrossoverOperator
    parent_selection: ParentSelection
    speciator: Speciator
    innovation_tracker: GlobalInnovationTracker
    _phenotype_builder: NEATPhenotypeBuilder
    _species_stagnation_bookkeeping: dict[SpeciesId, _SpeciesReproductionState] = field(
        default_factory=dict
    )

    @property
    def phenotype_builder(self) -> PhenotypeBuilder:
        return self._phenotype_builder

    @classmethod
    def from_config(
        cls,
        config: NEATConfig,
        device_for_phenotype_computation: torch.device | None = None,
    ) -> "NEATAlgorithm":
        resolved_device = device_for_phenotype_computation or torch.device(
            config.device_for_phenotype_evaluation
        )

        composite_mutation = CompositeNEATMutation(
            ordered_individual_mutations=[
                WeightModificationMutation(
                    probability_of_perturbation=config.probability_of_weight_perturbation,
                    probability_of_replacement=config.probability_of_weight_replacement,
                    perturbation_strength_sigma=config.weight_perturbation_strength_sigma,
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
        neat_crossover = NEATCrossover(
            probability_of_inheriting_from_fitter_parent_for_matching_genes=(
                config.probability_of_inheriting_from_fitter_parent_for_matching_genes
            ),
        )
        tournament_selection = TournamentParentSelection(
            tournament_size=config.tournament_size_for_parent_selection
        )
        speciator = CompatibilityDistanceSpeciator(
            coefficient_excess_c1=config.compatibility_distance_coefficient_excess_c1,
            coefficient_disjoint_c2=config.compatibility_distance_coefficient_disjoint_c2,
            coefficient_weight_difference_c3=(
                config.compatibility_distance_coefficient_weight_difference_c3
            ),
            compatibility_threshold=config.compatibility_distance_threshold,
        )
        innovation_tracker = GlobalInnovationTracker()
        phenotype_builder = NEATPhenotypeBuilder(
            device_for_computation=resolved_device,
        )

        return cls(
            config=config,
            mutation=composite_mutation,
            crossover=neat_crossover,
            parent_selection=tournament_selection,
            speciator=speciator,
            innovation_tracker=innovation_tracker,
            _phenotype_builder=phenotype_builder,
        )

    def create_initial_population(self, rng: Generator) -> Population:
        input_node_id_range = range(self.config.number_of_input_nodes)
        bias_node_id = self.config.number_of_input_nodes
        output_node_id_range = range(
            self.config.number_of_input_nodes + 1,
            self.config.number_of_input_nodes + 1 + self.config.number_of_output_nodes,
        )

        input_node_genes = tuple(
            NodeGene(
                node_id=node_id,
                node_type="input",
                activation_function_name="identity",
            )
            for node_id in input_node_id_range
        )
        bias_node_gene = NodeGene(
            node_id=bias_node_id,
            node_type="bias",
            activation_function_name="identity",
        )
        output_node_genes = tuple(
            NodeGene(
                node_id=node_id,
                node_type="output",
                activation_function_name=(
                    self.config.default_activation_function_for_output_nodes
                ),
            )
            for node_id in output_node_id_range
        )

        template_node_genes: tuple[NodeGene, ...] = (
            input_node_genes + (bias_node_gene,) + output_node_genes
        )

        initial_genomes: list[NEATGenome] = []
        for _individual_index in range(self.config.population_size):
            initial_connection_genes: list[ConnectionGene] = []
            for input_or_bias_node_id in [*input_node_id_range, bias_node_id]:
                for output_node_id in output_node_id_range:
                    innovation_id = (
                        self.innovation_tracker.get_or_assign_innovation_id_for_connection(
                            source_node_id=input_or_bias_node_id,
                            target_node_id=output_node_id,
                        )
                    )
                    initial_weight = float(
                        rng.uniform(
                            self.config.initial_weight_range_min,
                            self.config.initial_weight_range_max,
                        )
                    )
                    initial_connection_genes.append(
                        ConnectionGene(
                            innovation_id=innovation_id,
                            source_node_id=input_or_bias_node_id,
                            target_node_id=output_node_id,
                            weight=initial_weight,
                            is_enabled=True,
                        )
                    )
            initial_genomes.append(
                NEATGenome(
                    node_genes=template_node_genes,
                    connection_genes=tuple(initial_connection_genes),
                )
            )

        self.innovation_tracker.reset_for_new_generation()

        return Population(
            genomes=initial_genomes,
            species_assignments=None,
            generation_number=0,
        )

    def advance_one_generation(
        self,
        current_population: Population,
        fitnesses_of_current_population: list[FitnessValue],
        rng: Generator,
    ) -> tuple[Population, GenerationStatistics]:
        generation_start_wall_time = time.perf_counter()

        neat_genomes_in_current_population: list[NEATGenome] = [
            genome  # type: ignore[misc]
            for genome in current_population.genomes
        ]

        species_id_per_genome_index = self.speciator.assign_genomes_to_species(
            neat_genomes_in_current_population  # type: ignore[arg-type]
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
        for species_state in non_stagnated_species_states:
            member_genomes_in_species = [
                neat_genomes_in_current_population[member_index]
                for member_index in species_state.member_indices_in_current_population
            ]
            member_fitnesses_in_species = species_state.raw_fitnesses

            elite_genomes_from_species = self._pick_elite_genomes_from_species(
                member_genomes_in_species=member_genomes_in_species,
                member_fitnesses_in_species=member_fitnesses_in_species,
            )
            offspring_genomes.extend(elite_genomes_from_species)

            remaining_offspring_slots_to_fill_for_species = (
                species_state.offspring_slot_count - len(elite_genomes_from_species)
            )
            for _slot_index in range(max(0, remaining_offspring_slots_to_fill_for_species)):
                child_genome = self._produce_single_child_from_species(
                    member_genomes_in_species=member_genomes_in_species,
                    member_fitnesses_in_species=member_fitnesses_in_species,
                    rng=rng,
                )
                offspring_genomes.append(child_genome)

        while len(offspring_genomes) < self.config.population_size:
            offspring_genomes.append(neat_genomes_in_current_population[0])
        offspring_genomes = offspring_genomes[: self.config.population_size]

        next_generation_population = Population(
            genomes=offspring_genomes,  # type: ignore[arg-type]
            species_assignments=None,
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
        current_generation_species_ids: set[SpeciesId] = set()
        for species_state in per_species_reproduction_states:
            current_generation_species_ids.add(species_state.species_id)
            existing_bookkeeping = self._species_stagnation_bookkeeping.get(species_state.species_id)
            if existing_bookkeeping is None:
                self._species_stagnation_bookkeeping[species_state.species_id] = species_state
                species_state.best_raw_fitness_ever = species_state.best_raw_fitness
                species_state.generations_since_last_improvement = 0
                continue

            if species_state.best_raw_fitness > existing_bookkeeping.best_raw_fitness_ever:
                species_state.best_raw_fitness_ever = species_state.best_raw_fitness
                species_state.generations_since_last_improvement = 0
            else:
                species_state.best_raw_fitness_ever = (
                    existing_bookkeeping.best_raw_fitness_ever
                )
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
        sum_of_mean_adjusted_fitnesses = sum(
            max(0.0, species_state.mean_adjusted_fitness)
            for species_state in non_stagnated_species_states
        )
        if sum_of_mean_adjusted_fitnesses <= 0.0:
            equal_share = total_offspring_slots_available // len(non_stagnated_species_states)
            for species_state in non_stagnated_species_states:
                species_state.offspring_slot_count = equal_share
            return

        for species_state in non_stagnated_species_states:
            proportional_share = (
                max(0.0, species_state.mean_adjusted_fitness) / sum_of_mean_adjusted_fitnesses
            )
            species_state.offspring_slot_count = int(
                round(proportional_share * total_offspring_slots_available)
            )

    def _pick_elite_genomes_from_species(
        self,
        member_genomes_in_species: list[NEATGenome],
        member_fitnesses_in_species: list[FitnessValue],
    ) -> list[NEATGenome]:
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

    def _produce_single_child_from_species(
        self,
        member_genomes_in_species: list[NEATGenome],
        member_fitnesses_in_species: list[FitnessValue],
        rng: Generator,
    ) -> NEATGenome:
        should_use_crossover = (
            len(member_genomes_in_species) >= 2
            and rng.random() < self.config.probability_of_crossover_vs_mutation_only
        )
        if should_use_crossover:
            selected_parents = self.parent_selection.select_parents(
                candidate_genomes=member_genomes_in_species,  # type: ignore[arg-type]
                candidate_fitnesses=member_fitnesses_in_species,
                number_of_parents_to_select=2,
                rng=rng,
            )
            first_parent_genome, second_parent_genome = selected_parents
            first_parent_fitness = member_fitnesses_in_species[
                member_genomes_in_species.index(first_parent_genome)  # type: ignore[arg-type]
            ]
            second_parent_fitness = member_fitnesses_in_species[
                member_genomes_in_species.index(second_parent_genome)  # type: ignore[arg-type]
            ]
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
            selected_parents = self.parent_selection.select_parents(
                candidate_genomes=member_genomes_in_species,  # type: ignore[arg-type]
                candidate_fitnesses=member_fitnesses_in_species,
                number_of_parents_to_select=1,
                rng=rng,
            )
            crossover_child_genome = selected_parents[0]

        mutated_child_genome = self.mutation.apply_to_genome(
            genome=crossover_child_genome,
            rng=rng,
            innovation_tracker=self.innovation_tracker,
        )
        return mutated_child_genome  # type: ignore[return-value]
