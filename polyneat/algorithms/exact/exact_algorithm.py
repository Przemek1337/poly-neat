"""EXACT as a ``NEATAlgorithm`` subclass (Desell, 2017).

EXACT evolves CNN structure at the filter level. Its genetics (genome
encoding, the eight mutation operators of section III-B, the crossover of
section III-C, epigenetic weight inheritance of section III-D) replace NEAT's
via the ``_build_*`` factories; the paper's asynchronous master/worker
loop and BOINC distribution (section VI) are out of scope — reproduction is
driven by the inherited generational loop with a single species, since
EXACT has no speciation. Training (section VII) runs Lamarckian-style on
every untrained genome before its evaluation, the same hook style as
L-NEAT. ``tournament_size_for_parent_selection = 1`` (config default)
reproduces the paper's uniform-random genome selection.

References:
    Desell, T. (2017). Developing a Volunteer Computing Project to Evolve
        Convolutional Neural Networks and Their Hyperparameters. 2017 IEEE
        13th International Conference on e-Science, pp. 19-28.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import cast

import torch
from numpy.random import Generator

from polyneat.algorithms.exact.exact_backpropagation_trainer import (
    EXACTBackpropagationTrainer,
)
from polyneat.algorithms.exact.exact_crossover import EXACTCrossover
from polyneat.algorithms.exact.exact_genome import EXACTGenome
from polyneat.algorithms.exact.exact_initial_population import (
    build_minimal_cnn_initial_population,  # noqa: F401  (import registers the strategy)
)
from polyneat.algorithms.exact.exact_innovation_tracker import EXACTInnovationTracker
from polyneat.algorithms.exact.exact_phenotype_decoder import EXACTPhenotypeDecoder
from polyneat.algorithms.exact.mutations.add_convolution_edge_mutation import (
    AddConvolutionEdgeMutation,
)
from polyneat.algorithms.exact.mutations.add_convolution_node_mutation import (
    AddConvolutionNodeMutation,
)
from polyneat.algorithms.exact.mutations.change_filter_size_mutation import (
    ChangeFilterSizeMutation,
)
from polyneat.algorithms.exact.mutations.disable_convolution_edge_mutation import (
    DisableConvolutionEdgeMutation,
)
from polyneat.algorithms.exact.mutations.enable_convolution_edge_mutation import (
    EnableConvolutionEdgeMutation,
)
from polyneat.algorithms.exact.mutations.exact_composite_mutation import (
    EXACTCompositeMutation,
)
from polyneat.algorithms.exact.mutations.split_convolution_edge_mutation import (
    SplitConvolutionEdgeMutation,
)
from polyneat.algorithms.exact.simplex_hyperparameter_optimizer import (
    SimplexHyperparameterOptimizer,
)
from polyneat.algorithms.exact.single_species_speciator import SingleSpeciesSpeciator
from polyneat.configs.exact.exact_config import EXACTConfig
from polyneat.configs.neat.neat_config import NEATConfig
from polyneat.core.component_protocols import (
    CrossoverOperator,
    Genome,
    MutationOperator,
    PhenotypeDecoder,
    Speciator,
)
from polyneat.core.generation_statistics import GenerationStatistics
from polyneat.core.neat.global_innovation_tracker import GlobalInnovationTracker
from polyneat.core.neat.neat_algorithm import NEATAlgorithm
from polyneat.core.neat.neat_genome import NEATGenome
from polyneat.core.population import Population
from polyneat.core.type_aliases import FitnessValue
from polyneat.logging_utils.custom_logger import get_logger

logger = get_logger(__name__)


@dataclass
class EXACTAlgorithm(NEATAlgorithm):
    """EXACT: evolve CNN filter topologies, train each genome before evaluation.

    Build it with the inherited ``EXACTAlgorithm.from_config(exact_config)``
    and attach ``backpropagation_trainer`` afterwards — it depends on the
    dataset, which the config does not carry. Without a trainer the
    algorithm still evolves structure, but phenotypes keep their He-random
    kernels (useful only for tests).

    Attributes:
        config: Validated EXACT hyperparameters.
        backpropagation_trainer: Trainer applied to every untrained genome;
            ``None`` disables training.
        simplex_optimizer: Generates each offspring's training
            hyperparameters (section IV); ``None`` leaves every genome on the
            config's fixed vector.
    """

    config: EXACTConfig
    backpropagation_trainer: EXACTBackpropagationTrainer | None = None
    simplex_optimizer: SimplexHyperparameterOptimizer | None = None

    @classmethod
    def from_config(
        cls,
        config: NEATConfig,
        device_for_phenotype_computation: torch.device | None = None,
    ) -> EXACTAlgorithm:
        """Build EXACT, attaching the SHO component when the config enables it.

        Args:
            config: Validated EXACT hyperparameters.
            device_for_phenotype_computation: Device override for the
                phenotype decoder.

        Returns:
            The wired algorithm.
        """
        algorithm = cast(
            "EXACTAlgorithm",
            super().from_config(config, device_for_phenotype_computation),
        )
        exact_config = cast("EXACTConfig", config)
        if exact_config.use_simplex_hyperparameter_optimization:
            algorithm.simplex_optimizer = SimplexHyperparameterOptimizer.from_config(
                exact_config
            )
            logger.debug("EXACT: simplex hyperparameter optimization enabled")
        return algorithm

    @classmethod
    def _build_innovation_tracker(cls, config: NEATConfig) -> GlobalInnovationTracker:
        """EXACT keeps a persistent master innovation list (section III-B)."""
        return EXACTInnovationTracker()

    @classmethod
    def _build_mutation(cls, config: NEATConfig) -> MutationOperator[NEATGenome]:
        """Compose the eight EXACT operators behind one categorical draw (section III-B)."""
        exact_config = cast("EXACTConfig", config)
        composite_mutation = EXACTCompositeMutation(
            mutation_operators=[
                DisableConvolutionEdgeMutation(),
                EnableConvolutionEdgeMutation(),
                SplitConvolutionEdgeMutation(),
                AddConvolutionEdgeMutation(),
                AddConvolutionNodeMutation(
                    minimum_hidden_filter_size=exact_config.minimum_hidden_filter_size
                ),
                ChangeFilterSizeMutation(
                    change_height=True,
                    change_width=True,
                    filter_size_change_options=exact_config.filter_size_change_options,
                    minimum_filter_size=exact_config.minimum_hidden_filter_size,
                ),
                ChangeFilterSizeMutation(
                    change_height=False,
                    change_width=True,
                    filter_size_change_options=exact_config.filter_size_change_options,
                    minimum_filter_size=exact_config.minimum_hidden_filter_size,
                ),
                ChangeFilterSizeMutation(
                    change_height=True,
                    change_width=False,
                    filter_size_change_options=exact_config.filter_size_change_options,
                    minimum_filter_size=exact_config.minimum_hidden_filter_size,
                ),
            ],
            operator_selection_probabilities=[
                exact_config.probability_of_disable_edge_mutation,
                exact_config.probability_of_enable_edge_mutation,
                exact_config.probability_of_split_edge_mutation,
                exact_config.probability_of_add_edge_mutation,
                exact_config.probability_of_add_node_mutation,
                exact_config.probability_of_change_filter_size_mutation,
                exact_config.probability_of_change_filter_size_x_mutation,
                exact_config.probability_of_change_filter_size_y_mutation,
            ],
            number_of_mutations_per_genome=exact_config.number_of_mutations_per_genome,
            maximum_attempts_for_reachable_child=(
                exact_config.maximum_mutation_attempts_for_reachable_child
            ),
        )
        return cast("MutationOperator[NEATGenome]", composite_mutation)

    @classmethod
    def _build_crossover(cls, config: NEATConfig) -> CrossoverOperator[NEATGenome]:
        """Build the fitness-asymmetric EXACT crossover (section III-C)."""
        exact_config = cast("EXACTConfig", config)
        exact_crossover = EXACTCrossover(
            more_fit_parent_edge_inclusion_rate=(
                exact_config.more_fit_parent_edge_inclusion_rate
            ),
            less_fit_parent_edge_inclusion_rate=(
                exact_config.less_fit_parent_edge_inclusion_rate
            ),
            maximum_attempts_for_reachable_child=(
                exact_config.maximum_mutation_attempts_for_reachable_child
            ),
        )
        return cast("CrossoverOperator[NEATGenome]", exact_crossover)

    @classmethod
    def _build_speciator(cls, config: NEATConfig) -> Speciator[NEATGenome]:
        """EXACT has no speciation: everything is species 0."""
        return cast("Speciator[NEATGenome]", SingleSpeciesSpeciator())

    @classmethod
    def _build_phenotype_decoder(
        cls, config: NEATConfig, device: torch.device
    ) -> PhenotypeDecoder[NEATGenome]:
        """Build the convolutional decoder bound to the config's image geometry."""
        exact_config = cast("EXACTConfig", config)
        exact_decoder = EXACTPhenotypeDecoder(
            input_image_height=exact_config.input_image_height,
            input_image_width=exact_config.input_image_width,
            leaky_relu_negative_slope=exact_config.leaky_relu_negative_slope,
            activation_clamp_maximum=exact_config.activation_clamp_maximum,
            device_for_computation=device,
            use_batch_normalization=exact_config.use_batch_normalization,
        )
        return cast("PhenotypeDecoder[NEATGenome]", exact_decoder)

    def create_initial_population(self, rng: Generator) -> Population:
        """Build generation 0 and train it when a trainer is attached.

        The minimal-CNN strategy produces identical genomes, so training is
        cached by object identity: each distinct genome instance is trained
        once and the trained result shared across its copies. With SHO on,
        every genome first receives its own random-in-range hyperparameter
        vector (section IV), which makes each one a distinct object — the
        identity cache then trains them separately, as intended, because
        they no longer train alike.

        Args:
            rng: Source of randomness for the configured strategy.

        Returns:
            Generation-0 population, trained when a trainer is attached.
        """
        population = super().create_initial_population(rng)
        if self.simplex_optimizer is not None:
            population = Population(
                genomes=[
                    replace(
                        cast("EXACTGenome", genome),
                        training_hyperparameters=(
                            self.simplex_optimizer.draw_initial_hyperparameters(rng)
                        ),
                    )
                    for genome in population.genomes
                ],
                species_assignments=population.species_assignments,
                generation_number=population.generation_number,
            )
        if self.backpropagation_trainer is None:
            return population
        trained_genome_by_source_id: dict[int, Genome] = {}
        trained_genomes: list[Genome] = []
        for genome in cast("list[EXACTGenome]", population.genomes):
            trained_genome = trained_genome_by_source_id.get(id(genome))
            if trained_genome is None:
                trained_genome = self.backpropagation_trainer.train_genome(genome)
                trained_genome_by_source_id[id(genome)] = trained_genome
            trained_genomes.append(trained_genome)
        return Population(
            genomes=trained_genomes,
            species_assignments=population.species_assignments,
            generation_number=population.generation_number,
        )

    def advance_one_generation(
        self,
        current_population: Population,
        fitnesses_of_current_population: list[FitnessValue],
        rng: Generator,
    ) -> tuple[Population, GenerationStatistics]:
        """Run one reproduction cycle, then train every untrained offspring.

        Mirrors the paper's worker step (a genome is trained before its
        fitness is reported, section III) inside the generational loop: with
        SHO on, every untrained offspring first receives a simplex-generated
        hyperparameter vector (section IV); after
        the inherited reproduction, each offspring with ``is_trained=False``
        gets a full backpropagation session and its trained kernels are
        carried in the next population (epigenetic inheritance, section
        2.6). Elites arrive with ``is_trained=True`` and are skipped.

        Args:
            current_population: The population to reproduce from.
            fitnesses_of_current_population: Raw fitness per genome, aligned
                with ``current_population.genomes``.
            rng: Source of randomness for the whole cycle.

        Returns:
            The next-generation population (trained when a trainer is
            attached) and the evaluated generation's statistics, with
            ``extra_metrics["number_of_genomes_trained"]`` set whenever a
            trainer ran.
        """
        next_population, generation_statistics = super().advance_one_generation(
            current_population=current_population,
            fitnesses_of_current_population=fitnesses_of_current_population,
            rng=rng,
        )
        if self.simplex_optimizer is not None:
            next_population = self._assign_simplex_hyperparameters_to_offspring(
                current_population=current_population,
                fitnesses_of_current_population=fitnesses_of_current_population,
                next_population=next_population,
                rng=rng,
            )
        if self.backpropagation_trainer is None:
            return next_population, generation_statistics

        genomes_after_training: list[Genome] = []
        number_of_genomes_trained = 0
        for genome in cast("list[EXACTGenome]", next_population.genomes):
            genome_after_training = self.backpropagation_trainer.train_genome(genome)
            if genome_after_training is not genome:
                number_of_genomes_trained += 1
            genomes_after_training.append(genome_after_training)

        generation_statistics.extra_metrics["number_of_genomes_trained"] = float(
            number_of_genomes_trained
        )
        population_after_training = Population(
            genomes=genomes_after_training,
            species_assignments=next_population.species_assignments,
            generation_number=next_population.generation_number,
        )
        return population_after_training, generation_statistics

    def _assign_simplex_hyperparameters_to_offspring(
        self,
        current_population: Population,
        fitnesses_of_current_population: list[FitnessValue],
        next_population: Population,
        rng: Generator,
    ) -> Population:
        """Give every untrained offspring a simplex-generated vector (section IV).

        ``number_of_selected_genomes`` distinct parents are drawn per
        offspring from the evaluated generation; their hyperparameters and
        fitnesses drive eqs. 1-2. Elites arrive trained and keep the vector
        they were trained with.

        Args:
            current_population: The evaluated generation SHO selects from.
            fitnesses_of_current_population: Raw fitness per evaluated genome.
            next_population: The offspring produced by reproduction.
            rng: Source of randomness for the selection and the ``r`` draw.

        Returns:
            ``next_population`` with hyperparameters attached, or unchanged
            when fewer than two evaluated genomes carry a vector.
        """
        assert self.simplex_optimizer is not None
        candidate_pool = [
            (genome.training_hyperparameters, fitness)
            for genome, fitness in zip(
                cast("list[EXACTGenome]", current_population.genomes),
                fitnesses_of_current_population,
                strict=True,
            )
            if genome.training_hyperparameters is not None
        ]
        if len(candidate_pool) < 2:
            return next_population

        genomes_with_hyperparameters: list[Genome] = []
        for genome in cast("list[EXACTGenome]", next_population.genomes):
            if genome.is_trained:
                genomes_with_hyperparameters.append(genome)
                continue
            number_to_select = min(
                self.simplex_optimizer.number_of_selected_genomes, len(candidate_pool)
            )
            selected_indices = rng.choice(
                len(candidate_pool), size=number_to_select, replace=False
            )
            selected_candidates = [
                candidate_pool[int(candidate_index)]
                for candidate_index in selected_indices
            ]
            genomes_with_hyperparameters.append(
                replace(
                    genome,
                    training_hyperparameters=(
                        self.simplex_optimizer.generate_offspring_hyperparameters(
                            selected_candidates, rng
                        )
                    ),
                )
            )
        return Population(
            genomes=genomes_with_hyperparameters,
            species_assignments=next_population.species_assignments,
            generation_number=next_population.generation_number,
        )
