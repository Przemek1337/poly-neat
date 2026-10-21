from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from numpy.random import Generator

from polyneat.algorithms.lneat.backpropagation_weight_trainer import (
    BackpropagationWeightTrainer,
)
from polyneat.configs.lneat.lneat_config import LNEATConfig
from polyneat.core.component_protocols import Genome
from polyneat.core.generation_statistics import GenerationStatistics
from polyneat.core.neat.neat_algorithm import NEATAlgorithm
from polyneat.core.neat.neat_genome import NEATGenome
from polyneat.core.population import Population
from polyneat.core.type_aliases import FitnessValue


@dataclass
class LNEATAlgorithm(NEATAlgorithm):
    """L-NEAT (Chen & Alahakoon, 2006): NEAT hybridized with backpropagation.

    L-NEAT keeps NEAT's genetics untouched — this class inherits every
    operator, factory, and the generational loop from
    :class:`~polyneat.core.neat.neat_algorithm.NEATAlgorithm` — and adds the
    paper's "learning with backpropagation" (section IV.B) on top: every
    ``config.learning_interval_generations`` generations, offspring genomes
    that are not Type 1 networks undergo a backpropagation session and their
    trained weights are inherited by the next generation (Lamarckian).

    Constructed through the inherited ``from_config``;
    ``backpropagation_trainer`` is assigned after construction because it
    depends on the dataset, which the config does not carry. Without a trainer
    attached, the algorithm behaves exactly like NEAT.

    The paper's divide-and-conquer split (section IV.A) is part of the
    surrounding wiring rather than the genetics: one evolution runs per class
    label, scored by a
    :class:`~polyneat.evaluators.binary_recognizer_evaluator.BinaryRecognizerFitnessEvaluator`,
    and the best recognizer of each run is assembled into a
    :class:`~polyneat.algorithms.lneat.recognizer_ensemble_phenotype.RecognizerEnsemblePhenotype`,
    the argmax classifier over the per-class recognizer outputs.

    Attributes:
        config: Validated L-NEAT hyperparameters, carrying the learning
            schedule on top of every NEAT field.
        backpropagation_trainer: Trainer applied to the offspring on learning
            generations; ``None`` disables learning and reduces the algorithm
            to vanilla NEAT.

    References:
        Chen, L., & Alahakoon, D. (2006). NeuroEvolution of Augmenting Topologies with Learning
            for Data Classification. *ICIA 2006: 2nd International Conference on Information and
            Automation*, pp. 367-371.
    """

    config: LNEATConfig
    backpropagation_trainer: BackpropagationWeightTrainer | None = None

    def advance_one_generation(
        self,
        current_population: Population,
        fitnesses_of_current_population: list[FitnessValue],
        rng: Generator,
    ) -> tuple[Population, GenerationStatistics]:
        """Run one NEAT reproduction cycle, then a backpropagation session.

        The session runs only on generations that are a multiple of
        ``config.learning_interval_generations`` (the paper's interval ``I``)
        and only when a trainer is attached. It applies to the offspring,
        before their evaluation: the paper trains during the evolution stage
        so that networks with correct structure gain fitness and survive
        (section IV.B.4). Large weight changes may move a genome outside its
        parents' species, which the paper describes as exploring a new search
        area (section IV.B.3); the next generation's speciation pass
        reassigns every genome regardless.

        Species champions carried over by the inherited elitism are trained
        along with the rest of the offspring — the paper's selective learning
        exempts only Type 1 networks (section IV.B.2), not elites — so a
        champion's weights can change across a learning generation. The
        all-time best genome is unaffected: ``EvolutionRunner`` records it from
        the evaluated population before reproduction.

        The learning session is keyed on the *offspring's* generation number,
        while ``generation_statistics`` describes the generation that was just
        evaluated. ``extra_metrics["number_of_genomes_backpropagated"]``
        therefore appears one step earlier than the generation whose genomes
        were trained: with ``I = 5`` it is recorded at generations 4, 9, 14.

        Args:
            current_population: The population to reproduce from.
            fitnesses_of_current_population: Raw fitness per genome, aligned
                with ``current_population.genomes``.
            rng: Source of randomness for the whole cycle.

        Returns:
            The next-generation population (weights trained on learning
            generations) and the statistics of the evaluated generation, with
            ``extra_metrics["number_of_genomes_backpropagated"]`` set whenever
            a learning session ran.
        """
        next_population, generation_statistics = super().advance_one_generation(
            current_population=current_population,
            fitnesses_of_current_population=fitnesses_of_current_population,
            rng=rng,
        )
        is_learning_generation = (
            self.backpropagation_trainer is not None
            and next_population.generation_number % self.config.learning_interval_generations
            == 0
        )
        if not is_learning_generation:
            return next_population, generation_statistics

        genomes_after_learning: list[Genome] = []
        number_of_genomes_backpropagated = 0
        for genome in cast("list[NEATGenome]", next_population.genomes):
            genome_after_learning = self.backpropagation_trainer.train_genome_if_learning_needed(
                genome
            )
            if genome_after_learning is not genome:
                number_of_genomes_backpropagated += 1
            genomes_after_learning.append(genome_after_learning)

        generation_statistics.extra_metrics["number_of_genomes_backpropagated"] = float(
            number_of_genomes_backpropagated
        )
        population_after_learning = Population(
            genomes=genomes_after_learning,
            species_assignments=next_population.species_assignments,
            generation_number=next_population.generation_number,
        )
        return population_after_learning, generation_statistics
