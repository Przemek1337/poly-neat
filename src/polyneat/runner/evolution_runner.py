from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

from polyneat.core.population import Population
from polyneat.core.protocols import FitnessEvaluator, Genome, NeuroevolutionAlgorithm
from polyneat.core.statistics import GenerationStatistics
from polyneat.core.type_aliases import FitnessValue
from polyneat.runner.callbacks import BaseEvolutionCallback, EvolutionCallback
from polyneat.runner.context import RunContext
from polyneat.runner.termination import TerminationCriterion
from polyneat.utils.rng import create_rng

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvolutionResult:
    final_population: Population
    best_genome_ever_found: Genome
    best_fitness_ever_achieved: FitnessValue
    full_generation_history: list[GenerationStatistics]
    total_runtime_seconds: float
    termination_reason: Literal[
        "max_generations_reached",
        "target_fitness_reached",
        "stagnation_limit_reached",
        "manually_stopped",
    ]


class EvolutionRunner:
    """Convenience wrapper around algorithm.advance_one_generation().

    Users who need a custom loop should call advance_one_generation directly
    rather than wrapping this runner.
    """

    def __init__(
        self,
        algorithm: NeuroevolutionAlgorithm,
        fitness_evaluator: FitnessEvaluator,
        termination_criterion: TerminationCriterion,
        callbacks: list[EvolutionCallback] | None = None,
        random_seed: int | None = None,
    ) -> None:
        self._algorithm = algorithm
        self._fitness_evaluator = fitness_evaluator
        self._termination_criterion = termination_criterion
        self._callbacks: list[EvolutionCallback] = callbacks or []
        self._random_seed = random_seed

    def run_evolution(self) -> EvolutionResult:
        rng = create_rng(self._random_seed)
        run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        run_started_at = datetime.now()
        wall_start = time.perf_counter()

        context = RunContext(
            run_id=run_id,
            run_started_at=run_started_at,
            current_generation_number=0,
            total_generations_planned=-1,  # unknown until termination
            history_of_generation_statistics=[],
            current_best_genome=None,
            current_best_fitness=None,
            configuration_snapshot={},
        )

        for callback in self._callbacks:
            callback.on_run_started(context)

        population = self._algorithm.create_initial_population(rng)
        termination_reason: str = "manually_stopped"

        while True:
            context.current_generation_number = population.generation_number

            for callback in self._callbacks:
                callback.on_generation_started(context, population)

            # Build phenotypes
            phenotypes = [
                self._algorithm.phenotype_builder.build_phenotype_from_genome(genome)
                for genome in population.genomes
            ]

            # Evaluate
            fitnesses: list[FitnessValue] = (
                self._fitness_evaluator.evaluate_batch_of_phenotypes(phenotypes)
            )

            for callback in self._callbacks:
                callback.on_population_evaluated(context, population, fitnesses)

            # Track best
            generation_best_fitness = max(fitnesses)
            generation_best_genome = population.genomes[fitnesses.index(generation_best_fitness)]
            is_new_best = (
                context.current_best_fitness is None
                or generation_best_fitness > context.current_best_fitness
            )
            if is_new_best:
                context.current_best_fitness = generation_best_fitness
                context.current_best_genome = generation_best_genome
                for callback in self._callbacks:
                    callback.on_new_best_genome_found(
                        context, generation_best_genome, generation_best_fitness
                    )

            # Advance generation
            new_population, statistics = self._algorithm.advance_one_generation(
                current_population=population,
                fitnesses_of_current_population=fitnesses,
                rng=rng,
            )

            context.history_of_generation_statistics.append(statistics)

            for callback in self._callbacks:
                callback.on_generation_completed(context, new_population, statistics)

            # Termination check
            if self._termination_criterion.should_terminate_evolution(context):
                termination_reason = self._termination_criterion.termination_reason_label
                population = new_population
                break

            population = new_population

        total_runtime_seconds = time.perf_counter() - wall_start

        result = EvolutionResult(
            final_population=population,
            best_genome_ever_found=context.current_best_genome,  # type: ignore[arg-type]
            best_fitness_ever_achieved=context.current_best_fitness,  # type: ignore[arg-type]
            full_generation_history=context.history_of_generation_statistics,
            total_runtime_seconds=total_runtime_seconds,
            termination_reason=termination_reason,  # type: ignore[arg-type]
        )

        for callback in self._callbacks:
            callback.on_run_completed(context, result)

        logger.info(
            "Run %s completed in %.1fs. Termination: %s. Best fitness: %.6f",
            run_id,
            total_runtime_seconds,
            termination_reason,
            result.best_fitness_ever_achieved,
        )

        return result
