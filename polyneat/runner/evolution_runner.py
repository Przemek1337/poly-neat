from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, cast

from polyneat.core.component_protocols import (
    FitnessEvaluator,
    Genome,
    NeuroevolutionAlgorithm,
)
from polyneat.core.generation_statistics import GenerationStatistics
from polyneat.core.population import Population
from polyneat.core.type_aliases import FitnessValue
from polyneat.logging_utils.custom_logger import get_logger
from polyneat.runner.evolution_callback_protocol import EvolutionCallback
from polyneat.runner.run_context import RunContext
from polyneat.runner.termination_criteria import TerminationCriterion
from polyneat.utils.random_generator_factory import create_seeded_random_generator

logger = get_logger(__name__)

TerminationReason = Literal[
    "max_generations_reached",
    "target_fitness_reached",
    "stagnation_limit_reached",
    "manually_stopped",
]


@dataclass(frozen=True)
class EvolutionResult:
    final_population: Population
    best_genome_ever_found: Genome
    best_fitness_ever_achieved: FitnessValue
    full_generation_history: list[GenerationStatistics]
    total_runtime_seconds: float
    termination_reason: TerminationReason


class EvolutionRunner:
    """Convenience wrapper around ``algorithm.advance_one_generation``.

    Users who need a custom evolution loop should call
    ``advance_one_generation`` directly rather than wrapping this runner.
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
        random_generator = create_seeded_random_generator(self._random_seed)
        run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        run_started_at = datetime.now()
        wall_clock_start_time = time.perf_counter()

        run_context = RunContext(
            run_id=run_id,
            run_started_at=run_started_at,
            current_generation_number=0,
            total_generations_planned=-1,
            history_of_generation_statistics=[],
            current_best_genome=None,
            current_best_fitness=None,
            configuration_snapshot={},
        )

        for callback in self._callbacks:
            callback.on_run_started(run_context)

        current_population = self._algorithm.create_initial_population(random_generator)
        termination_reason_label: TerminationReason = "manually_stopped"

        while True:
            run_context.current_generation_number = current_population.generation_number

            for callback in self._callbacks:
                callback.on_generation_started(run_context, current_population)

            current_phenotypes = [
                self._algorithm.phenotype_builder.build_phenotype_from_genome(genome)
                for genome in current_population.genomes
            ]

            current_fitnesses: list[FitnessValue] = (
                self._fitness_evaluator.evaluate_batch_of_phenotypes(current_phenotypes)
            )

            for callback in self._callbacks:
                callback.on_population_evaluated(
                    run_context, current_population, current_fitnesses
                )

            generation_best_fitness = max(current_fitnesses)
            generation_best_genome = current_population.genomes[
                current_fitnesses.index(generation_best_fitness)
            ]
            is_new_all_time_best = (
                run_context.current_best_fitness is None
                or generation_best_fitness > run_context.current_best_fitness
            )
            if is_new_all_time_best:
                run_context.current_best_fitness = generation_best_fitness
                run_context.current_best_genome = generation_best_genome
                for callback in self._callbacks:
                    callback.on_new_best_genome_found(
                        run_context, generation_best_genome, generation_best_fitness
                    )

            new_population, generation_statistics = self._algorithm.advance_one_generation(
                current_population=current_population,
                fitnesses_of_current_population=current_fitnesses,
                rng=random_generator,
            )

            run_context.history_of_generation_statistics.append(generation_statistics)

            for callback in self._callbacks:
                callback.on_generation_completed(
                    run_context, new_population, generation_statistics
                )

            if self._termination_criterion.should_terminate_evolution(run_context):
                termination_reason_label = cast(
                    TerminationReason,
                    self._termination_criterion.termination_reason_label,
                )
                current_population = new_population
                break

            current_population = new_population

        total_runtime_seconds = time.perf_counter() - wall_clock_start_time

        assert run_context.current_best_genome is not None
        assert run_context.current_best_fitness is not None

        evolution_result = EvolutionResult(
            final_population=current_population,
            best_genome_ever_found=run_context.current_best_genome,
            best_fitness_ever_achieved=run_context.current_best_fitness,
            full_generation_history=run_context.history_of_generation_statistics,
            total_runtime_seconds=total_runtime_seconds,
            termination_reason=termination_reason_label,
        )

        for callback in self._callbacks:
            callback.on_run_completed(run_context, evolution_result)

        logger.info(
            "Run %s completed in %.1fs. Termination: %s. Best fitness: %.6f",
            run_id,
            total_runtime_seconds,
            termination_reason_label,
            evolution_result.best_fitness_ever_achieved,
        )

        return evolution_result
