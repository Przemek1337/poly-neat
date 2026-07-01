from __future__ import annotations

import logging

from joblib import Parallel, delayed

from polyneat.core.protocols import FitnessEvaluator, Phenotype
from polyneat.core.type_aliases import FitnessValue

logger = logging.getLogger(__name__)


class ParallelFitnessEvaluatorWrapper:
    """Wraps any FitnessEvaluator and evaluates chunks in parallel via joblib."""

    def __init__(
        self,
        wrapped_evaluator: FitnessEvaluator,
        number_of_parallel_workers: int = -1,
    ) -> None:
        self._wrapped_evaluator = wrapped_evaluator
        self._number_of_parallel_workers = number_of_parallel_workers

    def evaluate_batch_of_phenotypes(
        self, phenotypes: list[Phenotype]
    ) -> list[FitnessValue]:
        if not phenotypes:
            return []

        logger.debug(
            "Evaluating %d phenotypes with %d workers",
            len(phenotypes),
            self._number_of_parallel_workers,
        )

        fitnesses: list[FitnessValue] = Parallel(
            n_jobs=self._number_of_parallel_workers, prefer="threads"
        )(
            delayed(self._wrapped_evaluator.evaluate_batch_of_phenotypes)([phenotype])
            for phenotype in phenotypes
        )

        # Each call returns a list of one; flatten
        return [fitness_list[0] for fitness_list in fitnesses]
