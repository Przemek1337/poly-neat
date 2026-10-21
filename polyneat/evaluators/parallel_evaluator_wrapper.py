from __future__ import annotations

from joblib import Parallel, delayed

from polyneat.core.component_protocols import FitnessEvaluator, Phenotype
from polyneat.core.type_aliases import FitnessValue
from polyneat.logging_utils.custom_logger import get_logger

logger = get_logger(__name__)


class ParallelFitnessEvaluatorWrapper:
    """Wraps any FitnessEvaluator and evaluates phenotypes in parallel via joblib."""

    def __init__(
        self,
        wrapped_evaluator: FitnessEvaluator,
        number_of_parallel_workers: int = -1,
    ) -> None:
        """Wrap ``wrapped_evaluator`` for thread-parallel batch evaluation.

        Args:
            wrapped_evaluator: Evaluator that scores individual phenotypes.
            number_of_parallel_workers: joblib ``n_jobs``; ``-1`` uses all cores.
        """
        self._wrapped_evaluator = wrapped_evaluator
        self._number_of_parallel_workers = number_of_parallel_workers

    def evaluate_batch_of_phenotypes(self, phenotypes: list[Phenotype]) -> list[FitnessValue]:
        """Evaluate all phenotypes in parallel threads, preserving order."""
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
