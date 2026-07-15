from __future__ import annotations

from polyneat.core.component_protocols import Phenotype
from polyneat.core.type_aliases import FitnessValue


class SequentialFitnessEvaluator:
    """Base class for evaluators that process phenotypes one at a time.

    Subclasses implement ``evaluate_single_phenotype``; this class handles
    batching so the subclass satisfies the ``FitnessEvaluator`` protocol.
    """

    def evaluate_single_phenotype(self, phenotype: Phenotype) -> FitnessValue:
        """Return the fitness of one phenotype. Must be overridden."""
        raise NotImplementedError

    def evaluate_batch_of_phenotypes(self, phenotypes: list[Phenotype]) -> list[FitnessValue]:
        """Evaluate ``phenotypes`` sequentially, preserving order."""
        return [self.evaluate_single_phenotype(phenotype) for phenotype in phenotypes]
