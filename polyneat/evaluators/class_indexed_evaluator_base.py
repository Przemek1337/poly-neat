from __future__ import annotations

from polyneat.core.component_protocols import Phenotype
from polyneat.core.type_aliases import FitnessValue


class ClassIndexedFitnessEvaluator:
    """Base class for C-NEAT evaluators that score organisms per class label.

    C-NEAT (Alfaham et al., 2024, section IV) assigns each organism the class
    label ``organism_index mod number_of_class_labels`` and scores it only on
    recognizing that class. The batch index received here equals the
    organism's population index because
    :class:`~polyneat.runner.evolution_runner.EvolutionRunner` builds
    phenotypes positionally from ``population.genomes``.

    Subclasses implement ``evaluate_phenotype_for_class``.
    """

    def __init__(self, number_of_class_labels: int) -> None:
        """Fix the class count used for the modulo assignment.

        Args:
            number_of_class_labels: Number of classes in the task.

        Raises:
            ValueError: If ``number_of_class_labels`` is below 2.
        """
        if number_of_class_labels < 2:
            raise ValueError(
                f"number_of_class_labels must be >= 2, got {number_of_class_labels}"
            )
        self._number_of_class_labels = number_of_class_labels

    @property
    def number_of_class_labels(self) -> int:
        return self._number_of_class_labels

    def evaluate_phenotype_for_class(
        self, phenotype: Phenotype, class_label_index: int
    ) -> FitnessValue:
        """Return the fitness of ``phenotype`` as a recognizer for one class label.

        Subclasses must override this method.
        """
        raise NotImplementedError

    def evaluate_batch_of_phenotypes(
        self, phenotypes: list[Phenotype]
    ) -> list[FitnessValue]:
        """Score each phenotype on the class label its batch index maps to."""
        return [
            self.evaluate_phenotype_for_class(
                phenotype, organism_index % self._number_of_class_labels
            )
            for organism_index, phenotype in enumerate(phenotypes)
        ]
