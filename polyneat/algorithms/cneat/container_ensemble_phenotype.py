from __future__ import annotations

import torch

from polyneat.algorithms.cneat.class_genome_container import ClassGenomeContainer
from polyneat.core.component_protocols import Phenotype, PhenotypeDecoder


class ContainerEnsemblePhenotype:
    """Executable multi-class classifier assembled from a C-NEAT container.

    Each container cell holds a single-output recognizer network for one class
    label. ``forward_pass`` runs every network on the same inputs and stacks
    their outputs into a ``[batch, number_of_class_labels]`` score tensor; the
    predicted class is the argmax over that dimension (figure 3 of Alfaham
    et al., 2024).

    Conforms to the :class:`~polyneat.core.component_protocols.Phenotype`
    protocol, so the ensemble can be scored by any evaluator that expects a
    multi-output phenotype — e.g.
    :class:`~polyneat.evaluators.classification_accuracy_evaluator.ClassificationAccuracyEvaluator`.
    """

    def __init__(self, per_class_phenotypes: list[Phenotype]) -> None:
        """Assemble the ensemble from one phenotype per class label.

        Args:
            per_class_phenotypes: Single-output phenotypes, one per class,
                ordered by class label index.

        Raises:
            ValueError: If fewer than two phenotypes are given.
        """
        if len(per_class_phenotypes) < 2:
            raise ValueError(
                f"an ensemble needs at least 2 per-class networks, "
                f"got {len(per_class_phenotypes)}"
            )
        self._per_class_phenotypes = per_class_phenotypes

    @classmethod
    def from_container(
        cls,
        container: ClassGenomeContainer,
        phenotype_decoder: PhenotypeDecoder,
    ) -> ContainerEnsemblePhenotype:
        """Decode one phenotype per container cell and assemble the ensemble.

        Args:
            container: Fully populated best-genome-per-class container.
            phenotype_decoder: Decoder used to build each per-class network,
                typically ``algorithm.phenotype_decoder``.

        Returns:
            The assembled ensemble classifier.

        Raises:
            ValueError: If any container cell is still empty.
        """
        if not container.is_fully_populated():
            empty_class_labels = [
                class_label_index
                for class_label_index in range(container.number_of_class_labels)
                if container.best_genome_for_class(class_label_index) is None
            ]
            raise ValueError(
                f"container is missing genomes for class labels {empty_class_labels}; "
                f"run evolution longer so every class gets a genome"
            )
        per_class_phenotypes = [
            phenotype_decoder.build_phenotype_from_genome(
                container.best_genome_for_class(class_label_index)
            )
            for class_label_index in range(container.number_of_class_labels)
        ]
        return cls(per_class_phenotypes=per_class_phenotypes)

    def forward_pass(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Return per-class scores of shape ``[batch, number_of_class_labels]``."""
        per_class_score_columns = [
            phenotype.forward_pass(input_tensor)[:, 0]
            for phenotype in self._per_class_phenotypes
        ]
        return torch.stack(per_class_score_columns, dim=1)

    def reset_recurrent_state(self) -> None:
        """Reset the recurrent state of every member network."""
        for phenotype in self._per_class_phenotypes:
            phenotype.reset_recurrent_state()

    def predict_class_labels(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Return the argmax class label per sample, shape ``[batch]``."""
        with torch.no_grad():
            return torch.argmax(self.forward_pass(input_tensor), dim=1)
