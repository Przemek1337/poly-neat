from __future__ import annotations

from polyneat.core.component_protocols import Genome
from polyneat.core.type_aliases import FitnessValue


class ClassGenomeContainer:
    """Best-genome-per-class store used by C-NEAT.

    The container has one cell per class label. Organisms are assigned to
    cells by their index in the population modulo the container size
    (equation 1 of the paper), and each cell keeps the best genome offered for
    its class so far. Cells only ever improve, so per-class learning progress
    is preserved across generations even when the population drifts.

    References:
        Alfaham, A., Van Raemdonck, S., & Mercelis, S. (2024). Genetic
        NEAT-Based Method for Multi-class Classification. *ACAI 2024*,
        section IV.
    """

    def __init__(self, number_of_class_labels: int) -> None:
        """Create an empty container with one cell per class label.

        Args:
            number_of_class_labels: Number of classes; also the container size.

        Raises:
            ValueError: If ``number_of_class_labels`` is below 2.
        """
        if number_of_class_labels < 2:
            raise ValueError(
                f"number_of_class_labels must be >= 2, got {number_of_class_labels}"
            )
        self._number_of_class_labels = number_of_class_labels
        self._best_genome_per_class: list[Genome | None] = [None] * number_of_class_labels
        self._best_fitness_per_class: list[FitnessValue | None] = (
            [None] * number_of_class_labels
        )

    @property
    def number_of_class_labels(self) -> int:
        return self._number_of_class_labels

    def assigned_class_for_organism_index(self, organism_index: int) -> int:
        """Return the class label the organism at ``organism_index`` is assigned to."""
        return organism_index % self._number_of_class_labels

    def offer_genome_for_class(
        self,
        class_label_index: int,
        genome: Genome,
        fitness: FitnessValue,
    ) -> bool:
        """Store a clone of ``genome`` if it strictly improves the cell's fitness.

        Args:
            class_label_index: Cell to offer the genome to.
            genome: Candidate genome; cloned before storing.
            fitness: The genome's per-class fitness.

        Returns:
            ``True`` when the genome was stored, ``False`` when the cell
            already holds a genome with equal or better fitness.

        Raises:
            ValueError: If ``class_label_index`` is out of range.
        """
        self._validate_class_label_index(class_label_index)
        current_best_fitness = self._best_fitness_per_class[class_label_index]
        if current_best_fitness is not None and fitness <= current_best_fitness:
            return False
        self._best_genome_per_class[class_label_index] = genome.clone_genome()
        self._best_fitness_per_class[class_label_index] = fitness
        return True

    def best_genome_for_class(self, class_label_index: int) -> Genome | None:
        """Return the best genome stored for ``class_label_index``, or ``None``."""
        self._validate_class_label_index(class_label_index)
        return self._best_genome_per_class[class_label_index]

    def best_fitness_for_class(self, class_label_index: int) -> FitnessValue | None:
        """Return the fitness of the stored genome for ``class_label_index``, or ``None``."""
        self._validate_class_label_index(class_label_index)
        return self._best_fitness_per_class[class_label_index]

    def is_fully_populated(self) -> bool:
        """Return whether every class label has at least one stored genome."""
        return all(genome is not None for genome in self._best_genome_per_class)

    def to_serializable_dict(self) -> dict:
        """Return a JSON-serializable representation of the container."""
        cells: list[dict | None] = []
        for genome, fitness in zip(
            self._best_genome_per_class, self._best_fitness_per_class, strict=True
        ):
            if genome is None:
                cells.append(None)
            else:
                cells.append({"fitness": fitness, "genome": genome.to_serializable_dict()})
        return {
            "number_of_class_labels": self._number_of_class_labels,
            "cells": cells,
        }

    @classmethod
    def from_serializable_dict(
        cls, payload: dict, genome_class: type
    ) -> ClassGenomeContainer:
        """Rebuild a container from :meth:`to_serializable_dict` output.

        Args:
            payload: Dictionary produced by :meth:`to_serializable_dict`.
            genome_class: Concrete genome class providing
                ``from_serializable_dict`` (for example
                :class:`~polyneat.core.neat.neat_genome.NEATGenome`).

        Returns:
            A container with the serialized cells restored.
        """
        container = cls(number_of_class_labels=payload["number_of_class_labels"])
        for class_label_index, cell in enumerate(payload["cells"]):
            if cell is None:
                continue
            container._best_genome_per_class[class_label_index] = (
                genome_class.from_serializable_dict(cell["genome"])
            )
            container._best_fitness_per_class[class_label_index] = cell["fitness"]
        return container

    def _validate_class_label_index(self, class_label_index: int) -> None:
        if not (0 <= class_label_index < self._number_of_class_labels):
            raise ValueError(
                f"class_label_index must be in [0, {self._number_of_class_labels}), "
                f"got {class_label_index}"
            )
