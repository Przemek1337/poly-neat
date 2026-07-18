from __future__ import annotations

from dataclasses import dataclass

import pytest

from polyneat.algorithms.cneat.class_genome_container import ClassGenomeContainer


@dataclass(frozen=True)
class FakeGenome:
    label: str

    def clone_genome(self) -> FakeGenome:
        return FakeGenome(self.label)

    def to_serializable_dict(self) -> dict:
        return {"label": self.label}

    @classmethod
    def from_serializable_dict(cls, payload: dict) -> FakeGenome:
        return cls(payload["label"])


def test_rejects_fewer_than_two_class_labels() -> None:
    with pytest.raises(ValueError):
        ClassGenomeContainer(number_of_class_labels=1)


def test_class_assignment_wraps_by_modulo() -> None:
    container = ClassGenomeContainer(number_of_class_labels=3)
    assert [container.assigned_class_for_organism_index(i) for i in range(7)] == [
        0, 1, 2, 0, 1, 2, 0,
    ]


def test_starts_empty() -> None:
    container = ClassGenomeContainer(number_of_class_labels=2)
    assert container.best_genome_for_class(0) is None
    assert container.best_fitness_for_class(0) is None
    assert not container.is_fully_populated()


def test_offer_stores_into_empty_cell_and_reports_true() -> None:
    container = ClassGenomeContainer(number_of_class_labels=2)
    assert container.offer_genome_for_class(0, FakeGenome("a"), fitness=0.5) is True
    assert container.best_genome_for_class(0) == FakeGenome("a")
    assert container.best_fitness_for_class(0) == 0.5


def test_offer_replaces_only_on_strict_improvement() -> None:
    container = ClassGenomeContainer(number_of_class_labels=2)
    container.offer_genome_for_class(0, FakeGenome("a"), fitness=0.5)
    assert container.offer_genome_for_class(0, FakeGenome("worse"), fitness=0.4) is False
    assert container.offer_genome_for_class(0, FakeGenome("equal"), fitness=0.5) is False
    assert container.best_genome_for_class(0) == FakeGenome("a")
    assert container.offer_genome_for_class(0, FakeGenome("better"), fitness=0.6) is True
    assert container.best_genome_for_class(0) == FakeGenome("better")


def test_stored_genome_is_a_clone_not_the_caller_reference() -> None:
    container = ClassGenomeContainer(number_of_class_labels=2)
    offered_genome = FakeGenome("a")
    container.offer_genome_for_class(1, offered_genome, fitness=1.0)
    assert container.best_genome_for_class(1) is not offered_genome


def test_rejects_out_of_range_class_index() -> None:
    container = ClassGenomeContainer(number_of_class_labels=2)
    with pytest.raises(ValueError):
        container.offer_genome_for_class(2, FakeGenome("a"), fitness=1.0)
    with pytest.raises(ValueError):
        container.best_genome_for_class(-1)


def test_is_fully_populated_once_every_class_has_a_genome() -> None:
    container = ClassGenomeContainer(number_of_class_labels=2)
    container.offer_genome_for_class(0, FakeGenome("a"), fitness=0.1)
    assert not container.is_fully_populated()
    container.offer_genome_for_class(1, FakeGenome("b"), fitness=0.2)
    assert container.is_fully_populated()


def test_serialization_round_trip() -> None:
    container = ClassGenomeContainer(number_of_class_labels=3)
    container.offer_genome_for_class(0, FakeGenome("a"), fitness=0.25)
    container.offer_genome_for_class(2, FakeGenome("c"), fitness=0.75)

    payload = container.to_serializable_dict()
    restored = ClassGenomeContainer.from_serializable_dict(payload, genome_class=FakeGenome)

    assert restored.number_of_class_labels == 3
    assert restored.best_genome_for_class(0) == FakeGenome("a")
    assert restored.best_fitness_for_class(0) == 0.25
    assert restored.best_genome_for_class(1) is None
    assert restored.best_genome_for_class(2) == FakeGenome("c")
