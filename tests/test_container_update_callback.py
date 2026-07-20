from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import cast

import pytest

from polyneat.algorithms.cneat.class_genome_container import ClassGenomeContainer
from polyneat.algorithms.cneat.container_update_callback import ContainerUpdateCallback
from polyneat.core.component_protocols import Genome
from polyneat.core.population import Population
from polyneat.evaluators.class_indexed_evaluator_base import ClassIndexedFitnessEvaluator
from polyneat.runner.run_context import RunContext


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


class _DummyEvaluator(ClassIndexedFitnessEvaluator):
    def evaluate_phenotype_for_class(self, phenotype, class_label_index: int) -> float:
        return 0.0


def _make_population(labels: list[str], generation_number: int = 0) -> Population:
    return Population(
        genomes=cast("list[Genome]", [FakeGenome(label) for label in labels]),
        species_assignments=None,
        generation_number=generation_number,
    )


def _make_run_context() -> RunContext:
    return RunContext(
        run_id="test_run",
        run_started_at=datetime.now(),
        current_generation_number=0,
        total_generations_planned=-1,
        history_of_generation_statistics=[],
        current_best_genome=None,
        current_best_fitness=None,
        configuration_snapshot={},
    )


def test_rejects_mismatched_class_counts() -> None:
    container = ClassGenomeContainer(number_of_class_labels=2)
    evaluator = _DummyEvaluator(number_of_class_labels=3)
    with pytest.raises(ValueError):
        ContainerUpdateCallback(container, evaluator)


def test_stores_best_genome_per_class_by_index_modulo() -> None:
    container = ClassGenomeContainer(number_of_class_labels=3)
    callback = ContainerUpdateCallback(container, _DummyEvaluator(number_of_class_labels=3))
    # indices 0..5 -> classes [0, 1, 2, 0, 1, 2]
    population = _make_population(["a", "b", "c", "d", "e", "f"])
    fitnesses = [0.1, 0.9, 0.3, 0.7, 0.2, 0.4]

    callback.on_population_evaluated(_make_run_context(), population, fitnesses)

    assert container.best_genome_for_class(0) == FakeGenome("d")  # 0.7 > 0.1
    assert container.best_genome_for_class(1) == FakeGenome("b")  # 0.9 > 0.2
    assert container.best_genome_for_class(2) == FakeGenome("f")  # 0.4 > 0.3
    assert container.is_fully_populated()


def test_container_never_regresses_across_generations() -> None:
    container = ClassGenomeContainer(number_of_class_labels=2)
    callback = ContainerUpdateCallback(container, _DummyEvaluator(number_of_class_labels=2))
    context = _make_run_context()

    callback.on_population_evaluated(
        context, _make_population(["gen0_c0", "gen0_c1"]), [0.8, 0.9]
    )
    callback.on_population_evaluated(
        context, _make_population(["gen1_c0", "gen1_c1"], generation_number=1), [0.5, 0.95]
    )

    assert container.best_genome_for_class(0) == FakeGenome("gen0_c0")  # kept: 0.8 > 0.5
    assert container.best_fitness_for_class(0) == 0.8
    assert container.best_genome_for_class(1) == FakeGenome("gen1_c1")  # improved
    assert container.best_fitness_for_class(1) == 0.95
