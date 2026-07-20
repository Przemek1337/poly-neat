from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from polyneat.algorithms.cneat.class_genome_container import ClassGenomeContainer
from polyneat.algorithms.cneat.container_ensemble_phenotype import ContainerEnsemblePhenotype


@dataclass(frozen=True)
class FakeGenome:
    constant_output: float

    def clone_genome(self) -> FakeGenome:
        return FakeGenome(self.constant_output)

    def to_serializable_dict(self) -> dict:
        return {"constant_output": self.constant_output}

    @classmethod
    def from_serializable_dict(cls, payload: dict) -> FakeGenome:
        return cls(payload["constant_output"])


class _ConstantOutputPhenotype:
    def __init__(self, constant_value: float) -> None:
        self._constant_value = constant_value
        self.reset_call_count = 0

    def forward_pass(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return torch.full((input_tensor.shape[0], 1), self._constant_value)

    def reset_recurrent_state(self) -> None:
        self.reset_call_count += 1


class _FakePhenotypeDecoder:
    def build_phenotype_from_genome(self, genome: FakeGenome) -> _ConstantOutputPhenotype:
        return _ConstantOutputPhenotype(genome.constant_output)


def test_requires_at_least_two_networks() -> None:
    with pytest.raises(ValueError):
        ContainerEnsemblePhenotype(per_class_phenotypes=[_ConstantOutputPhenotype(0.5)])


def test_forward_pass_stacks_one_column_per_class() -> None:
    ensemble = ContainerEnsemblePhenotype(
        per_class_phenotypes=[
            _ConstantOutputPhenotype(0.1),
            _ConstantOutputPhenotype(0.9),
            _ConstantOutputPhenotype(0.4),
        ]
    )
    scores = ensemble.forward_pass(torch.zeros((5, 4)))
    assert scores.shape == (5, 3)
    assert torch.allclose(scores[0], torch.tensor([0.1, 0.9, 0.4]))


def test_predict_class_labels_is_argmax_over_networks() -> None:
    ensemble = ContainerEnsemblePhenotype(
        per_class_phenotypes=[
            _ConstantOutputPhenotype(0.1),
            _ConstantOutputPhenotype(0.9),
            _ConstantOutputPhenotype(0.4),
        ]
    )
    predictions = ensemble.predict_class_labels(torch.zeros((5, 4)))
    assert predictions.tolist() == [1, 1, 1, 1, 1]


def test_reset_recurrent_state_propagates_to_every_network() -> None:
    member_phenotypes = [_ConstantOutputPhenotype(0.1), _ConstantOutputPhenotype(0.9)]
    ensemble = ContainerEnsemblePhenotype(per_class_phenotypes=member_phenotypes)
    ensemble.reset_recurrent_state()
    assert [phenotype.reset_call_count for phenotype in member_phenotypes] == [1, 1]


def test_from_container_builds_one_phenotype_per_class() -> None:
    container = ClassGenomeContainer(number_of_class_labels=2)
    container.offer_genome_for_class(0, FakeGenome(0.2), fitness=0.5)
    container.offer_genome_for_class(1, FakeGenome(0.8), fitness=0.5)

    ensemble = ContainerEnsemblePhenotype.from_container(container, _FakePhenotypeDecoder())

    predictions = ensemble.predict_class_labels(torch.zeros((3, 4)))
    assert predictions.tolist() == [1, 1, 1]


def test_from_container_rejects_partially_populated_container() -> None:
    container = ClassGenomeContainer(number_of_class_labels=2)
    container.offer_genome_for_class(0, FakeGenome(0.2), fitness=0.5)
    with pytest.raises(ValueError):
        ContainerEnsemblePhenotype.from_container(container, _FakePhenotypeDecoder())
