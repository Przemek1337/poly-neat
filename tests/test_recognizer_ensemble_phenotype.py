from __future__ import annotations

import pytest
import torch

from polyneat.algorithms.lneat.recognizer_ensemble_phenotype import (
    RecognizerEnsemblePhenotype,
)
from polyneat.core.neat.neat_genome import ConnectionGene, NEATGenome, NodeGene
from polyneat.core.neat.neat_phenotype_decoder import NEATPhenotypeDecoder
from polyneat.evaluators.classification_accuracy_evaluator import (
    ClassificationAccuracyEvaluator,
)


class _ConstantOutputPhenotype:
    """Phenotype stub returning a fixed column of outputs."""

    def __init__(self, outputs_per_sample: list[float]) -> None:
        self._outputs_per_sample = outputs_per_sample
        self.reset_call_count = 0

    def forward_pass(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return torch.tensor(self._outputs_per_sample).unsqueeze(1)

    def reset_recurrent_state(self) -> None:
        self.reset_call_count += 1


def _single_connection_genome(weight: float) -> NEATGenome:
    return NEATGenome(
        node_genes=(
            NodeGene(node_id=0, node_type="input", activation_function_name="identity"),
            NodeGene(node_id=1, node_type="bias", activation_function_name="identity"),
            NodeGene(node_id=2, node_type="output", activation_function_name="sigmoid"),
        ),
        connection_genes=(
            ConnectionGene(
                innovation_id=0, source_node_id=0, target_node_id=2,
                weight=weight, is_enabled=True,
            ),
        ),
    )


def test_forward_stacks_recognizer_outputs_into_columns() -> None:
    ensemble = RecognizerEnsemblePhenotype(
        class_recognizer_phenotypes=[
            _ConstantOutputPhenotype([0.9, 0.1]),
            _ConstantOutputPhenotype([0.2, 0.8]),
        ]
    )
    outputs = ensemble.forward_pass(torch.zeros((2, 3)))
    torch.testing.assert_close(outputs, torch.tensor([[0.9, 0.2], [0.1, 0.8]]))


def test_argmax_prediction_matches_strongest_recognizer() -> None:
    ensemble = RecognizerEnsemblePhenotype(
        class_recognizer_phenotypes=[
            _ConstantOutputPhenotype([0.9, 0.1]),
            _ConstantOutputPhenotype([0.2, 0.8]),
        ]
    )
    accuracy = ClassificationAccuracyEvaluator(
        input_features=torch.zeros((2, 3)),
        target_labels=torch.tensor([0, 1]),
    ).evaluate_single_phenotype(ensemble)
    assert accuracy == pytest.approx(1.0)


def test_from_genomes_builds_executable_ensemble() -> None:
    decoder = NEATPhenotypeDecoder(device_for_computation=torch.device("cpu"))
    ensemble = RecognizerEnsemblePhenotype.from_genomes(
        class_recognizer_genomes=[
            _single_connection_genome(weight=5.0),
            _single_connection_genome(weight=-5.0),
        ],
        phenotype_decoder=decoder,
    )
    outputs = ensemble.forward_pass(torch.tensor([[1.0]]))
    assert outputs.shape == (1, 2)
    # sigmoid(5.0) > 0.5 > sigmoid(-5.0): class 0 wins on this input
    assert outputs[0, 0] > outputs[0, 1]


def test_rejects_fewer_than_two_recognizers() -> None:
    with pytest.raises(ValueError):
        RecognizerEnsemblePhenotype(
            class_recognizer_phenotypes=[_ConstantOutputPhenotype([0.5])]
        )


def test_rejects_multi_output_recognizer_at_forward() -> None:
    class _TwoColumnPhenotype:
        def forward_pass(self, input_tensor: torch.Tensor) -> torch.Tensor:
            return torch.zeros((input_tensor.shape[0], 2))

        def reset_recurrent_state(self) -> None:
            return None

    ensemble = RecognizerEnsemblePhenotype(
        class_recognizer_phenotypes=[_TwoColumnPhenotype(), _TwoColumnPhenotype()]
    )
    with pytest.raises(ValueError):
        ensemble.forward_pass(torch.zeros((1, 1)))


def test_reset_recurrent_state_propagates_to_every_recognizer() -> None:
    recognizers = [
        _ConstantOutputPhenotype([0.5]),
        _ConstantOutputPhenotype([0.5]),
    ]
    ensemble = RecognizerEnsemblePhenotype(class_recognizer_phenotypes=recognizers)
    ensemble.reset_recurrent_state()
    assert all(recognizer.reset_call_count == 1 for recognizer in recognizers)
