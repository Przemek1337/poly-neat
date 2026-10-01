"""The sweep-side output sums must reproduce the library phenotype exactly."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from gpu_sweep.output_sums import (
    count_tied_maxima,
    evaluate_genome_with_output_sums,
    evaluate_networks_with_output_sums,
)
from polyneat.core.neat.torch_feedforward_phenotype import TorchFeedForwardPhenotype


def _node(node_id: int, node_type: str, activation: str) -> SimpleNamespace:
    return SimpleNamespace(
        node_id=node_id, node_type=node_type, activation_function_name=activation
    )


def _connection(source: int, target: int, weight: float, enabled: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        source_node_id=source, target_node_id=target, weight=weight, is_enabled=enabled
    )


def _two_output_genome() -> SimpleNamespace:
    """Two inputs, a bias, one hidden node, two outputs - one of them saturating."""
    return SimpleNamespace(
        node_genes=[
            _node(0, "input", "identity"),
            _node(1, "input", "identity"),
            _node(2, "bias", "identity"),
            _node(3, "output", "steepened_sigmoid"),
            _node(4, "output", "steepened_sigmoid"),
            _node(5, "hidden", "steepened_sigmoid"),
        ],
        connection_genes=[
            _connection(0, 3, 40.0),
            _connection(1, 3, -3.0),
            _connection(0, 5, 1.5),
            _connection(5, 4, 60.0),
            _connection(2, 4, 0.7),
            _connection(1, 4, 9.0, enabled=False),
        ],
    )


def test_activations_equal_the_library_forward_pass_bit_for_bit() -> None:
    genome = _two_output_genome()
    features = torch.randn(64, 2, generator=torch.Generator().manual_seed(0))

    library_outputs = TorchFeedForwardPhenotype(genome, torch.device("cpu")).forward_pass(features)
    activations, sums = evaluate_genome_with_output_sums(genome, features, "cpu")

    assert torch.equal(activations, library_outputs)
    assert sums.shape == (64, 2)
    # The activation is the steepened sigmoid of the sum.
    assert torch.equal(torch.sigmoid(4.9 * sums), activations)


def test_sums_separate_outputs_that_saturate_to_the_same_value() -> None:
    genome = _two_output_genome()
    features = torch.tensor([[5.0, 0.0]])

    activations, sums = evaluate_genome_with_output_sums(genome, features, "cpu")

    assert activations[0, 0] == activations[0, 1] == 1.0
    assert sums[0, 0] != sums[0, 1]
    assert count_tied_maxima(activations) == 1
    assert count_tied_maxima(sums) == 0


def test_an_unconnected_output_has_a_zero_sum() -> None:
    genome = SimpleNamespace(
        node_genes=[
            _node(0, "input", "identity"),
            _node(1, "output", "steepened_sigmoid"),
        ],
        connection_genes=[],
    )

    activations, sums = evaluate_genome_with_output_sums(genome, torch.ones(3, 1), "cpu")

    assert torch.equal(sums, torch.zeros(3, 1))
    assert torch.equal(activations, torch.full((3, 1), 0.5))


def test_an_ensemble_stacks_one_column_per_recognizer() -> None:
    recognizer = SimpleNamespace(
        node_genes=[_node(0, "input", "identity"), _node(1, "output", "steepened_sigmoid")],
        connection_genes=[_connection(0, 1, 2.0)],
    )
    other = SimpleNamespace(
        node_genes=[_node(0, "input", "identity"), _node(1, "output", "steepened_sigmoid")],
        connection_genes=[_connection(0, 1, -1.0)],
    )
    features = torch.tensor([[1.0], [-2.0]])

    _, sums = evaluate_networks_with_output_sums([recognizer, other], features, "cpu")

    assert torch.equal(sums, torch.tensor([[2.0, -1.0], [-4.0, 2.0]]))
