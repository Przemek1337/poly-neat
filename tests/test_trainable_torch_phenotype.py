from __future__ import annotations

import pytest
import torch

from polyneat.algorithms.lneat.trainable_torch_phenotype import (
    TrainableTorchFeedForwardPhenotype,
)
from polyneat.core.neat.neat_genome import ConnectionGene, NEATGenome, NodeGene
from polyneat.core.neat.torch_feedforward_phenotype import TorchFeedForwardPhenotype

_CPU = torch.device("cpu")


def _build_genome_with_hidden_node() -> NEATGenome:
    return NEATGenome(
        node_genes=(
            NodeGene(node_id=0, node_type="input", activation_function_name="identity"),
            NodeGene(node_id=1, node_type="bias", activation_function_name="identity"),
            NodeGene(node_id=2, node_type="hidden", activation_function_name="sigmoid"),
            NodeGene(node_id=3, node_type="output", activation_function_name="sigmoid"),
        ),
        connection_genes=(
            ConnectionGene(
                innovation_id=0, source_node_id=0, target_node_id=2, weight=0.7, is_enabled=True
            ),
            ConnectionGene(
                innovation_id=1, source_node_id=1, target_node_id=2, weight=-0.3, is_enabled=True
            ),
            ConnectionGene(
                innovation_id=2, source_node_id=2, target_node_id=3, weight=1.1, is_enabled=True
            ),
            ConnectionGene(
                innovation_id=3, source_node_id=0, target_node_id=3, weight=9.9, is_enabled=False
            ),
        ),
    )


def test_forward_matches_non_trainable_phenotype() -> None:
    genome = _build_genome_with_hidden_node()
    trainable = TrainableTorchFeedForwardPhenotype(genome, _CPU)
    reference = TorchFeedForwardPhenotype(genome, _CPU)
    inputs = torch.tensor([[0.0], [0.5], [1.0]])
    torch.testing.assert_close(
        trainable.forward_pass(inputs), reference.forward_pass(inputs)
    )


def test_weights_are_trainable_parameters() -> None:
    genome = _build_genome_with_hidden_node()
    trainable = TrainableTorchFeedForwardPhenotype(genome, _CPU)
    parameters = list(trainable.parameters())
    # one parameter per ENABLED connection (3 of 4)
    assert len(parameters) == 3
    assert all(parameter.requires_grad for parameter in parameters)


def test_gradients_flow_to_connection_weights() -> None:
    genome = _build_genome_with_hidden_node()
    trainable = TrainableTorchFeedForwardPhenotype(genome, _CPU)
    outputs = trainable.forward_pass(torch.tensor([[1.0]]))
    loss = torch.mean((outputs - 1.0) ** 2)
    loss.backward()
    assert all(parameter.grad is not None for parameter in trainable.parameters())


def test_extract_genome_writes_back_trained_weights() -> None:
    genome = _build_genome_with_hidden_node()
    trainable = TrainableTorchFeedForwardPhenotype(genome, _CPU)
    with torch.no_grad():
        for parameter in trainable.parameters():
            parameter.add_(0.5)
    extracted = trainable.extract_genome_with_trained_weights()

    assert extracted is not genome
    extracted_by_innovation = {
        gene.innovation_id: gene for gene in extracted.connection_genes
    }
    original_by_innovation = {
        gene.innovation_id: gene for gene in genome.connection_genes
    }
    for innovation_id in (0, 1, 2):  # enabled: weight shifted by +0.5
        assert extracted_by_innovation[innovation_id].weight == pytest.approx(
            original_by_innovation[innovation_id].weight + 0.5
        )
    # disabled connection untouched, flag preserved
    assert extracted_by_innovation[3].weight == 9.9
    assert extracted_by_innovation[3].is_enabled is False
    # structure preserved
    assert extracted.node_genes == genome.node_genes
