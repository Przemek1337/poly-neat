from __future__ import annotations

import torch

from polyneat.core.neat.neat_genome import ConnectionGene, NEATGenome, NodeGene
from polyneat.core.neat.torch_feedforward_phenotype import TorchFeedForwardPhenotype

_CPU = torch.device("cpu")


def test_output_without_incoming_connections_yields_activation_of_zero() -> None:
    """Not zeros: an unwired output still sits in the topological order.

    Kahn's algorithm queues every node of in-degree 0, so the node sums an empty
    input and passes it through its activation. For sigmoid that is 0.5.
    """
    genome = NEATGenome(
        node_genes=(
            NodeGene(node_id=0, node_type="input", activation_function_name="identity"),
            NodeGene(node_id=1, node_type="output", activation_function_name="sigmoid"),
            NodeGene(node_id=2, node_type="output", activation_function_name="sigmoid"),
        ),
        connection_genes=(
            ConnectionGene(
                innovation_id=0, source_node_id=0, target_node_id=1, weight=0.5, is_enabled=True
            ),
        ),
    )
    phenotype = TorchFeedForwardPhenotype(genome, _CPU)

    outputs = phenotype.forward_pass(torch.tensor([[1.0]]))

    assert outputs[0, 1].item() == 0.5


def test_output_reachable_only_through_disabled_connections_still_activates() -> None:
    genome = NEATGenome(
        node_genes=(
            NodeGene(node_id=0, node_type="input", activation_function_name="identity"),
            NodeGene(node_id=1, node_type="output", activation_function_name="sigmoid"),
        ),
        connection_genes=(
            ConnectionGene(
                innovation_id=0, source_node_id=0, target_node_id=1, weight=9.0, is_enabled=False
            ),
        ),
    )
    phenotype = TorchFeedForwardPhenotype(genome, _CPU)

    outputs = phenotype.forward_pass(torch.tensor([[1.0]]))

    assert outputs[0, 0].item() == 0.5


def test_output_feeding_another_output_is_evaluated_in_order() -> None:
    """Output nodes may be chained; the later one must see the earlier's value."""
    genome = NEATGenome(
        node_genes=(
            NodeGene(node_id=0, node_type="bias", activation_function_name="identity"),
            NodeGene(node_id=1, node_type="output", activation_function_name="identity"),
            NodeGene(node_id=2, node_type="output", activation_function_name="identity"),
        ),
        connection_genes=(
            ConnectionGene(
                innovation_id=0, source_node_id=0, target_node_id=1, weight=3.0, is_enabled=True
            ),
            ConnectionGene(
                innovation_id=1, source_node_id=1, target_node_id=2, weight=2.0, is_enabled=True
            ),
        ),
    )
    phenotype = TorchFeedForwardPhenotype(genome, _CPU)

    outputs = phenotype.forward_pass(torch.zeros(1, 0))

    assert outputs[0, 0].item() == 3.0
    assert outputs[0, 1].item() == 6.0


def test_output_columns_follow_node_registration_order() -> None:
    genome = NEATGenome(
        node_genes=(
            NodeGene(node_id=0, node_type="bias", activation_function_name="identity"),
            NodeGene(node_id=5, node_type="output", activation_function_name="identity"),
            NodeGene(node_id=3, node_type="output", activation_function_name="identity"),
        ),
        connection_genes=(
            ConnectionGene(
                innovation_id=0, source_node_id=0, target_node_id=5, weight=1.0, is_enabled=True
            ),
            ConnectionGene(
                innovation_id=1, source_node_id=0, target_node_id=3, weight=7.0, is_enabled=True
            ),
        ),
    )
    phenotype = TorchFeedForwardPhenotype(genome, _CPU)

    outputs = phenotype.forward_pass(torch.zeros(1, 0))

    assert outputs[0, 0].item() == 1.0
    assert outputs[0, 1].item() == 7.0


def test_bias_node_is_a_constant_one() -> None:
    genome = NEATGenome(
        node_genes=(
            NodeGene(node_id=0, node_type="bias", activation_function_name="identity"),
            NodeGene(node_id=1, node_type="output", activation_function_name="identity"),
        ),
        connection_genes=(
            ConnectionGene(
                innovation_id=0, source_node_id=0, target_node_id=1, weight=2.5, is_enabled=True
            ),
        ),
    )
    phenotype = TorchFeedForwardPhenotype(genome, _CPU)

    outputs = phenotype.forward_pass(torch.zeros(4, 0))

    assert outputs.shape == (4, 1)
    assert torch.allclose(outputs, torch.full((4, 1), 2.5))


def test_one_dimensional_input_is_treated_as_a_batch_of_one() -> None:
    genome = NEATGenome(
        node_genes=(
            NodeGene(node_id=0, node_type="input", activation_function_name="identity"),
            NodeGene(node_id=1, node_type="output", activation_function_name="identity"),
        ),
        connection_genes=(
            ConnectionGene(
                innovation_id=0, source_node_id=0, target_node_id=1, weight=2.0, is_enabled=True
            ),
        ),
    )
    phenotype = TorchFeedForwardPhenotype(genome, _CPU)

    outputs = phenotype.forward_pass(torch.tensor([3.0]))

    assert outputs.shape == (1, 1)
    assert outputs[0, 0].item() == 6.0


def test_phenotype_registers_no_parameters_or_buffers() -> None:
    """Weights are plain floats; the class carries no ``nn.Module`` state."""
    genome = NEATGenome(
        node_genes=(
            NodeGene(node_id=0, node_type="input", activation_function_name="identity"),
            NodeGene(node_id=1, node_type="output", activation_function_name="sigmoid"),
        ),
        connection_genes=(
            ConnectionGene(
                innovation_id=0, source_node_id=0, target_node_id=1, weight=0.5, is_enabled=True
            ),
        ),
    )
    phenotype = TorchFeedForwardPhenotype(genome, _CPU)

    assert list(phenotype.parameters()) == []
    assert list(phenotype.buffers()) == []
    assert phenotype.state_dict() == {}


def test_outputs_are_created_on_the_requested_device() -> None:
    genome = NEATGenome(
        node_genes=(
            NodeGene(node_id=0, node_type="input", activation_function_name="identity"),
            NodeGene(node_id=1, node_type="output", activation_function_name="sigmoid"),
        ),
        connection_genes=(
            ConnectionGene(
                innovation_id=0, source_node_id=0, target_node_id=1, weight=0.5, is_enabled=True
            ),
        ),
    )
    phenotype = TorchFeedForwardPhenotype(genome, _CPU)

    outputs = phenotype.forward_pass(torch.tensor([[1.0]]))

    assert outputs.device.type == _CPU.type
