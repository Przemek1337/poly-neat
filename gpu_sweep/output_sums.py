"""Output-node sums before the activation, computed from a network genome.

Every output in this sweep goes through ``steepened_sigmoid``, which in float32
returns exactly 1.0 for any sum above about 3.4 and exactly 0.0 below about
-20.3. Distinct sums therefore collapse onto the same output value, and the
``argmax`` in :mod:`gpu_sweep.metrics` breaks such ties towards the lowest class
index. The sums keep the ordering the outputs lose, so the sweep stores them per
patient: any other prediction rule can then be applied after the run without
evolving anything again.

The library's ``TorchFeedForwardPhenotype.forward_pass`` returns only the
activations, and the sweep does not modify the library. This module therefore
walks the genome itself, with the same node order and the same tensor
operations in the same sequence, so its activations are identical to the
library's. :func:`gpu_sweep.algorithm_runners._split_metrics` checks that on
every run and records the result.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from polyneat.nn.activation_functions import resolve_activation_function_by_name
from polyneat.nn.topology_utilities import compute_topological_order_of_node_ids


def evaluate_genome_with_output_sums(
    network_genome: object,
    features: torch.Tensor,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run one feed-forward network genome and return activations and output sums.

    Mirrors ``polyneat/core/neat/torch_feedforward_phenotype.py``: inputs in
    node registration order, every bias node a constant 1.0, nodes evaluated in
    topological order, each node's sum accumulated over its enabled incoming
    connections in connection-gene order, starting from zeros.

    Args:
        network_genome: Anything with ``node_genes`` (``node_id``,
            ``node_type``, ``activation_function_name``) and
            ``connection_genes`` (``source_node_id``, ``target_node_id``,
            ``weight``, ``is_enabled``) - a live genome or a stored topology
            record.
        features: ``[n_samples, n_features]`` float tensor.
        device: Device to compute on.

    Returns:
        ``(activations, sums)``, both ``[n_samples, n_outputs]`` with output
        columns in node registration order.
    """
    device = torch.device(device)
    input_node_ids: list[int] = []
    output_node_ids: list[int] = []
    bias_node_ids: list[int] = []
    node_type_by_id: dict[int, str] = {}
    activation_by_id = {}
    for node_gene in network_genome.node_genes:
        node_type_by_id[node_gene.node_id] = node_gene.node_type
        activation_by_id[node_gene.node_id] = resolve_activation_function_by_name(
            node_gene.activation_function_name
        )
        if node_gene.node_type == "input":
            input_node_ids.append(node_gene.node_id)
        elif node_gene.node_type == "output":
            output_node_ids.append(node_gene.node_id)
        elif node_gene.node_type == "bias":
            bias_node_ids.append(node_gene.node_id)

    all_node_ids = [node_gene.node_id for node_gene in network_genome.node_genes]
    enabled_edges = [
        (connection_gene.source_node_id, connection_gene.target_node_id)
        for connection_gene in network_genome.connection_genes
        if connection_gene.is_enabled
    ]
    topological_order = compute_topological_order_of_node_ids(
        all_node_ids=all_node_ids, enabled_directed_edges=enabled_edges
    )
    incoming_by_target: dict[int, list[tuple[int, float]]] = {
        node_id: [] for node_id in all_node_ids
    }
    for connection_gene in network_genome.connection_genes:
        if connection_gene.is_enabled:
            incoming_by_target[connection_gene.target_node_id].append(
                (connection_gene.source_node_id, connection_gene.weight)
            )

    inputs = features.to(device)
    if inputs.dim() == 1:
        inputs = inputs.unsqueeze(0)
    batch_size = inputs.shape[0]

    activations: dict[int, torch.Tensor] = {}
    output_sums: dict[int, torch.Tensor] = {}
    with torch.no_grad():
        for slot, node_id in enumerate(input_node_ids):
            activations[node_id] = inputs[:, slot]
        for node_id in bias_node_ids:
            activations[node_id] = torch.ones(batch_size, device=device)
        for node_id in topological_order:
            if node_type_by_id[node_id] in ("input", "bias"):
                continue
            weighted_input_sum = torch.zeros(batch_size, device=device)
            for source_node_id, weight in incoming_by_target[node_id]:
                if source_node_id not in activations:
                    continue
                weighted_input_sum = weighted_input_sum + (activations[source_node_id] * weight)
            activations[node_id] = activation_by_id[node_id](weighted_input_sum)
            if node_type_by_id[node_id] == "output":
                output_sums[node_id] = weighted_input_sum

    return (
        torch.stack([activations[node_id] for node_id in output_node_ids], dim=1),
        torch.stack([output_sums[node_id] for node_id in output_node_ids], dim=1),
    )


def evaluate_networks_with_output_sums(
    network_genomes: Sequence[object],
    features: torch.Tensor,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Activations and output sums of a classifier made of one or more networks.

    One genome is a single network with one output per class. Several genomes
    are the C-NEAT / L-NEAT recognizer ensembles: one single-output network per
    class, in class order, whose columns are stacked the way the library's
    ensemble phenotypes stack them.

    Args:
        network_genomes: The classifier's network genomes, in class order.
        features: ``[n_samples, n_features]`` float tensor.
        device: Device to compute on.

    Returns:
        ``(activations, sums)``, both ``[n_samples, n_classes]``.
    """
    if len(network_genomes) == 1:
        return evaluate_genome_with_output_sums(network_genomes[0], features, device)
    activation_columns: list[torch.Tensor] = []
    sum_columns: list[torch.Tensor] = []
    for network_genome in network_genomes:
        member_activations, member_sums = evaluate_genome_with_output_sums(
            network_genome, features, device
        )
        activation_columns.append(member_activations[:, 0])
        sum_columns.append(member_sums[:, 0])
    return torch.stack(activation_columns, dim=1), torch.stack(sum_columns, dim=1)


def count_tied_maxima(scores: torch.Tensor) -> int:
    """Number of rows whose largest value occurs in more than one column."""
    row_maxima = scores.max(dim=1, keepdim=True).values
    return int(((scores == row_maxima).sum(dim=1) > 1).sum().item())
