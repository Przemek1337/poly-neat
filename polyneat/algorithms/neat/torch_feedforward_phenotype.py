from __future__ import annotations

import torch
from torch import nn

from polyneat.algorithms.neat.neat_genome import NEATGenome, NodeGene
from polyneat.nn.activation_functions import (
    ActivationFunction,
    resolve_activation_function_by_name,
)
from polyneat.nn.topology_utilities import compute_topological_order_of_node_ids


class TorchFeedForwardPhenotype(nn.Module):
    """Executes a NEAT feed-forward network by evaluating nodes in topological order.

    Subclassing ``nn.Module`` is deliberate: it grants ``.to(device)``,
    ``.state_dict()``, and ``.parameters()`` even though the ``Phenotype``
    protocol only requires ``forward_pass``. Weights are stored as a *fixed*
    parameter tensor per input connection, not as ``nn.Linear`` layers — the
    NEAT topology is irregular so per-connection weights are the natural
    representation.

    Recurrent phenotypes will use a separate class; this one is strictly
    feed-forward and its ``reset_recurrent_state`` is a no-op.
    """

    def __init__(
        self,
        neat_genome: NEATGenome,
        device_for_computation: torch.device,
    ) -> None:
        super().__init__()
        self._device_for_computation = device_for_computation

        input_node_ids_in_registration_order: list[int] = []
        output_node_ids_in_registration_order: list[int] = []
        bias_node_ids_in_registration_order: list[int] = []
        node_id_to_activation_function: dict[int, ActivationFunction] = {}
        node_id_to_node_type: dict[int, str] = {}

        for node_gene in neat_genome.node_genes:
            node_id_to_node_type[node_gene.node_id] = node_gene.node_type
            node_id_to_activation_function[node_gene.node_id] = (
                resolve_activation_function_by_name(node_gene.activation_function_name)
            )
            if node_gene.node_type == "input":
                input_node_ids_in_registration_order.append(node_gene.node_id)
            elif node_gene.node_type == "output":
                output_node_ids_in_registration_order.append(node_gene.node_id)
            elif node_gene.node_type == "bias":
                bias_node_ids_in_registration_order.append(node_gene.node_id)

        enabled_directed_edges = [
            (connection_gene.source_node_id, connection_gene.target_node_id)
            for connection_gene in neat_genome.connection_genes
            if connection_gene.is_enabled
        ]
        all_node_ids_in_genome = [node_gene.node_id for node_gene in neat_genome.node_genes]
        topologically_sorted_node_ids = compute_topological_order_of_node_ids(
            all_node_ids=all_node_ids_in_genome,
            enabled_directed_edges=enabled_directed_edges,
        )

        incoming_connections_by_target_node_id: dict[int, list[tuple[int, float]]] = {
            node_id: [] for node_id in all_node_ids_in_genome
        }
        for connection_gene in neat_genome.connection_genes:
            if connection_gene.is_enabled:
                incoming_connections_by_target_node_id[connection_gene.target_node_id].append(
                    (connection_gene.source_node_id, connection_gene.weight)
                )

        self._input_node_ids_in_registration_order = input_node_ids_in_registration_order
        self._output_node_ids_in_registration_order = output_node_ids_in_registration_order
        self._bias_node_ids_in_registration_order = bias_node_ids_in_registration_order
        self._topologically_sorted_node_ids = topologically_sorted_node_ids
        self._incoming_connections_by_target_node_id = incoming_connections_by_target_node_id
        self._node_id_to_activation_function = node_id_to_activation_function
        self._node_id_to_node_type = node_id_to_node_type

        self.to(device_for_computation)

    def forward_pass(self, input_tensor: torch.Tensor) -> torch.Tensor:
        input_tensor_on_target_device = input_tensor.to(self._device_for_computation)
        if input_tensor_on_target_device.dim() == 1:
            input_tensor_on_target_device = input_tensor_on_target_device.unsqueeze(0)
        batch_size = input_tensor_on_target_device.shape[0]

        node_activations_by_node_id: dict[int, torch.Tensor] = {}

        for input_slot_position, input_node_id in enumerate(
            self._input_node_ids_in_registration_order
        ):
            node_activations_by_node_id[input_node_id] = (
                input_tensor_on_target_device[:, input_slot_position]
            )

        for bias_node_id in self._bias_node_ids_in_registration_order:
            node_activations_by_node_id[bias_node_id] = torch.ones(
                batch_size, device=self._device_for_computation
            )

        for current_node_id in self._topologically_sorted_node_ids:
            node_type_for_current_node = self._node_id_to_node_type[current_node_id]
            if node_type_for_current_node in ("input", "bias"):
                continue

            weighted_input_sum = torch.zeros(batch_size, device=self._device_for_computation)
            for (
                source_node_id,
                connection_weight,
            ) in self._incoming_connections_by_target_node_id[current_node_id]:
                if source_node_id not in node_activations_by_node_id:
                    continue
                weighted_input_sum = weighted_input_sum + (
                    node_activations_by_node_id[source_node_id] * connection_weight
                )

            activation_function_for_current_node = (
                self._node_id_to_activation_function[current_node_id]
            )
            node_activations_by_node_id[current_node_id] = (
                activation_function_for_current_node(weighted_input_sum)
            )

        output_columns_in_registration_order = [
            node_activations_by_node_id.get(
                output_node_id,
                torch.zeros(batch_size, device=self._device_for_computation),
            )
            for output_node_id in self._output_node_ids_in_registration_order
        ]
        output_tensor = torch.stack(output_columns_in_registration_order, dim=1)
        return output_tensor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``nn.Module`` compatibility shim — delegates to ``forward_pass``."""
        return self.forward_pass(x)

    def reset_recurrent_state(self) -> None:
        return None
