from __future__ import annotations

from dataclasses import replace

import torch
from torch import nn

from polyneat.core.neat.neat_genome import NEATGenome
from polyneat.nn.activation_functions import (
    ActivationFunction,
    resolve_activation_function_by_name,
)
from polyneat.nn.topology_utilities import compute_topological_order_of_node_ids


class TrainableTorchFeedForwardPhenotype(nn.Module):
    """Executes a NEAT feed-forward network with trainable connection weights.

    :class:`~polyneat.core.neat.torch_feedforward_phenotype.TorchFeedForwardPhenotype`
    stores connection weights as plain floats, so gradients cannot reach them.
    This class stores one ``nn.Parameter`` per enabled connection, which lets
    L-NEAT's backpropagation sessions (Chen & Alahakoon, 2006, section IV.B)
    train the weights and write them back into a new genome via
    :meth:`extract_genome_with_trained_weights`. Evaluation semantics match
    the non-trainable phenotype exactly: nodes are evaluated in topological
    order, the bias node is a constant 1.0, and output nodes with no enabled
    incoming path yield zeros.

    Recurrent phenotypes will use a separate class; this one is strictly
    feed-forward and its ``reset_recurrent_state`` is a no-op.

    References:
        Chen, L., & Alahakoon, D. (2006). NeuroEvolution of Augmenting
        Topologies with Learning for Data Classification. *ICIA 2006*.
    """

    def __init__(
        self,
        neat_genome: NEATGenome,
        device_for_computation: torch.device,
    ) -> None:
        """Precompute the evaluation order and register per-connection parameters.

        Args:
            neat_genome: Genome to express; only enabled connections become
                trainable parameters, disabled genes are carried through
                untouched on extraction.
            device_for_computation: Device all tensors are created on.
        """
        super().__init__()
        self._source_genome = neat_genome
        self._device_for_computation = device_for_computation

        input_node_ids_in_registration_order: list[int] = []
        output_node_ids_in_registration_order: list[int] = []
        bias_node_ids_in_registration_order: list[int] = []
        node_id_to_activation_function: dict[int, ActivationFunction] = {}
        node_id_to_node_type: dict[int, str] = {}

        for node_gene in neat_genome.node_genes:
            node_id_to_node_type[node_gene.node_id] = node_gene.node_type
            node_id_to_activation_function[node_gene.node_id] = resolve_activation_function_by_name(
                node_gene.activation_function_name
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

        self._connection_weight_parameters = nn.ParameterDict()
        incoming_connections_by_target_node_id: dict[int, list[tuple[int, str]]] = {
            node_id: [] for node_id in all_node_ids_in_genome
        }
        for connection_gene in neat_genome.connection_genes:
            if not connection_gene.is_enabled:
                continue
            parameter_key = str(connection_gene.innovation_id)
            self._connection_weight_parameters[parameter_key] = nn.Parameter(
                torch.tensor(connection_gene.weight, dtype=torch.float32)
            )
            incoming_connections_by_target_node_id[connection_gene.target_node_id].append(
                (connection_gene.source_node_id, parameter_key)
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
        """Evaluate the network on a batch of inputs, keeping the autograd graph.

        Args:
            input_tensor: Shape ``(batch, num_inputs)`` — or ``(num_inputs,)``,
                which is treated as a batch of one.

        Returns:
            Output activations of shape ``(batch, num_outputs)``, with output
            columns in node registration order. Outputs with no enabled
            incoming path yield zeros.
        """
        input_tensor_on_target_device = input_tensor.to(self._device_for_computation)
        if input_tensor_on_target_device.dim() == 1:
            input_tensor_on_target_device = input_tensor_on_target_device.unsqueeze(0)
        batch_size = input_tensor_on_target_device.shape[0]

        node_activations_by_node_id: dict[int, torch.Tensor] = {}

        for input_slot_position, input_node_id in enumerate(
            self._input_node_ids_in_registration_order
        ):
            node_activations_by_node_id[input_node_id] = input_tensor_on_target_device[
                :, input_slot_position
            ]

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
                parameter_key,
            ) in self._incoming_connections_by_target_node_id[current_node_id]:
                if source_node_id not in node_activations_by_node_id:
                    continue
                weighted_input_sum = weighted_input_sum + (
                    node_activations_by_node_id[source_node_id]
                    * self._connection_weight_parameters[parameter_key]
                )

            activation_function_for_current_node = self._node_id_to_activation_function[
                current_node_id
            ]
            node_activations_by_node_id[current_node_id] = activation_function_for_current_node(
                weighted_input_sum
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

    def extract_genome_with_trained_weights(self) -> NEATGenome:
        """Return a new genome whose enabled connections carry the trained weights.

        Disabled connection genes and all node genes are carried over
        unchanged, so historical markings, structure, and expression flags
        survive the learning session (only weights are Lamarckian).

        Returns:
            A structurally identical :class:`NEATGenome` with updated weights.
        """
        connection_genes_with_trained_weights = tuple(
            replace(
                connection_gene,
                weight=float(
                    self._connection_weight_parameters[str(connection_gene.innovation_id)].item()
                ),
            )
            if connection_gene.is_enabled
            else connection_gene
            for connection_gene in self._source_genome.connection_genes
        )
        return NEATGenome(
            node_genes=self._source_genome.node_genes,
            connection_genes=connection_genes_with_trained_weights,
        )
