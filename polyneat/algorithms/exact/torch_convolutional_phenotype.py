"""Executable CNN expressed from an EXACT genome, with trainable kernels."""

from __future__ import annotations

import math
from dataclasses import replace

import torch
import torch.nn.functional as functional
from torch import nn

from polyneat.algorithms.exact.exact_genome import (
    EXACTGenome,
    FilterNodeGene,
    compute_convolution_kernel_size,
)


class TorchConvolutionalPhenotype(nn.Module):
    """Executes an EXACT CNN; one trainable ``nn.Parameter`` kernel per enabled edge.

    Forward propagation follows the paper (Desell, 2017, section VII): nodes
    are processed in depth order (a linear sweep, no graph traversal); each
    edge convolves its source filter with a kernel of size
    ``|out_d - in_d| + 1`` and zero padding ``max(0, out_d - in_d)``, which
    maps the source size exactly onto the target size; a node sums its
    incoming convolutions and hidden nodes apply a leaky ReLU clamped at
    ``activation_clamp_maximum``. Output nodes are 1x1 and linear — their
    stacked values are class logits (the softmax lives in the loss and in
    the evaluator). Kernels stored in the genome are reused (epigenetic
    initialization, section III-D); ``None`` kernels are He-initialized with
    variance ``2 / n``, ``n`` being the number of weights feeding the target
    node (section VII, "Weight Initialization"; He et al., 2015).

    With ``use_batch_normalization`` every hidden node gets its own
    ``nn.BatchNorm2d`` applied between summation and activation, its running
    statistics tracked with momentum ``batch_normalization_alpha`` and seeded
    from the node gene's stored state (section VII). Dropout is applied at
    two sites, also section VII: on the input feature map
    (``input_dropout_probability``) and on every hidden filter after its
    activation (``hidden_dropout_probability``). Both dropout and the
    batch-norm statistics update are inert outside training mode, and
    construction leaves the module in eval mode — the trainer opts into
    ``train()`` explicitly.

    Strictly feed-forward; ``reset_recurrent_state`` is a no-op.

    References:
        Desell, T. (2017). Developing a Volunteer Computing Project to Evolve
            Convolutional Neural Networks and Their Hyperparameters. 2017 IEEE
            13th International Conference on e-Science, pp. 19-28.
    """

    def __init__(
        self,
        exact_genome: EXACTGenome,
        input_image_height: int,
        input_image_width: int,
        leaky_relu_negative_slope: float,
        activation_clamp_maximum: float,
        device_for_computation: torch.device,
        use_batch_normalization: bool = False,
        batch_normalization_alpha: float = 0.1,
        input_dropout_probability: float = 0.0,
        hidden_dropout_probability: float = 0.0,
    ) -> None:
        """Precompute the depth-ordered sweep and register per-edge kernels.

        Args:
            exact_genome: Genome to express; only enabled edges become
                trainable parameters, disabled genes are carried through
                untouched on extraction.
            input_image_height: Height the flat input rows are reshaped to.
            input_image_width: Width the flat input rows are reshaped to.
            leaky_relu_negative_slope: Hidden activation leak (paper: 0.1).
            activation_clamp_maximum: Hidden activation cap (paper: 5.5).
            device_for_computation: Device all tensors are created on.
            use_batch_normalization: Whether hidden nodes are batch
                normalized (section VII).
            batch_normalization_alpha: Momentum of the batch-norm running
                statistics.
            input_dropout_probability: Dropout on the input feature map.
            hidden_dropout_probability: Dropout on every hidden filter.

        Raises:
            ValueError: If the genome's input node size disagrees with the
                given image dimensions.
        """
        super().__init__()
        self._source_genome = exact_genome
        self._input_image_height = input_image_height
        self._input_image_width = input_image_width
        self._leaky_relu_negative_slope = leaky_relu_negative_slope
        self._activation_clamp_maximum = activation_clamp_maximum
        self._device_for_computation = device_for_computation
        self._input_dropout_probability = input_dropout_probability
        self._hidden_dropout_probability = hidden_dropout_probability

        node_gene_by_id = {
            node_gene.node_id: node_gene for node_gene in exact_genome.node_genes
        }
        input_node = next(
            node_gene
            for node_gene in exact_genome.node_genes
            if node_gene.node_type == "input"
        )
        if (input_node.filter_height, input_node.filter_width) != (
            input_image_height,
            input_image_width,
        ):
            raise ValueError(
                f"genome input node is {input_node.filter_height}x"
                f"{input_node.filter_width} but the phenotype was given "
                f"{input_image_height}x{input_image_width} images"
            )
        self._input_node_id = input_node.node_id
        self._output_node_ids_in_id_order = sorted(
            node_gene.node_id
            for node_gene in exact_genome.node_genes
            if node_gene.node_type == "output"
        )
        self._node_genes_in_depth_order: list[FilterNodeGene] = sorted(
            exact_genome.node_genes,
            key=lambda node_gene: (node_gene.depth, node_gene.node_id),
        )

        # He fan-in per target node: total weight count feeding it (section VII).
        fan_in_by_target_node_id: dict[int, int] = {}
        for edge_gene in exact_genome.edge_genes:
            if not edge_gene.is_enabled:
                continue
            kernel_height, kernel_width = compute_convolution_kernel_size(
                node_gene_by_id[edge_gene.source_node_id],
                node_gene_by_id[edge_gene.target_node_id],
            )
            fan_in_by_target_node_id[edge_gene.target_node_id] = (
                fan_in_by_target_node_id.get(edge_gene.target_node_id, 0)
                + kernel_height * kernel_width
            )

        self._kernel_parameters = nn.ParameterDict()
        # target node id -> [(source node id, parameter key, (pad_h, pad_w))]
        self._incoming_edges_by_target_node_id: dict[int, list[tuple[int, str, tuple[int, int]]]]
        self._incoming_edges_by_target_node_id = {
            node_gene.node_id: [] for node_gene in exact_genome.node_genes
        }
        for edge_gene in exact_genome.edge_genes:
            if not edge_gene.is_enabled:
                continue
            source_node = node_gene_by_id[edge_gene.source_node_id]
            target_node = node_gene_by_id[edge_gene.target_node_id]
            kernel_height, kernel_width = compute_convolution_kernel_size(
                source_node, target_node
            )
            if edge_gene.kernel_weights is not None:
                kernel_tensor = torch.tensor(
                    edge_gene.kernel_weights, dtype=torch.float32
                ).reshape(1, 1, kernel_height, kernel_width)
            else:
                he_standard_deviation = math.sqrt(
                    2.0 / fan_in_by_target_node_id[edge_gene.target_node_id]
                )
                kernel_tensor = (
                    torch.randn(1, 1, kernel_height, kernel_width) * he_standard_deviation
                )
            parameter_key = str(edge_gene.innovation_id)
            self._kernel_parameters[parameter_key] = nn.Parameter(kernel_tensor)
            padding = (
                max(0, target_node.filter_height - source_node.filter_height),
                max(0, target_node.filter_width - source_node.filter_width),
            )
            self._incoming_edges_by_target_node_id[edge_gene.target_node_id].append(
                (edge_gene.source_node_id, parameter_key, padding)
            )

        # Section VII: one batch-norm module per hidden node, seeded from the
        # state the genome carries so evaluation reproduces training.
        self._batch_normalization_by_node_id = nn.ModuleDict()
        if use_batch_normalization:
            for node_gene in exact_genome.node_genes:
                if node_gene.node_type != "hidden":
                    continue
                batch_normalization = nn.BatchNorm2d(
                    1, momentum=batch_normalization_alpha
                )
                if node_gene.batch_normalization_state is not None:
                    gamma, beta, running_mean, running_variance = (
                        node_gene.batch_normalization_state
                    )
                    with torch.no_grad():
                        batch_normalization.weight.fill_(gamma)
                        batch_normalization.bias.fill_(beta)
                        batch_normalization.running_mean.fill_(running_mean)
                        batch_normalization.running_var.fill_(running_variance)
                self._batch_normalization_by_node_id[str(node_gene.node_id)] = (
                    batch_normalization
                )

        self.to(device_for_computation)
        self.eval()

    def convolution_kernel_parameters(self) -> list[nn.Parameter]:
        """The per-edge kernels only.

        Returns:
            Every trainable convolution kernel — eq. 6's weight decay applies
            to these, not to the batch-normalization scale and shift.
        """
        return list(self._kernel_parameters.values())

    def _batch_normalization_for_node_id(self, node_id: int) -> nn.Module | None:
        """Return the node's batch-norm module, or ``None`` when it has none."""
        parameter_key = str(node_id)
        if parameter_key not in self._batch_normalization_by_node_id:
            return None
        return self._batch_normalization_by_node_id[parameter_key]

    def forward_pass(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Evaluate the CNN on a batch of flat images, keeping the autograd graph.

        Args:
            input_tensor: Shape ``(batch, height * width)`` — or a single
                flat image ``(height * width,)``, treated as a batch of one.

        Returns:
            Class logits of shape ``(batch, num_outputs)``, output columns
            in ascending output node id order. An output with no enabled
            incoming path yields 0.
        """
        input_tensor_on_device = input_tensor.to(self._device_for_computation)
        if input_tensor_on_device.dim() == 1:
            input_tensor_on_device = input_tensor_on_device.unsqueeze(0)
        batch_size = input_tensor_on_device.shape[0]
        feature_map_by_node_id: dict[int, torch.Tensor] = {
            self._input_node_id: input_tensor_on_device.reshape(
                batch_size, 1, self._input_image_height, self._input_image_width
            )
        }
        if self._input_dropout_probability > 0.0:
            feature_map_by_node_id[self._input_node_id] = functional.dropout(
                feature_map_by_node_id[self._input_node_id],
                p=self._input_dropout_probability,
                training=self.training,
            )

        for node_gene in self._node_genes_in_depth_order:
            if node_gene.node_type == "input":
                continue
            accumulated_map = torch.zeros(
                batch_size,
                1,
                node_gene.filter_height,
                node_gene.filter_width,
                device=self._device_for_computation,
            )
            for source_node_id, parameter_key, padding in (
                self._incoming_edges_by_target_node_id[node_gene.node_id]
            ):
                accumulated_map = accumulated_map + functional.conv2d(
                    feature_map_by_node_id[source_node_id],
                    self._kernel_parameters[parameter_key],
                    padding=padding,
                )
            if node_gene.node_type == "hidden":
                batch_normalization = self._batch_normalization_for_node_id(
                    node_gene.node_id
                )
                if batch_normalization is not None:
                    accumulated_map = batch_normalization(accumulated_map)
                accumulated_map = torch.clamp(
                    functional.leaky_relu(
                        accumulated_map, negative_slope=self._leaky_relu_negative_slope
                    ),
                    max=self._activation_clamp_maximum,
                )
                if self._hidden_dropout_probability > 0.0:
                    accumulated_map = functional.dropout(
                        accumulated_map,
                        p=self._hidden_dropout_probability,
                        training=self.training,
                    )
            feature_map_by_node_id[node_gene.node_id] = accumulated_map

        output_columns = [
            feature_map_by_node_id[output_node_id].reshape(batch_size)
            for output_node_id in self._output_node_ids_in_id_order
        ]
        return torch.stack(output_columns, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``nn.Module`` compatibility shim — delegates to ``forward_pass``."""
        return self.forward_pass(x)

    def reset_recurrent_state(self) -> None:
        return None

    def extract_genome_with_trained_weights(self) -> EXACTGenome:
        """Return a new genome whose enabled edges carry the current kernels.

        Disabled edge genes are carried over unchanged and every batch
        normalized node picks up its learned scale, shift and running
        statistics (section VII); the result is marked ``is_trained=True`` so
        the trainer skips it until reproduction resets the flag.

        Returns:
            A structurally identical :class:`EXACTGenome` with updated kernels
            and batch-normalization state.
        """
        edge_genes_with_trained_kernels = tuple(
            replace(
                edge_gene,
                kernel_weights=tuple(
                    tuple(float(value) for value in kernel_row)
                    for kernel_row in self._kernel_parameters[str(edge_gene.innovation_id)]
                    .detach()
                    .cpu()
                    .reshape(
                        self._kernel_parameters[str(edge_gene.innovation_id)].shape[2],
                        self._kernel_parameters[str(edge_gene.innovation_id)].shape[3],
                    )
                    .tolist()
                ),
            )
            if edge_gene.is_enabled
            else edge_gene
            for edge_gene in self._source_genome.edge_genes
        )
        node_genes_with_batch_normalization_state = tuple(
            self._node_gene_with_batch_normalization_state(node_gene)
            for node_gene in self._source_genome.node_genes
        )
        return replace(
            self._source_genome,
            node_genes=node_genes_with_batch_normalization_state,
            edge_genes=edge_genes_with_trained_kernels,
            is_trained=True,
        )

    def _node_gene_with_batch_normalization_state(
        self, node_gene: FilterNodeGene
    ) -> FilterNodeGene:
        """Copy a node's live batch-norm state back onto its gene.

        Args:
            node_gene: Node gene to update.

        Returns:
            The node gene with its state refreshed, or unchanged when the
            node has no batch-normalization module.
        """
        batch_normalization = self._batch_normalization_for_node_id(node_gene.node_id)
        if batch_normalization is None:
            return node_gene
        return replace(
            node_gene,
            batch_normalization_state=(
                float(batch_normalization.weight.item()),
                float(batch_normalization.bias.item()),
                float(batch_normalization.running_mean.item()),
                float(batch_normalization.running_var.item()),
            ),
        )
