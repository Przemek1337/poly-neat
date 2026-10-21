"""Executable network built from a DeepNEAT genome: one ``nn.Module`` per layer node.

A DeepNEAT genome is decoded once, in the constructor, into a stack of real
``torch.nn`` modules wired together exactly as the genome's tensor edges say;
``forward_pass`` then replays that wiring in a single topological sweep. This
mirrors the shape of ``polyneat.algorithms.exact.torch_convolutional_phenotype
.TorchConvolutionalPhenotype``: build once, execute many times, keep the
autograd graph so a downstream trainer can call ``.backward()`` on the result.

Two things happen here that are not decided elsewhere:

- **Degeneracy without execution.** A genome is rejected when no enabled path
  from input to output survives pruning, its encoded layer types have
  incompatible tensor shapes, or an optional caller-supplied parameter budget
  is exceeded. Each case sets ``is_degenerate`` before ``forward_pass`` runs.
  The parameter limit is a library safety extension and can be disabled for a
  source-only run.
- **Faithful layer decoding.** Every retained node is expressed as the layer
  type encoded in the chromosome. A graph mixing incompatible spatial and flat
  tensors is marked degenerate; it is never made executable by silently
  replacing a convolution with a linear layer.

References:
    Miikkulainen, R., Liang, J., Meyerson, E., Rawal, A., Fink, D., Francon, O., Raju, B.,
        Shahrzad, H., Navruzyan, A., Duffy, N., & Hodjat, B. (2017). Evolving Deep Neural
        Networks. *arXiv:1703.00548*. Published in *Artificial Intelligence in the Age of
        Neural Networks and Brain Computing* (2019), pp. 293-312.
        DOI: 10.1016/B978-0-12-815480-9.00015-3
"""

from __future__ import annotations

from collections import defaultdict

import torch
import torch.nn.functional as functional
from torch import nn

from polyneat.algorithms.deepneat.deepneat_genome import DeepNEATGenome, LayerNodeGene
from polyneat.algorithms.deepneat.layer_shape_propagation import (
    IncompatibleTensorShapesError,
    MergeStrategy,
    TensorShape,
    compute_merge_strategy,
    propagate_tensor_shapes,
    prune_to_nodes_on_an_input_output_path,
)
from polyneat.logging_utils.custom_logger import get_logger
from polyneat.nn.topology_utilities import compute_topological_order_of_node_ids

logger = get_logger(__name__)


class TorchLayerStackPhenotype(nn.Module):
    """A DeepNEAT genome expressed as trainable ``torch.nn`` modules.

    Construction prunes the genome to the nodes that sit on some enabled
    input-to-output path, propagates tensor shapes along that pruned
    graph, and builds one ``nn.Module`` per retained non-input node from its
    :class:`~polyneat.algorithms.deepneat.deepneat_genome.LayerNodeGene`
    hyperparameters and its incoming
    :class:`~polyneat.algorithms.deepneat.layer_shape_propagation.MergeStrategy`.
    Modules live in an ``nn.ModuleDict`` keyed by ``str(node_id)`` so PyTorch
    registers their parameters. ``forward_pass`` then walks the same
    topological order once, merging each node's incoming tensors per its
    stored :class:`MergeStrategy` before applying its module.

    Strictly feed-forward; ``reset_recurrent_state`` is a no-op, present only
    because the :class:`~polyneat.core.component_protocols.Phenotype`
    protocol requires it.

    Attributes:
        is_degenerate: ``True`` when the genome has no enabled input-to-output
            path, has incompatible encoded layer shapes, or exceeds an enabled
            parameter budget.
            A degenerate phenotype's ``forward_pass`` returns zeros without
            executing any module.
        total_parameter_count: Number of scalar parameters across every built
            module, counted before the budget check — so it reports the true
            size of a genome even when that size is why it was rejected.
        number_of_layer_modules: How many ``nn.Module`` layers were built,
            counted the same way and at the same moment as
            ``total_parameter_count`` so the two can never disagree about
            what counts as a layer. This is the *expressed* layer count: the
            input node carries no module and is not counted, so it is one
            less than the number of retained nodes on the input-to-output
            path.
    """

    def __init__(
        self,
        genome: DeepNEATGenome,
        input_shape: TensorShape,
        number_of_classes: int,
        maximum_total_parameter_count: int | None,
        device_for_computation: torch.device,
    ) -> None:
        """Prune, propagate shapes, and build one module per surviving layer node.

        Args:
            genome: DeepNEAT genome to express.
            input_shape: Shape of the tensor ``forward_pass`` will be given,
                excluding the batch dimension.
            number_of_classes: Width of the output layer's logits.
            maximum_total_parameter_count: Optional parameter budget.
                Exceeding it marks the phenotype degenerate instead of
                raising. ``None`` disables this library safety extension.
            device_for_computation: Device every module and tensor lives on.
        """
        super().__init__()
        self._number_of_classes = number_of_classes
        self._device_for_computation = device_for_computation
        self.is_degenerate = False
        self.total_parameter_count = 0
        self.number_of_layer_modules = 0
        self._layer_modules_by_node_id = nn.ModuleDict()
        self.global_hyperparameters = genome.global_hyperparameters
        self._initial_weight_scaling_by_node_id: dict[int, float] = {}

        retained_node_ids = prune_to_nodes_on_an_input_output_path(genome)
        if not retained_node_ids:
            logger.info(
                "genome has no enabled input-to-output path; phenotype is degenerate"
            )
            self.is_degenerate = True
            self.to(device_for_computation)
            return

        effective_input_shape = input_shape
        cropped_image_size = genome.global_hyperparameters.cropped_image_size
        if input_shape.is_spatial and cropped_image_size > 0:
            effective_input_shape = TensorShape.spatial(
                channels=input_shape.channels,
                height=cropped_image_size,
                width=cropped_image_size,
            )
        try:
            shapes_by_node_id = propagate_tensor_shapes(
                genome=genome,
                retained_node_ids=retained_node_ids,
                input_shape=effective_input_shape,
                number_of_classes=number_of_classes,
            )
        except IncompatibleTensorShapesError as error:
            logger.info("genome has incompatible layer shapes; phenotype is degenerate: %s", error)
            self.is_degenerate = True
            self.to(device_for_computation)
            return

        enabled_edges_within_retained_nodes = [
            (edge.source_node_id, edge.target_node_id)
            for edge in genome.edge_genes
            if edge.is_enabled
            and edge.source_node_id in retained_node_ids
            and edge.target_node_id in retained_node_ids
        ]
        incoming_source_ids_by_target: dict[int, list[int]] = defaultdict(list)
        for source_node_id, target_node_id in enabled_edges_within_retained_nodes:
            incoming_source_ids_by_target[target_node_id].append(source_node_id)

        topological_order = compute_topological_order_of_node_ids(
            all_node_ids=retained_node_ids,
            enabled_directed_edges=enabled_edges_within_retained_nodes,
        )

        input_node_id = genome.input_node_id
        merge_strategy_by_node_id: dict[int, MergeStrategy] = {}
        layer_modules_by_node_id = nn.ModuleDict()

        for node_id in topological_order:
            if node_id == input_node_id:
                continue
            node_gene = genome.get_node_gene_by_id(node_id)
            incoming_shapes = [
                shapes_by_node_id[source_node_id]
                for source_node_id in incoming_source_ids_by_target.get(node_id, [])
            ]
            merge_strategy = compute_merge_strategy(incoming_shapes, node_gene.layer_type)
            merge_strategy_by_node_id[node_id] = merge_strategy
            layer_modules_by_node_id[str(node_id)] = self._build_layer_module(
                node_gene, merge_strategy
            )
            self._initial_weight_scaling_by_node_id[node_id] = (
                node_gene.initial_weight_scaling
            )

        total_parameter_count = sum(
            parameter.numel() for parameter in layer_modules_by_node_id.parameters()
        )

        self._topological_order = topological_order
        self._incoming_source_ids_by_target = dict(incoming_source_ids_by_target)
        self._merge_strategy_by_node_id = merge_strategy_by_node_id
        self._input_node_id = input_node_id
        self._output_node_id = genome.output_node_id
        self._layer_modules_by_node_id = layer_modules_by_node_id
        self.total_parameter_count = total_parameter_count
        self.number_of_layer_modules = len(layer_modules_by_node_id)

        if (
            maximum_total_parameter_count is not None
            and total_parameter_count > maximum_total_parameter_count
        ):
            logger.info(
                "phenotype has %d parameters, over the budget of %d; marking degenerate",
                total_parameter_count,
                maximum_total_parameter_count,
            )
            self.is_degenerate = True
            # Drop the built modules so the oversized genome's parameters are not
            # kept alive in memory once it has been rejected.
            self._layer_modules_by_node_id = nn.ModuleDict()

        self.to(device_for_computation)
        self._apply_initial_weight_scaling()

    def _build_layer_module(
        self, node_gene: LayerNodeGene, merge_strategy: MergeStrategy
    ) -> nn.Module:
        """Build the ``nn.Module`` for one retained non-input node.

        Args:
            node_gene: The node's hyperparameters.
            merge_strategy: How the node's incoming tensors are combined,
                from :func:`compute_merge_strategy`.

        Returns:
            An ``nn.Sequential`` for ``conv``/``dense`` nodes, or a bare
            ``nn.Linear`` for the ``output`` node (no activation, since
            ``CrossEntropyLoss`` expects logits).
        """
        if node_gene.layer_type == "conv":
            return self._build_conv_module(node_gene, merge_strategy)
        if node_gene.layer_type == "dense":
            return self._build_dense_like_module(
                in_features=merge_strategy.merged_features,
                out_features=node_gene.number_of_units,
                node_gene=node_gene,
            )
        # "output": bare linear projection to class logits, no activation.
        return nn.Linear(merge_strategy.merged_features, self._number_of_classes)

    def _build_conv_module(
        self, node_gene: LayerNodeGene, merge_strategy: MergeStrategy
    ) -> nn.Module:
        """Build a ``Conv2d`` block for a ``conv`` node on a spatial input."""
        layers: list[nn.Module] = [
            nn.Conv2d(
                merge_strategy.merged_channels,
                node_gene.number_of_filters,
                node_gene.kernel_size,
                padding="same",
            )
        ]
        if node_gene.uses_batch_normalization:
            layers.append(nn.BatchNorm2d(node_gene.number_of_filters))
        layers.append(nn.ReLU())
        if node_gene.dropout_rate > 0.0:
            layers.append(nn.Dropout(node_gene.dropout_rate))
        if (
            node_gene.is_followed_by_max_pooling
            and merge_strategy.pooled_height >= 2
            and merge_strategy.pooled_width >= 2
        ):
            layers.append(nn.MaxPool2d(2))
        return nn.Sequential(*layers)

    def _build_dense_like_module(
        self, in_features: int, out_features: int, node_gene: LayerNodeGene
    ) -> nn.Module:
        """Build a ``Linear`` block for a dense node."""
        layers: list[nn.Module] = [nn.Linear(in_features, out_features)]
        if node_gene.uses_batch_normalization:
            layers.append(nn.BatchNorm1d(out_features))
        layers.append(nn.ReLU())
        if node_gene.dropout_rate > 0.0:
            layers.append(nn.Dropout(node_gene.dropout_rate))
        return nn.Sequential(*layers)

    def _merge_incoming_tensors(
        self, incoming_tensors: list[torch.Tensor], merge_strategy: MergeStrategy
    ) -> torch.Tensor:
        """Combine a node's incoming tensors per its stored :class:`MergeStrategy`."""
        if merge_strategy.pooled_height > 0:
            pooled_tensors = [
                functional.adaptive_max_pool2d(
                    tensor, (merge_strategy.pooled_height, merge_strategy.pooled_width)
                )
                for tensor in incoming_tensors
            ]
            merged_tensor = torch.cat(pooled_tensors, dim=1)
        else:
            merged_tensor = torch.cat(incoming_tensors, dim=1)
        if merge_strategy.flatten_inputs:
            return torch.flatten(merged_tensor, 1)
        return merged_tensor

    def forward_pass(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Evaluate the network on a batch of images, keeping the autograd graph.

        Args:
            input_tensor: Shape ``(batch, channels, height, width)``.

        Returns:
            Class logits of shape ``(batch, number_of_classes)``. A degenerate
            phenotype returns zeros of that shape without executing any
            module.
        """
        if self.is_degenerate:
            batch_size = input_tensor.shape[0]
            return torch.zeros(
                batch_size, self._number_of_classes, device=self._device_for_computation
            )

        input_tensor_on_device = input_tensor.to(self._device_for_computation)
        tensors_by_node_id: dict[int, torch.Tensor] = {
            self._input_node_id: input_tensor_on_device
        }
        for node_id in self._topological_order:
            if node_id == self._input_node_id:
                continue
            incoming_tensors = [
                tensors_by_node_id[source_node_id]
                for source_node_id in self._incoming_source_ids_by_target.get(node_id, [])
            ]
            merged_tensor = self._merge_incoming_tensors(
                incoming_tensors, self._merge_strategy_by_node_id[node_id]
            )
            tensors_by_node_id[node_id] = self._layer_modules_by_node_id[str(node_id)](
                merged_tensor
            )

        return tensors_by_node_id[self._output_node_id]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``nn.Module`` compatibility shim — delegates to ``forward_pass``."""
        return self.forward_pass(x)

    def reinitialize_parameters(self) -> None:
        """Reset every trainable layer and normalization buffer in place.

        Phenotypes are constructed by the generic evolution runner before the
        fitness evaluator derives its per-generation seed.  Resetting here,
        after that seed has been installed, makes initial weights part of the
        evaluator's reproducibility contract instead of an accidental product
        of the global RNG state during phenotype decoding.
        """
        if self.is_degenerate:
            return
        for module in self.modules():
            if module is self:
                continue
            reset_parameters = getattr(module, "reset_parameters", None)
            if callable(reset_parameters):
                reset_parameters()
        self._apply_initial_weight_scaling()

    def _apply_initial_weight_scaling(self) -> None:
        """Apply every hidden layer's evolved initial-weight multiplier."""
        with torch.no_grad():
            for node_id, scaling in self._initial_weight_scaling_by_node_id.items():
                module_key = str(node_id)
                if module_key not in self._layer_modules_by_node_id:
                    continue
                module = self._layer_modules_by_node_id[module_key]
                if not isinstance(module, nn.Sequential):
                    continue
                trainable_layer = module[0]
                weight = getattr(trainable_layer, "weight", None)
                if weight is not None:
                    weight.mul_(scaling)

    def reset_recurrent_state(self) -> None:
        return None
