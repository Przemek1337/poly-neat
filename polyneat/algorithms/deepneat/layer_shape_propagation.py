"""Tensor-shape propagation and multi-input merge rules for DeepNEAT phenotypes.

This module is purely computational — no PyTorch — so the merge and degeneracy
rules are testable without building a network. Given a genome, it prunes the
nodes that do not sit on any input-to-output path and then walks the survivors
in topological order to compute the tensor shape flowing out of each layer.
That per-node shape is what lets the next stage build real ``torch.nn`` modules
with the right channel/feature counts.

Two of the rules implemented here are this library's own reconstruction, not
rules taken from the source paper. The paper settles that DeepNEAT nodes are
layers, edges carry no weight, and evolution operates over a graph of them —
but it describes DeepNEAT at a level that does not settle how multiple inputs
to one node are combined, or what happens when a convolutional layer receives
a non-spatial input. Both gaps must be resolved for any concrete genome to be
buildable at all, so this library fills them itself:

- **Multi-input merge (Decision #3).** A node with more than one enabled
  incoming edge: if every incoming tensor is spatial (4-D) and the node is a
  ``conv`` layer, each is adaptively average-pooled down to the smallest
  ``(height, width)`` among them and concatenated along the channel
  dimension. If any incoming tensor is flat, or the node is ``dense`` or
  ``output``, every incoming tensor is flattened and concatenated along the
  feature dimension instead.
- **Conv-on-flat coercion (Decision #4).** A ``conv`` layer that receives a
  flat input behaves as a ``dense`` layer with ``units = number_of_filters``.
  This guarantees that no mutation can ever produce a genome that cannot be
  built into a network, so the mutation operators never have to inspect the
  graph to avoid that outcome.

References:
    Miikkulainen, R., Liang, J., Meyerson, E., Rawal, A., Fink, D., Francon, O., Raju, B.,
        Shahrzad, H., Navruzyan, A., Duffy, N., & Hodjat, B. (2017). Evolving Deep Neural
        Networks. *arXiv:1703.00548*. Published in *Artificial Intelligence in the Age of
        Neural Networks and Brain Computing* (2019), pp. 293-312.
        DOI: 10.1016/B978-0-12-815480-9.00015-3
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass

from polyneat.algorithms.deepneat.deepneat_genome import DeepNEATGenome
from polyneat.nn.topology_utilities import (
    compute_topological_order_of_node_ids,
    find_node_ids_with_enabled_path_to_any_target,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TensorShape:
    """The shape of a tensor flowing along one edge of a DeepNEAT phenotype.

    A shape is either spatial (``[channels, height, width]``, as produced by a
    ``conv`` layer or the image input) or flat (``[features]``, as produced by
    a ``dense`` or ``output`` layer). Use :meth:`spatial` or :meth:`flat`
    rather than the constructor directly, so the unused fields are always
    zeroed consistently and two shapes describing the same tensor compare equal.

    Attributes:
        is_spatial: ``True`` for a ``[channels, height, width]`` tensor,
            ``False`` for a ``[features]`` tensor.
        channels: Channel count; ``0`` when not spatial.
        height: Spatial height; ``0`` when not spatial.
        width: Spatial width; ``0`` when not spatial.
        features: Feature count; ``0`` when spatial.
    """

    is_spatial: bool
    channels: int
    height: int
    width: int
    features: int

    @classmethod
    def spatial(cls, channels: int, height: int, width: int) -> TensorShape:
        """Build a ``[channels, height, width]`` shape.

        Args:
            channels: Channel count.
            height: Spatial height.
            width: Spatial width.

        Returns:
            The spatial shape.
        """
        return cls(is_spatial=True, channels=channels, height=height, width=width, features=0)

    @classmethod
    def flat(cls, features: int) -> TensorShape:
        """Build a ``[features]`` shape.

        Args:
            features: Feature count.

        Returns:
            The flat shape.
        """
        return cls(is_spatial=False, channels=0, height=0, width=0, features=features)

    @property
    def element_count(self) -> int:
        """Total number of scalar elements the tensor carries (batch excluded).

        Returns:
            ``channels * height * width`` when spatial, else ``features``.
        """
        if self.is_spatial:
            return self.channels * self.height * self.width
        return self.features


@dataclass(frozen=True)
class MergeStrategy:
    """How to combine several incoming tensors into the single input a layer needs.

    See the module docstring for Decision #3, the merge rule this describes.

    Attributes:
        flatten_inputs: ``True`` when every incoming tensor must be flattened
            and concatenated along features; ``False`` when every incoming
            tensor is spatial and can instead be pooled to a common
            ``(height, width)`` and concatenated along channels.
        pooled_height: Common height each spatial input is pooled to. Only
            meaningful when ``flatten_inputs`` is ``False``.
        pooled_width: Common width each spatial input is pooled to. Only
            meaningful when ``flatten_inputs`` is ``False``.
        merged_channels: Total channel count after concatenation. Only
            meaningful when ``flatten_inputs`` is ``False``.
        merged_features: Total feature count after concatenation. Only
            meaningful when ``flatten_inputs`` is ``True``.
    """

    flatten_inputs: bool
    pooled_height: int
    pooled_width: int
    merged_channels: int
    merged_features: int


def prune_to_nodes_on_an_input_output_path(genome: DeepNEATGenome) -> set[int]:
    """Return the node ids that sit on some enabled input-to-output path.

    A node contributes to the phenotype only if it is both reachable from the
    input and able to reach the output, along enabled edges. Anything else —
    a dead end fed by the input, or an orphan feeding into nothing that reaches
    the output — is dropped. The two reachability sets are computed with
    :func:`~polyneat.nn.topology_utilities.find_node_ids_with_enabled_path_to_any_target`:
    once directly for "reaches the output", and once over the reversed edge
    set for "is reachable from the input".

    Args:
        genome: The genome to prune.

    Returns:
        The retained node ids. Empty when no enabled path from input to output
        exists at all.
    """
    all_node_ids = [node.node_id for node in genome.node_genes]
    enabled_edges = [
        (edge.source_node_id, edge.target_node_id)
        for edge in genome.edge_genes
        if edge.is_enabled
    ]

    reaches_output = set(
        find_node_ids_with_enabled_path_to_any_target(
            candidate_source_node_ids=all_node_ids,
            target_node_ids=[genome.output_node_id],
            enabled_directed_edges=enabled_edges,
        )
    )
    reachable_from_input = set(
        find_node_ids_with_enabled_path_to_any_target(
            candidate_source_node_ids=all_node_ids,
            target_node_ids=[genome.input_node_id],
            enabled_directed_edges=[(target, source) for source, target in enabled_edges],
        )
    )

    retained_node_ids = reaches_output & reachable_from_input
    logger.info(
        "pruned genome to %d/%d nodes on an input-output path",
        len(retained_node_ids),
        len(all_node_ids),
    )
    return retained_node_ids


def compute_merge_strategy(
    incoming_shapes: list[TensorShape],
    target_layer_type: str,
) -> MergeStrategy:
    """Decide how to combine several incoming tensors for one target layer.

    Implements Decision #3 from the module docstring: flatten-and-concatenate
    unless every input is spatial and the target is a ``conv`` layer, in which
    case each input is pooled to the smallest ``(height, width)`` among them
    and concatenated along channels.

    Args:
        incoming_shapes: The shapes of the target's enabled incoming edges, in
            any order.
        target_layer_type: The target node's ``layer_type``.

    Returns:
        The merge strategy for combining ``incoming_shapes``.
    """
    must_flatten = target_layer_type in ("dense", "output") or any(
        not shape.is_spatial for shape in incoming_shapes
    )

    if must_flatten:
        return MergeStrategy(
            flatten_inputs=True,
            pooled_height=0,
            pooled_width=0,
            merged_channels=0,
            merged_features=sum(shape.element_count for shape in incoming_shapes),
        )

    return MergeStrategy(
        flatten_inputs=False,
        pooled_height=min(shape.height for shape in incoming_shapes),
        pooled_width=min(shape.width for shape in incoming_shapes),
        merged_channels=sum(shape.channels for shape in incoming_shapes),
        merged_features=0,
    )


def propagate_tensor_shapes(
    genome: DeepNEATGenome,
    retained_node_ids: set[int],
    input_shape: TensorShape,
    number_of_classes: int,
) -> dict[int, TensorShape]:
    """Compute the output tensor shape of every retained node.

    Walks ``retained_node_ids`` in topological order so that a node's incoming
    shapes are always known before the node itself is computed. Each node's
    incoming shapes are combined with :func:`compute_merge_strategy`, and then
    the node's own layer rule is applied:

    - ``input`` — always ``input_shape``.
    - ``conv`` on a spatial merge — ``spatial(number_of_filters, height, width)``,
      where ``(height, width)`` is the merge's pooled size, unchanged by the
      convolution itself because kernels use ``"same"`` padding (Task 2). If
      ``is_followed_by_max_pooling`` and both dimensions are at least 2, the
      size is halved (2x2 max pool, stride 2); otherwise pooling is skipped
      rather than shrinking a map to nothing.
    - ``conv`` on a flattened merge — coerced per Decision #4 to
      ``flat(number_of_filters)``.
    - ``dense`` — always ``flat(number_of_units)``, regardless of the merge.
    - ``output`` — always ``flat(number_of_classes)``, regardless of the merge.
      ``number_of_classes`` has no default: it is a property of the
      classification task being evolved, not of the genome, so a caller must
      supply it explicitly every time rather than have it silently default.

    Args:
        genome: The genome whose nodes to compute shapes for.
        retained_node_ids: The node ids to process, typically the result of
            :func:`prune_to_nodes_on_an_input_output_path`. Nodes outside this
            set, and edges touching them, are ignored.
        input_shape: The shape produced by the genome's ``input`` node.
        number_of_classes: The width of the ``output`` layer's tensor. Required,
            with no default, because it is task-dependent.

    Returns:
        A mapping from each retained node id to its output :class:`TensorShape`.
    """
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

    shapes_by_node_id: dict[int, TensorShape] = {}
    for node_id in topological_order:
        node_gene = genome.get_node_gene_by_id(node_id)

        if node_gene.layer_type == "input":
            shapes_by_node_id[node_id] = input_shape
            continue

        incoming_shapes = [
            shapes_by_node_id[source_id]
            for source_id in incoming_source_ids_by_target.get(node_id, [])
        ]
        merge_strategy = compute_merge_strategy(incoming_shapes, node_gene.layer_type)

        if node_gene.layer_type == "conv":
            if merge_strategy.flatten_inputs:
                # Decision #4: coerce to a dense layer so the genome stays buildable.
                shapes_by_node_id[node_id] = TensorShape.flat(features=node_gene.number_of_filters)
            else:
                height = merge_strategy.pooled_height
                width = merge_strategy.pooled_width
                if node_gene.is_followed_by_max_pooling and height >= 2 and width >= 2:
                    height //= 2
                    width //= 2
                shapes_by_node_id[node_id] = TensorShape.spatial(
                    channels=node_gene.number_of_filters, height=height, width=width
                )
        elif node_gene.layer_type == "dense":
            shapes_by_node_id[node_id] = TensorShape.flat(features=node_gene.number_of_units)
        else:  # "output"
            shapes_by_node_id[node_id] = TensorShape.flat(features=number_of_classes)

    return shapes_by_node_id
