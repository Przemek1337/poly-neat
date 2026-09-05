"""DeepNEAT's genome: nodes are layers, edges are tensor flow.

The difference from a NEAT genome is what a node and an edge *mean*. A node is a
whole layer carrying its own hyperparameters, not a single neuron; an edge is a
tensor being handed from one layer to another and carries **no weight**, because
weights live inside the layers and are trained by gradient descent while fitness
is being measured.

References:
    Miikkulainen, R., Liang, J., Meyerson, E., Rawal, A., Fink, D., Francon, O., Raju, B.,
        Shahrzad, H., Navruzyan, A., Duffy, N., & Hodjat, B. (2017). Evolving Deep Neural
        Networks. *arXiv:1703.00548*. Published in *Artificial Intelligence in the Age of
        Neural Networks and Brain Computing* (2019), pp. 293-312.
        DOI: 10.1016/B978-0-12-815480-9.00015-3
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field, replace
from typing import Literal

from polyneat.nn.topology_utilities import compute_topological_order_of_node_ids

LayerType = Literal["input", "conv", "dense", "output"]

_LAYER_TYPES: frozenset[str] = frozenset({"input", "conv", "dense", "output"})
_HYPERPARAMETER_FREE_LAYER_TYPES: frozenset[str] = frozenset({"input", "output"})


class InvalidDeepNEATGenomeError(ValueError):
    """Raised when a genome or a gene violates a structural invariant."""


@dataclass(frozen=True)
class DeepNEATGlobalHyperparameters:
    """Training and image-preprocessing genes carried by a chromosome.

    These are the global hyperparameters evolved in the CIFAR-10 DeepNEAT
    experiment (Liang, 2018, Table 3.1). A crop size of ``0`` is the
    backwards-compatible sentinel for "use the complete input image".
    """

    learning_rate: float = 1e-3
    momentum: float = 0.9
    hue_shift_degrees: float = 0.0
    saturation_value_shift: float = 0.0
    saturation_value_scale: float = 0.0
    cropped_image_size: int = 0
    spatial_scaling: float = 0.0
    uses_horizontal_flips: bool = False
    uses_variance_normalization: bool = False
    uses_nesterov_momentum: bool = False

    def __post_init__(self) -> None:
        if self.learning_rate <= 0.0:
            raise InvalidDeepNEATGenomeError(
                f"learning_rate must be > 0.0, got {self.learning_rate}"
            )
        if not (0.0 <= self.momentum < 1.0):
            raise InvalidDeepNEATGenomeError(
                f"momentum must be in [0.0, 1.0), got {self.momentum}"
            )
        for field_name in (
            "hue_shift_degrees",
            "saturation_value_shift",
            "saturation_value_scale",
            "spatial_scaling",
        ):
            value = getattr(self, field_name)
            if value < 0.0:
                raise InvalidDeepNEATGenomeError(
                    f"{field_name} must be >= 0.0, got {value}"
                )
        if self.cropped_image_size < 0:
            raise InvalidDeepNEATGenomeError(
                f"cropped_image_size must be >= 0, got {self.cropped_image_size}"
            )


@dataclass(frozen=True)
class LayerNodeGene:
    """One layer of the network, with the hyperparameters that define it.

    Fields are explicit rather than a dictionary so that invalid combinations
    are caught at construction, serialization stays trivial, and the speciator
    has a fixed field set to measure hyperparameter distance over.

    Attributes:
        node_id: Unique id within the genome; globally meaningful, issued by the
            innovation tracker.
        layer_type: ``"input"``, ``"conv"``, ``"dense"`` or ``"output"``.
        number_of_filters: Conv output channels; ``None`` for other types.
        kernel_size: Conv kernel side, odd so ``padding="same"`` is symmetric;
            ``None`` for other types.
        number_of_units: Dense output width; ``None`` for other types.
        dropout_rate: Dropout applied after the layer's activation.
        uses_batch_normalization: Whether the layer is batch-normalized.
        is_followed_by_max_pooling: Whether a 2x2 max pool follows a conv layer.
    """

    node_id: int
    layer_type: LayerType
    number_of_filters: int | None = None
    kernel_size: int | None = None
    number_of_units: int | None = None
    dropout_rate: float = 0.0
    initial_weight_scaling: float = 1.0
    uses_batch_normalization: bool = False
    is_followed_by_max_pooling: bool = False

    def __post_init__(self) -> None:
        """Reject hyperparameter combinations that do not describe a layer.

        Raises:
            InvalidDeepNEATGenomeError: On an unknown layer type, a missing
                required hyperparameter, or one that does not belong to the type.
        """
        if self.layer_type not in _LAYER_TYPES:
            raise InvalidDeepNEATGenomeError(
                f"unknown layer_type {self.layer_type!r}; known: {sorted(_LAYER_TYPES)}"
            )
        if not (0.0 <= self.dropout_rate < 1.0):
            raise InvalidDeepNEATGenomeError(
                f"dropout_rate must be in [0.0, 1.0), got {self.dropout_rate}"
            )
        if self.initial_weight_scaling < 0.0:
            raise InvalidDeepNEATGenomeError(
                "initial_weight_scaling must be >= 0.0, got "
                f"{self.initial_weight_scaling}"
            )

        if self.layer_type == "conv":
            if self.number_of_filters is None or self.kernel_size is None:
                raise InvalidDeepNEATGenomeError(
                    f"a conv layer needs both number_of_filters and kernel_size, got "
                    f"filters={self.number_of_filters}, kernel={self.kernel_size}"
                )
            if self.number_of_units is not None:
                raise InvalidDeepNEATGenomeError(
                    "a conv layer must not carry number_of_units"
                )
        elif self.layer_type == "dense":
            if self.number_of_units is None:
                raise InvalidDeepNEATGenomeError("a dense layer needs number_of_units")
            if self.number_of_filters is not None or self.kernel_size is not None:
                raise InvalidDeepNEATGenomeError(
                    "a dense layer must not carry conv hyperparameters"
                )
            if self.is_followed_by_max_pooling:
                raise InvalidDeepNEATGenomeError(
                    "max pooling only applies to conv layers"
                )
        else:
            carries_any = (
                self.number_of_filters is not None
                or self.kernel_size is not None
                or self.number_of_units is not None
                or self.dropout_rate != 0.0
                or self.initial_weight_scaling != 1.0
                or self.uses_batch_normalization
                or self.is_followed_by_max_pooling
            )
            if carries_any:
                raise InvalidDeepNEATGenomeError(
                    f"a {self.layer_type} layer carries no hyperparameters"
                )


@dataclass(frozen=True)
class TensorEdgeGene:
    """A tensor handed from one layer to another. Carries no weight.

    Attributes:
        innovation_id: Global historical marking, used to align genes in
            crossover exactly as in NEAT.
        source_node_id: Layer the tensor comes from.
        target_node_id: Layer the tensor goes to.
        is_enabled: Whether the edge participates in the phenotype.
    """

    innovation_id: int
    source_node_id: int
    target_node_id: int
    is_enabled: bool


@dataclass(frozen=True)
class DeepNEATGenome:
    """A directed acyclic graph of layers.

    Attributes:
        node_genes: The layers, exactly one ``input`` and one ``output``.
        edge_genes: The tensor flows between them.
    """

    node_genes: tuple[LayerNodeGene, ...]
    edge_genes: tuple[TensorEdgeGene, ...]
    global_hyperparameters: DeepNEATGlobalHyperparameters = field(
        default_factory=DeepNEATGlobalHyperparameters
    )

    def __post_init__(self) -> None:
        """Validate the graph invariants.

        Raises:
            InvalidDeepNEATGenomeError: On duplicate node ids or innovation ids,
                an edge endpoint that is not a node, a node count other than one
                input and one output, or a cycle among enabled edges.
        """
        node_ids = [node.node_id for node in self.node_genes]
        if len(set(node_ids)) != len(node_ids):
            raise InvalidDeepNEATGenomeError(f"duplicate node_id in {sorted(node_ids)}")

        innovation_ids = [edge.innovation_id for edge in self.edge_genes]
        if len(set(innovation_ids)) != len(innovation_ids):
            raise InvalidDeepNEATGenomeError(
                f"duplicate innovation_id in {sorted(innovation_ids)}"
            )

        known_node_ids = set(node_ids)
        for edge in self.edge_genes:
            if edge.source_node_id not in known_node_ids:
                raise InvalidDeepNEATGenomeError(
                    f"edge innov={edge.innovation_id} has unknown source "
                    f"{edge.source_node_id}"
                )
            if edge.target_node_id not in known_node_ids:
                raise InvalidDeepNEATGenomeError(
                    f"edge innov={edge.innovation_id} has unknown target "
                    f"{edge.target_node_id}"
                )

        input_node_count = sum(1 for node in self.node_genes if node.layer_type == "input")
        output_node_count = sum(1 for node in self.node_genes if node.layer_type == "output")
        if input_node_count != 1 or output_node_count != 1:
            raise InvalidDeepNEATGenomeError(
                f"a genome needs exactly one input and one output layer, got "
                f"{input_node_count} input(s) and {output_node_count} output(s)"
            )

        try:
            compute_topological_order_of_node_ids(
                all_node_ids=node_ids,
                enabled_directed_edges=[
                    (edge.source_node_id, edge.target_node_id)
                    for edge in self.edge_genes
                    if edge.is_enabled
                ],
            )
        except ValueError as cycle_error:
            raise InvalidDeepNEATGenomeError(
                f"enabled edges form a directed cycle: {cycle_error}"
            ) from cycle_error

    @property
    def input_node_id(self) -> int:
        """Node id of the single input layer."""
        return next(node.node_id for node in self.node_genes if node.layer_type == "input")

    @property
    def output_node_id(self) -> int:
        """Node id of the single output layer."""
        return next(node.node_id for node in self.node_genes if node.layer_type == "output")

    def count_structural_elements(self) -> tuple[int, int]:
        """Return ``(number_of_nodes, number_of_enabled_edges)``.

        The genome's structural complexity, aggregated per generation into the
        node/connection growth curves logged by the base generational loop.
        Named to match :meth:`NEATGenome.count_structural_elements` (and
        :meth:`EXACTGenome.count_structural_elements`) so
        ``NEATAlgorithm.advance_one_generation`` can call it on any genome
        type, DeepNEAT's included - this is what lets DeepNEAT reuse that loop
        verbatim without overriding it.
        """
        number_of_enabled_edges = sum(1 for edge_gene in self.edge_genes if edge_gene.is_enabled)
        return len(self.node_genes), number_of_enabled_edges

    def get_node_gene_by_id(self, node_id: int) -> LayerNodeGene | None:
        """Return the layer with ``node_id``, or ``None`` when absent.

        Args:
            node_id: Id to look up.

        Returns:
            The matching layer gene, or ``None``.
        """
        for node_gene in self.node_genes:
            if node_gene.node_id == node_id:
                return node_gene
        return None

    def is_output_reachable_from_input(self) -> bool:
        """Return whether a path of enabled edges runs from input to output.

        Returns:
            ``True`` when the graph computes anything at all. A genome without
            such a path decodes to a degenerate phenotype scoring zero.
        """
        outgoing_by_source: dict[int, list[int]] = {}
        for edge in self.edge_genes:
            if edge.is_enabled:
                outgoing_by_source.setdefault(edge.source_node_id, []).append(
                    edge.target_node_id
                )

        target_node_id = self.output_node_id
        frontier: deque[int] = deque([self.input_node_id])
        visited: set[int] = {self.input_node_id}
        while frontier:
            current_node_id = frontier.popleft()
            if current_node_id == target_node_id:
                return True
            for next_node_id in outgoing_by_source.get(current_node_id, []):
                if next_node_id not in visited:
                    visited.add(next_node_id)
                    frontier.append(next_node_id)
        return False

    def clone_genome(self) -> DeepNEATGenome:
        """Return an independent copy.

        Returns:
            A new genome equal to this one. Genes are frozen, so the tuples can
            be shared safely.
        """
        return replace(self)

    def to_serializable_dict(self) -> dict:
        """Return a JSON-friendly representation.

        Returns:
            A dict with ``layer_node_genes`` and ``tensor_edge_genes`` lists.
        """
        return {
            "layer_node_genes": [
                {
                    "node_id": node.node_id,
                    "layer_type": node.layer_type,
                    "number_of_filters": node.number_of_filters,
                    "kernel_size": node.kernel_size,
                    "number_of_units": node.number_of_units,
                    "dropout_rate": node.dropout_rate,
                    "initial_weight_scaling": node.initial_weight_scaling,
                    "uses_batch_normalization": node.uses_batch_normalization,
                    "is_followed_by_max_pooling": node.is_followed_by_max_pooling,
                }
                for node in self.node_genes
            ],
            "tensor_edge_genes": [
                {
                    "innovation_id": edge.innovation_id,
                    "source_node_id": edge.source_node_id,
                    "target_node_id": edge.target_node_id,
                    "is_enabled": edge.is_enabled,
                }
                for edge in self.edge_genes
            ],
            "global_hyperparameters": asdict(self.global_hyperparameters),
        }

    @classmethod
    def from_serializable_dict(cls, payload: dict) -> DeepNEATGenome:
        """Rebuild a genome from :meth:`to_serializable_dict` output.

        Args:
            payload: The dict produced by :meth:`to_serializable_dict`.

        Returns:
            The restored genome, revalidated by ``__post_init__``.
        """
        return cls(
            node_genes=tuple(
                LayerNodeGene(**node_payload) for node_payload in payload["layer_node_genes"]
            ),
            edge_genes=tuple(
                TensorEdgeGene(**edge_payload) for edge_payload in payload["tensor_edge_genes"]
            ),
            global_hyperparameters=DeepNEATGlobalHyperparameters(
                **payload.get("global_hyperparameters", {})
            ),
        )
