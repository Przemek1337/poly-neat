"""EXACT's genetic encoding: filter node genes and convolution edge genes.

Implements the genome representation of Desell (2017), section III: a CNN's
structure is captured by the sizes of its filters and how they are
connected, since any two filters are connectable by a convolution of size
``|out_d - in_d| + 1`` in each dimension. Unlike NEAT, both nodes and edges
carry innovation numbers (section III-B); the node id doubles as the node's
historical marking. Edges optionally carry trained kernels so children can
reuse parental weights (epigenetic weight initialization, section III-D).

References:
    Desell, T. (2017). Developing a Volunteer Computing Project to Evolve
        Convolutional Neural Networks and Their Hyperparameters. 2017 IEEE
        13th International Conference on e-Science, pp. 19-28.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from polyneat.algorithms.exact.exact_training_hyperparameters import (
    EXACTTrainingHyperparameters,
)
from polyneat.configs.configuration_errors import ConfigurationError
from polyneat.core.type_aliases import InnovationId

FilterNodeType = Literal["input", "hidden", "output"]
VALID_FILTER_NODE_TYPES: tuple[FilterNodeType, ...] = ("input", "hidden", "output")

# One convolution kernel as an immutable (rows, columns) grid of floats.
KernelWeights = tuple[tuple[float, ...], ...]


class InvalidEXACTGenomeError(ValueError):
    """Raised when an EXACTGenome is structurally invalid."""


def compute_convolution_kernel_size(
    source_node: FilterNodeGene, target_node: FilterNodeGene
) -> tuple[int, int]:
    """Kernel size connecting two filters: ``|out_d - in_d| + 1`` per dimension.

    The observation EXACT is built on (paper, section III): with this kernel
    size (and zero padding of ``max(0, out_d - in_d)``) a convolution maps a
    filter of size ``in_d`` exactly onto a filter of size ``out_d``.

    Args:
        source_node: The edge's input filter.
        target_node: The edge's output filter.

    Returns:
        ``(kernel_height, kernel_width)``.
    """
    return (
        abs(target_node.filter_height - source_node.filter_height) + 1,
        abs(target_node.filter_width - source_node.filter_width) + 1,
    )


def _rebuild_batch_normalization_state(
    state_payload: list | None,
) -> tuple[float, float, float, float] | None:
    """Rebuild a node's ``(gamma, beta, running_mean, running_var)`` tuple.

    Args:
        state_payload: The serialized four-element list, or ``None`` — which
            is also what artifacts written before the field existed yield.

    Returns:
        The state tuple, or ``None`` when the node carries no state.
    """
    if state_payload is None:
        return None
    gamma, beta, running_mean, running_variance = (
        float(value) for value in state_payload
    )
    return (gamma, beta, running_mean, running_variance)


@dataclass(frozen=True)
class FilterNodeGene:
    """One filter of the CNN (paper, section III).

    Attributes:
        node_id: Unique id within the genome, issued by the innovation
            tracker — EXACT requires innovation numbers for nodes as well as
            edges (section III-B), and the node id is that marking.
        node_type: ``input`` (the image), ``hidden`` (a filter), or
            ``output`` (a 1x1 softmax class node).
        filter_height: Filter size in y; the image height for the input node.
        filter_width: Filter size in x; the image width for the input node.
        depth: Position in the forward pass, assigned at creation and never
            mutated. Edges only run from lower to higher depth, which keeps
            the network feed-forward and lets propagation run in a linear
            sweep without graph traversal (section III-B).
        batch_normalization_state: Per-node batch normalization state
            ``(gamma, beta, running_mean, running_var)`` learned during
            training (section VII); ``None`` until first trained.
    """

    node_id: int
    node_type: FilterNodeType
    filter_height: int
    filter_width: int
    depth: float
    batch_normalization_state: tuple[float, float, float, float] | None = None

    def __post_init__(self) -> None:
        if self.node_type not in VALID_FILTER_NODE_TYPES:
            raise InvalidEXACTGenomeError(
                f"FilterNodeGene.node_type must be one of {VALID_FILTER_NODE_TYPES}, "
                f"got '{self.node_type}' for node_id={self.node_id}"
            )
        if self.filter_height < 1 or self.filter_width < 1:
            raise InvalidEXACTGenomeError(
                f"FilterNodeGene filter size must be >= 1 in both dimensions, got "
                f"{self.filter_height}x{self.filter_width} for node_id={self.node_id}"
            )
        if self.node_type == "output" and (self.filter_height, self.filter_width) != (1, 1):
            raise InvalidEXACTGenomeError(
                f"output nodes are 1x1 softmax class nodes (paper, section III-A), got "
                f"{self.filter_height}x{self.filter_width} for node_id={self.node_id}"
            )


@dataclass(frozen=True)
class ConvolutionEdgeGene:
    """One convolution between two filters (paper, section III).

    Attributes:
        innovation_id: Historical marking aligning edges in crossover
            (section III-C).
        source_node_id: Id of the filter the convolution reads from.
        target_node_id: Id of the filter the convolution feeds.
        is_enabled: Expression flag; the disable-edge mutation keeps the
            gene in the genome (section III-B).
        kernel_weights: Trained kernel of shape ``|out_d - in_d| + 1`` per
            dimension, carried for epigenetic weight initialization
            (section III-D). ``None`` means structurally new or invalidated —
            the phenotype He-initializes it at decode time.
    """

    innovation_id: InnovationId
    source_node_id: int
    target_node_id: int
    is_enabled: bool
    kernel_weights: KernelWeights | None = None


@dataclass(frozen=True)
class EXACTGenome:
    """EXACT genome: filter nodes + convolution edges, immutable.

    Every operator returns a new genome instead of mutating in place.
    ``__post_init__`` validates structural integrity: unique ids, existing
    endpoints, strict depth ordering on every edge, and stored kernels
    matching the ``|out_d - in_d| + 1`` shape their endpoints imply.

    Attributes:
        node_genes: All filter node genes.
        edge_genes: All convolution edge genes, enabled or not.
        is_trained: Whether the stored kernels are the result of a completed
            backpropagation session. Reproduction operators reset this to
            ``False``; the trainer skips genomes where it is ``True`` so
            elites are not retrained.
        training_hyperparameters: The genome's own backpropagation
            hyperparameters (section IV); ``None`` means the trainer's
            config-derived defaults apply.

    Raises:
        InvalidEXACTGenomeError: If the genome is structurally invalid.
    """

    node_genes: tuple[FilterNodeGene, ...]
    edge_genes: tuple[ConvolutionEdgeGene, ...]
    is_trained: bool = False
    training_hyperparameters: EXACTTrainingHyperparameters | None = None

    def __post_init__(self) -> None:
        node_gene_by_id: dict[int, FilterNodeGene] = {}
        for node_gene in self.node_genes:
            if node_gene.node_id in node_gene_by_id:
                raise InvalidEXACTGenomeError(
                    f"EXACTGenome contains duplicate node_id={node_gene.node_id}"
                )
            node_gene_by_id[node_gene.node_id] = node_gene

        innovation_ids_seen: set[InnovationId] = set()
        for edge_gene in self.edge_genes:
            if edge_gene.innovation_id in innovation_ids_seen:
                raise InvalidEXACTGenomeError(
                    f"EXACTGenome contains duplicate innovation_id={edge_gene.innovation_id}"
                )
            innovation_ids_seen.add(edge_gene.innovation_id)
            source_node = node_gene_by_id.get(edge_gene.source_node_id)
            target_node = node_gene_by_id.get(edge_gene.target_node_id)
            if source_node is None or target_node is None:
                raise InvalidEXACTGenomeError(
                    f"ConvolutionEdgeGene innovation_id={edge_gene.innovation_id} references "
                    f"missing node ({edge_gene.source_node_id} -> {edge_gene.target_node_id})"
                )
            if source_node.depth >= target_node.depth:
                raise InvalidEXACTGenomeError(
                    f"ConvolutionEdgeGene innovation_id={edge_gene.innovation_id} does not "
                    f"feed forward: source depth {source_node.depth} >= target depth "
                    f"{target_node.depth}"
                )
            if edge_gene.kernel_weights is not None:
                expected_kernel_size = compute_convolution_kernel_size(source_node, target_node)
                actual_height = len(edge_gene.kernel_weights)
                actual_widths = {len(kernel_row) for kernel_row in edge_gene.kernel_weights}
                if actual_height != expected_kernel_size[0] or actual_widths != {
                    expected_kernel_size[1]
                }:
                    raise InvalidEXACTGenomeError(
                        f"ConvolutionEdgeGene innovation_id={edge_gene.innovation_id} kernel "
                        f"shape does not match |out_d - in_d| + 1 = {expected_kernel_size}"
                    )

    def clone_genome(self) -> EXACTGenome:
        """Return a structural copy (gene tuples are immutable and shared)."""
        return EXACTGenome(
            node_genes=self.node_genes,
            edge_genes=self.edge_genes,
            is_trained=self.is_trained,
            training_hyperparameters=self.training_hyperparameters,
        )

    def get_node_gene_by_id(self, node_id: int) -> FilterNodeGene | None:
        """Return the node gene with ``node_id``, or ``None`` when absent."""
        for node_gene in self.node_genes:
            if node_gene.node_id == node_id:
                return node_gene
        return None

    def get_edge_gene_by_innovation_id(
        self, innovation_id: InnovationId
    ) -> ConvolutionEdgeGene | None:
        """Return the edge gene with ``innovation_id``, or ``None``."""
        for edge_gene in self.edge_genes:
            if edge_gene.innovation_id == innovation_id:
                return edge_gene
        return None

    def are_all_output_nodes_reachable(self) -> bool:
        """Whether every output node is reachable from the input via enabled edges.

        The disable-edge mutation can strand an output; such child genomes
        are discarded and regenerated (paper, section III-B).
        """
        enabled_target_ids_by_source_id: dict[int, list[int]] = {}
        for edge_gene in self.edge_genes:
            if edge_gene.is_enabled:
                enabled_target_ids_by_source_id.setdefault(
                    edge_gene.source_node_id, []
                ).append(edge_gene.target_node_id)

        frontier = [
            node_gene.node_id
            for node_gene in self.node_genes
            if node_gene.node_type == "input"
        ]
        reachable_node_ids = set(frontier)
        while frontier:
            current_node_id = frontier.pop()
            for target_node_id in enabled_target_ids_by_source_id.get(current_node_id, ()):
                if target_node_id not in reachable_node_ids:
                    reachable_node_ids.add(target_node_id)
                    frontier.append(target_node_id)
        return all(
            node_gene.node_id in reachable_node_ids
            for node_gene in self.node_genes
            if node_gene.node_type == "output"
        )

    def to_serializable_dict(self) -> dict:
        """Return a JSON-serializable dict round-trippable via ``from_serializable_dict``."""
        return {
            "node_genes": [
                {
                    "node_id": node_gene.node_id,
                    "node_type": node_gene.node_type,
                    "filter_height": node_gene.filter_height,
                    "filter_width": node_gene.filter_width,
                    "depth": node_gene.depth,
                    "batch_normalization_state": (
                        None
                        if node_gene.batch_normalization_state is None
                        else list(node_gene.batch_normalization_state)
                    ),
                }
                for node_gene in self.node_genes
            ],
            "edge_genes": [
                {
                    "innovation_id": edge_gene.innovation_id,
                    "source_node_id": edge_gene.source_node_id,
                    "target_node_id": edge_gene.target_node_id,
                    "is_enabled": edge_gene.is_enabled,
                    "kernel_weights": (
                        None
                        if edge_gene.kernel_weights is None
                        else [list(kernel_row) for kernel_row in edge_gene.kernel_weights]
                    ),
                }
                for edge_gene in self.edge_genes
            ],
            "is_trained": self.is_trained,
            "training_hyperparameters": (
                None
                if self.training_hyperparameters is None
                else self.training_hyperparameters.to_serializable_dict()
            ),
        }

    @classmethod
    def from_serializable_dict(cls, payload: dict) -> EXACTGenome:
        """Rebuild a genome from ``to_serializable_dict`` output.

        Args:
            payload: Dict with ``node_genes``, ``edge_genes`` and ``is_trained``.

        Returns:
            The reconstructed genome.

        Raises:
            ConfigurationError: If a required key is missing.
            InvalidEXACTGenomeError: If the payload describes an invalid genome.
        """
        try:
            node_gene_payloads = payload["node_genes"]
            edge_gene_payloads = payload["edge_genes"]
            is_trained = payload["is_trained"]
        except KeyError as missing_key:
            raise ConfigurationError(
                f"EXACTGenome.from_serializable_dict: missing key {missing_key}"
            ) from missing_key

        node_genes = tuple(
            FilterNodeGene(
                node_id=node_payload["node_id"],
                node_type=node_payload["node_type"],
                filter_height=node_payload["filter_height"],
                filter_width=node_payload["filter_width"],
                depth=node_payload["depth"],
                batch_normalization_state=_rebuild_batch_normalization_state(
                    node_payload.get("batch_normalization_state")
                ),
            )
            for node_payload in node_gene_payloads
        )
        edge_genes = tuple(
            ConvolutionEdgeGene(
                innovation_id=edge_payload["innovation_id"],
                source_node_id=edge_payload["source_node_id"],
                target_node_id=edge_payload["target_node_id"],
                is_enabled=edge_payload["is_enabled"],
                kernel_weights=(
                    None
                    if edge_payload["kernel_weights"] is None
                    else tuple(
                        tuple(float(value) for value in kernel_row)
                        for kernel_row in edge_payload["kernel_weights"]
                    )
                ),
            )
            for edge_payload in edge_gene_payloads
        )
        training_hyperparameters_payload = payload.get("training_hyperparameters")
        return cls(
            node_genes=node_genes,
            edge_genes=edge_genes,
            is_trained=is_trained,
            training_hyperparameters=(
                None
                if training_hyperparameters_payload is None
                else EXACTTrainingHyperparameters.from_serializable_dict(
                    training_hyperparameters_payload
                )
            ),
        )
