"""NEAT's direct genetic encoding: node genes and connection genes.

Implements the genome representation of Stanley & Miikkulainen (2002),
section 3.1: a genome is a list of node genes plus a list of connection
genes, where each connection gene carries an innovation number (historical
marking, section 3.2) that lets crossover align corresponding genes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from polyneat.config.configuration_errors import ConfigurationError
from polyneat.core.type_aliases import InnovationId

NodeType = Literal["input", "hidden", "output", "bias"]
VALID_NODE_TYPES: tuple[NodeType, ...] = ("input", "hidden", "output", "bias")


class InvalidGenomeError(ValueError):
    """Raised when a NEATGenome is structurally invalid (missing node, cycle, ...)."""


@dataclass(frozen=True)
class NodeGene:
    """One node of the network (paper, section 3.1).

    Attributes:
        node_id: Unique id within the genome.
        node_type: One of ``input``, ``hidden``, ``output``, ``bias``. The
            bias node is an input that is always set to 1.0 (section 4.2).
        activation_function_name: Name resolved via
            :func:`polyneat.nn.activation_functions.resolve_activation_function_by_name`.
    """

    node_id: int
    node_type: NodeType
    activation_function_name: str

    def __post_init__(self) -> None:
        if self.node_type not in VALID_NODE_TYPES:
            raise InvalidGenomeError(
                f"NodeGene.node_type must be one of {VALID_NODE_TYPES}, "
                f"got '{self.node_type}' for node_id={self.node_id}"
            )


@dataclass(frozen=True)
class ConnectionGene:
    """One weighted directed connection (paper, section 3.1).

    Attributes:
        innovation_id: Historical marking (section 3.2). Genes carrying the
            same innovation id in two genomes represent the same structure
            and are aligned by crossover.
        source_node_id: Id of the node the connection leaves.
        target_node_id: Id of the node the connection enters.
        weight: Connection weight.
        is_enabled: Expression flag; ``AddNodeMutation`` disables the split
            connection instead of deleting it, so the gene can be re-enabled
            later.
    """

    innovation_id: InnovationId
    source_node_id: int
    target_node_id: int
    weight: float
    is_enabled: bool


@dataclass(frozen=True)
class NEATGenome:
    """Direct-encoding NEAT genome: tuple of nodes + tuple of connections.

    Immutable — every operator returns a new genome instead of mutating in
    place. ``__post_init__`` validates structural integrity (no duplicate
    node ids or innovation ids, no dangling connection endpoints).

    Attributes:
        node_genes: All node genes of the genome.
        connection_genes: All connection genes, enabled or not.

    Raises:
        InvalidGenomeError: If the genome is structurally invalid.
    """

    node_genes: tuple[NodeGene, ...]
    connection_genes: tuple[ConnectionGene, ...]

    def __post_init__(self) -> None:
        node_ids_seen: set[int] = set()
        for node_gene in self.node_genes:
            if node_gene.node_id in node_ids_seen:
                raise InvalidGenomeError(
                    f"NEATGenome contains duplicate node_id={node_gene.node_id}"
                )
            node_ids_seen.add(node_gene.node_id)

        innovation_ids_seen: set[InnovationId] = set()
        for connection_gene in self.connection_genes:
            if connection_gene.innovation_id in innovation_ids_seen:
                raise InvalidGenomeError(
                    f"NEATGenome contains duplicate innovation_id={connection_gene.innovation_id}"
                )
            innovation_ids_seen.add(connection_gene.innovation_id)
            if connection_gene.source_node_id not in node_ids_seen:
                raise InvalidGenomeError(
                    f"ConnectionGene innovation_id={connection_gene.innovation_id} "
                    f"references missing source_node_id={connection_gene.source_node_id}"
                )
            if connection_gene.target_node_id not in node_ids_seen:
                raise InvalidGenomeError(
                    f"ConnectionGene innovation_id={connection_gene.innovation_id} "
                    f"references missing target_node_id={connection_gene.target_node_id}"
                )

    def clone_genome(self) -> NEATGenome:
        """Return a structural copy (gene tuples are immutable and shared)."""
        return NEATGenome(
            node_genes=self.node_genes,
            connection_genes=self.connection_genes,
        )

    def get_node_gene_by_id(self, node_id: int) -> NodeGene | None:
        """Return the node gene with ``node_id``, or ``None`` when absent."""
        for node_gene in self.node_genes:
            if node_gene.node_id == node_id:
                return node_gene
        return None

    def get_connection_gene_by_innovation_id(
        self, innovation_id: InnovationId
    ) -> ConnectionGene | None:
        """Return the connection gene with ``innovation_id``, or ``None``."""
        for connection_gene in self.connection_genes:
            if connection_gene.innovation_id == innovation_id:
                return connection_gene
        return None

    def to_serializable_dict(self) -> dict:
        """Return a JSON-serializable dict round-trippable via ``from_serializable_dict``."""
        return {
            "node_genes": [
                {
                    "node_id": node_gene.node_id,
                    "node_type": node_gene.node_type,
                    "activation_function_name": node_gene.activation_function_name,
                }
                for node_gene in self.node_genes
            ],
            "connection_genes": [
                {
                    "innovation_id": connection_gene.innovation_id,
                    "source_node_id": connection_gene.source_node_id,
                    "target_node_id": connection_gene.target_node_id,
                    "weight": connection_gene.weight,
                    "is_enabled": connection_gene.is_enabled,
                }
                for connection_gene in self.connection_genes
            ],
        }

    @classmethod
    def from_serializable_dict(cls, payload: dict) -> NEATGenome:
        """Rebuild a genome from ``to_serializable_dict`` output.

        Args:
            payload: Dict with ``node_genes`` and ``connection_genes`` lists.

        Returns:
            The reconstructed genome.

        Raises:
            ConfigurationError: If a required key is missing.
            InvalidGenomeError: If the payload describes an invalid genome.
        """
        try:
            node_gene_payloads = payload["node_genes"]
            connection_gene_payloads = payload["connection_genes"]
        except KeyError as missing_key:
            raise ConfigurationError(
                f"NEATGenome.from_serializable_dict: missing key {missing_key}"
            ) from missing_key

        node_genes = tuple(
            NodeGene(
                node_id=node_payload["node_id"],
                node_type=node_payload["node_type"],
                activation_function_name=node_payload["activation_function_name"],
            )
            for node_payload in node_gene_payloads
        )
        connection_genes = tuple(
            ConnectionGene(
                innovation_id=connection_payload["innovation_id"],
                source_node_id=connection_payload["source_node_id"],
                target_node_id=connection_payload["target_node_id"],
                weight=connection_payload["weight"],
                is_enabled=connection_payload["is_enabled"],
            )
            for connection_payload in connection_gene_payloads
        )
        return cls(node_genes=node_genes, connection_genes=connection_genes)
