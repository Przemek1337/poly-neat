"""Innovation-aligned crossover for DeepNEAT chromosomes."""

from __future__ import annotations

from dataclasses import fields

from numpy.random import Generator

from polyneat.algorithms.deepneat.deepneat_genome import (
    DeepNEATGenome,
    DeepNEATGlobalHyperparameters,
    LayerNodeGene,
    TensorEdgeGene,
)
from polyneat.logging_utils.custom_logger import get_logger
from polyneat.nn.topology_utilities import would_directed_edge_create_cycle

logger = get_logger(__name__)


class DeepNEATCrossover:
    """Cross layer chromosomes using NEAT's historical markings."""

    def __init__(
        self,
        probability_of_inheriting_from_fitter_parent_for_matching_genes: float,
        probability_of_child_gene_remaining_disabled_when_either_parent_disabled: float = 0.75,
    ) -> None:
        self._probability_of_inheriting_from_fitter_parent = (
            probability_of_inheriting_from_fitter_parent_for_matching_genes
        )
        self._probability_of_child_gene_remaining_disabled = (
            probability_of_child_gene_remaining_disabled_when_either_parent_disabled
        )

    def apply_to_parents(
        self,
        fitter_parent: DeepNEATGenome,
        less_fit_parent: DeepNEATGenome,
        rng: Generator,
        *,
        parents_have_equal_fitness: bool = False,
    ) -> DeepNEATGenome:
        """Return an acyclic child, handling equal-fitness parents symmetrically."""
        fitter_edges = {edge.innovation_id: edge for edge in fitter_parent.edge_genes}
        other_edges = {edge.innovation_id: edge for edge in less_fit_parent.edge_genes}
        inherited_edges: list[TensorEdgeGene] = []

        for innovation_id in sorted(fitter_edges.keys() | other_edges.keys()):
            fitter_edge = fitter_edges.get(innovation_id)
            other_edge = other_edges.get(innovation_id)
            if fitter_edge is None or other_edge is None:
                unique_edge = fitter_edge or other_edge
                if unique_edge is None:
                    continue
                if fitter_edge is None and not parents_have_equal_fitness:
                    continue
                if parents_have_equal_fitness and rng.random() >= 0.5:
                    continue
                inherited_edges.append(unique_edge)
                continue

            take_from_fitter = (
                rng.random() < self._probability_of_inheriting_from_fitter_parent
            )
            chosen_edge = fitter_edge if take_from_fitter else other_edge
            if not fitter_edge.is_enabled or not other_edge.is_enabled:
                chosen_edge = TensorEdgeGene(
                    innovation_id=chosen_edge.innovation_id,
                    source_node_id=chosen_edge.source_node_id,
                    target_node_id=chosen_edge.target_node_id,
                    is_enabled=not (
                        rng.random()
                        < self._probability_of_child_gene_remaining_disabled
                    ),
                )
            inherited_edges.append(chosen_edge)

        inherited_edges = self._drop_edges_closing_a_cycle(inherited_edges)
        fitter_nodes = {node.node_id: node for node in fitter_parent.node_genes}
        other_nodes = {node.node_id: node for node in less_fit_parent.node_genes}
        required_node_ids = {fitter_parent.input_node_id, fitter_parent.output_node_id}
        for edge in inherited_edges:
            required_node_ids.update((edge.source_node_id, edge.target_node_id))

        inherited_nodes: list[LayerNodeGene] = []
        for node_id in sorted(required_node_ids):
            fitter_node = fitter_nodes.get(node_id)
            other_node = other_nodes.get(node_id)
            if fitter_node is None:
                chosen_node = other_node
            elif other_node is None or other_node.layer_type != fitter_node.layer_type:
                chosen_node = fitter_node
            else:
                chosen_node = (
                    fitter_node
                    if rng.random() < self._probability_of_inheriting_from_fitter_parent
                    else other_node
                )
            if chosen_node is not None:
                inherited_nodes.append(chosen_node)

        return DeepNEATGenome(
            node_genes=tuple(inherited_nodes),
            edge_genes=tuple(inherited_edges),
            global_hyperparameters=self._cross_global_hyperparameters(
                fitter_parent.global_hyperparameters,
                less_fit_parent.global_hyperparameters,
                rng,
            ),
        )

    def _cross_global_hyperparameters(
        self,
        fitter: DeepNEATGlobalHyperparameters,
        other: DeepNEATGlobalHyperparameters,
        rng: Generator,
    ) -> DeepNEATGlobalHyperparameters:
        values = {
            descriptor.name: (
                getattr(fitter, descriptor.name)
                if rng.random() < self._probability_of_inheriting_from_fitter_parent
                else getattr(other, descriptor.name)
            )
            for descriptor in fields(DeepNEATGlobalHyperparameters)
        }
        return DeepNEATGlobalHyperparameters(**values)

    @staticmethod
    def _drop_edges_closing_a_cycle(
        candidate_edges: list[TensorEdgeGene],
    ) -> list[TensorEdgeGene]:
        accepted_enabled_edges: list[tuple[int, int]] = []
        resolved_edges: list[TensorEdgeGene] = []
        for edge in sorted(candidate_edges, key=lambda gene: gene.innovation_id):
            if not edge.is_enabled:
                resolved_edges.append(edge)
                continue
            if would_directed_edge_create_cycle(
                candidate_source_node_id=edge.source_node_id,
                candidate_target_node_id=edge.target_node_id,
                existing_enabled_edges=accepted_enabled_edges,
            ):
                logger.debug(
                    "DeepNEATCrossover disabled innov=%d to keep the child acyclic",
                    edge.innovation_id,
                )
                resolved_edges.append(
                    TensorEdgeGene(
                        innovation_id=edge.innovation_id,
                        source_node_id=edge.source_node_id,
                        target_node_id=edge.target_node_id,
                        is_enabled=False,
                    )
                )
                continue
            accepted_enabled_edges.append((edge.source_node_id, edge.target_node_id))
            resolved_edges.append(edge)
        return resolved_edges
