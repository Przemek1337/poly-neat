"""EXACT crossover (Desell, 2017, section III-C)."""

from __future__ import annotations

from dataclasses import dataclass, replace

from numpy.random import Generator

from polyneat.algorithms.exact.exact_genome import (
    ConvolutionEdgeGene,
    EXACTGenome,
    FilterNodeGene,
    compute_convolution_kernel_size,
)
from polyneat.logging_utils.custom_logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class EXACTCrossover:
    """Mates two EXACT genomes, biased toward the fitter parent.

    Per the paper (section III-C): edges appearing in both parents are always
    inherited (from the fitter parent here, so its kernels carry over);
    edges only in the fitter parent are added at
    ``more_fit_parent_edge_inclusion_rate``, edges only in the less fit
    parent at ``less_fit_parent_edge_inclusion_rate``; edges skipped by the
    rate draw are still carried into the child **disabled**. Nodes are then
    added for each edge endpoint — from the fitter parent when it has the
    node, from the less fit parent otherwise. A kernel whose shape no longer
    matches ``|out_d - in_d| + 1`` under the inherited node sizes is reset to
    ``None`` (He-reinitialized at the child's next training). Node depths
    cannot disagree between parents: depth is fixed at node creation and no
    operator mutates it. The paper does not state the two rates' values.

    Carrying skipped edges over disabled can strand an output node, and the
    paper's discard rule applies to any such child (section III-B): the
    inclusion draw is retried up to ``maximum_attempts_for_reachable_child``
    times, and a clone of the fitter parent is returned when every attempt
    strands an output.

    Attributes:
        more_fit_parent_edge_inclusion_rate: Chance a fitter-parent-only
            edge is inherited enabled.
        less_fit_parent_edge_inclusion_rate: Chance a less-fit-parent-only
            edge is inherited enabled.
        maximum_attempts_for_reachable_child: Retry budget for the
            reachability discard rule.
    """

    more_fit_parent_edge_inclusion_rate: float
    less_fit_parent_edge_inclusion_rate: float
    maximum_attempts_for_reachable_child: int = 10

    def apply_to_parents(
        self,
        fitter_parent: EXACTGenome,
        less_fit_parent: EXACTGenome,
        rng: Generator,
        *,
        parents_have_equal_fitness: bool = False,
    ) -> EXACTGenome:
        """Return a child with every output reachable (section III-B discard rule).

        Args:
            fitter_parent: Parent with the higher fitness.
            less_fit_parent: Parent with the lower fitness.
            rng: Source of randomness for the inclusion draws.

        Returns:
            The child genome, marked untrained. When every attempt strands
            an output node, a clone of the fitter parent (marked untrained)
            is returned instead.
        """
        del parents_have_equal_fitness
        for _attempt in range(self.maximum_attempts_for_reachable_child):
            child_candidate = self._produce_child_candidate(
                fitter_parent=fitter_parent,
                less_fit_parent=less_fit_parent,
                rng=rng,
            )
            if child_candidate.are_all_output_nodes_reachable():
                return child_candidate
        logger.debug(
            "EXACT crossover exhausted %d attempts without a reachable child; "
            "returning a clone of the fitter parent",
            self.maximum_attempts_for_reachable_child,
        )
        return replace(fitter_parent, is_trained=False)

    def _produce_child_candidate(
        self,
        fitter_parent: EXACTGenome,
        less_fit_parent: EXACTGenome,
        rng: Generator,
    ) -> EXACTGenome:
        """Run one inclusion draw over both parents' edges, no reachability check.

        Args:
            fitter_parent: Parent with the higher fitness.
            less_fit_parent: Parent with the lower fitness.
            rng: Source of randomness for the inclusion draws.

        Returns:
            The candidate child genome, marked untrained.
        """
        fitter_edge_by_innovation_id = {
            edge_gene.innovation_id: edge_gene for edge_gene in fitter_parent.edge_genes
        }
        less_fit_edge_by_innovation_id = {
            edge_gene.innovation_id: edge_gene for edge_gene in less_fit_parent.edge_genes
        }

        child_edge_genes: list[ConvolutionEdgeGene] = []
        for innovation_id in sorted(
            fitter_edge_by_innovation_id.keys() | less_fit_edge_by_innovation_id.keys()
        ):
            fitter_edge = fitter_edge_by_innovation_id.get(innovation_id)
            less_fit_edge = less_fit_edge_by_innovation_id.get(innovation_id)
            if fitter_edge is not None and less_fit_edge is not None:
                child_edge_genes.append(fitter_edge)
            elif fitter_edge is not None:
                if rng.random() < self.more_fit_parent_edge_inclusion_rate:
                    child_edge_genes.append(fitter_edge)
                else:
                    child_edge_genes.append(replace(fitter_edge, is_enabled=False))
            else:
                if rng.random() < self.less_fit_parent_edge_inclusion_rate:
                    child_edge_genes.append(less_fit_edge)
                else:
                    child_edge_genes.append(replace(less_fit_edge, is_enabled=False))

        child_node_ids = {
            node_id
            for edge_gene in child_edge_genes
            for node_id in (edge_gene.source_node_id, edge_gene.target_node_id)
        }
        # Inputs and outputs always exist in the child, edge or no edge.
        child_node_ids.update(
            node_gene.node_id
            for node_gene in fitter_parent.node_genes
            if node_gene.node_type in ("input", "output")
        )
        child_node_genes: list[FilterNodeGene] = []
        for node_id in sorted(child_node_ids):
            node_from_fitter = fitter_parent.get_node_gene_by_id(node_id)
            child_node_genes.append(
                node_from_fitter
                if node_from_fitter is not None
                else less_fit_parent.get_node_gene_by_id(node_id)
            )
        child_node_gene_by_id = {
            node_gene.node_id: node_gene for node_gene in child_node_genes
        }

        sanitized_edge_genes: list[ConvolutionEdgeGene] = []
        for edge_gene in child_edge_genes:
            if edge_gene.kernel_weights is None:
                sanitized_edge_genes.append(edge_gene)
                continue
            expected_kernel_size = compute_convolution_kernel_size(
                child_node_gene_by_id[edge_gene.source_node_id],
                child_node_gene_by_id[edge_gene.target_node_id],
            )
            actual_kernel_size = (
                len(edge_gene.kernel_weights),
                len(edge_gene.kernel_weights[0]),
            )
            if actual_kernel_size == expected_kernel_size:
                sanitized_edge_genes.append(edge_gene)
            else:
                sanitized_edge_genes.append(replace(edge_gene, kernel_weights=None))

        return EXACTGenome(
            node_genes=tuple(child_node_genes),
            edge_genes=tuple(sanitized_edge_genes),
            is_trained=False,
        )
