"""Re-draw one layer's hyperparameters from the configured search space.

References:
    Miikkulainen, R., et al. (2017). Evolving Deep Neural Networks. *arXiv:1703.00548*.
        DOI: 10.1016/B978-0-12-815480-9.00015-3
"""

from __future__ import annotations

from numpy.random import Generator

from polyneat.algorithms.deepneat.deepneat_genome import DeepNEATGenome, LayerNodeGene
from polyneat.core.component_protocols import InnovationTracker
from polyneat.logging_utils.custom_logger import get_logger

logger = get_logger(__name__)


def draw_conv_layer_hyperparameters(
    node_id: int,
    rng: Generator,
    available_filter_counts: tuple[int, ...],
    available_kernel_sizes: tuple[int, ...],
    dropout_rate_min: float,
    dropout_rate_max: float,
) -> LayerNodeGene:
    """Sample a conv layer uniformly from the search space.

    Args:
        node_id: Id to give the new gene.
        rng: Source of randomness.
        available_filter_counts: Allowed output-channel counts.
        available_kernel_sizes: Allowed kernel sides.
        dropout_rate_min: Lower end of the dropout range.
        dropout_rate_max: Upper end of the dropout range.

    Returns:
        A freshly sampled conv `LayerNodeGene`.
    """
    return LayerNodeGene(
        node_id=node_id,
        layer_type="conv",
        number_of_filters=int(
            available_filter_counts[int(rng.integers(0, len(available_filter_counts)))]
        ),
        kernel_size=int(
            available_kernel_sizes[int(rng.integers(0, len(available_kernel_sizes)))]
        ),
        dropout_rate=float(rng.uniform(dropout_rate_min, dropout_rate_max)),
        uses_batch_normalization=bool(rng.random() < 0.5),
        is_followed_by_max_pooling=bool(rng.random() < 0.5),
    )


def draw_dense_layer_hyperparameters(
    node_id: int,
    rng: Generator,
    available_dense_unit_counts: tuple[int, ...],
    dropout_rate_min: float,
    dropout_rate_max: float,
) -> LayerNodeGene:
    """Sample a dense layer uniformly from the search space.

    Args:
        node_id: Id to give the new gene.
        rng: Source of randomness.
        available_dense_unit_counts: Allowed widths.
        dropout_rate_min: Lower end of the dropout range.
        dropout_rate_max: Upper end of the dropout range.

    Returns:
        A freshly sampled dense `LayerNodeGene`.
    """
    return LayerNodeGene(
        node_id=node_id,
        layer_type="dense",
        number_of_units=int(
            available_dense_unit_counts[
                int(rng.integers(0, len(available_dense_unit_counts)))
            ]
        ),
        dropout_rate=float(rng.uniform(dropout_rate_min, dropout_rate_max)),
        uses_batch_normalization=bool(rng.random() < 0.5),
    )


class LayerHyperparameterMutation:
    """Re-draws the hyperparameters of one randomly chosen hidden layer.

    Input and output layers carry no hyperparameters, so they are never
    candidates. The layer's *type* is preserved: changing conv into dense would
    be a structural change, and structure is the other operators' business.
    """

    def __init__(
        self,
        probability_of_application: float,
        available_filter_counts: tuple[int, ...],
        available_kernel_sizes: tuple[int, ...],
        available_dense_unit_counts: tuple[int, ...],
        dropout_rate_min: float,
        dropout_rate_max: float,
    ) -> None:
        """Store the firing probability and the search space.

        Args:
            probability_of_application: Chance the operator fires on a genome.
            available_filter_counts: Allowed conv output-channel counts.
            available_kernel_sizes: Allowed conv kernel sides.
            available_dense_unit_counts: Allowed dense widths.
            dropout_rate_min: Lower end of the dropout range.
            dropout_rate_max: Upper end of the dropout range.
        """
        self._probability_of_application = probability_of_application
        self._available_filter_counts = available_filter_counts
        self._available_kernel_sizes = available_kernel_sizes
        self._available_dense_unit_counts = available_dense_unit_counts
        self._dropout_rate_min = dropout_rate_min
        self._dropout_rate_max = dropout_rate_max

    def apply_to_genome(
        self,
        genome: DeepNEATGenome,
        rng: Generator,
        innovation_tracker: InnovationTracker,
    ) -> DeepNEATGenome:
        """Return a copy with one hidden layer's hyperparameters re-drawn.

        Args:
            genome: Genome to mutate; never modified in place.
            rng: Source of randomness.
            innovation_tracker: Unused - no structure is created.

        Returns:
            The mutated genome, or the original when the operator does not fire
            or the genome has no hidden layer.
        """
        if rng.random() >= self._probability_of_application:
            return genome

        mutable_positions = [
            position
            for position, node in enumerate(genome.node_genes)
            if node.layer_type in ("conv", "dense")
        ]
        if not mutable_positions:
            return genome

        position_to_mutate = mutable_positions[
            int(rng.integers(0, len(mutable_positions)))
        ]
        existing_node = genome.node_genes[position_to_mutate]
        if existing_node.layer_type == "conv":
            replacement_node = draw_conv_layer_hyperparameters(
                node_id=existing_node.node_id,
                rng=rng,
                available_filter_counts=self._available_filter_counts,
                available_kernel_sizes=self._available_kernel_sizes,
                dropout_rate_min=self._dropout_rate_min,
                dropout_rate_max=self._dropout_rate_max,
            )
        else:
            replacement_node = draw_dense_layer_hyperparameters(
                node_id=existing_node.node_id,
                rng=rng,
                available_dense_unit_counts=self._available_dense_unit_counts,
                dropout_rate_min=self._dropout_rate_min,
                dropout_rate_max=self._dropout_rate_max,
            )

        logger.debug(
            "LayerHyperparameterMutation re-drew node %d (%s)",
            existing_node.node_id,
            existing_node.layer_type,
        )
        return DeepNEATGenome(
            node_genes=tuple(
                replacement_node if position == position_to_mutate else node
                for position, node in enumerate(genome.node_genes)
            ),
            edge_genes=genome.edge_genes,
        )
