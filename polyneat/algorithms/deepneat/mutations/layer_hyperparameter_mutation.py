"""DeepNEAT layer-hyperparameter initialization and mutation.

Real-valued genes are perturbed with Gaussian noise and binary genes are
flipped, following Liang (2018), Chapter 3. Structural layer addition still
draws an initial value uniformly from the configured search space.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from numpy.random import Generator

from polyneat.algorithms.deepneat.deepneat_genome import DeepNEATGenome, LayerNodeGene
from polyneat.core.component_protocols import InnovationTracker
from polyneat.logging_utils.custom_logger import get_logger

logger = get_logger(__name__)


def _draw_choice(values: tuple, rng: Generator):  # noqa: ANN202
    return values[int(rng.integers(0, len(values)))]


def _clip_gaussian(
    value: float,
    minimum: float,
    maximum: float,
    standard_deviation_fraction: float,
    rng: Generator,
) -> float:
    span = maximum - minimum
    if span == 0.0:
        return minimum
    return float(
        np.clip(
            value + rng.normal(0.0, standard_deviation_fraction * span),
            minimum,
            maximum,
        )
    )


def _mutate_discrete_numeric(
    value: int,
    choices: tuple[int, ...],
    standard_deviation_fraction: float,
    rng: Generator,
) -> int:
    if len(choices) == 1:
        return choices[0]
    perturbed = _clip_gaussian(
        float(value),
        float(min(choices)),
        float(max(choices)),
        standard_deviation_fraction,
        rng,
    )
    return min(choices, key=lambda choice: abs(choice - perturbed))


def draw_conv_layer_hyperparameters(
    node_id: int,
    rng: Generator,
    available_filter_counts: tuple[int, ...],
    available_kernel_sizes: tuple[int, ...],
    dropout_rate_min: float,
    dropout_rate_max: float,
    initial_weight_scaling_min: float = 0.0,
    initial_weight_scaling_max: float = 2.0,
    available_batch_normalization_options: tuple[bool, ...] = (False,),
    number_of_filters_min: int | None = None,
    number_of_filters_max: int | None = None,
) -> LayerNodeGene:
    """Draw the initial genes of a newly inserted convolutional layer."""
    if number_of_filters_min is None:
        number_of_filters = int(_draw_choice(available_filter_counts, rng))
    else:
        assert number_of_filters_max is not None
        number_of_filters = int(rng.integers(number_of_filters_min, number_of_filters_max + 1))
    return LayerNodeGene(
        node_id=node_id,
        layer_type="conv",
        number_of_filters=number_of_filters,
        kernel_size=int(_draw_choice(available_kernel_sizes, rng)),
        dropout_rate=float(rng.uniform(dropout_rate_min, dropout_rate_max)),
        initial_weight_scaling=float(
            rng.uniform(initial_weight_scaling_min, initial_weight_scaling_max)
        ),
        uses_batch_normalization=bool(_draw_choice(available_batch_normalization_options, rng)),
        is_followed_by_max_pooling=bool(rng.random() < 0.5),
    )


def draw_dense_layer_hyperparameters(
    node_id: int,
    rng: Generator,
    available_dense_unit_counts: tuple[int, ...],
    dropout_rate_min: float,
    dropout_rate_max: float,
    initial_weight_scaling_min: float = 0.0,
    initial_weight_scaling_max: float = 2.0,
    available_batch_normalization_options: tuple[bool, ...] = (False,),
) -> LayerNodeGene:
    """Draw the initial genes of a newly inserted fully-connected layer."""
    return LayerNodeGene(
        node_id=node_id,
        layer_type="dense",
        number_of_units=int(_draw_choice(available_dense_unit_counts, rng)),
        dropout_rate=float(rng.uniform(dropout_rate_min, dropout_rate_max)),
        initial_weight_scaling=float(
            rng.uniform(initial_weight_scaling_min, initial_weight_scaling_max)
        ),
        uses_batch_normalization=bool(_draw_choice(available_batch_normalization_options, rng)),
    )


class LayerHyperparameterMutation:
    """Mutate one gene of one hidden layer with type-appropriate semantics."""

    def __init__(
        self,
        probability_of_application: float,
        available_filter_counts: tuple[int, ...],
        available_kernel_sizes: tuple[int, ...],
        available_dense_unit_counts: tuple[int, ...],
        dropout_rate_min: float,
        dropout_rate_max: float,
        initial_weight_scaling_min: float = 0.0,
        initial_weight_scaling_max: float = 2.0,
        available_batch_normalization_options: tuple[bool, ...] = (False,),
        probability_of_new_conv_layer: float = 0.5,
        gaussian_mutation_standard_deviation_fraction: float = 0.1,
        number_of_filters_min: int | None = None,
        number_of_filters_max: int | None = None,
    ) -> None:
        self._probability_of_application = probability_of_application
        self._available_filter_counts = available_filter_counts
        self._available_kernel_sizes = available_kernel_sizes
        self._available_dense_unit_counts = available_dense_unit_counts
        self._dropout_rate_min = dropout_rate_min
        self._dropout_rate_max = dropout_rate_max
        self._initial_weight_scaling_min = initial_weight_scaling_min
        self._initial_weight_scaling_max = initial_weight_scaling_max
        self._available_batch_normalization_options = available_batch_normalization_options
        self._probability_of_new_conv_layer = probability_of_new_conv_layer
        self._standard_deviation_fraction = gaussian_mutation_standard_deviation_fraction
        self._number_of_filters_min = number_of_filters_min
        self._number_of_filters_max = number_of_filters_max

    def _draw_layer(self, node_id: int, layer_type: str, rng: Generator) -> LayerNodeGene:
        if layer_type == "conv":
            return draw_conv_layer_hyperparameters(
                node_id=node_id,
                rng=rng,
                available_filter_counts=self._available_filter_counts,
                available_kernel_sizes=self._available_kernel_sizes,
                dropout_rate_min=self._dropout_rate_min,
                dropout_rate_max=self._dropout_rate_max,
                initial_weight_scaling_min=self._initial_weight_scaling_min,
                initial_weight_scaling_max=self._initial_weight_scaling_max,
                available_batch_normalization_options=(self._available_batch_normalization_options),
                number_of_filters_min=self._number_of_filters_min,
                number_of_filters_max=self._number_of_filters_max,
            )
        return draw_dense_layer_hyperparameters(
            node_id=node_id,
            rng=rng,
            available_dense_unit_counts=self._available_dense_unit_counts,
            dropout_rate_min=self._dropout_rate_min,
            dropout_rate_max=self._dropout_rate_max,
            initial_weight_scaling_min=self._initial_weight_scaling_min,
            initial_weight_scaling_max=self._initial_weight_scaling_max,
            available_batch_normalization_options=(self._available_batch_normalization_options),
        )

    def _mutate_node(self, node: LayerNodeGene, rng: Generator) -> LayerNodeGene:
        mutable_fields = ["dropout_rate", "initial_weight_scaling"]
        if len(self._available_batch_normalization_options) > 1:
            mutable_fields.append("uses_batch_normalization")
        if 0.0 < self._probability_of_new_conv_layer < 1.0:
            mutable_fields.append("layer_type")
        if node.layer_type == "conv":
            mutable_fields.extend(
                ["number_of_filters", "kernel_size", "is_followed_by_max_pooling"]
            )
        else:
            mutable_fields.append("number_of_units")

        field_name = str(_draw_choice(tuple(mutable_fields), rng))
        if field_name == "layer_type":
            replacement_type = "dense" if node.layer_type == "conv" else "conv"
            return self._draw_layer(node.node_id, replacement_type, rng)
        if field_name == "dropout_rate":
            return replace(
                node,
                dropout_rate=_clip_gaussian(
                    node.dropout_rate,
                    self._dropout_rate_min,
                    self._dropout_rate_max,
                    self._standard_deviation_fraction,
                    rng,
                ),
            )
        if field_name == "initial_weight_scaling":
            return replace(
                node,
                initial_weight_scaling=_clip_gaussian(
                    node.initial_weight_scaling,
                    self._initial_weight_scaling_min,
                    self._initial_weight_scaling_max,
                    self._standard_deviation_fraction,
                    rng,
                ),
            )
        if field_name == "uses_batch_normalization":
            alternatives = tuple(
                value
                for value in self._available_batch_normalization_options
                if value != node.uses_batch_normalization
            )
            return replace(
                node,
                uses_batch_normalization=bool(_draw_choice(alternatives, rng)),
            )
        if field_name == "is_followed_by_max_pooling":
            return replace(
                node,
                is_followed_by_max_pooling=not node.is_followed_by_max_pooling,
            )
        if field_name == "kernel_size":
            alternatives = tuple(
                value for value in self._available_kernel_sizes if value != node.kernel_size
            )
            if not alternatives:
                return node
            return replace(node, kernel_size=int(_draw_choice(alternatives, rng)))
        if field_name == "number_of_filters":
            if self._number_of_filters_min is not None:
                assert self._number_of_filters_max is not None
                value = int(
                    round(
                        _clip_gaussian(
                            float(node.number_of_filters),
                            float(self._number_of_filters_min),
                            float(self._number_of_filters_max),
                            self._standard_deviation_fraction,
                            rng,
                        )
                    )
                )
            else:
                value = _mutate_discrete_numeric(
                    int(node.number_of_filters),
                    self._available_filter_counts,
                    self._standard_deviation_fraction,
                    rng,
                )
            return replace(node, number_of_filters=value)
        return replace(
            node,
            number_of_units=_mutate_discrete_numeric(
                int(node.number_of_units),
                self._available_dense_unit_counts,
                self._standard_deviation_fraction,
                rng,
            ),
        )

    def apply_to_genome(
        self,
        genome: DeepNEATGenome,
        rng: Generator,
        innovation_tracker: InnovationTracker,
    ) -> DeepNEATGenome:
        del innovation_tracker
        if rng.random() >= self._probability_of_application:
            return genome
        mutable_positions = [
            position
            for position, node in enumerate(genome.node_genes)
            if node.layer_type in ("conv", "dense")
        ]
        if not mutable_positions:
            return genome
        position = int(_draw_choice(tuple(mutable_positions), rng))
        existing_node = genome.node_genes[position]
        replacement_node = self._mutate_node(existing_node, rng)
        logger.debug(
            "LayerHyperparameterMutation changed node %d (%s -> %s)",
            existing_node.node_id,
            existing_node.layer_type,
            replacement_node.layer_type,
        )
        return replace(
            genome,
            node_genes=tuple(
                replacement_node if index == position else node
                for index, node in enumerate(genome.node_genes)
            ),
        )
