"""DeepNEAT as a ``NEATAlgorithm`` subclass (Miikkulainen et al., 2017).

DeepNEAT evolves network *topology*: each genome node is a whole layer with its
own hyperparameters, and edges carry tensors, not weights. Weights are never
inherited across generations - every genome is trained from scratch during
fitness evaluation, so all training lives in the evaluator (see
``polyneat.evaluators.trained_network_accuracy_evaluator``), not in the
generational loop. That is the reason ``advance_one_generation`` is **not**
overridden here, unlike EXACT (which is Lamarckian and writes trained weights
back into the genotype): DeepNEAT reuses the library's NEAT generational loop.
This supplies the source-required speciation, per-species truncation,
tournament selection and fitness-proportional species growth. The genetics
(genome encoding, mutation operators, historical-marking crossover and the
genome-to-phenotype decoder) are swapped in through the ``_build_*`` factories.
The optional c3 layer-hyperparameter distance remains a library extension and
is disabled by the source profile.

References:
    Miikkulainen, R., Liang, J., Meyerson, E., Rawal, A., Fink, D., Francon, O., Raju, B.,
        Shahrzad, H., Navruzyan, A., Duffy, N., & Hodjat, B. (2017). Evolving Deep Neural
        Networks. *arXiv:1703.00548*. Published in *Artificial Intelligence in the Age of
        Neural Networks and Brain Computing* (2019), pp. 293-312.
        DOI: 10.1016/B978-0-12-815480-9.00015-3
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
from numpy.random import Generator

from polyneat.algorithms.deepneat.deepneat_crossover import DeepNEATCrossover
from polyneat.algorithms.deepneat.deepneat_initial_population import (
    build_deepneat_initial_population,
)
from polyneat.algorithms.deepneat.deepneat_innovation_tracker import (
    DeepNEATInnovationTracker,
)
from polyneat.algorithms.deepneat.deepneat_phenotype_decoder import (
    DeepNEATPhenotypeDecoder,
)
from polyneat.algorithms.deepneat.deepneat_speciator import DeepNEATSpeciator
from polyneat.algorithms.deepneat.layer_shape_propagation import TensorShape
from polyneat.algorithms.deepneat.mutations.add_layer_node_mutation import (
    AddLayerNodeMutation,
)
from polyneat.algorithms.deepneat.mutations.add_tensor_edge_mutation import (
    AddTensorEdgeMutation,
)
from polyneat.algorithms.deepneat.mutations.deepneat_composite_mutation import (
    DeepNEATCompositeMutation,
)
from polyneat.algorithms.deepneat.mutations.global_hyperparameter_mutation import (
    GlobalHyperparameterMutation,
)
from polyneat.algorithms.deepneat.mutations.layer_hyperparameter_mutation import (
    LayerHyperparameterMutation,
)
from polyneat.algorithms.deepneat.mutations.toggle_tensor_edge_mutation import (
    ToggleTensorEdgeMutation,
)
from polyneat.configs.deepneat.deepneat_config import DeepNEATConfig
from polyneat.configs.neat.neat_config import NEATConfig
from polyneat.core.component_protocols import (
    CrossoverOperator,
    MutationOperator,
    PhenotypeDecoder,
    Speciator,
)
from polyneat.core.neat.global_innovation_tracker import GlobalInnovationTracker
from polyneat.core.neat.neat_algorithm import NEATAlgorithm
from polyneat.core.neat.neat_genome import NEATGenome
from polyneat.core.population import Population
from polyneat.logging_utils.custom_logger import get_logger

logger = get_logger(__name__)


@dataclass
class DeepNEATAlgorithm(NEATAlgorithm):
    """DeepNEAT: evolve layer-level topology, train each genome from scratch.

    Build it with ``DeepNEATAlgorithm.from_config(deepneat_config)``; the
    trainer that turns a decoded phenotype into a fitness value lives in the
    evaluator, not on this class - this class only ever hands out untrained
phenotypes via ``phenotype_decoder``.

    Generation 0 is fixed to a single input->output topology (a linear
    classifier), while each chromosome draws its global training and
    preprocessing genes independently. Because of this,
    ``create_initial_population`` is overridden outright rather than
    dispatched through the ``initial_population_strategy`` registry, and the
    inherited ``NEATConfig.initial_population_strategy`` field is therefore
    **not consulted** - see ``from_config``, which logs a warning when a
    config sets it away from its default so the setting does not silently
    do nothing.

    Attributes:
        config: Validated DeepNEAT hyperparameters.
    """

    config: DeepNEATConfig

    @classmethod
    def from_config(
        cls,
        config: NEATConfig,
        device_for_phenotype_computation: torch.device | None = None,
    ) -> DeepNEATAlgorithm:
        """Build DeepNEAT directly from its ``_build_*`` factories.

        Unlike the inherited template method, this does not resolve
        ``initial_population_strategy`` against the registry and does not set
        ``initial_population_factory``: DeepNEAT fixes its own generation-0
        topology (see ``create_initial_population``) and never consults it.
        A configured value other than the inherited default
        (``"fully_connected"``) is therefore logged as a warning - it is
        otherwise accepted and validated by ``DeepNEATConfig`` and then has no
        effect at all.

        Args:
            config: Validated DeepNEAT hyperparameters.
            device_for_phenotype_computation: Device override for the
                phenotype decoder. ``None`` falls back to
                ``config.device_for_phenotype_evaluation``.

        Returns:
            The wired algorithm.
        """
        deepneat_config = cast("DeepNEATConfig", config)

        # Compare against the config's *own* class default, not NEATConfig's:
        # DeepNEATConfig does not currently redeclare the field, so today the
        # two are the same value, but reading it off type(deepneat_config)
        # means this cannot silently start warning on every default config if
        # DeepNEATConfig ever does redeclare it with a different default.
        config_class_default_strategy = type(deepneat_config).initial_population_strategy
        if deepneat_config.initial_population_strategy != config_class_default_strategy:
            logger.warning(
                "DeepNEATConfig.initial_population_strategy=%r has no effect: DeepNEAT "
                "fixes generation 0 to the source-required minimal input->output "
                "topology and never consults this field.",
                deepneat_config.initial_population_strategy,
            )

        resolved_device = device_for_phenotype_computation or torch.device(
            deepneat_config.device_for_phenotype_evaluation
        )
        logger.info(
            "Building DeepNEATAlgorithm: population_size=%d, device=%s",
            deepneat_config.population_size,
            resolved_device,
        )

        # Mirrors NEATAlgorithm.from_config's construction (neat_algorithm.py)
        # field for field, minus initial_population_factory (deliberately
        # left at its dataclass default of None - see create_initial_population
        # below) - keep the two in step if NEATAlgorithm ever gains a new
        # required field.
        return cls(
            config=deepneat_config,
            mutation=cls._build_mutation(deepneat_config),
            crossover=cls._build_crossover(deepneat_config),
            parent_selection=cls._build_parent_selection(deepneat_config),
            speciator=cls._build_speciator(deepneat_config),
            innovation_tracker=cls._build_innovation_tracker(deepneat_config),
            _phenotype_decoder=cls._build_phenotype_decoder(deepneat_config, resolved_device),
        )

    @classmethod
    def _build_innovation_tracker(cls, config: NEATConfig) -> GlobalInnovationTracker:
        """Build the NEAT-style tracker with generation-local deduplication."""
        return DeepNEATInnovationTracker()

    @classmethod
    def _build_mutation(cls, config: NEATConfig) -> MutationOperator[NEATGenome]:
        """Compose the configured DeepNEAT mutation operators."""
        deepneat_config = cast("DeepNEATConfig", config)
        composite_mutation = DeepNEATCompositeMutation(
            ordered_individual_mutations=[
                GlobalHyperparameterMutation(
                    probability_of_application=(
                        deepneat_config.probability_of_global_hyperparameter_mutation
                    ),
                    config=deepneat_config,
                ),
                LayerHyperparameterMutation(
                    probability_of_application=(
                        deepneat_config.probability_of_layer_hyperparameter_mutation
                    ),
                    available_filter_counts=deepneat_config.available_filter_counts,
                    available_kernel_sizes=deepneat_config.available_kernel_sizes,
                    available_dense_unit_counts=deepneat_config.available_dense_unit_counts,
                    dropout_rate_min=deepneat_config.dropout_rate_min,
                    dropout_rate_max=deepneat_config.dropout_rate_max,
                    initial_weight_scaling_min=(
                        deepneat_config.initial_weight_scaling_min
                    ),
                    initial_weight_scaling_max=(
                        deepneat_config.initial_weight_scaling_max
                    ),
                    available_batch_normalization_options=(
                        deepneat_config.available_batch_normalization_options
                    ),
                    probability_of_new_conv_layer=(
                        deepneat_config.probability_of_new_conv_layer
                    ),
                    gaussian_mutation_standard_deviation_fraction=(
                        deepneat_config.gaussian_mutation_standard_deviation_fraction
                    ),
                    number_of_filters_min=deepneat_config.number_of_filters_min,
                    number_of_filters_max=deepneat_config.number_of_filters_max,
                ),
                AddTensorEdgeMutation(
                    probability_of_application=(
                        deepneat_config.probability_of_add_tensor_edge_mutation
                    ),
                ),
                AddLayerNodeMutation(
                    probability_of_application=(
                        deepneat_config.probability_of_add_layer_node_mutation
                    ),
                    available_filter_counts=deepneat_config.available_filter_counts,
                    available_kernel_sizes=deepneat_config.available_kernel_sizes,
                    available_dense_unit_counts=deepneat_config.available_dense_unit_counts,
                    dropout_rate_min=deepneat_config.dropout_rate_min,
                    dropout_rate_max=deepneat_config.dropout_rate_max,
                    probability_of_new_conv_layer=deepneat_config.probability_of_new_conv_layer,
                    initial_weight_scaling_min=(
                        deepneat_config.initial_weight_scaling_min
                    ),
                    initial_weight_scaling_max=(
                        deepneat_config.initial_weight_scaling_max
                    ),
                    available_batch_normalization_options=(
                        deepneat_config.available_batch_normalization_options
                    ),
                    number_of_filters_min=deepneat_config.number_of_filters_min,
                    number_of_filters_max=deepneat_config.number_of_filters_max,
                ),
                ToggleTensorEdgeMutation(
                    probability_of_application=(
                        deepneat_config.probability_of_toggle_tensor_edge_mutation
                    ),
                ),
            ]
        )
        return cast("MutationOperator[NEATGenome]", composite_mutation)

    @classmethod
    def _build_crossover(cls, config: NEATConfig) -> CrossoverOperator[NEATGenome]:
        """Build the innovation-aligned DeepNEAT crossover operator."""
        deepneat_config = cast("DeepNEATConfig", config)
        deepneat_crossover = DeepNEATCrossover(
            probability_of_inheriting_from_fitter_parent_for_matching_genes=(
                deepneat_config.probability_of_inheriting_from_fitter_parent_for_matching_genes
            ),
        )
        return cast("CrossoverOperator[NEATGenome]", deepneat_crossover)

    @classmethod
    def _build_speciator(cls, config: NEATConfig) -> Speciator[NEATGenome]:
        """Build NEAT-style speciation with an optional library-defined c3 term."""
        deepneat_config = cast("DeepNEATConfig", config)
        deepneat_speciator = DeepNEATSpeciator(
            coefficient_excess_c1=deepneat_config.compatibility_distance_coefficient_excess_c1,
            coefficient_disjoint_c2=(
                deepneat_config.compatibility_distance_coefficient_disjoint_c2
            ),
            coefficient_hyperparameter_c3=(
                deepneat_config.compatibility_distance_coefficient_weight_difference_c3
            ),
            compatibility_distance_threshold=(
                deepneat_config.compatibility_distance_threshold
            ),
            available_filter_counts=deepneat_config.available_filter_counts,
            available_kernel_sizes=deepneat_config.available_kernel_sizes,
            available_dense_unit_counts=deepneat_config.available_dense_unit_counts,
            dropout_rate_min=deepneat_config.dropout_rate_min,
            dropout_rate_max=deepneat_config.dropout_rate_max,
            initial_weight_scaling_min=deepneat_config.initial_weight_scaling_min,
            initial_weight_scaling_max=deepneat_config.initial_weight_scaling_max,
        )
        return cast("Speciator[NEATGenome]", deepneat_speciator)

    @classmethod
    def _build_phenotype_decoder(
        cls, config: NEATConfig, device: torch.device
    ) -> PhenotypeDecoder[NEATGenome]:
        """Build the layer-stack decoder bound to the config's image geometry."""
        deepneat_config = cast("DeepNEATConfig", config)
        deepneat_decoder = DeepNEATPhenotypeDecoder(
            input_shape=TensorShape.spatial(
                channels=deepneat_config.input_image_channels,
                height=deepneat_config.input_image_height,
                width=deepneat_config.input_image_width,
            ),
            number_of_classes=deepneat_config.number_of_classes,
            maximum_total_parameter_count=deepneat_config.maximum_total_parameter_count,
            device_for_computation=device,
        )
        return cast("PhenotypeDecoder[NEATGenome]", deepneat_decoder)

    def create_initial_population(self, rng: Generator) -> Population:
        """Build generation 0: minimal topology plus global hyperparameter genes.

        Every genome starts with the same minimal topology but independently
        sampled chromosome-wide hyperparameters. This bypasses
        ``initial_population_strategy``/``initial_population_factory``
        entirely; see the class docstring and ``from_config``.

        Args:
            rng: Source of randomness for global hyperparameter initialization.

        Returns:
            Generation-0 population of ``config.population_size`` minimal
            chromosomes.
        """
        deepneat_config = cast("DeepNEATConfig", self.config)
        return build_deepneat_initial_population(deepneat_config, self.innovation_tracker, rng)
