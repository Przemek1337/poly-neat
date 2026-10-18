"""Tests for the track B reset of EXACT's Lamarckian parameters."""

from __future__ import annotations

import torch

from polyneat.algorithms.exact.exact_backpropagation_trainer import (
    EXACTBackpropagationTrainer,
)
from polyneat.algorithms.exact.exact_genome import (
    ConvolutionEdgeGene,
    EXACTGenome,
    FilterNodeGene,
)
from polyneat.algorithms.exact.exact_parameter_reset import (
    genome_carries_learned_parameters,
    reset_genome_for_fresh_training,
)
from polyneat.algorithms.exact.exact_training_hyperparameters import (
    EXACTTrainingHyperparameters,
)

_CPU = torch.device("cpu")


def _two_class_genome() -> EXACTGenome:
    return EXACTGenome(
        node_genes=(
            FilterNodeGene(
                node_id=0, node_type="input", filter_height=2, filter_width=2, depth=0.0
            ),
            FilterNodeGene(
                node_id=3, node_type="hidden", filter_height=2, filter_width=2, depth=0.5
            ),
            FilterNodeGene(
                node_id=1, node_type="output", filter_height=1, filter_width=1, depth=1.0
            ),
            FilterNodeGene(
                node_id=2, node_type="output", filter_height=1, filter_width=1, depth=1.0
            ),
        ),
        edge_genes=(
            ConvolutionEdgeGene(
                innovation_id=0, source_node_id=0, target_node_id=3, is_enabled=True
            ),
            ConvolutionEdgeGene(
                innovation_id=1, source_node_id=3, target_node_id=1, is_enabled=True
            ),
            ConvolutionEdgeGene(
                innovation_id=2, source_node_id=3, target_node_id=2, is_enabled=True
            ),
        ),
    )


def _separable_data() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    features = torch.cat([torch.randn(16, 4) - 2.0, torch.randn(16, 4) + 2.0])
    labels = torch.cat([torch.zeros(16, dtype=torch.long), torch.ones(16, dtype=torch.long)])
    return features, labels


def _trainer(features: torch.Tensor, labels: torch.Tensor) -> EXACTBackpropagationTrainer:
    return EXACTBackpropagationTrainer(
        training_features=features,
        training_labels=labels,
        input_image_height=2,
        input_image_width=2,
        number_of_training_epochs=2,
        leaky_relu_negative_slope=0.01,
        activation_clamp_maximum=50.0,
        use_batch_normalization=True,
        default_hyperparameters=EXACTTrainingHyperparameters(
            learning_rate=0.05,
            learning_rate_decay_factor=0.98,
            momentum=0.5,
            momentum_decay_factor=0.95,
            weight_decay=0.00001,
            weight_decay_decay_factor=0.98,
            velocity_reset_interval=0,
            input_dropout_probability=0.0,
            hidden_dropout_probability=0.0,
            batch_size=8,
            batch_normalization_alpha=0.1,
        ),
        minimum_learning_rate=1e-5,
        maximum_momentum=0.99,
        minimum_weight_decay=0.0,
        device_for_computation=_CPU,
        use_epigenetic_weight_initialization=True,
    )


def _trained_genome() -> EXACTGenome:
    features, labels = _separable_data()
    return _trainer(features, labels).train_genome(_two_class_genome())


class TestExactParameterReset:
    def test_a_trained_genome_really_carries_learned_parameters(self) -> None:
        trained = _trained_genome()
        assert trained.is_trained
        assert genome_carries_learned_parameters(trained)
        assert any(edge.kernel_weights is not None for edge in trained.edge_genes)
        assert any(node.batch_normalization_state is not None for node in trained.node_genes)

    def test_reset_clears_every_learned_field(self) -> None:
        reset = reset_genome_for_fresh_training(_trained_genome())
        assert reset.is_trained is False
        assert all(edge.kernel_weights is None for edge in reset.edge_genes)
        assert all(node.batch_normalization_state is None for node in reset.node_genes)
        assert not genome_carries_learned_parameters(reset)

    def test_reset_leaves_the_track_a_genome_untouched(self) -> None:
        trained = _trained_genome()
        original_payload = trained.to_serializable_dict()
        reset_genome_for_fresh_training(trained)
        assert trained.to_serializable_dict() == original_payload
        assert genome_carries_learned_parameters(trained)

    def test_reset_preserves_the_topology(self) -> None:
        trained = _trained_genome()
        reset = reset_genome_for_fresh_training(trained)
        assert [node.node_id for node in reset.node_genes] == [
            node.node_id for node in trained.node_genes
        ]
        assert [(edge.source_node_id, edge.target_node_id, edge.is_enabled) for edge in
                reset.edge_genes] == [
            (edge.source_node_id, edge.target_node_id, edge.is_enabled)
            for edge in trained.edge_genes
        ]

    def test_clearing_kernels_alone_would_not_be_enough(self) -> None:
        """The documented gap: dropping kernels leaves batch-norm state behind."""
        from dataclasses import replace

        trained = _trained_genome()
        kernels_only = replace(
            trained,
            edge_genes=tuple(
                replace(edge, kernel_weights=None) for edge in trained.edge_genes
            ),
        )
        assert genome_carries_learned_parameters(kernels_only)
        assert not genome_carries_learned_parameters(
            reset_genome_for_fresh_training(trained)
        )

    def test_a_reset_genome_is_trained_again_rather_than_skipped(self) -> None:
        features, labels = _separable_data()
        trained = _trained_genome()
        reset = reset_genome_for_fresh_training(trained)

        retrained = _trainer(features, labels).train_genome(reset)
        assert retrained is not reset, "a reset genome must not be skipped as already trained"
        assert retrained.is_trained
        assert any(edge.kernel_weights is not None for edge in retrained.edge_genes)

        original_kernels = [edge.kernel_weights for edge in trained.edge_genes]
        retrained_kernels = [edge.kernel_weights for edge in retrained.edge_genes]
        assert original_kernels != retrained_kernels

    def test_a_still_trained_genome_is_skipped_by_the_trainer(self) -> None:
        features, labels = _separable_data()
        trained = _trained_genome()
        assert _trainer(features, labels).train_genome(trained) is trained
