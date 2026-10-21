from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from polyneat.algorithms.exact.exact_backpropagation_trainer import (
    EXACTBackpropagationTrainer,
)
from polyneat.algorithms.exact.exact_genome import (
    ConvolutionEdgeGene,
    EXACTGenome,
    FilterNodeGene,
)
from polyneat.algorithms.exact.exact_training_hyperparameters import (
    EXACTTrainingHyperparameters,
)
from polyneat.configs.exact.exact_config import EXACTConfig

_CPU = torch.device("cpu")


def _two_class_genome() -> EXACTGenome:
    return EXACTGenome(
        node_genes=(
            FilterNodeGene(
                node_id=0, node_type="input", filter_height=2, filter_width=2, depth=0.0
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
                innovation_id=0, source_node_id=0, target_node_id=1, is_enabled=True
            ),
            ConvolutionEdgeGene(
                innovation_id=1, source_node_id=0, target_node_id=2, is_enabled=True
            ),
        ),
    )


def _linearly_separable_data() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    class_zero = torch.randn(20, 4) - 2.0
    class_one = torch.randn(20, 4) + 2.0
    features = torch.cat([class_zero, class_one])
    labels = torch.cat([torch.zeros(20, dtype=torch.long), torch.ones(20, dtype=torch.long)])
    return features, labels


def _trainer(
    features: torch.Tensor, labels: torch.Tensor, use_epigenetic: bool = True
) -> EXACTBackpropagationTrainer:
    return EXACTBackpropagationTrainer(
        training_features=features,
        training_labels=labels,
        input_image_height=2,
        input_image_width=2,
        leaky_relu_negative_slope=0.1,
        activation_clamp_maximum=5.5,
        number_of_training_epochs=20,
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
        maximum_momentum=0.99,
        minimum_learning_rate=0.00001,
        minimum_weight_decay=0.000001,
        use_batch_normalization=False,
        use_epigenetic_weight_initialization=use_epigenetic,
        device_for_computation=_CPU,
    )


def _build_single_edge_genome_with_kernel_value(kernel_value: float) -> EXACTGenome:
    """2x2 input directly convolved into one 1x1 output (kernel |1-2|+1 = 2)."""
    return EXACTGenome(
        node_genes=(
            FilterNodeGene(
                node_id=0, node_type="input", filter_height=2, filter_width=2, depth=0.0
            ),
            FilterNodeGene(
                node_id=1, node_type="output", filter_height=1, filter_width=1, depth=1.0
            ),
        ),
        edge_genes=(
            ConvolutionEdgeGene(
                innovation_id=0,
                source_node_id=0,
                target_node_id=1,
                is_enabled=True,
                kernel_weights=(
                    (kernel_value, kernel_value),
                    (kernel_value, kernel_value),
                ),
            ),
        ),
    )


def _build_trainer_for_two_by_two_images(
    learning_rate: float = 0.01,
    learning_rate_decay_factor: float = 1.0,
    weight_decay: float = 0.0,
    weight_decay_decay_factor: float = 1.0,
    number_of_training_epochs: int = 1,
    training_batch_size: int = 1,
    number_of_samples: int = 1,
    velocity_reset_interval: int = 0,
    use_epigenetic_weight_initialization: bool = True,
) -> EXACTBackpropagationTrainer:
    """Trainer over all-zero 2x2 images, with the paper's fixed activations."""
    return EXACTBackpropagationTrainer(
        training_features=torch.zeros((number_of_samples, 4)),
        training_labels=torch.zeros((number_of_samples,), dtype=torch.long),
        input_image_height=2,
        input_image_width=2,
        leaky_relu_negative_slope=0.1,
        activation_clamp_maximum=5.5,
        number_of_training_epochs=number_of_training_epochs,
        default_hyperparameters=EXACTTrainingHyperparameters(
            learning_rate=learning_rate,
            learning_rate_decay_factor=learning_rate_decay_factor,
            momentum=0.5,
            momentum_decay_factor=0.95,
            weight_decay=weight_decay,
            weight_decay_decay_factor=weight_decay_decay_factor,
            velocity_reset_interval=velocity_reset_interval,
            input_dropout_probability=0.0,
            hidden_dropout_probability=0.0,
            batch_size=training_batch_size,
            batch_normalization_alpha=0.1,
        ),
        maximum_momentum=0.99,
        minimum_learning_rate=1e-15,
        minimum_weight_decay=0.0,
        use_batch_normalization=False,
        use_epigenetic_weight_initialization=use_epigenetic_weight_initialization,
        device_for_computation=_CPU,
    )


def _mean_cross_entropy(genome: EXACTGenome, features: torch.Tensor,
                        labels: torch.Tensor) -> float:
    from polyneat.algorithms.exact.torch_convolutional_phenotype import (
        TorchConvolutionalPhenotype,
    )

    phenotype = TorchConvolutionalPhenotype(
        exact_genome=genome,
        input_image_height=2,
        input_image_width=2,
        leaky_relu_negative_slope=0.1,
        activation_clamp_maximum=5.5,
        device_for_computation=_CPU,
    )
    with torch.no_grad():
        return float(torch.nn.functional.cross_entropy(
            phenotype.forward_pass(features), labels
        ))


def test_training_reduces_cross_entropy_and_marks_trained() -> None:
    features, labels = _linearly_separable_data()
    trainer = _trainer(features, labels)
    torch.manual_seed(1)
    untrained = _two_class_genome()
    trained = trainer.train_genome(untrained)
    assert trained.is_trained is True
    assert all(edge.kernel_weights is not None for edge in trained.edge_genes)
    torch.manual_seed(1)
    loss_before = _mean_cross_entropy(untrained, features, labels)
    loss_after = _mean_cross_entropy(trained, features, labels)
    assert loss_after < loss_before


def test_already_trained_genome_is_returned_by_identity() -> None:
    features, labels = _linearly_separable_data()
    trainer = _trainer(features, labels)
    trained = trainer.train_genome(_two_class_genome())
    assert trainer.train_genome(trained) is trained


def test_epigenetic_off_discards_inherited_kernels() -> None:
    features, labels = _linearly_separable_data()
    trainer_with_epigenetics = _trainer(features, labels, use_epigenetic=True)
    first_pass = trainer_with_epigenetics.train_genome(_two_class_genome())
    reproduced = EXACTGenome(
        node_genes=first_pass.node_genes,
        edge_genes=first_pass.edge_genes,
        is_trained=False,
    )
    torch.manual_seed(2)
    randomized = _trainer(features, labels, use_epigenetic=False).train_genome(reproduced)
    torch.manual_seed(2)
    epigenetic = _trainer(features, labels, use_epigenetic=True).train_genome(reproduced)
    # Same torch seed: any difference comes from the discarded starting kernels.
    assert (
        randomized.get_edge_gene_by_innovation_id(0).kernel_weights
        != epigenetic.get_edge_gene_by_innovation_id(0).kernel_weights
    )


def test_from_config_reads_the_training_fields() -> None:
    features, labels = _linearly_separable_data()
    config = EXACTConfig(
        number_of_output_nodes=2,
        input_image_height=2,
        input_image_width=2,
        number_of_training_epochs_per_genome=1,
        training_batch_size=4,
    )
    trainer = EXACTBackpropagationTrainer.from_config(
        config, features, labels, device_for_computation=_CPU
    )
    trained = trainer.train_genome(_two_class_genome())
    assert trained.is_trained is True


def test_momentum_schedule_follows_equation_7() -> None:
    """µ' = µ_max - ((µ_max - µ)·Δµ): 0.99 - (0.49·0.90) = 0.549."""
    next_momentum = EXACTBackpropagationTrainer.compute_next_momentum(
        current_momentum=0.5, maximum_momentum=0.99, momentum_decay_factor=0.90
    )
    assert next_momentum == pytest.approx(0.549)


def test_learning_rate_schedule_follows_equation_8() -> None:
    assert EXACTBackpropagationTrainer.compute_next_learning_rate(
        current_learning_rate=0.01,
        learning_rate_decay_factor=0.95,
        minimum_learning_rate=0.00001,
    ) == pytest.approx(0.0095)
    assert EXACTBackpropagationTrainer.compute_next_learning_rate(
        current_learning_rate=0.00001,
        learning_rate_decay_factor=0.95,
        minimum_learning_rate=0.00001,
    ) == pytest.approx(0.00001)


def test_weight_decay_schedule_follows_equation_9() -> None:
    assert EXACTBackpropagationTrainer.compute_next_weight_decay(
        current_weight_decay=0.001,
        weight_decay_decay_factor=0.95,
        minimum_weight_decay=0.000001,
    ) == pytest.approx(0.00095)


def test_weight_decay_is_decoupled_from_learning_rate() -> None:
    """Eq. 6 applies w -= w·λ directly: with a vanishing learning rate the
    kernel must still shrink by (1 - λ) per step. Torch's gradient-coupled
    weight_decay would shrink it by η·λ ≈ 0 instead."""
    genome = _build_single_edge_genome_with_kernel_value(1.0)
    trainer = _build_trainer_for_two_by_two_images(
        learning_rate=1e-12,
        weight_decay=0.5,
        weight_decay_decay_factor=1.0,
        number_of_training_epochs=1,
        training_batch_size=1,
        number_of_samples=1,
        use_epigenetic_weight_initialization=True,
    )
    trained_genome = trainer.train_genome(genome)
    trained_kernel = trained_genome.edge_genes[0].kernel_weights
    for kernel_row in trained_kernel:
        for kernel_value in kernel_row:
            assert kernel_value == pytest.approx(0.5, rel=1e-3)


def test_reset_optimizer_velocities_clears_momentum_buffers() -> None:
    parameter = torch.nn.Parameter(torch.ones(2, 2))
    optimizer = torch.optim.SGD([parameter], lr=0.1, momentum=0.9, nesterov=True)
    parameter.sum().backward()
    optimizer.step()
    assert len(optimizer.state) > 0
    EXACTBackpropagationTrainer.reset_optimizer_velocities(optimizer)
    assert len(optimizer.state) == 0


def test_genome_hyperparameters_override_defaults() -> None:
    """A genome carrying its own hyperparameters trains with them: with a
    per-genome weight_decay of 0.5 and a vanishing learning rate the kernel
    halves, while the trainer's default weight_decay is 0."""
    hyperparameters = EXACTTrainingHyperparameters(
        learning_rate=1e-12,
        learning_rate_decay_factor=1.0,
        momentum=0.5,
        momentum_decay_factor=0.95,
        weight_decay=0.5,
        weight_decay_decay_factor=1.0,
        velocity_reset_interval=0,
        input_dropout_probability=0.0,
        hidden_dropout_probability=0.0,
        batch_size=1,
        batch_normalization_alpha=0.1,
    )
    genome = replace(
        _build_single_edge_genome_with_kernel_value(1.0),
        training_hyperparameters=hyperparameters,
    )
    trainer = _build_trainer_for_two_by_two_images(weight_decay=0.0)
    trained_genome = trainer.train_genome(genome)
    for kernel_row in trained_genome.edge_genes[0].kernel_weights:
        for kernel_value in kernel_row:
            assert kernel_value == pytest.approx(0.5, rel=1e-3)
    assert trained_genome.training_hyperparameters == hyperparameters
