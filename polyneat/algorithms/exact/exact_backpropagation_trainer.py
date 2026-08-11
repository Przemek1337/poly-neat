"""Backpropagation of EXACT CNN genomes (Desell, 2017, section VII)."""

from __future__ import annotations

from dataclasses import replace

import torch
import torch.nn.functional as functional

from polyneat.algorithms.exact.exact_genome import EXACTGenome
from polyneat.algorithms.exact.exact_training_hyperparameters import (
    EXACTTrainingHyperparameters,
)
from polyneat.algorithms.exact.torch_convolutional_phenotype import (
    TorchConvolutionalPhenotype,
)
from polyneat.configs.exact.exact_config import EXACTConfig


class EXACTBackpropagationTrainer:
    """Trains an EXACT genome's kernels and writes them back into a new genome.

    Implements the paper's fitness-evaluation training (section VII): SGD
    with Nesterov momentum (eqs. 3-5), cross-entropy over the softmax
    outputs, shuffled epochs, decoupled L2 decay applied directly to the
    weights (eq. 6, ``w -= w·λ``, independent of the learning rate), and the
    per-epoch hyperparameter schedule of eqs. 7-9 — momentum approaches
    ``maximum_momentum`` exponentially while learning rate and weight decay
    shrink toward their floors. Eq. 6's decay applies to the convolution
    kernels only, never to the batch-normalization scale and shift, and the
    momentum buffers are dropped every ``ω`` examples (section VII's velocity
    reset). A genome carrying its own ``training_hyperparameters`` trains
    with them (section IV); the rest use the config-derived defaults. With
    ``use_epigenetic_weight_initialization``
    the genome's stored kernels are the starting point (section III-D);
    without it they are discarded and every kernel is He-initialized, the
    paper's randomized control arm.

    Genomes with ``is_trained=True`` are returned by identity, so elites
    carried over by the generational loop are not retrained.

    References:
        Desell, T. (2017). Developing a Volunteer Computing Project to Evolve
            Convolutional Neural Networks and Their Hyperparameters. 2017 IEEE
            13th International Conference on e-Science, pp. 19-28.
    """

    def __init__(
        self,
        training_features: torch.Tensor,
        training_labels: torch.Tensor,
        input_image_height: int,
        input_image_width: int,
        leaky_relu_negative_slope: float,
        activation_clamp_maximum: float,
        number_of_training_epochs: int,
        default_hyperparameters: EXACTTrainingHyperparameters,
        maximum_momentum: float,
        minimum_learning_rate: float,
        minimum_weight_decay: float,
        use_batch_normalization: bool,
        use_epigenetic_weight_initialization: bool,
        device_for_computation: torch.device,
    ) -> None:
        """Fix the dataset, the geometry and the fallback hyperparameters.

        Args:
            training_features: Float tensor ``(num_samples, height * width)``.
            training_labels: Long tensor ``(num_samples,)`` of class indices.
            input_image_height: Height flat feature rows are reshaped to.
            input_image_width: Width flat feature rows are reshaped to.
            leaky_relu_negative_slope: Hidden activation leak (paper: 0.1).
            activation_clamp_maximum: Hidden activation cap (paper: 5.5).
            number_of_training_epochs: Epochs per training session.
            default_hyperparameters: Vector used for genomes that carry none
                of their own (section IV).
            maximum_momentum: Cap ``µ_max`` (eq. 7).
            minimum_learning_rate: Floor ``η_min`` (eq. 8).
            minimum_weight_decay: Floor ``λ_min`` (eq. 9).
            use_batch_normalization: Whether hidden nodes are batch
                normalized during training (section VII).
            use_epigenetic_weight_initialization: Whether stored kernels seed
                the training (section III-D).
            device_for_computation: Device the phenotypes train on.

        Raises:
            ValueError: If tensor shapes disagree with the image geometry or
                a hyperparameter is out of range.
        """
        if training_features.dim() != 2:
            raise ValueError(
                "EXACTBackpropagationTrainer: training_features must be 2-D "
                f"(num_samples, height * width), got {tuple(training_features.shape)}"
            )
        if training_features.shape[1] != input_image_height * input_image_width:
            raise ValueError(
                f"EXACTBackpropagationTrainer: training_features has "
                f"{training_features.shape[1]} columns but the image geometry is "
                f"{input_image_height}x{input_image_width}"
            )
        if training_features.shape[0] != training_labels.shape[0]:
            raise ValueError(
                f"EXACTBackpropagationTrainer: {training_features.shape[0]} feature rows "
                f"but {training_labels.shape[0]} labels"
            )
        if number_of_training_epochs < 1:
            raise ValueError(
                "EXACTBackpropagationTrainer: number_of_training_epochs must be >= 1, "
                f"got {number_of_training_epochs}"
            )
        if default_hyperparameters.batch_size < 1:
            raise ValueError(
                "EXACTBackpropagationTrainer: batch_size must be >= 1, "
                f"got {default_hyperparameters.batch_size}"
            )
        if default_hyperparameters.learning_rate <= 0.0 or not (
            0.0 < default_hyperparameters.momentum < 1.0
        ):
            raise ValueError(
                "EXACTBackpropagationTrainer: learning_rate must be > 0 and momentum "
                f"in (0, 1) (nesterov), got {default_hyperparameters.learning_rate} "
                f"and {default_hyperparameters.momentum}"
            )
        self._training_features = training_features.to(torch.float32).to(
            device_for_computation
        )
        self._training_labels = training_labels.to(torch.long).to(device_for_computation)
        self._input_image_height = input_image_height
        self._input_image_width = input_image_width
        self._leaky_relu_negative_slope = leaky_relu_negative_slope
        self._activation_clamp_maximum = activation_clamp_maximum
        self._number_of_training_epochs = number_of_training_epochs
        self._default_hyperparameters = default_hyperparameters
        self._maximum_momentum = maximum_momentum
        self._minimum_learning_rate = minimum_learning_rate
        self._minimum_weight_decay = minimum_weight_decay
        self._use_batch_normalization = use_batch_normalization
        self._use_epigenetic_weight_initialization = use_epigenetic_weight_initialization
        self._device_for_computation = device_for_computation

    @classmethod
    def from_config(
        cls,
        config: EXACTConfig,
        training_features: torch.Tensor,
        training_labels: torch.Tensor,
        device_for_computation: torch.device,
    ) -> EXACTBackpropagationTrainer:
        """Build a trainer from the config's geometry and training fields.

        Args:
            config: Validated EXACT hyperparameters.
            training_features: Float tensor ``(num_samples, height * width)``.
            training_labels: Long tensor ``(num_samples,)`` of class indices.
            device_for_computation: Device the phenotypes train on.

        Returns:
            The wired trainer.
        """
        return cls(
            training_features=training_features,
            training_labels=training_labels,
            input_image_height=config.input_image_height,
            input_image_width=config.input_image_width,
            leaky_relu_negative_slope=config.leaky_relu_negative_slope,
            activation_clamp_maximum=config.activation_clamp_maximum,
            number_of_training_epochs=config.number_of_training_epochs_per_genome,
            default_hyperparameters=EXACTTrainingHyperparameters.from_config(config),
            maximum_momentum=config.maximum_momentum,
            minimum_learning_rate=config.minimum_learning_rate,
            minimum_weight_decay=config.minimum_weight_decay,
            use_batch_normalization=config.use_batch_normalization,
            use_epigenetic_weight_initialization=(
                config.use_epigenetic_weight_initialization
            ),
            device_for_computation=device_for_computation,
        )

    @staticmethod
    def compute_next_momentum(
        current_momentum: float, maximum_momentum: float, momentum_decay_factor: float
    ) -> float:
        """Eq. 7: exponentially approach ``µ_max`` — the gap shrinks by ``Δµ``."""
        return maximum_momentum - (
            (maximum_momentum - current_momentum) * momentum_decay_factor
        )

    @staticmethod
    def compute_next_learning_rate(
        current_learning_rate: float,
        learning_rate_decay_factor: float,
        minimum_learning_rate: float,
    ) -> float:
        """Eq. 8: ``η = max(η·Δη, η_min)``."""
        return max(
            current_learning_rate * learning_rate_decay_factor, minimum_learning_rate
        )

    @staticmethod
    def compute_next_weight_decay(
        current_weight_decay: float,
        weight_decay_decay_factor: float,
        minimum_weight_decay: float,
    ) -> float:
        """Eq. 9: ``λ = max(λ·Δλ, λ_min)``."""
        return max(
            current_weight_decay * weight_decay_decay_factor, minimum_weight_decay
        )

    @staticmethod
    def reset_optimizer_velocities(optimizer: torch.optim.Optimizer) -> None:
        """Drop all momentum buffers — the paper's velocity reset (section VII)."""
        optimizer.state.clear()

    def train_genome(self, genome: EXACTGenome) -> EXACTGenome:
        """Run one full training session and return the trained genome.

        Args:
            genome: Genome whose enabled kernels are trained. Returned by
                identity when already marked trained; callers can detect a
                skip by object identity.

        Returns:
            A new genome with trained kernels and ``is_trained=True``, or
            ``genome`` itself when it was already trained or has no enabled
            edges.
        """
        if genome.is_trained:
            return genome
        genome_to_express = genome
        if not self._use_epigenetic_weight_initialization:
            genome_to_express = replace(
                genome,
                edge_genes=tuple(
                    replace(edge_gene, kernel_weights=None)
                    for edge_gene in genome.edge_genes
                ),
            )
        hyperparameters = genome.training_hyperparameters or self._default_hyperparameters
        phenotype = TorchConvolutionalPhenotype(
            exact_genome=genome_to_express,
            input_image_height=self._input_image_height,
            input_image_width=self._input_image_width,
            leaky_relu_negative_slope=self._leaky_relu_negative_slope,
            activation_clamp_maximum=self._activation_clamp_maximum,
            use_batch_normalization=self._use_batch_normalization,
            batch_normalization_alpha=hyperparameters.batch_normalization_alpha,
            input_dropout_probability=hyperparameters.input_dropout_probability,
            hidden_dropout_probability=hyperparameters.hidden_dropout_probability,
            device_for_computation=self._device_for_computation,
        )
        trainable_parameters = list(phenotype.parameters())
        if not trainable_parameters:
            return genome
        phenotype.train()

        optimizer = torch.optim.SGD(
            trainable_parameters,
            lr=hyperparameters.learning_rate,
            momentum=hyperparameters.momentum,
            nesterov=True,
        )
        kernel_parameters = phenotype.convolution_kernel_parameters()
        current_learning_rate = hyperparameters.learning_rate
        current_momentum = hyperparameters.momentum
        current_weight_decay = hyperparameters.weight_decay
        examples_processed_since_velocity_reset = 0
        number_of_samples = self._training_features.shape[0]

        for _epoch in range(self._number_of_training_epochs):
            shuffled_indices = torch.randperm(
                number_of_samples, device=self._device_for_computation
            )
            for batch_start in range(0, number_of_samples, hyperparameters.batch_size):
                batch_indices = shuffled_indices[
                    batch_start : batch_start + hyperparameters.batch_size
                ]
                optimizer.zero_grad()
                batch_logits = phenotype.forward_pass(
                    self._training_features[batch_indices]
                )
                batch_loss = functional.cross_entropy(
                    batch_logits, self._training_labels[batch_indices]
                )
                batch_loss.backward()
                optimizer.step()
                # Eq. 6: decoupled L2 decay on the kernels.
                with torch.no_grad():
                    for kernel_parameter in kernel_parameters:
                        kernel_parameter.mul_(1.0 - current_weight_decay)
                examples_processed_since_velocity_reset += int(batch_indices.shape[0])
                if (
                    hyperparameters.velocity_reset_interval > 0
                    and examples_processed_since_velocity_reset
                    >= hyperparameters.velocity_reset_interval
                ):
                    self.reset_optimizer_velocities(optimizer)
                    examples_processed_since_velocity_reset = 0
            # Paper eqs. 7-9: per-epoch hyperparameter schedule.
            current_learning_rate = self.compute_next_learning_rate(
                current_learning_rate,
                hyperparameters.learning_rate_decay_factor,
                self._minimum_learning_rate,
            )
            current_momentum = self.compute_next_momentum(
                current_momentum,
                self._maximum_momentum,
                hyperparameters.momentum_decay_factor,
            )
            current_weight_decay = self.compute_next_weight_decay(
                current_weight_decay,
                hyperparameters.weight_decay_decay_factor,
                self._minimum_weight_decay,
            )
            for parameter_group in optimizer.param_groups:
                parameter_group["lr"] = current_learning_rate
                parameter_group["momentum"] = current_momentum

        phenotype.eval()
        return phenotype.extract_genome_with_trained_weights()
