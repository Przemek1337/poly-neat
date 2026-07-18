from __future__ import annotations

import torch

from polyneat.algorithms.lneat.trainable_torch_phenotype import (
    TrainableTorchFeedForwardPhenotype,
)
from polyneat.core.neat.neat_genome import NEATGenome


def _validate_feature_and_binary_target_pair(
    features: torch.Tensor,
    binary_targets: torch.Tensor,
    feature_argument_name: str,
    target_argument_name: str,
) -> None:
    """Check one feature/target pair's shapes and target values.

    Args:
        features: Tensor expected to be ``(num_samples, num_features)``.
        binary_targets: Tensor expected to be ``(num_samples, 1)`` holding only
            ``0.0`` and ``1.0``.
        feature_argument_name: Name of the feature argument, for error text.
        target_argument_name: Name of the target argument, for error text.

    Raises:
        ValueError: If either tensor has the wrong rank or shape, the sample
            counts disagree, or the targets are not binary.
    """
    if features.dim() != 2:
        raise ValueError(
            f"BackpropagationWeightTrainer: {feature_argument_name} must be 2-D "
            f"(num_samples, num_features), got shape {tuple(features.shape)}"
        )
    if binary_targets.dim() != 2 or binary_targets.shape[1] != 1:
        raise ValueError(
            f"BackpropagationWeightTrainer: {target_argument_name} must have shape "
            f"(num_samples, 1), got {tuple(binary_targets.shape)}"
        )
    if features.shape[0] != binary_targets.shape[0]:
        raise ValueError(
            f"BackpropagationWeightTrainer: {feature_argument_name} has "
            f"{features.shape[0]} samples but {target_argument_name} has "
            f"{binary_targets.shape[0]}"
        )
    target_values = set(binary_targets.unique().tolist())
    if not target_values <= {0.0, 1.0}:
        raise ValueError(
            f"BackpropagationWeightTrainer: {target_argument_name} must only contain "
            f"0.0 and 1.0, got values {sorted(target_values)}"
        )


class BackpropagationWeightTrainer:
    """Trains a genome's connection weights by selective backpropagation.

    Implements the learning stage of L-NEAT (Chen & Alahakoon, 2006, section
    IV.B): a genome's enabled connection weights are trained by gradient
    descent on a small fixed set of perfect examples (the paper's learning
    samples, section IV.B.5) and the result is written into a new genome.
    Selective learning (section IV.B.2) skips Type 1 networks — those that
    classify every sample correctly and whose mean output distance is at most
    ``training_indicator`` — since further training costs computation without
    changing their classification.

    The trainer holds two datasets. The Type-1 test runs on the classification
    set, the full sample set the equation-1 fitness is computed over, because
    the paper's four network types are defined on the fitness surface (Table I;
    Fig. 2, whose axes are equation 1's correct-classification and
    output-difference terms). Backpropagation itself runs on the learning set,
    the small fixed subset of section IV.B.5. Judging Type 1 on the learning
    subset alone would let a genome that fits those few samples skip training
    while misclassifying the rest of the data.

    Two of the paper's three learning-amount parameters are fixed here: the
    amount of learning ``A`` is the number of learning samples this trainer
    holds, and the quality ``L`` is ``number_of_iterations``. The interval
    ``I`` is applied by
    :class:`~polyneat.algorithms.lneat.lneat_algorithm.LNEATAlgorithm`.

    References:
        Chen, L., & Alahakoon, D. (2006). NeuroEvolution of Augmenting
        Topologies with Learning for Data Classification. *ICIA 2006*.
    """

    def __init__(
        self,
        classification_features: torch.Tensor,
        classification_binary_targets: torch.Tensor,
        learning_sample_features: torch.Tensor,
        learning_sample_binary_targets: torch.Tensor,
        learning_rate: float,
        number_of_iterations: int,
        training_indicator: float,
        classification_threshold: float,
        device_for_computation: torch.device,
    ) -> None:
        """Fix both datasets and the training hyperparameters.

        Args:
            classification_features: Float tensor of shape
                ``(num_samples, num_features)`` — the full sample set the
                equation-1 fitness is computed over, used for the Type-1 test.
            classification_binary_targets: Float tensor of shape
                ``(num_samples, 1)`` with values in ``{0.0, 1.0}``.
            learning_sample_features: Float tensor of shape
                ``(num_samples, num_features)`` — the fixed learning subset
                (paper's ``A`` samples), reused for every session.
            learning_sample_binary_targets: Float tensor of shape
                ``(num_samples, 1)`` with values in ``{0.0, 1.0}``.
            learning_rate: SGD learning rate.
            number_of_iterations: Gradient steps per session (paper's ``L``).
            training_indicator: Maximum mean output distance for a correctly
                classifying network to count as Type 1 and skip training.
            classification_threshold: Output level at or above which a sample
                counts as classified into the class.
            device_for_computation: Device the phenotypes are built on.

        Raises:
            ValueError: If tensor shapes disagree, targets are not binary,
                or a hyperparameter is out of range.
        """
        _validate_feature_and_binary_target_pair(
            features=classification_features,
            binary_targets=classification_binary_targets,
            feature_argument_name="classification_features",
            target_argument_name="classification_binary_targets",
        )
        _validate_feature_and_binary_target_pair(
            features=learning_sample_features,
            binary_targets=learning_sample_binary_targets,
            feature_argument_name="learning_sample_features",
            target_argument_name="learning_sample_binary_targets",
        )
        if learning_rate <= 0.0:
            raise ValueError(
                f"BackpropagationWeightTrainer: learning_rate must be > 0, got {learning_rate}"
            )
        if number_of_iterations < 1:
            raise ValueError(
                "BackpropagationWeightTrainer: number_of_iterations must be >= 1, "
                f"got {number_of_iterations}"
            )
        if not (0.0 <= training_indicator <= 1.0):
            raise ValueError(
                "BackpropagationWeightTrainer: training_indicator must be in [0.0, 1.0], "
                f"got {training_indicator}"
            )
        if not (0.0 < classification_threshold < 1.0):
            raise ValueError(
                "BackpropagationWeightTrainer: classification_threshold must be in "
                f"(0.0, 1.0), got {classification_threshold}"
            )
        self._classification_features = classification_features.to(torch.float32).to(
            device_for_computation
        )
        self._classification_binary_targets = classification_binary_targets.to(
            torch.float32
        ).to(device_for_computation)
        self._learning_sample_features = learning_sample_features.to(
            torch.float32
        ).to(device_for_computation)
        self._learning_sample_binary_targets = learning_sample_binary_targets.to(
            torch.float32
        ).to(device_for_computation)
        self._learning_rate = learning_rate
        self._number_of_iterations = number_of_iterations
        self._training_indicator = training_indicator
        self._classification_threshold = classification_threshold
        self._device_for_computation = device_for_computation

    def genome_requires_training(self, genome: NEATGenome) -> bool:
        """Return whether the genome is not a Type 1 network (section IV.B.2).

        A Type 1 network classifies every sample of the classification set
        correctly and keeps its mean output distance to the binary target at or
        below ``training_indicator``. Type 2 to Type 4 networks (wrong
        classification, large output distance, or both) require training. The
        test runs on the classification set rather than the learning subset
        because the paper's types are positions on the equation-1 fitness
        surface, which is measured over every sample.

        Args:
            genome: Genome whose learning need is assessed.

        Returns:
            ``True`` when the genome requires a backpropagation session,
            ``False`` for a Type 1 network.
        """
        phenotype = TrainableTorchFeedForwardPhenotype(genome, self._device_for_computation)
        with torch.no_grad():
            outputs = phenotype.forward_pass(self._classification_features)
        predicted_memberships = (outputs >= self._classification_threshold).float()
        classifies_all_samples_correctly = bool(
            torch.all(predicted_memberships == self._classification_binary_targets)
        )
        mean_output_distance = float(
            torch.mean(torch.abs(outputs - self._classification_binary_targets))
        )
        is_type_one_network = (
            classifies_all_samples_correctly
            and mean_output_distance <= self._training_indicator
        )
        return not is_type_one_network

    def train_genome(self, genome: NEATGenome) -> NEATGenome:
        """Run one full backpropagation session and return the trained genome.

        Args:
            genome: Genome whose enabled connection weights are trained.

        Returns:
            A new genome with trained weights — or ``genome`` itself when it
            has no enabled connections (nothing to train).
        """
        phenotype = TrainableTorchFeedForwardPhenotype(genome, self._device_for_computation)
        trainable_parameters = list(phenotype.parameters())
        if not trainable_parameters:
            return genome
        optimizer = torch.optim.SGD(trainable_parameters, lr=self._learning_rate)
        for _iteration in range(self._number_of_iterations):
            optimizer.zero_grad()
            outputs = phenotype.forward_pass(self._learning_sample_features)
            loss = torch.mean((outputs - self._learning_sample_binary_targets) ** 2)
            loss.backward()
            optimizer.step()
        return phenotype.extract_genome_with_trained_weights()

    def train_genome_if_learning_needed(self, genome: NEATGenome) -> NEATGenome:
        """Apply selective learning: train Types 2 to 4, leave Type 1 unchanged.

        Args:
            genome: Genome offered for a backpropagation session.

        Returns:
            A new trained genome, or ``genome`` itself when the session is
            skipped. Callers identify skipped genomes by object identity.
        """
        if not self.genome_requires_training(genome):
            return genome
        return self.train_genome(genome)
