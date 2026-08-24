"""Fitness evaluator that trains each phenotype from scratch, as DeepNEAT prescribes.

DeepNEAT genomes carry no weights, only topology and layer hyperparameters: the
source paper's genotype/phenotype split puts weight learning entirely on the
phenotype side, so *every* fitness evaluation has to train the network from a
fresh random initialization by backpropagation. That is why training lives
**here**, in the evaluator, and not in ``advance_one_generation`` the way EXACT
does it (see ``polyneat.algorithms.exact.exact_backpropagation_trainer``): EXACT
is Lamarckian and writes its trained weights back into the genotype, so its
generational loop must call a trainer as a distinct step between generations.
DeepNEAT has no weights to write back anywhere -- the genotype is stateless
with respect to training -- so ``DeepNEATAlgorithm`` never needs to override
the generational loop at all; evaluating fitness and training the network are
the same call.

A degenerate phenotype (``is_degenerate = True``, set by
:class:`~polyneat.algorithms.deepneat.torch_layer_stack_phenotype
.TorchLayerStackPhenotype` when a genome has no enabled input-to-output path,
or exceeds its parameter budget) is scored ``0.0`` without being trained *or
otherwise touched*: its mode (``training``/``eval``) is left exactly as its
constructor set it. Only a phenotype that is actually trained is guaranteed to
end evaluation in ``eval()`` mode.

References:
    Miikkulainen, R., Liang, J., Meyerson, E., Rawal, A., Fink, D., Francon, O., Raju, B.,
        Shahrzad, H., Navruzyan, A., Duffy, N., & Hodjat, B. (2017). Evolving Deep Neural
        Networks. *arXiv:1703.00548*. Published in *Artificial Intelligence in the Age of
        Neural Networks and Brain Computing* (2019), pp. 293-312.
        DOI: 10.1016/B978-0-12-815480-9.00015-3
"""

from __future__ import annotations

import torch
from torch import nn

from polyneat.core.component_protocols import Phenotype
from polyneat.core.type_aliases import FitnessValue
from polyneat.logging_utils.custom_logger import get_logger

logger = get_logger(__name__)

# Large, mutually coprime-ish multipliers keep the per-phenotype seed derived
# from (base seed, generation counter, position in batch) well spread across
# torch.manual_seed's range, so the three inputs' low bits do not collide.
_GENERATION_SEED_MULTIPLIER = 1_000_003
_POSITION_SEED_MULTIPLIER = 7_919


class TrainedNetworkAccuracyEvaluator:
    """Trains each phenotype from scratch, then scores it by validation accuracy.

    Implements the ``FitnessEvaluator`` protocol
    (:class:`~polyneat.core.component_protocols.FitnessEvaluator`) directly
    rather than through ``SequentialFitnessEvaluator``: that base class maps a
    per-phenotype method over the batch and cannot express the two things this
    evaluator needs across the whole batch -- a generation counter bumped
    exactly once per call, and a random seed derived from each phenotype's
    position within the batch so a phenotype's fitness does not depend on
    where in the batch it happens to sit.

    For each non-degenerate phenotype, in order: seed the RNG from
    ``(base_random_seed, generation_counter, position_in_batch)``; train with
    Adam and cross-entropy loss for ``number_of_epochs`` passes over shuffled
    minibatches of the training set; switch to evaluation mode and measure
    accuracy on the validation set in minibatches, under ``torch.no_grad()``.
    Trained weights are never written anywhere -- DeepNEAT does not inherit
    them (see module docstring) -- so nothing here can leak back into a
    genome. A degenerate phenotype is scored ``0.0`` without training.

    ``nn.BatchNorm1d``/``BatchNorm2d`` cannot compute batch statistics from a
    single sample in training mode and raise if asked to. Any minibatch of
    exactly one sample -- whether it is the trailing remainder against
    ``batch_size``, or (with ``batch_size=1``) *every* minibatch -- is dropped
    during training only, never during validation: validation runs under
    ``eval()`` + ``no_grad()`` and uses running statistics, so a
    validation-set batch of one is not a problem and is not dropped, since
    dropping it would silently change the reported accuracy. The constructor
    counts, up front, how many minibatches the training loop will actually
    keep after that drop -- simulating the loop's own per-batch sizes rather
    than only checking the trailing remainder, so a configuration like
    ``batch_size=1`` (where the remainder-based check is always zero and
    would miss that every minibatch is a singleton) is caught the same way.
    If zero minibatches would remain, the constructor raises instead of
    silently "training" on nothing and reporting an untrained network's
    accuracy as if it were a real fitness.
    """

    def __init__(
        self,
        train_features: torch.Tensor,
        train_labels: torch.Tensor,
        validation_features: torch.Tensor,
        validation_labels: torch.Tensor,
        number_of_epochs: int,
        learning_rate: float,
        batch_size: int,
        device_for_computation: torch.device,
        base_random_seed: int,
        use_deterministic_algorithms: bool = False,
    ) -> None:
        """Fix the datasets, training hyperparameters and seeding scheme.

        Args:
            train_features: Training inputs, batch dimension first. Any
                further shape is accepted -- flat features or spatial images
                -- since it is dictated by the phenotypes this evaluator will
                receive, not by this class.
            train_labels: Long tensor of class indices, shape
                ``(num_train_samples,)``.
            validation_features: Validation inputs, same shape convention as
                ``train_features``.
            validation_labels: Long tensor of class indices, shape
                ``(num_validation_samples,)``.
            number_of_epochs: Passes over the training set per phenotype.
            learning_rate: Adam's learning rate.
            batch_size: Minibatch size for both training and validation.
            device_for_computation: Device the datasets and every phenotype
                train on.
            base_random_seed: Root of the per-phenotype seed derivation; see
                the class docstring.
            use_deterministic_algorithms: When ``True``, calls
                ``torch.use_deterministic_algorithms(True)`` once here, making
                every evaluation reproducible bit-for-bit given the same seed
                at the cost of performance: some CUDA kernels lose their fast
                non-deterministic implementation, and any operation with no
                deterministic implementation raises ``RuntimeError`` instead
                of running. Off by default.

        Raises:
            ValueError: If a features/labels sample count disagrees within
                either the training or the validation set, if
                ``number_of_epochs``, ``learning_rate`` or ``batch_size`` is
                out of range, or if dropping every size-one training minibatch
                (see class docstring -- this also catches ``batch_size=1``)
                would leave zero usable minibatches.
        """
        if train_features.shape[0] != train_labels.shape[0]:
            raise ValueError(
                "TrainedNetworkAccuracyEvaluator: train_features has "
                f"{train_features.shape[0]} samples but train_labels has "
                f"{train_labels.shape[0]}"
            )
        if validation_features.shape[0] != validation_labels.shape[0]:
            raise ValueError(
                "TrainedNetworkAccuracyEvaluator: validation_features has "
                f"{validation_features.shape[0]} samples but validation_labels has "
                f"{validation_labels.shape[0]}"
            )
        if number_of_epochs < 1:
            raise ValueError(
                "TrainedNetworkAccuracyEvaluator: number_of_epochs must be >= 1, "
                f"got {number_of_epochs}"
            )
        if learning_rate <= 0.0:
            raise ValueError(
                "TrainedNetworkAccuracyEvaluator: learning_rate must be > 0, "
                f"got {learning_rate}"
            )
        if batch_size < 1:
            raise ValueError(
                "TrainedNetworkAccuracyEvaluator: batch_size must be >= 1, "
                f"got {batch_size}"
            )

        # Count usable minibatches the same way _train_phenotype's loop will: a
        # minibatch of exactly one sample is dropped (nn.BatchNorm1d cannot compute
        # batch statistics from a single sample in train mode). Simulating the actual
        # per-batch sizes -- rather than only special-casing the trailing remainder --
        # is what catches batch_size=1, where *every* minibatch has size one and the
        # remainder-based check (n % batch_size) is always 0 and would miss it entirely.
        number_of_training_samples = train_features.shape[0]
        number_of_usable_training_minibatches = sum(
            1
            for batch_start in range(0, number_of_training_samples, batch_size)
            if min(batch_size, number_of_training_samples - batch_start) != 1
        )
        if number_of_usable_training_minibatches < 1:
            raise ValueError(
                "TrainedNetworkAccuracyEvaluator: train_features has "
                f"{number_of_training_samples} samples and batch_size is {batch_size}; "
                "after dropping every minibatch of exactly one sample (nn.BatchNorm1d "
                "cannot compute batch statistics from a single sample in train mode) "
                "there are no usable training minibatches left"
            )

        self._train_features = train_features.to(torch.float32).to(device_for_computation)
        self._train_labels = train_labels.to(torch.long).to(device_for_computation)
        self._validation_features = validation_features.to(torch.float32).to(
            device_for_computation
        )
        self._validation_labels = validation_labels.to(torch.long).to(device_for_computation)
        self._number_of_epochs = number_of_epochs
        self._learning_rate = learning_rate
        self._batch_size = batch_size
        self._device_for_computation = device_for_computation
        self._base_random_seed = base_random_seed
        self._generation_counter = 0

        if use_deterministic_algorithms:
            logger.warning(
                "use_deterministic_algorithms=True: evaluation is now reproducible "
                "bit-for-bit given the same seed, at the cost of disabling some fast "
                "non-deterministic CUDA kernels and raising RuntimeError for any "
                "operation with no deterministic implementation"
            )
            torch.use_deterministic_algorithms(True)

        logger.info(
            "TrainedNetworkAccuracyEvaluator ready: %d training samples, %d validation "
            "samples, %d epochs/phenotype, batch_size=%d, device=%s",
            number_of_training_samples,
            self._validation_features.shape[0],
            self._number_of_epochs,
            self._batch_size,
            self._device_for_computation,
        )

    def evaluate_batch_of_phenotypes(self, phenotypes: list[Phenotype]) -> list[FitnessValue]:
        """Train and score each phenotype in turn, in order.

        Args:
            phenotypes: Phenotypes to evaluate, typically
                :class:`~polyneat.algorithms.deepneat.torch_layer_stack_phenotype
                .TorchLayerStackPhenotype` instances. A phenotype with
                ``is_degenerate = True`` is scored ``0.0`` without training or
                any other interaction.

        Returns:
            One fitness (validation accuracy in ``[0.0, 1.0]``, or ``0.0`` for
            a degenerate phenotype) per input phenotype, in the same order.
        """
        logger.debug("Evaluating %d phenotypes sequentially, training each from scratch",
                      len(phenotypes))
        fitness_values: list[FitnessValue] = []
        for position_in_batch, phenotype in enumerate(phenotypes):
            if getattr(phenotype, "is_degenerate", False):
                logger.info(
                    "phenotype %d/%d is degenerate; scoring 0.0 without training",
                    position_in_batch + 1,
                    len(phenotypes),
                )
                fitness_values.append(0.0)
                continue

            derived_seed = (
                self._base_random_seed
                + self._generation_counter * _GENERATION_SEED_MULTIPLIER
                + position_in_batch * _POSITION_SEED_MULTIPLIER
            )
            torch.manual_seed(derived_seed)

            self._train_phenotype(phenotype)
            accuracy = self._measure_validation_accuracy(phenotype)
            logger.info(
                "phenotype %d/%d trained; validation accuracy %.4f",
                position_in_batch + 1,
                len(phenotypes),
                accuracy,
            )
            fitness_values.append(accuracy)

        self._generation_counter += 1
        return fitness_values

    def _train_phenotype(self, phenotype: Phenotype) -> None:
        """Run ``number_of_epochs`` passes of Adam over shuffled minibatches."""
        phenotype.train()
        optimizer = torch.optim.Adam(phenotype.parameters(), lr=self._learning_rate)
        loss_function = nn.CrossEntropyLoss()
        number_of_training_samples = self._train_features.shape[0]

        for _epoch in range(self._number_of_epochs):
            shuffled_indices = torch.randperm(
                number_of_training_samples, device=self._device_for_computation
            )
            for batch_start in range(0, number_of_training_samples, self._batch_size):
                batch_indices = shuffled_indices[batch_start : batch_start + self._batch_size]
                if batch_indices.shape[0] == 1:
                    # Ruling T9-A: nn.BatchNorm1d raises on a training minibatch of
                    # one. Drop it here, in training only -- never in validation.
                    continue
                optimizer.zero_grad()
                batch_logits = phenotype.forward_pass(self._train_features[batch_indices])
                loss = loss_function(batch_logits, self._train_labels[batch_indices])
                loss.backward()
                optimizer.step()

    def _measure_validation_accuracy(self, phenotype: Phenotype) -> float:
        """Fraction of correctly classified validation samples, in minibatches."""
        phenotype.eval()
        number_of_validation_samples = self._validation_features.shape[0]
        if number_of_validation_samples == 0:
            return 0.0
        number_of_correct_predictions = 0
        with torch.no_grad():
            for batch_start in range(0, number_of_validation_samples, self._batch_size):
                batch_end = batch_start + self._batch_size
                batch_logits = phenotype.forward_pass(
                    self._validation_features[batch_start:batch_end]
                )
                predicted_labels = torch.argmax(batch_logits, dim=1)
                number_of_correct_predictions += int(
                    (predicted_labels == self._validation_labels[batch_start:batch_end]).sum()
                )
        return number_of_correct_predictions / number_of_validation_samples
