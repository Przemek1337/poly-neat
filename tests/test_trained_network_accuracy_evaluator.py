from __future__ import annotations

import inspect

import torch

from polyneat.algorithms.deepneat.deepneat_genome import (
    DeepNEATGenome,
    LayerNodeGene,
    TensorEdgeGene,
)
from polyneat.algorithms.deepneat.deepneat_phenotype_decoder import (
    DeepNEATPhenotypeDecoder,
)
from polyneat.algorithms.deepneat.layer_shape_propagation import TensorShape
from polyneat.evaluators.trained_network_accuracy_evaluator import (
    TrainedNetworkAccuracyEvaluator,
)


class _RaisesOnForwardPassDouble:
    """Stand-in for a degenerate phenotype: fails loudly if ever trained.

    ``is_degenerate = True`` should make the evaluator skip straight to a 0.0
    fitness without calling ``forward_pass`` (and therefore without calling
    ``train()``/``parameters()``/``eval()`` either, since a plain test double
    like this one implements none of those ``nn.Module`` methods).
    """

    is_degenerate = True

    def forward_pass(self, input_tensor: torch.Tensor) -> torch.Tensor:
        raise AssertionError("forward_pass must not be called for a degenerate phenotype")

    def reset_recurrent_state(self) -> None:
        return None


def _edge(innovation_id: int, source: int, target: int, enabled: bool = True) -> TensorEdgeGene:
    return TensorEdgeGene(
        innovation_id=innovation_id,
        source_node_id=source,
        target_node_id=target,
        is_enabled=enabled,
    )


def _flat_decoder(number_of_features: int, number_of_classes: int = 2) -> DeepNEATPhenotypeDecoder:
    return DeepNEATPhenotypeDecoder(
        input_shape=TensorShape.flat(features=number_of_features),
        number_of_classes=number_of_classes,
        maximum_total_parameter_count=1_000_000,
        device_for_computation=torch.device("cpu"),
    )


def _linear_classifier_genome() -> DeepNEATGenome:
    return DeepNEATGenome(
        node_genes=(
            LayerNodeGene(node_id=0, layer_type="input"),
            LayerNodeGene(node_id=1, layer_type="output"),
        ),
        edge_genes=(_edge(0, 0, 1),),
    )


def _dense_with_batch_norm_genome() -> DeepNEATGenome:
    return DeepNEATGenome(
        node_genes=(
            LayerNodeGene(node_id=0, layer_type="input"),
            LayerNodeGene(node_id=1, layer_type="output"),
            LayerNodeGene(
                node_id=2,
                layer_type="dense",
                number_of_units=8,
                uses_batch_normalization=True,
            ),
        ),
        edge_genes=(_edge(0, 0, 2), _edge(1, 2, 1)),
    )


def _toy_dataset(
    number_of_samples: int, number_of_features: int = 4
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(0)
    features = torch.randn(number_of_samples, number_of_features, generator=generator)
    labels = (features.sum(dim=1) > 0).long()
    return features, labels


def _evaluator(
    train_size: int = 8,
    validation_size: int = 4,
    number_of_features: int = 4,
    number_of_epochs: int = 2,
    batch_size: int = 4,
    base_random_seed: int = 42,
) -> TrainedNetworkAccuracyEvaluator:
    train_features, train_labels = _toy_dataset(train_size, number_of_features)
    validation_features, validation_labels = _toy_dataset(validation_size, number_of_features)
    return TrainedNetworkAccuracyEvaluator(
        train_features=train_features,
        train_labels=train_labels,
        validation_features=validation_features,
        validation_labels=validation_labels,
        number_of_epochs=number_of_epochs,
        learning_rate=0.05,
        batch_size=batch_size,
        device_for_computation=torch.device("cpu"),
        base_random_seed=base_random_seed,
    )


def test_degenerate_phenotype_gets_zero_fitness_without_training() -> None:
    evaluator = _evaluator()
    fitnesses = evaluator.evaluate_batch_of_phenotypes([_RaisesOnForwardPassDouble()])
    assert fitnesses == [0.0]


def test_same_phenotype_and_seed_give_the_same_result_twice() -> None:
    genome = _dense_with_batch_norm_genome()
    decoder = _flat_decoder(number_of_features=4)

    # Deliberately construct from different global RNG states. The evaluator,
    # not the caller/decoder, owns the fresh initialization seed.
    torch.manual_seed(123)
    phenotype_one = decoder.build_phenotype_from_genome(genome)
    fitness_one = _evaluator(base_random_seed=7).evaluate_batch_of_phenotypes([phenotype_one])[0]

    torch.manual_seed(987654)
    phenotype_two = decoder.build_phenotype_from_genome(genome)
    fitness_two = _evaluator(base_random_seed=7).evaluate_batch_of_phenotypes([phenotype_two])[0]

    assert fitness_one == fitness_two
    assert all(
        torch.equal(first.detach(), second.detach())
        for first, second in zip(
            phenotype_one.parameters(), phenotype_two.parameters(), strict=True
        )
    )


def test_identical_genomes_receive_common_random_numbers_independent_of_position() -> None:
    genome = _dense_with_batch_norm_genome()
    decoder = _flat_decoder(number_of_features=4)
    torch.manual_seed(1)
    first = decoder.build_phenotype_from_genome(genome)
    torch.manual_seed(2)
    second = decoder.build_phenotype_from_genome(genome)

    evaluator = _evaluator(base_random_seed=17)
    fitnesses = evaluator.evaluate_batch_of_phenotypes([first, second])

    assert fitnesses[0] == fitnesses[1]
    assert all(
        torch.equal(first_parameter.detach(), second_parameter.detach())
        for first_parameter, second_parameter in zip(
            first.parameters(), second.parameters(), strict=True
        )
    )


def test_fitness_is_within_the_unit_interval() -> None:
    decoder = _flat_decoder(number_of_features=4)
    phenotype = decoder.build_phenotype_from_genome(_linear_classifier_genome())
    fitness = _evaluator().evaluate_batch_of_phenotypes([phenotype])[0]
    assert 0.0 <= fitness <= 1.0


def test_result_order_matches_phenotype_order_and_is_not_swapped() -> None:
    decoder = _flat_decoder(number_of_features=4)
    real_phenotype = decoder.build_phenotype_from_genome(_linear_classifier_genome())
    evaluator = _evaluator()

    degenerate_first = evaluator.evaluate_batch_of_phenotypes(
        [_RaisesOnForwardPassDouble(), real_phenotype]
    )
    assert degenerate_first[0] == 0.0
    assert 0.0 <= degenerate_first[1] <= 1.0

    degenerate_second = evaluator.evaluate_batch_of_phenotypes(
        [real_phenotype, _RaisesOnForwardPassDouble()]
    )
    assert degenerate_second[1] == 0.0
    assert 0.0 <= degenerate_second[0] <= 1.0


def test_evaluator_consumes_phenotypes_not_genomes() -> None:
    # The evaluator never sees a genome at all, so weights trained during
    # evaluation cannot be written back into one: there is no genome argument
    # to write them into. Checking the signature is the whole test.
    signature = inspect.signature(TrainedNetworkAccuracyEvaluator.evaluate_batch_of_phenotypes)
    parameter_names = list(signature.parameters)
    assert parameter_names == ["self", "phenotypes"]


def test_evaluation_leaves_a_trained_phenotype_in_eval_mode() -> None:
    # Scoped to the trained path: the degenerate short-circuit does not touch
    # the phenotype at all (see module docstring), so it is not covered here.
    decoder = _flat_decoder(number_of_features=4)
    phenotype = decoder.build_phenotype_from_genome(_linear_classifier_genome())
    _evaluator().evaluate_batch_of_phenotypes([phenotype])
    assert phenotype.training is False


def test_trailing_minibatch_of_one_is_dropped_during_training() -> None:
    # Ruling T9-A: 9 training samples with batch_size=4 leaves a trailing
    # minibatch of exactly 1 sample (9 % 4 == 1). nn.BatchNorm1d raises in
    # train mode on a batch of one, so without the drop this test fails with
    # a ValueError from inside PyTorch. Do not "simplify" 9 to a number whose
    # remainder is not 1 -- that would silently delete this coverage.
    decoder = _flat_decoder(number_of_features=4)
    phenotype = decoder.build_phenotype_from_genome(_dense_with_batch_norm_genome())
    evaluator = _evaluator(train_size=9, validation_size=4, batch_size=4, number_of_epochs=1)
    fitnesses = evaluator.evaluate_batch_of_phenotypes([phenotype])
    assert len(fitnesses) == 1
    assert 0.0 <= fitnesses[0] <= 1.0


def test_dropping_the_trailing_singleton_leaves_no_batches_raises() -> None:
    # A training set of exactly 1 sample with batch_size > 1: the single
    # trailing minibatch is a singleton and must be dropped, which leaves
    # zero usable minibatches -- a real misconfiguration, not something to
    # silently train on nothing about.
    train_features, train_labels = _toy_dataset(1, number_of_features=4)
    validation_features, validation_labels = _toy_dataset(4, number_of_features=4)
    try:
        TrainedNetworkAccuracyEvaluator(
            train_features=train_features,
            train_labels=train_labels,
            validation_features=validation_features,
            validation_labels=validation_labels,
            number_of_epochs=1,
            learning_rate=0.05,
            batch_size=4,
            device_for_computation=torch.device("cpu"),
            base_random_seed=1,
        )
    except ValueError as error:
        assert "train_features" in str(error) or "batch_size" in str(error)
    else:
        raise AssertionError("expected a ValueError for a training set with no usable minibatches")


def test_batch_size_of_one_is_rejected_at_construction() -> None:
    # Critical Finding 1: with batch_size=1 EVERY minibatch has size 1, so a
    # remainder-only check (n % batch_size) is always 0 and misses this case
    # entirely -- the training loop would then drop every single minibatch,
    # run zero optimizer steps, and report an untrained network's accuracy as
    # a real fitness with no error and nothing in the logs to flag it. This is
    # the single most dangerous value for batch_size and must be rejected
    # up front, not discovered by a silently-meaningless result. Do not
    # "simplify" this test away if the guard is ever refactored -- re-derive
    # the check from what the training loop actually drops (see the
    # constructor's own comment) rather than special-casing 1 again.
    train_features, train_labels = _toy_dataset(8, number_of_features=4)
    validation_features, validation_labels = _toy_dataset(4, number_of_features=4)
    try:
        TrainedNetworkAccuracyEvaluator(
            train_features=train_features,
            train_labels=train_labels,
            validation_features=validation_features,
            validation_labels=validation_labels,
            number_of_epochs=1,
            learning_rate=0.05,
            batch_size=1,
            device_for_computation=torch.device("cpu"),
            base_random_seed=1,
        )
    except ValueError as error:
        assert "train_features" in str(error) or "batch_size" in str(error)
    else:
        raise AssertionError("expected a ValueError for batch_size=1 (every minibatch dropped)")


def test_training_actually_changes_the_phenotypes_parameters() -> None:
    # Important Finding 2: none of the other tests would fail if
    # optimizer.step() were silently removed from _train_phenotype (forward,
    # loss and backward() still run; BatchNorm still gets a real minibatch to
    # compute statistics from; the seed still makes everything reproducible;
    # bounds and ordering are untouched). Only comparing parameters before and
    # after proves backpropagation actually updated the network. Snapshots are
    # cloned (not just referenced) since the live parameter tensors are the
    # exact objects being mutated in place by the optimizer.
    decoder = _flat_decoder(number_of_features=4)
    phenotype = decoder.build_phenotype_from_genome(_dense_with_batch_norm_genome())
    parameters_before_training = [
        parameter.detach().clone() for parameter in phenotype.parameters()
    ]

    _evaluator(number_of_epochs=3).evaluate_batch_of_phenotypes([phenotype])

    parameters_after_training = list(phenotype.parameters())
    assert len(parameters_before_training) == len(parameters_after_training)
    assert any(
        not torch.equal(before, after.detach())
        for before, after in zip(
            parameters_before_training, parameters_after_training, strict=True
        )
    )


def test_generation_counter_increments_once_per_batch_call() -> None:
    # Indirect check: two calls at the same batch position must derive
    # different seeds (because the generation counter advanced), so two
    # freshly-built, identically-seeded phenotypes trained by two separate
    # calls on the same evaluator diverge from what a single call with the
    # same base seed would reproduce a second time only via a *fresh*
    # evaluator (see test_same_phenotype_and_seed_give_the_same_result_twice).
    # Here we simply assert the private counter itself advances by exactly one.
    evaluator = _evaluator()
    assert evaluator._generation_counter == 0
    evaluator.evaluate_batch_of_phenotypes([_RaisesOnForwardPassDouble()])
    assert evaluator._generation_counter == 1
    evaluator.evaluate_batch_of_phenotypes([_RaisesOnForwardPassDouble()])
    assert evaluator._generation_counter == 2
    # Minor Finding 3: the empty batch is a distinct case from "every phenotype is
    # degenerate" -- there is no loop iteration at all -- and the increment sits
    # unconditionally after the loop, so it must still fire here.
    assert evaluator.evaluate_batch_of_phenotypes([]) == []
    assert evaluator._generation_counter == 3


def test_use_deterministic_algorithms_flag_is_accepted() -> None:
    train_features, train_labels = _toy_dataset(8, number_of_features=4)
    validation_features, validation_labels = _toy_dataset(4, number_of_features=4)
    try:
        evaluator = TrainedNetworkAccuracyEvaluator(
            train_features=train_features,
            train_labels=train_labels,
            validation_features=validation_features,
            validation_labels=validation_labels,
            number_of_epochs=1,
            learning_rate=0.05,
            batch_size=4,
            device_for_computation=torch.device("cpu"),
            base_random_seed=1,
            use_deterministic_algorithms=True,
        )
        decoder = _flat_decoder(number_of_features=4)
        phenotype = decoder.build_phenotype_from_genome(_linear_classifier_genome())
        fitnesses = evaluator.evaluate_batch_of_phenotypes([phenotype])
        assert len(fitnesses) == 1
    finally:
        torch.use_deterministic_algorithms(False)
