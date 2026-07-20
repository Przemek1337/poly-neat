from __future__ import annotations

import pytest
import torch

from polyneat.algorithms.lneat.backpropagation_weight_trainer import (
    BackpropagationWeightTrainer,
)
from polyneat.algorithms.lneat.trainable_torch_phenotype import (
    TrainableTorchFeedForwardPhenotype,
)
from polyneat.core.neat.neat_genome import ConnectionGene, NEATGenome, NodeGene

_CPU = torch.device("cpu")


def _build_recognizer_genome(
    input_weight: float, bias_weight: float, extra_disabled: bool = False
) -> NEATGenome:
    connection_genes = [
        ConnectionGene(
            innovation_id=0, source_node_id=0, target_node_id=2,
            weight=input_weight, is_enabled=True,
        ),
        ConnectionGene(
            innovation_id=1, source_node_id=1, target_node_id=2,
            weight=bias_weight, is_enabled=True,
        ),
    ]
    if extra_disabled:
        connection_genes.append(
            ConnectionGene(
                innovation_id=2, source_node_id=0, target_node_id=2,
                weight=7.7, is_enabled=False,
            )
        )
    return NEATGenome(
        node_genes=(
            NodeGene(node_id=0, node_type="input", activation_function_name="identity"),
            NodeGene(node_id=1, node_type="bias", activation_function_name="identity"),
            NodeGene(node_id=2, node_type="output", activation_function_name="sigmoid"),
        ),
        connection_genes=tuple(connection_genes),
    )


# a separable toy task: output 1 for x >= 0.5, output 0 for x < 0.5.
# The learning set (the paper's A samples) is a strict subset of the
# classification set, so the two can disagree about Type 1 membership.
_FEATURES = torch.tensor([[0.0], [0.2], [0.8], [1.0]])
_TARGETS = torch.tensor([[0.0], [0.0], [1.0], [1.0]])
_CLASSIFICATION_FEATURES = torch.tensor([[0.0], [0.2], [0.45], [0.55], [0.8], [1.0]])
_CLASSIFICATION_TARGETS = torch.tensor([[0.0], [0.0], [0.0], [1.0], [1.0], [1.0]])


def _make_trainer(
    training_indicator: float = 0.2, number_of_iterations: int = 50
) -> BackpropagationWeightTrainer:
    return BackpropagationWeightTrainer(
        classification_features=_CLASSIFICATION_FEATURES,
        classification_binary_targets=_CLASSIFICATION_TARGETS,
        learning_sample_features=_FEATURES,
        learning_sample_binary_targets=_TARGETS,
        learning_rate=1.0,
        number_of_iterations=number_of_iterations,
        training_indicator=training_indicator,
        classification_threshold=0.5,
        device_for_computation=_CPU,
    )


def _mean_squared_error_of(genome: NEATGenome) -> float:
    phenotype = TrainableTorchFeedForwardPhenotype(genome, _CPU)
    with torch.no_grad():
        outputs = phenotype.forward_pass(_FEATURES)
    return float(torch.mean((outputs - _TARGETS) ** 2))


def test_training_reduces_loss_and_returns_new_genome() -> None:
    genome = _build_recognizer_genome(input_weight=0.1, bias_weight=0.1)
    trainer = _make_trainer()
    trained_genome = trainer.train_genome(genome)
    assert trained_genome is not genome
    assert _mean_squared_error_of(trained_genome) < _mean_squared_error_of(genome)


def test_disabled_connections_survive_training_untouched() -> None:
    genome = _build_recognizer_genome(input_weight=0.1, bias_weight=0.1, extra_disabled=True)
    trained_genome = _make_trainer().train_genome(genome)
    disabled_gene = trained_genome.get_connection_gene_by_innovation_id(2)
    assert disabled_gene is not None
    assert disabled_gene.weight == 7.7
    assert disabled_gene.is_enabled is False


def test_type_one_network_is_skipped() -> None:
    # steep decision: large weights saturate the sigmoid, outputs near-binary
    genome = _build_recognizer_genome(input_weight=40.0, bias_weight=-20.0)
    trainer = _make_trainer(training_indicator=0.3)
    assert trainer.genome_requires_training(genome) is False
    # skip returns the SAME object (identity contract used by LNEATAlgorithm)
    assert trainer.train_genome_if_learning_needed(genome) is genome


def test_incorrectly_classifying_network_requires_training() -> None:
    genome = _build_recognizer_genome(input_weight=-40.0, bias_weight=20.0)  # inverted
    trainer = _make_trainer()
    assert trainer.genome_requires_training(genome) is True
    trained_genome = trainer.train_genome_if_learning_needed(genome)
    assert trained_genome is not genome


def test_type_one_check_uses_the_classification_set_not_the_learning_set() -> None:
    # Decision boundary at x = 0.8: correct on all four learning samples with a
    # mean output distance of ~0.125 (Type 1 by the learning set alone), but it
    # misclassifies x = 0.55, which only the classification set contains. The
    # paper's four types live on the fitness surface, so the wider set decides.
    genome = _build_recognizer_genome(input_weight=40.0, bias_weight=-32.0)
    trainer = _make_trainer(training_indicator=0.2)
    assert trainer.genome_requires_training(genome) is True


def test_genome_without_enabled_connections_is_returned_unchanged() -> None:
    genome = NEATGenome(
        node_genes=(
            NodeGene(node_id=0, node_type="input", activation_function_name="identity"),
            NodeGene(node_id=2, node_type="output", activation_function_name="sigmoid"),
        ),
        connection_genes=(
            ConnectionGene(
                innovation_id=0, source_node_id=0, target_node_id=2,
                weight=1.0, is_enabled=False,
            ),
        ),
    )
    assert _make_trainer().train_genome(genome) is genome


def test_rejects_mismatched_feature_and_target_counts() -> None:
    with pytest.raises(ValueError):
        BackpropagationWeightTrainer(
            classification_features=_CLASSIFICATION_FEATURES,
            classification_binary_targets=_CLASSIFICATION_TARGETS,
            learning_sample_features=torch.zeros((3, 1)),
            learning_sample_binary_targets=_TARGETS,
            learning_rate=0.1,
            number_of_iterations=1,
            training_indicator=0.2,
            classification_threshold=0.5,
            device_for_computation=_CPU,
        )


def test_rejects_targets_with_wrong_shape() -> None:
    with pytest.raises(ValueError):
        BackpropagationWeightTrainer(
            classification_features=_CLASSIFICATION_FEATURES,
            classification_binary_targets=_CLASSIFICATION_TARGETS,
            learning_sample_features=_FEATURES,
            learning_sample_binary_targets=torch.tensor([0.0, 0.0, 1.0, 1.0]),  # 1-D
            learning_rate=0.1,
            number_of_iterations=1,
            training_indicator=0.2,
            classification_threshold=0.5,
            device_for_computation=_CPU,
        )


def test_rejects_mismatched_classification_feature_and_target_counts() -> None:
    with pytest.raises(ValueError):
        BackpropagationWeightTrainer(
            classification_features=torch.zeros((3, 1)),
            classification_binary_targets=_CLASSIFICATION_TARGETS,
            learning_sample_features=_FEATURES,
            learning_sample_binary_targets=_TARGETS,
            learning_rate=0.1,
            number_of_iterations=1,
            training_indicator=0.2,
            classification_threshold=0.5,
            device_for_computation=_CPU,
        )
