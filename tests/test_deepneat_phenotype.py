from __future__ import annotations

import torch

from polyneat.algorithms.deepneat.deepneat_genome import (
    DeepNEATGenome,
    DeepNEATGlobalHyperparameters,
    LayerNodeGene,
    TensorEdgeGene,
)
from polyneat.algorithms.deepneat.deepneat_phenotype_decoder import (
    DeepNEATPhenotypeDecoder,
)
from polyneat.algorithms.deepneat.layer_shape_propagation import TensorShape


def _edge(innovation_id: int, source: int, target: int, enabled: bool = True):
    return TensorEdgeGene(
        innovation_id=innovation_id, source_node_id=source,
        target_node_id=target, is_enabled=enabled,
    )


def _decoder(maximum_parameters: int | None = 20_000_000) -> DeepNEATPhenotypeDecoder:
    return DeepNEATPhenotypeDecoder(
        input_shape=TensorShape.spatial(channels=1, height=8, width=8),
        number_of_classes=10,
        maximum_total_parameter_count=maximum_parameters,
        device_for_computation=torch.device("cpu"),
    )


def _linear_classifier() -> DeepNEATGenome:
    return DeepNEATGenome(
        node_genes=(
            LayerNodeGene(node_id=0, layer_type="input"),
            LayerNodeGene(node_id=1, layer_type="output"),
        ),
        edge_genes=(_edge(0, 0, 1),),
    )


def test_minimal_genome_produces_class_logits() -> None:
    phenotype = _decoder().build_phenotype_from_genome(_linear_classifier())
    assert not phenotype.is_degenerate
    outputs = phenotype.forward_pass(torch.randn(4, 1, 8, 8))
    assert outputs.shape == (4, 10)


def test_conv_stack_runs_end_to_end() -> None:
    genome = DeepNEATGenome(
        node_genes=(
            LayerNodeGene(node_id=0, layer_type="input"),
            LayerNodeGene(node_id=1, layer_type="output"),
            LayerNodeGene(
                node_id=2, layer_type="conv", number_of_filters=8, kernel_size=3,
                uses_batch_normalization=True, is_followed_by_max_pooling=True,
                dropout_rate=0.25,
            ),
            LayerNodeGene(node_id=3, layer_type="dense", number_of_units=32),
        ),
        edge_genes=(_edge(0, 0, 2), _edge(1, 2, 3), _edge(2, 3, 1)),
    )
    phenotype = _decoder().build_phenotype_from_genome(genome)
    assert not phenotype.is_degenerate
    assert phenotype.forward_pass(torch.randn(2, 1, 8, 8)).shape == (2, 10)


def test_skip_connection_merges_two_paths() -> None:
    genome = DeepNEATGenome(
        node_genes=(
            LayerNodeGene(node_id=0, layer_type="input"),
            LayerNodeGene(node_id=1, layer_type="output"),
            LayerNodeGene(
                node_id=2, layer_type="conv", number_of_filters=4, kernel_size=3,
                is_followed_by_max_pooling=True,
            ),
        ),
        edge_genes=(_edge(0, 0, 2), _edge(1, 2, 1), _edge(2, 0, 1)),
    )
    phenotype = _decoder().build_phenotype_from_genome(genome)
    output_layer = phenotype._layer_modules_by_node_id["1"]
    # The source requires the 8x8 shortcut to be max-pooled to the smallest
    # parent output (4x4) before concatenation: (1 + 4) * 4 * 4 = 80.
    assert output_layer.in_features == 80
    assert phenotype.forward_pass(torch.randn(3, 1, 8, 8)).shape == (3, 10)


def test_conv_after_dense_is_degenerate_instead_of_being_coerced() -> None:
    genome = DeepNEATGenome(
        node_genes=(
            LayerNodeGene(node_id=0, layer_type="input"),
            LayerNodeGene(node_id=1, layer_type="output"),
            LayerNodeGene(node_id=2, layer_type="dense", number_of_units=16),
            LayerNodeGene(node_id=3, layer_type="conv", number_of_filters=8, kernel_size=3),
        ),
        edge_genes=(_edge(0, 0, 2), _edge(1, 2, 3), _edge(2, 3, 1)),
    )
    phenotype = _decoder().build_phenotype_from_genome(genome)
    assert phenotype.is_degenerate
    assert phenotype.number_of_layer_modules == 0


def test_genome_without_a_path_is_degenerate() -> None:
    genome = DeepNEATGenome(
        node_genes=(
            LayerNodeGene(node_id=0, layer_type="input"),
            LayerNodeGene(node_id=1, layer_type="output"),
        ),
        edge_genes=(_edge(0, 0, 1, enabled=False),),
    )
    phenotype = _decoder().build_phenotype_from_genome(genome)
    assert phenotype.is_degenerate


def test_degenerate_phenotype_still_returns_a_correctly_shaped_tensor() -> None:
    genome = DeepNEATGenome(
        node_genes=(
            LayerNodeGene(node_id=0, layer_type="input"),
            LayerNodeGene(node_id=1, layer_type="output"),
        ),
        edge_genes=(_edge(0, 0, 1, enabled=False),),
    )
    phenotype = _decoder().build_phenotype_from_genome(genome)
    outputs = phenotype.forward_pass(torch.randn(5, 1, 8, 8))
    assert outputs.shape == (5, 10)
    assert torch.count_nonzero(outputs) == 0


def test_exceeding_the_parameter_budget_makes_the_phenotype_degenerate() -> None:
    phenotype = _decoder(maximum_parameters=10).build_phenotype_from_genome(
        _linear_classifier()
    )
    assert phenotype.is_degenerate


def test_none_parameter_budget_does_not_add_a_source_unspecified_fitness_cutoff() -> None:
    phenotype = _decoder(maximum_parameters=None).build_phenotype_from_genome(
        _linear_classifier()
    )
    assert not phenotype.is_degenerate


def test_parameter_count_is_reported() -> None:
    phenotype = _decoder().build_phenotype_from_genome(_linear_classifier())
    # one Linear(64, 10): 64*10 weights + 10 biases
    assert phenotype.total_parameter_count == 650


def test_layer_module_count_matches_the_modules_actually_built() -> None:
    """The layer count must mean the same thing the parameter count means.

    ``number_of_layer_modules`` is what the examples report as
    ``number_of_layers``, so it has to count modules, not retained nodes: the
    input node sits on the path but carries no module. A linear classifier is
    two nodes and exactly one ``Linear``.
    """
    phenotype = _decoder().build_phenotype_from_genome(_linear_classifier())

    assert phenotype.number_of_layer_modules == 1
    assert phenotype.total_parameter_count == 650


def test_degenerate_phenotype_reports_no_layer_modules() -> None:
    genome = DeepNEATGenome(
        node_genes=(
            LayerNodeGene(node_id=0, layer_type="input"),
            LayerNodeGene(node_id=1, layer_type="output"),
        ),
        edge_genes=(_edge(0, 0, 1, enabled=False),),
    )
    phenotype = _decoder().build_phenotype_from_genome(genome)

    assert phenotype.number_of_layer_modules == 0
    assert phenotype.total_parameter_count == 0


def test_over_budget_phenotype_still_reports_the_size_it_would_have_had() -> None:
    """Both size metrics survive the budget rejection, and survive together.

    The modules are dropped to free the memory, but the counts describe the
    network that was built, which is what makes the rejection legible.
    """
    phenotype = _decoder(maximum_parameters=10).build_phenotype_from_genome(
        _linear_classifier()
    )

    assert phenotype.is_degenerate
    assert phenotype.number_of_layer_modules == 1
    assert phenotype.total_parameter_count == 650


def test_dead_end_branch_contributes_no_parameters() -> None:
    with_dead_end = DeepNEATGenome(
        node_genes=(
            LayerNodeGene(node_id=0, layer_type="input"),
            LayerNodeGene(node_id=1, layer_type="output"),
            LayerNodeGene(node_id=2, layer_type="dense", number_of_units=256),
        ),
        edge_genes=(_edge(0, 0, 1), _edge(1, 0, 2)),
    )
    assert _decoder().build_phenotype_from_genome(
        with_dead_end
    ).total_parameter_count == _decoder().build_phenotype_from_genome(
        _linear_classifier()
    ).total_parameter_count


def test_phenotype_is_an_nn_module_and_trainable() -> None:
    phenotype = _decoder().build_phenotype_from_genome(_linear_classifier())
    assert isinstance(phenotype, torch.nn.Module)
    assert any(parameter.requires_grad for parameter in phenotype.parameters())
    loss = phenotype.forward_pass(torch.randn(2, 1, 8, 8)).sum()
    loss.backward()


def test_reset_recurrent_state_is_a_no_op() -> None:
    _decoder().build_phenotype_from_genome(_linear_classifier()).reset_recurrent_state()


def test_evolved_initial_weight_scaling_is_reapplied_after_reset() -> None:
    genome = DeepNEATGenome(
        node_genes=(
            LayerNodeGene(node_id=0, layer_type="input"),
            LayerNodeGene(node_id=1, layer_type="output"),
            LayerNodeGene(
                node_id=2,
                layer_type="dense",
                number_of_units=8,
                initial_weight_scaling=0.0,
            ),
        ),
        edge_genes=(_edge(0, 0, 2), _edge(1, 2, 1)),
    )
    phenotype = _decoder().build_phenotype_from_genome(genome)
    phenotype.reinitialize_parameters()
    hidden_linear = phenotype._layer_modules_by_node_id["2"][0]
    assert torch.count_nonzero(hidden_linear.weight) == 0


def test_evolved_crop_size_is_used_during_shape_propagation() -> None:
    genome = DeepNEATGenome(
        node_genes=(
            LayerNodeGene(node_id=0, layer_type="input"),
            LayerNodeGene(node_id=1, layer_type="output"),
        ),
        edge_genes=(_edge(0, 0, 1),),
        global_hyperparameters=DeepNEATGlobalHyperparameters(cropped_image_size=6),
    )
    phenotype = _decoder().build_phenotype_from_genome(genome)
    output_layer = phenotype._layer_modules_by_node_id["1"]
    assert output_layer.in_features == 36
    assert phenotype.forward_pass(torch.ones(2, 1, 6, 6)).shape == (2, 10)
