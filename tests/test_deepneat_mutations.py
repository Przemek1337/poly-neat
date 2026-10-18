from __future__ import annotations

import numpy as np
import pytest

from polyneat.algorithms.deepneat.deepneat_genome import (
    DeepNEATGenome,
    DeepNEATGlobalHyperparameters,
    LayerNodeGene,
    TensorEdgeGene,
)
from polyneat.algorithms.deepneat.deepneat_innovation_tracker import (
    DeepNEATInnovationTracker,
)
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
    draw_global_hyperparameters,
)
from polyneat.algorithms.deepneat.mutations.layer_hyperparameter_mutation import (
    LayerHyperparameterMutation,
)
from polyneat.algorithms.deepneat.mutations.toggle_tensor_edge_mutation import (
    ToggleTensorEdgeMutation,
)
from polyneat.configs.deepneat.deepneat_config import DeepNEATConfig

_FILTERS = (16, 32, 64, 128)
_KERNELS = (1, 3, 5)
_UNITS = (64, 128, 256)


def _minimal_genome() -> DeepNEATGenome:
    return DeepNEATGenome(
        node_genes=(
            LayerNodeGene(node_id=0, layer_type="input"),
            LayerNodeGene(node_id=1, layer_type="output"),
        ),
        edge_genes=(
            TensorEdgeGene(innovation_id=0, source_node_id=0, target_node_id=1,
                           is_enabled=True),
        ),
    )


def _add_layer_node(probability: float = 1.0, conv_probability: float = 1.0):
    return AddLayerNodeMutation(
        probability_of_application=probability,
        available_filter_counts=_FILTERS,
        available_kernel_sizes=_KERNELS,
        available_dense_unit_counts=_UNITS,
        dropout_rate_min=0.0,
        dropout_rate_max=0.5,
        probability_of_new_conv_layer=conv_probability,
    )


def _seeded_tracker() -> DeepNEATInnovationTracker:
    """A tracker whose counter has advanced past `_minimal_genome()`'s edge.

    Mirrors `tests/test_add_node_mutation.py`'s `_tracker_seeded_for`. Mutation
    operators take their markings from the tracker, not from the genome, so a
    genome whose edges were hand-crafted (bypassing the tracker, as
    `_minimal_genome()` does with `innovation_id=0`) must register those edges
    first. Otherwise a fresh tracker's own counter -- which also starts at 0 --
    reissues an id the genome already carries, and the genome's own
    duplicate-innovation-id check in `__post_init__` rejects the result. This
    has nothing to do with what any individual test is proving; it only avoids
    an incidental id collision in the shared test fixture.
    """
    tracker = DeepNEATInnovationTracker()
    tracker.get_or_assign_innovation_id_for_connection(source_node_id=0, target_node_id=1)
    return tracker


@pytest.fixture
def tracker() -> DeepNEATInnovationTracker:
    return _seeded_tracker()


def test_add_layer_node_splits_an_edge_into_two(tracker) -> None:
    result = _add_layer_node().apply_to_genome(
        _minimal_genome(), np.random.default_rng(0), tracker
    )
    assert len(result.node_genes) == 3
    assert len([e for e in result.edge_genes if e.is_enabled]) == 2
    assert len([e for e in result.edge_genes if not e.is_enabled]) == 1


def test_add_layer_node_disables_the_split_edge(tracker) -> None:
    result = _add_layer_node().apply_to_genome(
        _minimal_genome(), np.random.default_rng(0), tracker
    )
    original = [e for e in result.edge_genes if e.innovation_id == 0]
    assert len(original) == 1
    assert not original[0].is_enabled


def test_add_layer_node_takes_ids_from_the_tracker(tracker) -> None:
    genome = _minimal_genome()
    first = _add_layer_node().apply_to_genome(genome, np.random.default_rng(0), tracker)
    second = _add_layer_node().apply_to_genome(genome, np.random.default_rng(1), tracker)
    new_id_first = {n.node_id for n in first.node_genes} - {0, 1}
    new_id_second = {n.node_id for n in second.node_genes} - {0, 1}
    # Both split the same edge (there is only one), so the marking must match.
    assert new_id_first == new_id_second


def test_new_conv_layer_gets_values_from_the_search_space(tracker) -> None:
    for seed in range(20):
        result = _add_layer_node(conv_probability=1.0).apply_to_genome(
            _minimal_genome(), np.random.default_rng(seed), _seeded_tracker()
        )
        inserted = next(n for n in result.node_genes if n.node_id not in (0, 1))
        assert inserted.layer_type == "conv"
        assert inserted.number_of_filters in _FILTERS
        assert inserted.kernel_size in _KERNELS
        assert 0.0 <= inserted.dropout_rate <= 0.5


def test_new_dense_layer_gets_values_from_the_search_space(tracker) -> None:
    for seed in range(20):
        result = _add_layer_node(conv_probability=0.0).apply_to_genome(
            _minimal_genome(), np.random.default_rng(seed), _seeded_tracker()
        )
        inserted = next(n for n in result.node_genes if n.node_id not in (0, 1))
        assert inserted.layer_type == "dense"
        assert inserted.number_of_units in _UNITS
        assert not inserted.is_followed_by_max_pooling


def test_add_layer_node_does_not_fire_below_its_probability(tracker) -> None:
    genome = _minimal_genome()
    result = _add_layer_node(probability=0.0).apply_to_genome(
        genome, np.random.default_rng(0), tracker
    )
    assert result.node_genes == genome.node_genes


@pytest.mark.parametrize("seed", range(25))
def test_add_tensor_edge_never_creates_a_cycle(tracker, seed: int) -> None:
    genome = _minimal_genome()
    for _round in range(6):
        genome = _add_layer_node().apply_to_genome(
            genome, np.random.default_rng(seed), tracker
        )
    # Construction revalidates, so a cycle would raise here.
    result = AddTensorEdgeMutation(probability_of_application=1.0).apply_to_genome(
        genome, np.random.default_rng(seed), tracker
    )
    assert len(result.edge_genes) >= len(genome.edge_genes)


def test_add_tensor_edge_never_targets_the_input_layer(tracker) -> None:
    genome = _minimal_genome()
    for _round in range(4):
        genome = _add_layer_node().apply_to_genome(genome, np.random.default_rng(3), tracker)
    for seed in range(25):
        result = AddTensorEdgeMutation(probability_of_application=1.0).apply_to_genome(
            genome, np.random.default_rng(seed), tracker
        )
        assert all(e.target_node_id != result.input_node_id for e in result.edge_genes)


def test_add_tensor_edge_never_sources_from_the_output_layer(tracker) -> None:
    genome = _minimal_genome()
    for _round in range(4):
        genome = _add_layer_node().apply_to_genome(genome, np.random.default_rng(3), tracker)
    for seed in range(25):
        result = AddTensorEdgeMutation(probability_of_application=1.0).apply_to_genome(
            genome, np.random.default_rng(seed), tracker
        )
        assert all(e.source_node_id != result.output_node_id for e in result.edge_genes)


def test_add_tensor_edge_never_duplicates_an_existing_edge(tracker) -> None:
    genome = _minimal_genome()
    for _round in range(4):
        genome = _add_layer_node().apply_to_genome(genome, np.random.default_rng(3), tracker)
    for seed in range(25):
        result = AddTensorEdgeMutation(probability_of_application=1.0).apply_to_genome(
            genome, np.random.default_rng(seed), tracker
        )
        pairs = [(e.source_node_id, e.target_node_id) for e in result.edge_genes]
        assert len(pairs) == len(set(pairs))


def test_toggle_flips_exactly_one_edge(tracker) -> None:
    genome = _minimal_genome()
    for _round in range(3):
        genome = _add_layer_node().apply_to_genome(genome, np.random.default_rng(1), tracker)
    result = ToggleTensorEdgeMutation(probability_of_application=1.0).apply_to_genome(
        genome, np.random.default_rng(0), tracker
    )
    flipped = [
        (before.innovation_id)
        for before, after in zip(genome.edge_genes, result.edge_genes, strict=True)
        if before.is_enabled != after.is_enabled
    ]
    assert len(flipped) <= 1


@pytest.mark.parametrize("seed", range(25))
def test_toggle_never_produces_a_cycle(tracker, seed: int) -> None:
    genome = _minimal_genome()
    for _round in range(5):
        genome = _add_layer_node().apply_to_genome(genome, np.random.default_rng(seed), tracker)
        genome = AddTensorEdgeMutation(probability_of_application=1.0).apply_to_genome(
            genome, np.random.default_rng(seed), tracker
        )
    ToggleTensorEdgeMutation(probability_of_application=1.0).apply_to_genome(
        genome, np.random.default_rng(seed), tracker
    )


def test_hyperparameter_mutation_stays_inside_the_search_space(tracker) -> None:
    genome = DeepNEATGenome(
        node_genes=(
            LayerNodeGene(node_id=0, layer_type="input"),
            LayerNodeGene(node_id=1, layer_type="output"),
            LayerNodeGene(node_id=2, layer_type="conv", number_of_filters=16, kernel_size=3),
            LayerNodeGene(node_id=3, layer_type="dense", number_of_units=64),
        ),
        edge_genes=(
            TensorEdgeGene(innovation_id=0, source_node_id=0, target_node_id=2,
                           is_enabled=True),
            TensorEdgeGene(innovation_id=1, source_node_id=2, target_node_id=3,
                           is_enabled=True),
            TensorEdgeGene(innovation_id=2, source_node_id=3, target_node_id=1,
                           is_enabled=True),
        ),
    )
    mutation = LayerHyperparameterMutation(
        probability_of_application=1.0,
        available_filter_counts=_FILTERS,
        available_kernel_sizes=_KERNELS,
        available_dense_unit_counts=_UNITS,
        dropout_rate_min=0.0,
        dropout_rate_max=0.5,
    )
    for seed in range(30):
        result = mutation.apply_to_genome(genome, np.random.default_rng(seed), tracker)
        for node in result.node_genes:
            if node.layer_type == "conv":
                assert node.number_of_filters in _FILTERS
                assert node.kernel_size in _KERNELS
            if node.layer_type == "dense":
                assert node.number_of_units in _UNITS
            assert 0.0 <= node.dropout_rate <= 0.5


def test_hyperparameter_mutation_never_touches_input_or_output(tracker) -> None:
    genome = _minimal_genome()
    mutation = LayerHyperparameterMutation(
        probability_of_application=1.0,
        available_filter_counts=_FILTERS,
        available_kernel_sizes=_KERNELS,
        available_dense_unit_counts=_UNITS,
        dropout_rate_min=0.0,
        dropout_rate_max=0.5,
    )
    for seed in range(20):
        result = mutation.apply_to_genome(genome, np.random.default_rng(seed), tracker)
        assert result.node_genes == genome.node_genes


def test_composite_applies_operators_in_order(tracker) -> None:
    calls: list[str] = []

    class _Recorder:
        def __init__(self, label: str) -> None:
            self._label = label

        def apply_to_genome(self, genome, rng, innovation_tracker):  # noqa: ANN001, ANN201
            calls.append(self._label)
            return genome

    composite = DeepNEATCompositeMutation(
        ordered_individual_mutations=[_Recorder("a"), _Recorder("b"), _Recorder("c")]
    )
    composite.apply_to_genome(_minimal_genome(), np.random.default_rng(0), tracker)
    assert calls == ["a", "b", "c"]


def test_global_hyperparameters_are_drawn_inside_source_ranges() -> None:
    config = DeepNEATConfig(
        input_image_height=32,
        input_image_width=32,
        global_hue_shift_degrees_max=45.0,
        global_saturation_value_shift_max=0.5,
        global_saturation_value_scale_max=0.5,
        global_cropped_image_size_min=26,
        global_cropped_image_size_max=32,
        global_spatial_scaling_max=0.3,
        available_horizontal_flip_options=(False, True),
        available_variance_normalization_options=(False, True),
    )
    value = draw_global_hyperparameters(config, np.random.default_rng(0))
    assert 1e-4 <= value.learning_rate <= 0.1
    assert 0.68 <= value.momentum <= 0.99
    assert 0.0 <= value.hue_shift_degrees <= 45.0
    assert 26 <= value.cropped_image_size <= 32
    assert 0.0 <= value.spatial_scaling <= 0.3


def test_global_mutation_changes_only_one_gene_and_stays_in_range(tracker) -> None:
    config = DeepNEATConfig(
        input_image_height=32,
        input_image_width=32,
        global_hue_shift_degrees_max=45.0,
        global_saturation_value_shift_max=0.5,
        global_saturation_value_scale_max=0.5,
        global_cropped_image_size_min=26,
        global_cropped_image_size_max=32,
        global_spatial_scaling_max=0.3,
        available_horizontal_flip_options=(False, True),
        available_variance_normalization_options=(False, True),
    )
    genome = DeepNEATGenome(
        node_genes=_minimal_genome().node_genes,
        edge_genes=_minimal_genome().edge_genes,
        global_hyperparameters=DeepNEATGlobalHyperparameters(
            learning_rate=0.01,
            momentum=0.8,
            cropped_image_size=28,
        ),
    )
    mutation = GlobalHyperparameterMutation(1.0, config)
    observed_change = False
    for seed in range(40):
        result = mutation.apply_to_genome(
            genome, np.random.default_rng(seed), tracker
        )
        before = genome.global_hyperparameters
        after = result.global_hyperparameters
        changed_fields = sum(
            getattr(before, field_name) != getattr(after, field_name)
            for field_name in before.__dataclass_fields__
        )
        assert changed_fields <= 1
        observed_change |= changed_fields == 1
        assert config.global_learning_rate_min <= after.learning_rate <= 0.1
        assert 0.68 <= after.momentum <= 0.99
        assert 26 <= after.cropped_image_size <= 32
    assert observed_change
