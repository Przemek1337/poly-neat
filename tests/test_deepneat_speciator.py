from __future__ import annotations

import numpy as np

from polyneat.algorithms.deepneat.deepneat_genome import (
    DeepNEATGenome,
    LayerNodeGene,
    TensorEdgeGene,
)
from polyneat.algorithms.deepneat.deepneat_speciator import (
    DeepNEATSpeciator,
    compute_compatibility_distance,
    compute_layer_hyperparameter_distance,
)

_FILTERS = (16, 32, 64, 128)
_KERNELS = (1, 3, 5)
_UNITS = (64, 128, 256)

_DISTANCE_KWARGS = {
    "available_filter_counts": _FILTERS,
    "available_kernel_sizes": _KERNELS,
    "available_dense_unit_counts": _UNITS,
    "dropout_rate_min": 0.0,
    "dropout_rate_max": 0.5,
}


def _conv(node_id: int, filters: int = 16, kernel: int = 3, dropout: float = 0.0):
    return LayerNodeGene(
        node_id=node_id,
        layer_type="conv",
        number_of_filters=filters,
        kernel_size=kernel,
        dropout_rate=dropout,
    )


def test_identical_layers_are_at_distance_zero() -> None:
    assert compute_layer_hyperparameter_distance(
        _conv(2), _conv(2), **_DISTANCE_KWARGS
    ) == 0.0


def test_layers_of_different_types_are_at_maximum_distance() -> None:
    dense = LayerNodeGene(node_id=2, layer_type="dense", number_of_units=64)
    assert compute_layer_hyperparameter_distance(
        _conv(2), dense, **_DISTANCE_KWARGS
    ) == 1.0


def test_distance_grows_with_the_gap_in_the_search_space() -> None:
    near = compute_layer_hyperparameter_distance(
        _conv(2, filters=16), _conv(2, filters=32), **_DISTANCE_KWARGS
    )
    far = compute_layer_hyperparameter_distance(
        _conv(2, filters=16), _conv(2, filters=128), **_DISTANCE_KWARGS
    )
    assert 0.0 < near < far <= 1.0


def test_distance_is_symmetric() -> None:
    first, second = _conv(2, filters=16, dropout=0.1), _conv(2, filters=64, dropout=0.4)
    assert compute_layer_hyperparameter_distance(
        first, second, **_DISTANCE_KWARGS
    ) == compute_layer_hyperparameter_distance(second, first, **_DISTANCE_KWARGS)


def test_genomes_with_no_shared_nodes_have_maximum_hyperparameter_distance() -> None:
    """Spec property: genomes sharing no node id contribute H = 1.0.

    Coefficients on the excess/disjoint edge terms are zeroed out so the
    assertion isolates the hyperparameter term specifically.
    """
    first_genome = DeepNEATGenome(
        node_genes=(
            LayerNodeGene(node_id=0, layer_type="input"),
            LayerNodeGene(node_id=1, layer_type="output"),
            _conv(2, filters=16),
        ),
        edge_genes=(
            TensorEdgeGene(innovation_id=0, source_node_id=0, target_node_id=2,
                           is_enabled=True),
            TensorEdgeGene(innovation_id=1, source_node_id=2, target_node_id=1,
                           is_enabled=True),
        ),
    )
    second_genome = DeepNEATGenome(
        node_genes=(
            LayerNodeGene(node_id=10, layer_type="input"),
            LayerNodeGene(node_id=11, layer_type="output"),
            _conv(12, filters=16),
        ),
        edge_genes=(
            TensorEdgeGene(innovation_id=0, source_node_id=10, target_node_id=12,
                           is_enabled=True),
            TensorEdgeGene(innovation_id=1, source_node_id=12, target_node_id=11,
                           is_enabled=True),
        ),
    )
    distance = compute_compatibility_distance(
        first_genome,
        second_genome,
        coefficient_excess_c1=0.0,
        coefficient_disjoint_c2=0.0,
        coefficient_hyperparameter_c3=1.0,
        **_DISTANCE_KWARGS,
    )
    assert distance == 1.0


def test_two_shared_identical_layers_give_zero_hyperparameter_distance() -> None:
    """Spec property: identical genomes give H = 0.

    Proved directly on ``compute_compatibility_distance`` over two shared,
    identical conv layers, rather than only inferred from species clustering
    under a loose threshold, which a small H off-by-one could still pass.
    """

    def _twin() -> DeepNEATGenome:
        return DeepNEATGenome(
            node_genes=(
                LayerNodeGene(node_id=0, layer_type="input"),
                LayerNodeGene(node_id=1, layer_type="output"),
                _conv(2, filters=16, kernel=3, dropout=0.1),
                _conv(3, filters=64, kernel=5, dropout=0.3),
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

    distance = compute_compatibility_distance(
        _twin(),
        _twin(),
        coefficient_excess_c1=0.0,
        coefficient_disjoint_c2=0.0,
        coefficient_hyperparameter_c3=1.0,
        **_DISTANCE_KWARGS,
    )
    assert distance == 0.0


def _speciator(threshold: float = 3.0) -> DeepNEATSpeciator:
    return DeepNEATSpeciator(
        coefficient_excess_c1=1.0,
        coefficient_disjoint_c2=1.0,
        coefficient_hyperparameter_c3=0.4,
        compatibility_distance_threshold=threshold,
        **_DISTANCE_KWARGS,
    )


def _genome(filters: int) -> DeepNEATGenome:
    return DeepNEATGenome(
        node_genes=(
            LayerNodeGene(node_id=0, layer_type="input"),
            LayerNodeGene(node_id=1, layer_type="output"),
            _conv(2, filters=filters),
        ),
        edge_genes=(
            TensorEdgeGene(innovation_id=0, source_node_id=0, target_node_id=2,
                           is_enabled=True),
            TensorEdgeGene(innovation_id=1, source_node_id=2, target_node_id=1,
                           is_enabled=True),
        ),
    )


def test_identical_genomes_land_in_one_species() -> None:
    assignments = _speciator().assign_genomes_to_species(
        [_genome(16), _genome(16), _genome(16)], np.random.default_rng(0)
    )
    assert len(set(assignments)) == 1


def test_structurally_distant_genomes_split_into_species() -> None:
    big = DeepNEATGenome(
        node_genes=(
            LayerNodeGene(node_id=0, layer_type="input"),
            LayerNodeGene(node_id=1, layer_type="output"),
            _conv(2),
            _conv(3),
            _conv(4),
            _conv(5),
        ),
        edge_genes=tuple(
            TensorEdgeGene(innovation_id=index, source_node_id=source,
                           target_node_id=target, is_enabled=True)
            for index, (source, target) in enumerate(
                [(0, 2), (2, 3), (3, 4), (4, 5), (5, 1)]
            )
        ),
    )
    assignments = _speciator(threshold=0.5).assign_genomes_to_species(
        [_genome(16), big], np.random.default_rng(0)
    )
    assert len(set(assignments)) == 2


def test_one_assignment_is_returned_per_genome() -> None:
    genomes = [_genome(16), _genome(32), _genome(128)]
    assignments = _speciator().assign_genomes_to_species(genomes, np.random.default_rng(0))
    assert len(assignments) == len(genomes)


def test_empty_population_yields_no_assignments() -> None:
    assert _speciator().assign_genomes_to_species([], np.random.default_rng(0)) == []
