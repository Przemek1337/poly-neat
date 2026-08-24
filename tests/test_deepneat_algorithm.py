from __future__ import annotations

import numpy as np
import pytest

from polyneat.algorithms.deepneat.deepneat_algorithm import DeepNEATAlgorithm
from polyneat.algorithms.deepneat.deepneat_crossover import DeepNEATCrossover
from polyneat.algorithms.deepneat.deepneat_genome import (
    DeepNEATGenome,
    LayerNodeGene,
    TensorEdgeGene,
)
from polyneat.algorithms.deepneat.deepneat_innovation_tracker import (
    DeepNEATInnovationTracker,
)
from polyneat.algorithms.deepneat.deepneat_phenotype_decoder import (
    DeepNEATPhenotypeDecoder,
)
from polyneat.algorithms.deepneat.deepneat_speciator import DeepNEATSpeciator
from polyneat.algorithms.deepneat.mutations.add_layer_node_mutation import (
    AddLayerNodeMutation,
)
from polyneat.algorithms.deepneat.mutations.add_tensor_edge_mutation import (
    AddTensorEdgeMutation,
)
from polyneat.algorithms.deepneat.mutations.deepneat_composite_mutation import (
    DeepNEATCompositeMutation,
)
from polyneat.algorithms.deepneat.mutations.layer_hyperparameter_mutation import (
    LayerHyperparameterMutation,
)
from polyneat.algorithms.deepneat.mutations.toggle_tensor_edge_mutation import (
    ToggleTensorEdgeMutation,
)
from polyneat.configs.deepneat.deepneat_config import DeepNEATConfig
from polyneat.core.neat.neat_algorithm import NEATAlgorithm


def _small_config(**overrides: object) -> DeepNEATConfig:
    defaults: dict[str, object] = dict(
        population_size=8,
        input_image_channels=1,
        input_image_height=4,
        input_image_width=4,
        number_of_classes=3,
        available_filter_counts=(4, 8),
        available_kernel_sizes=(1, 3),
        available_dense_unit_counts=(8, 16),
        maximum_total_parameter_count=2_000_000,
        training_epochs_per_evaluation=1,
        training_batch_size=4,
        probability_of_add_layer_node_mutation=0.5,
        probability_of_add_tensor_edge_mutation=0.5,
        probability_of_toggle_tensor_edge_mutation=0.3,
        probability_of_layer_hyperparameter_mutation=0.5,
        random_seed=0,
    )
    defaults.update(overrides)
    return DeepNEATConfig(**defaults)


def test_advance_one_generation_is_not_overridden() -> None:
    # The payoff of the architecture: DeepNEAT does not inherit weights and does
    # not co-evolve training hyperparameters, so all training lives in the
    # evaluator and the generational loop is reused verbatim.
    assert (
        DeepNEATAlgorithm.advance_one_generation is NEATAlgorithm.advance_one_generation
    )


def test_from_config_wires_deepneat_components() -> None:
    algorithm = DeepNEATAlgorithm.from_config(_small_config())
    assert isinstance(algorithm.mutation, DeepNEATCompositeMutation)
    assert isinstance(algorithm.crossover, DeepNEATCrossover)
    assert isinstance(algorithm.speciator, DeepNEATSpeciator)
    assert isinstance(algorithm.innovation_tracker, DeepNEATInnovationTracker)
    assert isinstance(algorithm.phenotype_decoder, DeepNEATPhenotypeDecoder)

    # Pin Task 4's fixed operator order (DeepNEATCompositeMutation's own
    # docstring claims this exact order) so a future edit that swaps two
    # operators, or two of the four probabilities feeding them, cannot pass
    # silently behind an isinstance-only check.
    operator_types = [
        type(operator) for operator in algorithm.mutation._ordered_individual_mutations
    ]
    assert operator_types == [
        LayerHyperparameterMutation,
        AddTensorEdgeMutation,
        AddLayerNodeMutation,
        ToggleTensorEdgeMutation,
    ]


def test_from_config_does_not_warn_on_default_strategy(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level("WARNING"):
        DeepNEATAlgorithm.from_config(_small_config())
    assert not any(
        "initial_population_strategy" in record.message for record in caplog.records
    )


def test_from_config_warns_when_initial_population_strategy_is_overridden(
    caplog: pytest.LogCaptureFixture,
) -> None:
    config = _small_config(initial_population_strategy="fs_neat")
    with caplog.at_level("WARNING"):
        DeepNEATAlgorithm.from_config(config)
    assert any(
        "initial_population_strategy" in record.message and record.levelname == "WARNING"
        for record in caplog.records
    )


def test_initial_population_is_identical_linear_classifiers() -> None:
    config = _small_config()
    algorithm = DeepNEATAlgorithm.from_config(config)
    rng = np.random.default_rng(0)
    population = algorithm.create_initial_population(rng)

    assert population.size() == config.population_size
    assert population.generation_number == 0

    genomes = population.genomes
    assert all(isinstance(genome, DeepNEATGenome) for genome in genomes)
    for genome in genomes:
        assert len(genome.node_genes) == 2
        assert len(genome.edge_genes) == 1
        assert genome.edge_genes[0].is_enabled
        layer_types = {node.layer_type for node in genome.node_genes}
        assert layer_types == {"input", "output"}
    # Diversity comes from mutation, not from the starting population (decision #12).
    assert all(genome == genomes[0] for genome in genomes)


def test_yaml_initial_population_strategy_is_ignored() -> None:
    """Even an explicit, non-default ``initial_population_strategy`` has no effect:
    DeepNEAT always starts from the fixed input->output topology (decision #12)."""
    config = _small_config(initial_population_strategy="fs_neat")
    algorithm = DeepNEATAlgorithm.from_config(config)
    rng = np.random.default_rng(0)
    population = algorithm.create_initial_population(rng)
    for genome in population.genomes:
        assert len(genome.node_genes) == 2
        assert len(genome.edge_genes) == 1


def test_five_generations_run_without_exception_and_population_size_holds() -> None:
    config = _small_config(population_size=10)
    algorithm = DeepNEATAlgorithm.from_config(config)
    rng = np.random.default_rng(1)

    population = algorithm.create_initial_population(rng)
    assert population.size() == config.population_size

    saw_a_structural_change = False
    for _generation_index in range(5):
        fitnesses = [float(rng.random()) for _ in range(population.size())]
        population, _statistics = algorithm.advance_one_generation(population, fitnesses, rng)

        assert population.size() == config.population_size
        for genome in population.genomes:
            assert isinstance(genome, DeepNEATGenome)
            # Re-validating via clone_genome() re-runs DeepNEATGenome.__post_init__,
            # exercising the same structural invariants the genome was built under.
            genome.clone_genome()
            if len(genome.node_genes) > 2 or len(genome.edge_genes) > 1:
                saw_a_structural_change = True

    assert saw_a_structural_change, (
        "expected at least one mutation to have fired across 5 generations "
        "with a small, non-trivial population"
    )


def _add_layer_node_mutation() -> AddLayerNodeMutation:
    return AddLayerNodeMutation(
        probability_of_application=1.0,
        available_filter_counts=(8,),
        available_kernel_sizes=(3,),
        available_dense_unit_counts=(16,),
        dropout_rate_min=0.0,
        dropout_rate_max=0.5,
        probability_of_new_conv_layer=1.0,
    )


@pytest.mark.parametrize("second_split_seed", [0, 1, 2, 3, 4])
def test_add_layer_node_excludes_a_replayed_split_instead_of_forfeiting_the_event(
    second_split_seed: int,
) -> None:
    """Regression test for the guard in ``AddLayerNodeMutation.apply_to_genome``.

    Constructs the collision directly rather than relying on any one lucky
    seed: split an edge, re-enable the resulting disabled edge (what
    ``ToggleTensorEdgeMutation`` does), then split again with
    ``probability_of_application=1.0``. ``DeepNEATInnovationTracker`` never
    clears its split cache, so the re-split of the same edge would replay the
    node id from the first split; the fix must exclude that one edge and
    still split one of the others, rather than losing the mutation event
    entirely. Parametrized over several draw seeds for the second split to
    confirm the outcome does not depend on which of the three enabled edges
    is drawn first.
    """
    tracker = DeepNEATInnovationTracker()
    mutation = _add_layer_node_mutation()

    # Prime the edge's marking through the tracker itself (as
    # build_deepneat_initial_population does), rather than hardcoding an
    # innovation id, so the tracker's own id counter starts in sync with the
    # genome and the later fresh split cannot collide with it by accident.
    initial_edge_innovation_id = tracker.get_or_assign_innovation_id_for_connection(
        source_node_id=0, target_node_id=1
    )
    minimal_genome = DeepNEATGenome(
        node_genes=(
            LayerNodeGene(node_id=0, layer_type="input"),
            LayerNodeGene(node_id=1, layer_type="output"),
        ),
        edge_genes=(
            TensorEdgeGene(
                innovation_id=initial_edge_innovation_id,
                source_node_id=0,
                target_node_id=1,
                is_enabled=True,
            ),
        ),
    )

    once_split_genome = mutation.apply_to_genome(
        minimal_genome, np.random.default_rng(0), tracker
    )
    assert len(once_split_genome.node_genes) == 3
    original_edge = next(
        edge
        for edge in once_split_genome.edge_genes
        if edge.innovation_id == initial_edge_innovation_id
    )
    assert original_edge.is_enabled is False

    # Simulate ToggleTensorEdgeMutation re-enabling the disabled split edge:
    # both the direct edge and the through-the-new-node path are now enabled,
    # so 3 of the genome's edges are legal split candidates - one of them (the
    # replayed one) collides, two do not.
    re_enabled_original_edge = TensorEdgeGene(
        innovation_id=original_edge.innovation_id,
        source_node_id=original_edge.source_node_id,
        target_node_id=original_edge.target_node_id,
        is_enabled=True,
    )
    genome_with_edge_re_enabled = DeepNEATGenome(
        node_genes=once_split_genome.node_genes,
        edge_genes=tuple(
            re_enabled_original_edge
            if edge.innovation_id == initial_edge_innovation_id
            else edge
            for edge in once_split_genome.edge_genes
        ),
    )

    result = mutation.apply_to_genome(
        genome_with_edge_re_enabled, np.random.default_rng(second_split_seed), tracker
    )

    node_ids = [node.node_id for node in result.node_genes]
    assert len(node_ids) == len(set(node_ids)), f"duplicate node id produced: {node_ids}"
    # The event must not be lost: one of the two non-colliding edges is split
    # instead, regardless of which edge the rng happened to draw first.
    assert len(result.node_genes) == 4
    assert len(result.edge_genes) == 5


def test_add_layer_node_skips_only_when_every_enabled_edge_would_collide() -> None:
    """The "no legal move" fallback still exists for the fully-exhausted case.

    A genome whose only enabled edge is the replayed one has no legal split
    left at all, so the operator must fall back to returning the genome
    unchanged - the same contract it already used for "no enabled edge".
    """
    tracker = DeepNEATInnovationTracker()
    mutation = _add_layer_node_mutation()

    initial_edge_innovation_id = tracker.get_or_assign_innovation_id_for_connection(
        source_node_id=0, target_node_id=1
    )
    minimal_genome = DeepNEATGenome(
        node_genes=(
            LayerNodeGene(node_id=0, layer_type="input"),
            LayerNodeGene(node_id=1, layer_type="output"),
        ),
        edge_genes=(
            TensorEdgeGene(
                innovation_id=initial_edge_innovation_id,
                source_node_id=0,
                target_node_id=1,
                is_enabled=True,
            ),
        ),
    )
    once_split_genome = mutation.apply_to_genome(
        minimal_genome, np.random.default_rng(0), tracker
    )
    original_edge = next(
        edge
        for edge in once_split_genome.edge_genes
        if edge.innovation_id == initial_edge_innovation_id
    )

    # Re-enable the replayed edge but disable the two edges the first split
    # produced, so it is the *only* enabled edge left to draw.
    edges_with_only_the_replayed_edge_enabled = tuple(
        (
            TensorEdgeGene(
                innovation_id=edge.innovation_id,
                source_node_id=edge.source_node_id,
                target_node_id=edge.target_node_id,
                is_enabled=True,
            )
            if edge.innovation_id == original_edge.innovation_id
            else TensorEdgeGene(
                innovation_id=edge.innovation_id,
                source_node_id=edge.source_node_id,
                target_node_id=edge.target_node_id,
                is_enabled=False,
            )
        )
        for edge in once_split_genome.edge_genes
    )
    genome_with_only_the_replay_available = DeepNEATGenome(
        node_genes=once_split_genome.node_genes,
        edge_genes=edges_with_only_the_replayed_edge_enabled,
    )

    result = mutation.apply_to_genome(
        genome_with_only_the_replay_available, np.random.default_rng(0), tracker
    )
    assert result is genome_with_only_the_replay_available
