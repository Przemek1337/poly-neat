"""Tests for the persistent EXACT innovation tracker."""

from __future__ import annotations

from polyneat.algorithms.exact.exact_algorithm import EXACTAlgorithm
from polyneat.algorithms.exact.exact_innovation_tracker import EXACTInnovationTracker
from polyneat.configs.exact.exact_config import EXACTConfig


def test_edge_markings_survive_generation_reset() -> None:
    tracker = EXACTInnovationTracker()
    first_innovation_id = tracker.get_or_assign_innovation_id_for_connection(
        source_node_id=0, target_node_id=1
    )
    tracker.reset_for_new_generation()
    assert (
        tracker.get_or_assign_innovation_id_for_connection(
            source_node_id=0, target_node_id=1
        )
        == first_innovation_id
    )


def test_node_split_markings_survive_generation_reset() -> None:
    tracker = EXACTInnovationTracker()
    split_edge_innovation_id = tracker.get_or_assign_innovation_id_for_connection(
        source_node_id=0, target_node_id=1
    )
    first_record = tracker.get_or_assign_node_split(
        split_connection_innovation_id=split_edge_innovation_id, minimum_new_node_id=2
    )
    tracker.reset_for_new_generation()
    assert (
        tracker.get_or_assign_node_split(
            split_connection_innovation_id=split_edge_innovation_id,
            minimum_new_node_id=2,
        )
        == first_record
    )


def test_assign_fresh_node_id_is_monotonic_and_respects_minimum() -> None:
    tracker = EXACTInnovationTracker()
    first_node_id = tracker.assign_fresh_node_id(minimum_new_node_id=10)
    second_node_id = tracker.assign_fresh_node_id(minimum_new_node_id=0)
    assert first_node_id == 10
    assert second_node_id == 11


def test_exact_algorithm_uses_persistent_tracker() -> None:
    config = EXACTConfig()
    config.validate()
    algorithm = EXACTAlgorithm.from_config(config)
    assert isinstance(algorithm.innovation_tracker, EXACTInnovationTracker)
