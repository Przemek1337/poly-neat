"""Tests that one chosen device reaches every stage that runs a forward pass.

The benchmark measures wall-clock cost under a budget, so where the work
happens is part of the result rather than an implementation detail. These tests
cover the plumbing that carries the choice: the helper that moves a model, and
the protocol settings that hand one device to the search, both training tracks,
threshold scoring and the test evaluation.

``torch.device("meta")`` stands in for a second device on machines without
CUDA: parameters can be moved onto it and the move is observable, which is all
these tests assert.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import cast

import torch
from torch import nn

from examples.pediatric_pneumonia._protocol import (
    BenchmarkSettings,
    PreparedData,
    SearchContext,
    SelectedCandidate,
    _effective_configuration,
    run_search_stage,
)
from examples.pediatric_pneumonia._smoke import load_smoke_settings
from examples.pediatric_pneumonia.dataset import SplitTensors
from polyneat.training.trainable_model import TrainableModel, move_model_to_device
from polyneat.training.training_recipe import LearningRateSchedule, TrainingRecipe

_SIDE = 8
_SECOND_DEVICE = torch.device("meta")


class _SelfPlacingModel:
    """A trainable model that owns its placement, as PolyNEAT phenotypes do."""

    def __init__(self) -> None:
        self.layer = nn.Linear(2, 2)

    def forward_pass(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return self.layer(input_tensor)

    def parameters(self) -> Iterator[nn.Parameter]:
        return self.layer.parameters()

    def train(self, mode: bool = True) -> _SelfPlacingModel:
        self.layer.train(mode)
        return self

    def eval(self) -> _SelfPlacingModel:
        self.layer.eval()
        return self

    def state_dict(self) -> dict:
        return dict(self.layer.state_dict())

    def load_state_dict(self, state: dict) -> object:
        return self.layer.load_state_dict(state)


def _split(name: str, number_of_rows: int = 8) -> SplitTensors:
    """A tiny split with the columns the search stage reads."""
    return SplitTensors(
        images=torch.randint(0, 256, (number_of_rows, 1, _SIDE, _SIDE), dtype=torch.uint8),
        labels=torch.tensor([index % 2 for index in range(number_of_rows)], dtype=torch.long),
        example_ids=tuple(f"{name}-{index}" for index in range(number_of_rows)),
        group_ids=tuple(f"group-{index // 2}" for index in range(number_of_rows)),
        split_name=name,
    )


def _prepared_data() -> PreparedData:
    """Prepared data holding only what :func:`run_search_stage` touches."""
    return PreparedData(
        manifest=cast("object", None),  # never read before the freezing stage
        manifest_sha256="0" * 64,
        train=_split("train"),
        search_validation=_split("search_validation"),
        blocking_findings=(),
    )


def _settings(**overrides) -> BenchmarkSettings:
    recipe = TrainingRecipe(
        learning_rate=0.01,
        momentum=0.9,
        weight_decay=0.0,
        batch_size=4,
        number_of_epochs=1,
        schedule=LearningRateSchedule.CONSTANT,
    )
    defaults = dict(
        data_directory=Path("unused-by-the-search-stage"),
        protocol_id="device-test-v1",
        dataset_release="synthetic/fixture",
        dataset_license="not-applicable-synthetic",
        image_side=_SIDE,
        split_seed=1,
        search_seed=2,
        retraining_seeds=(),
        bootstrap_seed=3,
        candidate_recipe=recipe,
        track_b_recipe=recipe,
    )
    return BenchmarkSettings(**{**defaults, **overrides})


def _capturing_search(captured: list[SearchContext]):
    """A search that records its context instead of searching."""

    def search(context: SearchContext) -> SelectedCandidate:
        captured.append(context)
        model = _SelfPlacingModel()
        return SelectedCandidate(
            track_a_model=cast(TrainableModel, model),
            rebuild_model=lambda: cast(TrainableModel, _SelfPlacingModel()),
            genome_kind="none",
            genome_payload={},
            selection_fitness=0.5,
            evaluation_records=(),
            number_of_generations=1,
            search_seconds=0.0,
            parameter_count=6,
        )

    return search


def test_move_model_to_device_moves_a_torch_module() -> None:
    model = nn.Linear(2, 2)
    assert next(model.parameters()).device.type == "cpu"

    moved = move_model_to_device(model, _SECOND_DEVICE)

    assert next(moved.parameters()).device.type == _SECOND_DEVICE.type


def test_move_model_to_device_leaves_a_self_placing_model_alone() -> None:
    model = _SelfPlacingModel()

    moved = move_model_to_device(cast(TrainableModel, model), _SECOND_DEVICE)

    assert moved is model
    assert next(moved.parameters()).device.type == "cpu"


def test_search_stage_hands_the_configured_device_to_the_method() -> None:
    captured: list[SearchContext] = []
    settings = _settings(device_for_computation=_SECOND_DEVICE)

    run_search_stage(settings, _prepared_data(), _capturing_search(captured))

    assert captured[0].device_for_computation == _SECOND_DEVICE


def test_search_stage_defaults_to_the_cpu() -> None:
    captured: list[SearchContext] = []

    run_search_stage(_settings(), _prepared_data(), _capturing_search(captured))

    assert captured[0].device_for_computation == torch.device("cpu")


def test_search_context_builds_a_trainer_on_the_same_device() -> None:
    captured: list[SearchContext] = []
    settings = _settings(device_for_computation=_SECOND_DEVICE)

    run_search_stage(settings, _prepared_data(), _capturing_search(captured))
    trainer = captured[0].build_trainer()

    assert trainer.device_for_computation == _SECOND_DEVICE


def test_effective_configuration_records_the_device_that_ran() -> None:
    configuration = _effective_configuration(
        _settings(device_for_computation=_SECOND_DEVICE), "any_method"
    )

    assert configuration["device_for_computation"] == str(_SECOND_DEVICE)


def test_smoke_settings_take_the_device_from_the_flag(tmp_path: Path) -> None:
    settings = load_smoke_settings(
        Path("examples/pediatric_pneumonia/configs/fixed_cnn_smoke.yaml"),
        data_directory=tmp_path / "archive",
        device=_SECOND_DEVICE,
    )

    assert settings.device_for_computation == _SECOND_DEVICE


def test_smoke_settings_stay_on_the_cpu_without_a_flag(tmp_path: Path) -> None:
    settings = load_smoke_settings(
        Path("examples/pediatric_pneumonia/configs/fixed_cnn_smoke.yaml"),
        data_directory=tmp_path / "archive",
    )

    assert settings.device_for_computation == torch.device("cpu")
