from __future__ import annotations

import pytest
import torch

from benchmarks.retrain_selected_pneumonia import (
    StabilizedSGD,
    StabilizedTrainingRecipe,
    TrainingDivergedError,
)
from polyneat.training.training_recipe import TrainingRecipe


def _parameter_with_gradient(gradient: list[float]) -> torch.nn.Parameter:
    parameter = torch.nn.Parameter(torch.zeros(len(gradient)))
    parameter.grad = torch.tensor(gradient)
    return parameter


def test_step_clips_the_global_gradient_norm() -> None:
    parameter = _parameter_with_gradient([30.0, 40.0])
    optimizer = StabilizedSGD(
        [parameter], gradient_clipping_max_norm=1.0, warmup_steps=0, lr=1.0, momentum=0.0
    )

    optimizer.step()

    assert torch.allclose(parameter.detach(), torch.tensor([-0.6, -0.8]))


def test_warmup_scales_early_steps_and_leaves_the_scheduled_rate_alone() -> None:
    parameter = _parameter_with_gradient([1.0])
    optimizer = StabilizedSGD(
        [parameter], gradient_clipping_max_norm=None, warmup_steps=4, lr=0.4, momentum=0.0
    )

    optimizer.step()
    assert torch.allclose(parameter.detach(), torch.tensor([-0.1]))
    assert optimizer.param_groups[0]["lr"] == 0.4

    for _ in range(4):
        parameter.grad = torch.tensor([1.0])
        optimizer.step()
    # Steps 2..4 use 0.2, 0.3, 0.4; step 5 is past the warm-up and uses 0.4.
    assert torch.allclose(parameter.detach(), torch.tensor([-1.4]))


def test_non_finite_gradient_stops_the_step_without_touching_the_weights() -> None:
    parameter = _parameter_with_gradient([float("nan"), 1.0])
    optimizer = StabilizedSGD(
        [parameter], gradient_clipping_max_norm=1.0, warmup_steps=0, lr=0.1, momentum=0.9
    )

    with pytest.raises(TrainingDivergedError) as error:
        optimizer.step()

    assert error.value.optimizer_step == 0
    assert torch.equal(parameter.detach(), torch.zeros(2))


def test_recipe_keeps_the_series_recipe_and_records_the_stabilization() -> None:
    base = TrainingRecipe(learning_rate=0.01, batch_size=64, number_of_epochs=30)
    recipe = StabilizedTrainingRecipe(
        **{name: getattr(base, name) for name in TrainingRecipe.__dataclass_fields__},
        gradient_clipping_max_norm=1.0,
        warmup_steps=70,
    )

    serialized = recipe.to_serializable_dict()
    assert {key: serialized[key] for key in base.to_serializable_dict()} == (
        base.to_serializable_dict()
    )
    assert serialized["gradient_clipping_max_norm"] == 1.0
    assert serialized["warmup_steps"] == 70
    assert serialized["initialization"] == "shared"
    assert recipe.learning_rate_for_epoch(15) == base.learning_rate_for_epoch(15)
    optimizer = recipe.build_optimizer([torch.nn.Parameter(torch.zeros(1))])
    assert isinstance(optimizer, StabilizedSGD)
    assert optimizer.param_groups[0]["nesterov"] is True
