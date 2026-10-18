"""Smoke run of the pretrained ResNet-18 baseline, reported on its own.

Run it as a module from the repository root::

    uv run python -m examples.pediatric_pneumonia.transfer_learning

This baseline sits outside the DeepNEAT-EXACT comparison on purpose. Its
starting point was learned on a million labelled photographs whose cost appears
in no budget here, so equating its allowance with a search that started from
nothing would be misleading. It runs through the same split, the same threshold
rule and the same test evaluation as everything else, and is reported
separately.

It needs torchvision, which lives in the optional ``benchmark`` extra where its
version is pinned against the torch build in use.
"""

from __future__ import annotations

from pathlib import Path

import torch
import yaml

from examples._experiment import ExperimentReport
from examples.pediatric_pneumonia._execution import ExecutionOptions
from examples.pediatric_pneumonia._methods import make_transfer_learning_baseline
from examples.pediatric_pneumonia._profiles import (
    CONFIGS_DIRECTORY,
    run_profile_experiment,
    run_profile_main,
)
from polyneat.nn.pretrained_image_classifier import (
    PretrainedClassifierConfig,
    UnfreezingPolicy,
)

CONFIG_FILE_PATH = CONFIGS_DIRECTORY / "transfer_learning_smoke.yaml"
_ARTIFACTS_DIRECTORY = Path(__file__).parent / "artifacts" / "transfer_learning"


def run_experiment(
    device: torch.device | None = None,
    random_seed: int | None = None,
    artifacts_directory: Path | None = None,
    data_directory: Path | None = None,
    config_file_path: Path | None = None,
    execution: ExecutionOptions | None = None,
) -> ExperimentReport:
    """Fine-tune the pinned backbone and report it against the same test split."""
    resolved_config_file_path = config_file_path or CONFIG_FILE_PATH
    profile = yaml.safe_load(resolved_config_file_path.read_text(encoding="utf-8"))
    payload = profile["transfer_learning"]
    return run_profile_experiment(
        config_file_path=resolved_config_file_path,
        method_name="transfer_learning_resnet18",
        build_search=make_transfer_learning_baseline(
            PretrainedClassifierConfig(
                weights_identifier=payload["weights_identifier"],
                number_of_classes=2,
                unfreezing_policy=UnfreezingPolicy(payload["unfreezing_policy"]),
                uses_imagenet_normalization=bool(payload["uses_imagenet_normalization"]),
            )
        ),
        device=device,
        random_seed=random_seed,
        artifacts_directory=artifacts_directory,
        data_directory=data_directory,
        execution=execution,
    )


def main() -> None:
    """Command-line entry point."""
    run_profile_main(
        run_experiment,
        default_config_file_path=CONFIG_FILE_PATH,
        artifacts_directory=_ARTIFACTS_DIRECTORY,
    )


if __name__ == "__main__":
    main()
