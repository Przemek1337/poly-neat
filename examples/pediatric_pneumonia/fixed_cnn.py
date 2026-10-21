"""Smoke run of the hand-designed CNN baseline on the pediatric pneumonia protocol.

Run it as a module from the repository root::

    uv run python -m examples.pediatric_pneumonia.fixed_cnn

It uses a synthetic archive with the real directory layout and naming
conventions, so it exercises the audit, the grouped split, both tracks,
threshold selection and the bootstrap without downloading any radiographs. Its
numbers are meaningless as science and must not be reported.
"""

from __future__ import annotations

from pathlib import Path

import torch
import yaml

from examples._experiment import ExperimentReport
from examples.pediatric_pneumonia._execution import ExecutionOptions
from examples.pediatric_pneumonia._methods import make_fixed_cnn_baseline
from examples.pediatric_pneumonia._profiles import (
    CONFIGS_DIRECTORY,
    run_profile_experiment,
    run_profile_main,
)
from polyneat.nn.fixed_convolutional_network import FixedConvolutionalNetworkConfig

CONFIG_FILE_PATH = CONFIGS_DIRECTORY / "fixed_cnn_smoke.yaml"
_ARTIFACTS_DIRECTORY = Path(__file__).parent / "artifacts" / "fixed_cnn"


def _architecture_from_config(config_file_path: Path) -> FixedConvolutionalNetworkConfig:
    """The frozen baseline architecture, read from the profile."""
    payload = yaml.safe_load(config_file_path.read_text(encoding="utf-8"))["architecture"]
    return FixedConvolutionalNetworkConfig(
        input_channels=1,
        number_of_classes=2,
        convolution_channels=tuple(payload["convolution_channels"]),
        kernel_size=int(payload["kernel_size"]),
        uses_batch_normalization=bool(payload["uses_batch_normalization"]),
        dropout_probability=float(payload["dropout_probability"]),
    )


def run_experiment(
    device: torch.device | None = None,
    random_seed: int | None = None,
    artifacts_directory: Path | None = None,
    data_directory: Path | None = None,
    config_file_path: Path | None = None,
    execution: ExecutionOptions | None = None,
) -> ExperimentReport:
    """Run the baseline through the full protocol and report both tracks."""
    resolved_config_file_path = config_file_path or CONFIG_FILE_PATH
    return run_profile_experiment(
        config_file_path=resolved_config_file_path,
        method_name="fixed_cnn",
        build_search=make_fixed_cnn_baseline(_architecture_from_config(resolved_config_file_path)),
        device=device,
        random_seed=random_seed,
        artifacts_directory=artifacts_directory,
        data_directory=data_directory,
        execution=execution,
    )


def main() -> None:
    """Command-line entry point.

    The protocol writes its own artifacts (manifest, checkpoints, predictions,
    run report), so this profile does not go through the shared TensorBoard
    runner the search-curve examples use.
    """
    run_profile_main(
        run_experiment,
        default_config_file_path=CONFIG_FILE_PATH,
        artifacts_directory=_ARTIFACTS_DIRECTORY,
    )


if __name__ == "__main__":
    main()
