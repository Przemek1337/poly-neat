"""Smoke run of the hand-designed CNN baseline on the pediatric pneumonia protocol.

Run it as a module from the repository root::

    uv run python -m examples.pediatric_pneumonia.fixed_cnn_smoke

It uses a synthetic archive with the real directory layout and naming
conventions, so it exercises the audit, the grouped split, both tracks,
threshold selection and the bootstrap without downloading any radiographs. Its
numbers are meaningless as science and must not be reported.
"""

from __future__ import annotations

from pathlib import Path

import torch
import yaml

from examples._example_cli import parse_device_from_cli
from examples._experiment import ExperimentReport, print_experiment_report
from examples.pediatric_pneumonia._methods import make_fixed_cnn_baseline
from examples.pediatric_pneumonia._smoke import config_path_for, run_smoke_experiment
from polyneat.nn.fixed_convolutional_network import FixedConvolutionalNetworkConfig

CONFIG_FILE_PATH = config_path_for(__file__)
_ARTIFACTS_DIRECTORY = Path(__file__).parent / "artifacts" / "fixed_cnn_smoke"


def _architecture_from_config() -> FixedConvolutionalNetworkConfig:
    """The frozen baseline architecture, read from the profile."""
    payload = yaml.safe_load(CONFIG_FILE_PATH.read_text(encoding="utf-8"))["architecture"]
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
) -> ExperimentReport:
    """Run the baseline through the full protocol and report both tracks."""
    return run_smoke_experiment(
        config_file_path=CONFIG_FILE_PATH,
        method_name="fixed_cnn",
        build_search=make_fixed_cnn_baseline(_architecture_from_config()),
        device=device,
        random_seed=random_seed,
        artifacts_directory=artifacts_directory,
    )


def main() -> None:
    """Command-line entry point.

    The protocol writes its own artifacts (manifest, checkpoints, predictions,
    run report), so this profile does not go through the shared TensorBoard
    runner the search-curve examples use.
    """
    print_experiment_report(
        run_experiment(
            device=parse_device_from_cli(), artifacts_directory=_ARTIFACTS_DIRECTORY
        )
    )


if __name__ == "__main__":
    main()
