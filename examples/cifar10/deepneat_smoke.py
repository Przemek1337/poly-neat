"""Budgeted CIFAR-10 DeepNEAT reference run for one GPU.

Run from the repository root:
    uv run python -m examples.cifar10.deepneat_smoke --gpu --no-tensorboard

This profile is intended to validate the pipeline and compare repository
revisions under the same seed. It is not a reproduction of the paper budget.
"""

from __future__ import annotations

from pathlib import Path

import torch

from examples._experiment import ExperimentReport
from examples._run import run_example_main
from examples.cifar10._deepneat import run_cifar10_deepneat_experiment

CONFIG_FILE_PATH = Path(__file__).parent / "deepneat_smoke.yaml"
_ARTIFACTS_DIR = Path(__file__).parent / "artifacts" / "deepneat_smoke"


def run_experiment(
    device: torch.device | None = None,
    random_seed: int | None = None,
    artifacts_directory: Path | None = None,
) -> ExperimentReport:
    """Run the single-GPU CIFAR-10 smoke/reference profile."""
    return run_cifar10_deepneat_experiment(
        config_file_path=CONFIG_FILE_PATH,
        tensorboard_run_label="cifar10-deepneat-smoke",
        device=device,
        random_seed=random_seed,
        artifacts_directory=artifacts_directory,
    )


def main() -> None:
    run_example_main(run_experiment, _ARTIFACTS_DIR)


if __name__ == "__main__":
    main()
