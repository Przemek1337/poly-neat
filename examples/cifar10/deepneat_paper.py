"""DeepNEAT experiment using Liang (2018), Chapter 3's CIFAR-10 protocol.

Run from the repository root:
    uv run python -m examples.cifar10.deepneat_paper --gpu --no-tensorboard

The legacy ``paper`` name is retained for compatibility; the detailed source
is Liang's dissertation rather than the paper's CoDeepNEAT experiment. Values
not reported by the source remain explicit implementation choices in YAML.
"""

from __future__ import annotations

from pathlib import Path

import torch

from examples._experiment import ExperimentReport
from examples._run import run_example_main
from examples.cifar10._deepneat import run_cifar10_deepneat_experiment

CONFIG_FILE_PATH = Path(__file__).parent / "deepneat_paper.yaml"
_ARTIFACTS_DIR = Path(__file__).parent / "artifacts" / "deepneat_paper"


def run_experiment(
    device: torch.device | None = None,
    random_seed: int | None = None,
    artifacts_directory: Path | None = None,
) -> ExperimentReport:
    """Evolve DeepNEAT under the source-reported data/search/compute budget."""
    return run_cifar10_deepneat_experiment(
        config_file_path=CONFIG_FILE_PATH,
        tensorboard_run_label="cifar10-deepneat-paper",
        device=device,
        random_seed=random_seed,
        artifacts_directory=artifacts_directory,
    )


def main() -> None:
    run_example_main(run_experiment, _ARTIFACTS_DIR)


if __name__ == "__main__":
    main()
