"""DeepNEAT on the CIFAR-10 experiment protocol.

Run it as a module from the repository root::

    uv run python -m examples.cifar10.benchmark_deepneat --cpu
    uv run python -m examples.cifar10.benchmark_deepneat --gpu --mode pilot

This is the benchmark runner, distinct from ``examples.cifar10.deepneat_paper``
which reproduces the source experiment. CIFAR runs DeepNEAT only. DeepNEAT
genomes carry no weights, so the
checkpoint this profile freezes is the trained network that earned the selected
fitness; track B then rebuilds that topology with fresh weights and retrains it
under the shared recipe.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import torch
import yaml

from examples._experiment import ExperimentReport
from examples.cifar10._execution import ExecutionOptions
from examples.cifar10._methods import make_deepneat_search
from examples.cifar10._profiles import (
    CONFIGS_DIRECTORY,
    run_profile_experiment,
    run_profile_main,
)
from polyneat.configs.deepneat.deepneat_config import DeepNEATConfig

CONFIG_FILE_PATH = CONFIGS_DIRECTORY / "deepneat_smoke.yaml"
_ARTIFACTS_DIRECTORY = Path(__file__).parent / "artifacts" / "benchmark_deepneat"


def run_experiment(
    device: torch.device | None = None,
    random_seed: int | None = None,
    artifacts_directory: Path | None = None,
    data_directory: Path | None = None,
    config_file_path: Path | None = None,
    execution: ExecutionOptions | None = None,
) -> ExperimentReport:
    """Run one DeepNEAT search through the CIFAR-10 protocol and report both tracks."""
    resolved_config_file_path = config_file_path or CONFIG_FILE_PATH
    profile = yaml.safe_load(resolved_config_file_path.read_text(encoding="utf-8"))
    algorithm_config = cast(DeepNEATConfig, DeepNEATConfig.from_dict(profile["algorithm"]))
    if random_seed is not None:
        algorithm_config.random_seed = random_seed
    return run_profile_experiment(
        config_file_path=resolved_config_file_path,
        method_name="deepneat",
        build_search=make_deepneat_search(
            algorithm_config,
            number_of_generations=int(profile["search"]["number_of_generations"]),
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
