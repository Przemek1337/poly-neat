"""EXACT on the MNIST comparison protocol.

Run it as a module from the repository root::

    uv run python -m examples.mnist.benchmark_exact --cpu
    uv run python -m examples.mnist.benchmark_exact --gpu --mode pilot

This is the comparison runner, distinct from ``examples.mnist.exact`` which
reproduces the source experiment. EXACT is Lamarckian: it trains between generations and
writes the trained kernels back into the genotype, so its evaluator only scores.
Track B resets the selected genome first, so retraining really starts from
scratch under the shared recipe.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import torch
import yaml

from examples._experiment import ExperimentReport
from examples.mnist._execution import ExecutionOptions
from examples.mnist._methods import make_exact_search
from examples.mnist._profiles import (
    CONFIGS_DIRECTORY,
    run_profile_experiment,
    run_profile_main,
)
from polyneat.configs.exact.exact_config import EXACTConfig

CONFIG_FILE_PATH = CONFIGS_DIRECTORY / "exact_smoke.yaml"
_ARTIFACTS_DIRECTORY = Path(__file__).parent / "artifacts" / "benchmark_exact"


def run_experiment(
    device: torch.device | None = None,
    random_seed: int | None = None,
    artifacts_directory: Path | None = None,
    data_directory: Path | None = None,
    config_file_path: Path | None = None,
    execution: ExecutionOptions | None = None,
) -> ExperimentReport:
    """Run one EXACT search through the MNIST protocol and report both tracks."""
    resolved_config_file_path = config_file_path or CONFIG_FILE_PATH
    profile = yaml.safe_load(resolved_config_file_path.read_text(encoding="utf-8"))
    algorithm_config = cast(EXACTConfig, EXACTConfig.from_dict(profile["algorithm"]))
    if random_seed is not None:
        algorithm_config.random_seed = random_seed
    return run_profile_experiment(
        config_file_path=resolved_config_file_path,
        method_name="exact",
        build_search=make_exact_search(
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
