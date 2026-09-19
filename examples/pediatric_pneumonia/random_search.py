"""Smoke run of the random-search control over DeepNEAT's space.

Run it as a module from the repository root::

    uv run python -m examples.pediatric_pneumonia.random_search

This baseline exists to separate two claims that are otherwise easy to conflate:
that neuroevolution found a good architecture, and that the architecture space
contains good architectures. It shares DeepNEAT's decoder, gene ranges,
constraints, fitness, candidate training and budget, and differs only in having
no selection pressure and no inheritance.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import torch
import yaml

from examples._experiment import ExperimentReport
from examples.pediatric_pneumonia._execution import ExecutionOptions
from examples.pediatric_pneumonia._methods import make_random_search
from examples.pediatric_pneumonia._profiles import (
    CONFIGS_DIRECTORY,
    run_profile_experiment,
    run_profile_main,
)
from polyneat.configs.deepneat.deepneat_config import DeepNEATConfig

CONFIG_FILE_PATH = CONFIGS_DIRECTORY / "random_search_smoke.yaml"
_ARTIFACTS_DIRECTORY = Path(__file__).parent / "artifacts" / "random_search"


def run_experiment(
    device: torch.device | None = None,
    random_seed: int | None = None,
    artifacts_directory: Path | None = None,
    data_directory: Path | None = None,
    config_file_path: Path | None = None,
    execution: ExecutionOptions | None = None,
) -> ExperimentReport:
    """Draw independent candidates from DeepNEAT's space and report both tracks."""
    resolved_config_file_path = config_file_path or CONFIG_FILE_PATH
    profile = yaml.safe_load(resolved_config_file_path.read_text(encoding="utf-8"))
    # The strict loader is declared to return the base config type; the
    # concrete type is what the search factory needs.
    algorithm_config = cast(DeepNEATConfig, DeepNEATConfig.from_dict(profile["algorithm"]))
    if random_seed is not None:
        algorithm_config.random_seed = random_seed
    sampling = profile["random_search"]
    return run_profile_experiment(
        config_file_path=resolved_config_file_path,
        method_name="random_search",
        build_search=make_random_search(
            algorithm_config,
            number_of_candidates=int(sampling["number_of_candidates"]),
            minimum_structural_mutations=int(sampling["minimum_structural_mutations"]),
            maximum_structural_mutations=int(sampling["maximum_structural_mutations"]),
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
