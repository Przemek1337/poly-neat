"""Smoke run of EXACT on the pediatric pneumonia protocol.

Run it as a module from the repository root::

    uv run python -m examples.pediatric_pneumonia.exact

EXACT inherits its trained kernels between generations, so track A keeps the
genome as the search left it. Track B resets that genome first - kernels, batch
normalization state and the trained flag - and retrains the topology from
scratch under the shared recipe, because clearing the kernels alone would leave
the evolved normalization state in place and an already-trained genome would be
skipped by the trainer entirely.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import torch
import yaml

from examples._experiment import ExperimentReport
from examples.pediatric_pneumonia._execution import ExecutionOptions
from examples.pediatric_pneumonia._methods import make_exact_search
from examples.pediatric_pneumonia._profiles import (
    CONFIGS_DIRECTORY,
    run_profile_experiment,
    run_profile_main,
)
from polyneat.configs.exact.exact_config import EXACTConfig

CONFIG_FILE_PATH = CONFIGS_DIRECTORY / "exact_smoke.yaml"
_ARTIFACTS_DIRECTORY = Path(__file__).parent / "artifacts" / "exact"


def run_experiment(
    device: torch.device | None = None,
    random_seed: int | None = None,
    artifacts_directory: Path | None = None,
    data_directory: Path | None = None,
    config_file_path: Path | None = None,
    execution: ExecutionOptions | None = None,
) -> ExperimentReport:
    """Run one EXACT search through the full protocol and report both tracks."""
    resolved_config_file_path = config_file_path or CONFIG_FILE_PATH
    profile = yaml.safe_load(resolved_config_file_path.read_text(encoding="utf-8"))
    # The strict loader is declared to return the base config type; the
    # concrete type is what the search factory needs.
    algorithm_config = cast(EXACTConfig, EXACTConfig.from_dict(profile["algorithm"]))
    if random_seed is not None:
        algorithm_config.random_seed = random_seed
    return run_profile_experiment(
        config_file_path=resolved_config_file_path,
        method_name="exact",
        build_search=make_exact_search(
            algorithm_config,
            number_of_generations=int(profile["search"]["number_of_generations"]),
            image_side=int(profile["protocol"]["image_side"]),
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
