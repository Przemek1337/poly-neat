"""Smoke run of EXACT on the pediatric pneumonia protocol.

Run it as a module from the repository root::

    uv run python -m examples.pediatric_pneumonia.exact_smoke

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

from examples._example_cli import parse_device_from_cli
from examples._experiment import ExperimentReport, print_experiment_report
from examples.pediatric_pneumonia._methods import make_exact_search
from examples.pediatric_pneumonia._smoke import config_path_for, run_smoke_experiment
from polyneat.configs.exact.exact_config import EXACTConfig

CONFIG_FILE_PATH = config_path_for(__file__)
_ARTIFACTS_DIRECTORY = Path(__file__).parent / "artifacts" / "exact_smoke"


def run_experiment(
    device: torch.device | None = None,
    random_seed: int | None = None,
    artifacts_directory: Path | None = None,
) -> ExperimentReport:
    """Run one EXACT search through the full protocol and report both tracks."""
    profile = yaml.safe_load(CONFIG_FILE_PATH.read_text(encoding="utf-8"))
    # The strict loader is declared to return the base config type; the
    # concrete type is what the search factory needs.
    algorithm_config = cast(EXACTConfig, EXACTConfig.from_dict(profile["algorithm"]))
    if random_seed is not None:
        algorithm_config.random_seed = random_seed
    return run_smoke_experiment(
        config_file_path=CONFIG_FILE_PATH,
        method_name="exact",
        build_search=make_exact_search(
            algorithm_config,
            number_of_generations=int(profile["search"]["number_of_generations"]),
            image_side=int(profile["protocol"]["image_side"]),
        ),
        device=device,
        random_seed=random_seed,
        artifacts_directory=artifacts_directory,
    )


def main() -> None:
    """Command-line entry point."""
    print_experiment_report(
        run_experiment(
            device=parse_device_from_cli(), artifacts_directory=_ARTIFACTS_DIRECTORY
        )
    )


if __name__ == "__main__":
    main()
