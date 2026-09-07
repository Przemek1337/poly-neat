"""Smoke run of the random-search control over DeepNEAT's space.

Run it as a module from the repository root::

    uv run python -m examples.pediatric_pneumonia.random_search_smoke

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

from examples._example_cli import parse_device_from_cli
from examples._experiment import ExperimentReport, print_experiment_report
from examples.pediatric_pneumonia._methods import make_random_search
from examples.pediatric_pneumonia._smoke import config_path_for, run_smoke_experiment
from polyneat.configs.deepneat.deepneat_config import DeepNEATConfig

CONFIG_FILE_PATH = config_path_for(__file__)
_ARTIFACTS_DIRECTORY = Path(__file__).parent / "artifacts" / "random_search_smoke"


def run_experiment(
    device: torch.device | None = None,
    random_seed: int | None = None,
    artifacts_directory: Path | None = None,
) -> ExperimentReport:
    """Draw independent candidates from DeepNEAT's space and report both tracks."""
    profile = yaml.safe_load(CONFIG_FILE_PATH.read_text(encoding="utf-8"))
    # The strict loader is declared to return the base config type; the
    # concrete type is what the search factory needs.
    algorithm_config = cast(DeepNEATConfig, DeepNEATConfig.from_dict(profile["algorithm"]))
    if random_seed is not None:
        algorithm_config.random_seed = random_seed
    sampling = profile["random_search"]
    return run_smoke_experiment(
        config_file_path=CONFIG_FILE_PATH,
        method_name="random_search_deepneat",
        build_search=make_random_search(
            algorithm_config,
            number_of_candidates=int(sampling["number_of_candidates"]),
            minimum_structural_mutations=int(sampling["minimum_structural_mutations"]),
            maximum_structural_mutations=int(sampling["maximum_structural_mutations"]),
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
