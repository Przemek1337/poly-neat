"""Smoke run of DeepNEAT on the pediatric pneumonia protocol.

Run it as a module from the repository root::

    uv run python -m examples.pediatric_pneumonia.deepneat_smoke

DeepNEAT genomes carry no weights, so the checkpoint this profile freezes is
the trained network that earned the selected fitness, not one rebuilt from the
winning genome. Track B then rebuilds that same topology with fresh parameters
and retrains it under the shared recipe.

The archive is synthetic and the budget is tiny. This proves the plumbing and
measures nothing.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import torch
import yaml

from examples._example_cli import parse_device_from_cli
from examples._experiment import ExperimentReport, print_experiment_report
from examples.pediatric_pneumonia._methods import make_deepneat_search
from examples.pediatric_pneumonia._smoke import config_path_for, run_smoke_experiment
from polyneat.configs.deepneat.deepneat_config import DeepNEATConfig

CONFIG_FILE_PATH = config_path_for(__file__)
_ARTIFACTS_DIRECTORY = Path(__file__).parent / "artifacts" / "deepneat_smoke"


def _load_profile() -> dict:
    """Read the whole profile, protocol and algorithm sections together."""
    return yaml.safe_load(CONFIG_FILE_PATH.read_text(encoding="utf-8"))


def run_experiment(
    device: torch.device | None = None,
    random_seed: int | None = None,
    artifacts_directory: Path | None = None,
) -> ExperimentReport:
    """Run one DeepNEAT search through the full protocol and report both tracks."""
    profile = _load_profile()
    # The strict loader is declared to return the base config type; the
    # concrete type is what the search factory needs.
    algorithm_config = cast(DeepNEATConfig, DeepNEATConfig.from_dict(profile["algorithm"]))
    if random_seed is not None:
        algorithm_config.random_seed = random_seed
    return run_smoke_experiment(
        config_file_path=CONFIG_FILE_PATH,
        method_name="deepneat",
        build_search=make_deepneat_search(
            algorithm_config,
            number_of_generations=int(profile["search"]["number_of_generations"]),
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
