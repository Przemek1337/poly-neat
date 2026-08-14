"""HyperNEAT-LEO on the retina problem - does the locality seed yield modularity?

Run alongside ``examples/retina/hyperneat.py``: the same task, the same substrate
size, the same CPPN genetics, and the same run code (``_shared.py``) - the only
difference is the link expression output and its locality seed. Both report
``number_of_cross_hemisphere_connections``, so LEO's claim is directly testable:
fewer crossings at comparable fitness.

Run from the repository root:
    uv run python -m examples.retina.leo [--cpu | --gpu]

Artifacts are written to examples/retina/artifacts/leo/.

References:
    Verbancsics, P., & Stanley, K. O. (2011). Constraining Connectivity to Encourage
        Modularity in HyperNEAT. *GECCO '11*, pp. 1483-1490. DOI: 10.1145/2001576.2001776
"""

from __future__ import annotations

from pathlib import Path

import torch

import polyneat as pn
from examples._example_cli import parse_device_from_cli
from examples._experiment import ExperimentReport, print_experiment_report
from examples.retina._shared import run_retina_experiment

CONFIG_FILE_PATH = Path(__file__).parent / "leo.yaml"
_ARTIFACTS_DIR = Path(__file__).parent / "artifacts" / "leo"


def run_experiment(
    device: torch.device | None = None,
    random_seed: int | None = None,
    artifacts_directory: Path | None = None,
    max_generations: int | None = None,
) -> ExperimentReport:
    """Run the retina HyperNEAT-LEO experiment once.

    Args:
        device: Phenotype evaluation device; ``None`` uses the yaml value.
        random_seed: Evolution seed override; ``None`` uses the yaml value.
        artifacts_directory: Where to write artifacts; ``None`` writes none.
        max_generations: Generation cap override, for fast smoke runs.

    Returns:
        Fitness, connection counts, generation count and runtime.
    """
    return run_retina_experiment(
        config_class=pn.HyperNEATLEOConfig,
        algorithm_class=pn.HyperNEATLEOAlgorithm,
        config_file_path=CONFIG_FILE_PATH,
        variant_label="HyperNEAT-LEO",
        device=device,
        random_seed=random_seed,
        artifacts_directory=artifacts_directory,
        max_generations=max_generations,
    )


def main() -> None:
    device = parse_device_from_cli()
    report = run_experiment(device=device, artifacts_directory=_ARTIFACTS_DIR)
    print_experiment_report(report)


if __name__ == "__main__":
    main()
