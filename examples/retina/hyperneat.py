"""Plain HyperNEAT on the retina problem - the baseline HyperNEAT-LEO is measured against.

Identical to ``examples/retina/leo.py`` in task, substrate size, CPPN genetics and
run code (``_shared.py``). What differs is exactly what separates the two
algorithms: the CPPN here has four inputs and one output, a connection exists when
its queried weight clears a magnitude threshold, generation 0 carries no locality
seed, and the substrate nodes are spread evenly rather than clustered.

Run from the repository root:
    uv run python -m examples.retina.hyperneat [--cpu | --gpu]

Artifacts are written to examples/retina/artifacts/hyperneat/.

References:
    Stanley, K. O., D'Ambrosio, D. B., & Gauci, J. (2009). A Hypercube-Based Encoding for
        Evolving Large-Scale Neural Networks. *Artificial Life*, 15(2), 185-212.
        DOI: 10.1162/artl.2009.15.2.15202
"""

from __future__ import annotations

from pathlib import Path

import torch

import polyneat as pn
from examples._example_cli import parse_device_from_cli
from examples._experiment import ExperimentReport, print_experiment_report
from examples.retina._shared import run_retina_experiment

CONFIG_FILE_PATH = Path(__file__).parent / "hyperneat.yaml"
_ARTIFACTS_DIR = Path(__file__).parent / "artifacts" / "hyperneat"


def run_experiment(
    device: torch.device | None = None,
    random_seed: int | None = None,
    artifacts_directory: Path | None = None,
    max_generations: int | None = None,
) -> ExperimentReport:
    """Run the retina plain-HyperNEAT experiment once.

    Args:
        device: Phenotype evaluation device; ``None`` uses the yaml value.
        random_seed: Evolution seed override; ``None`` uses the yaml value.
        artifacts_directory: Where to write artifacts; ``None`` writes none.
        max_generations: Generation cap override, for fast smoke runs.

    Returns:
        Fitness, connection counts, generation count and runtime.
    """
    return run_retina_experiment(
        config_class=pn.HyperNEATConfig,
        algorithm_class=pn.HyperNEATAlgorithm,
        config_file_path=CONFIG_FILE_PATH,
        variant_label="HyperNEAT (baseline)",
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
