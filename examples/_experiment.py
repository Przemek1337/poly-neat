"""The contract between the example scripts and their consumers.

Every example module exposes a yaml path constant and a ``run_experiment``
function; :class:`ExampleModule` states that contract structurally (a module
with these attributes matches it - no inheritance), and
:class:`ExperimentReport` is what one full run produces, as the benchmark
harness sees it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import torch


@dataclass(frozen=True)
class ExperimentReport:
    """What one full example run produced, as the benchmark harness sees it."""

    metric_values: dict[str, float]
    number_of_generations: int
    runtime_seconds: float


def print_experiment_report(report: ExperimentReport) -> None:
    """Print a report the way every example's ``main()`` reports its run."""
    print(f"\nGenerations : {report.number_of_generations}")
    print(f"Runtime     : {report.runtime_seconds:.1f}s")
    for metric_name, metric_value in report.metric_values.items():
        print(f"{metric_name} : {metric_value:.4f}")


EXAMPLE_REGISTRY: dict[str, str] = {
    "iris/cneat": "examples.iris.cneat",
    "iris/lneat": "examples.iris.lneat",
    "iris/neatdbm": "examples.iris.neatdbm",
    "mnist/exact": "examples.mnist.exact",
    "mnist/hyperneat": "examples.mnist.hyperneat",
    "mnist/neat": "examples.mnist.neat",
    "visual_discrimination/hyperneat": "examples.visual_discrimination.hyperneat",
    "xor/fsneat": "examples.xor.fsneat",
    "xor/hyperneat": "examples.xor.hyperneat",
    "xor/neat": "examples.xor.neat",
    "xor/neatdbm": "examples.xor.neatdbm",
}


class ExampleModule(Protocol):
    """Structural contract every example module satisfies.

    Example modules do not inherit from anything - a module with these
    attributes matches the protocol as-is. The benchmark harness annotates
    the module returned by ``importlib.import_module`` with this type, so
    the contract is checked by a type checker and documented in one place
    instead of only in prose.
    """

    CONFIG_FILE_PATH: Path

    def run_experiment(
        self,
        device: torch.device | None = None,
        random_seed: int | None = None,
        artifacts_directory: Path | None = None,
    ) -> ExperimentReport: ...
