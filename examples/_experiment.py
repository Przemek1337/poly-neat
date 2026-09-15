"""The contract between the example scripts and their consumers.

Every example module exposes a yaml path constant and a ``run_experiment``
function; :class:`ExampleModule` states that contract structurally (a module
with these attributes matches it - no inheritance), and
:class:`ExperimentReport` is what one full run produces, as the benchmark
harness sees it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import torch


@dataclass(frozen=True)
class ExperimentReport:
    """What one full example run produced, as the benchmark harness sees it.

    The first three fields are the original contract and are unchanged. The
    rest were added for benchmarks that have to be reproducible from their own
    artifacts, and all of them default, so every existing example and every
    existing caller keeps working without knowing they exist.

    Attributes:
        metric_values: Scalar metrics of the run.
        number_of_generations: Generations the run completed.
        runtime_seconds: Wall-clock duration.
        status: ``succeeded`` or a failure label. A run that produced no
            usable model reports its failure here instead of returning
            plausible-looking metrics.
        failure_reason: Why the run failed, when it did.
        effective_configuration: The configuration after command-line and
            programmatic overrides were applied. The source yaml alone does not
            describe a run that overrode part of it.
        artifact_paths: Named paths this run wrote, so a result file points at
            its own checkpoints, predictions and manifest.
        undefined_metrics: Metrics that have no value, and why. A metric with a
            zero denominator is absent with a reason rather than reported as
            zero.
    """

    metric_values: dict[str, float]
    number_of_generations: int
    runtime_seconds: float
    status: str = "succeeded"
    failure_reason: str | None = None
    effective_configuration: dict = field(default_factory=dict)
    artifact_paths: dict[str, str] = field(default_factory=dict)
    undefined_metrics: dict[str, str] = field(default_factory=dict)

    @property
    def succeeded(self) -> bool:
        """Whether this run produced a usable model."""
        return self.status == "succeeded"


def print_experiment_report(report: ExperimentReport) -> None:
    """Print a report the way every example's ``main()`` reports its run."""
    print(f"\nGenerations : {report.number_of_generations}")
    print(f"Runtime     : {report.runtime_seconds:.1f}s")
    for metric_name, metric_value in report.metric_values.items():
        print(f"{metric_name} : {metric_value:.4f}")


EXAMPLE_REGISTRY: dict[str, str] = {
    "cifar10/deepneat_paper": "examples.cifar10.deepneat_paper",
    "cifar10/deepneat_smoke": "examples.cifar10.deepneat_smoke",
    "fashion_mnist/deepneat": "examples.fashion_mnist.deepneat",
    "iris/cneat": "examples.iris.cneat",
    "iris/lneat": "examples.iris.lneat",
    "iris/neatdbm": "examples.iris.neatdbm",
    "mnist/deepneat": "examples.mnist.deepneat",
    "mnist/exact": "examples.mnist.exact",
    "mnist/hyperneat": "examples.mnist.hyperneat",
    "mnist/neat": "examples.mnist.neat",
    "pediatric_pneumonia/deepneat": "examples.pediatric_pneumonia.deepneat",
    "pediatric_pneumonia/exact": "examples.pediatric_pneumonia.exact",
    "pediatric_pneumonia/fixed_cnn": "examples.pediatric_pneumonia.fixed_cnn",
    "pediatric_pneumonia/random_search": "examples.pediatric_pneumonia.random_search",
    "pediatric_pneumonia/transfer_learning": (
        "examples.pediatric_pneumonia.transfer_learning"
    ),
    "retina/hyperneat": "examples.retina.hyperneat",
    "retina/leo": "examples.retina.leo",
    "visual_discrimination/hyperneat": "examples.visual_discrimination.hyperneat",
    "xor/fdneat": "examples.xor.fdneat",
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
