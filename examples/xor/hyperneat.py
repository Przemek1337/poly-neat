"""XOR via HyperNEAT - PolyNEAT end-to-end demo.

HyperNEAT evolves a CPPN that generates the connection weights of a fixed
layered substrate. The substrate (2 inputs -> 3 hidden -> 1 output) is the
network that actually computes XOR; evolution searches CPPNs, not substrate
weights directly.

Run from the repository root:
    uv run python -m examples.xor.hyperneat [--cpu | --gpu]

Artifacts are written to examples/xor/artifacts/hyperneat/:
    best_genome.json - best CPPN genome of the run (JSON)
    best_genome.pkl  - best CPPN genome of the run (pickle)
"""
from __future__ import annotations

from pathlib import Path

import torch

import polyneat as pn
from examples._example_cli import parse_device_from_cli
from examples._experiment import ExperimentReport, print_experiment_report
from polyneat.evaluators.xor_evaluator import XORFitnessEvaluator

CONFIG_FILE_PATH = Path(__file__).parent / "hyperneat.yaml"
_ARTIFACTS_DIR = Path(__file__).parent / "artifacts" / "hyperneat"

_TARGET_FITNESS = 3.95
_MAX_GENERATION_NUMBER = 300


def run_experiment(
    device: torch.device | None = None,
    random_seed: int | None = None,
    artifacts_directory: Path | None = None,
) -> ExperimentReport:
    """Run the full XOR HyperNEAT experiment once.

    Args:
        device: Phenotype evaluation device; ``None`` uses the yaml value.
        random_seed: Evolution seed override; ``None`` uses the yaml value.
        artifacts_directory: Where to write artifacts; ``None`` writes none.

    Returns:
        The run's fitness, generation count and runtime.
    """
    config = pn.HyperNEATConfig.load_from_yaml_file(CONFIG_FILE_PATH)
    algorithm = pn.HyperNEATAlgorithm.from_config(
        config, device_for_phenotype_computation=device
    )

    callbacks: list = [pn.ConsoleStatisticsLogger()]
    if artifacts_directory is not None:
        callbacks.append(pn.BestGenomePersister(output_directory=artifacts_directory))

    runner = pn.EvolutionRunner(
        algorithm=algorithm,
        fitness_evaluator=XORFitnessEvaluator(),
        termination_criterion=pn.CompositeTermination(
            [
                pn.TargetFitnessTermination(target_fitness=_TARGET_FITNESS),
                pn.MaxGenerationsTermination(max_generations=_MAX_GENERATION_NUMBER),
            ]
        ),
        callbacks=callbacks,
        random_seed=config.random_seed if random_seed is None else random_seed,
    )
    result = runner.run_evolution()

    solved = result.best_fitness_ever_achieved >= _TARGET_FITNESS
    print(f"\nTermination : {result.termination_reason}")
    print(f"XOR solved  : {'YES' if solved else 'NO'}")
    return ExperimentReport(
        metric_values={"best_fitness": result.best_fitness_ever_achieved},
        number_of_generations=len(result.full_generation_history),
        runtime_seconds=result.total_runtime_seconds,
    )


def main() -> None:
    device = parse_device_from_cli()
    report = run_experiment(device=device, artifacts_directory=_ARTIFACTS_DIR)
    print_experiment_report(report)


if __name__ == "__main__":
    main()
