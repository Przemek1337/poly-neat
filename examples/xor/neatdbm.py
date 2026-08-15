"""XOR with NEAT-DBM - difference-based mutation demo.

Demonstrates NEAT-DBM (Stanovov et al., 2021) on the canonical XOR problem.
NEAT-DBM runs the standard NEAT loop and additionally recombines each fresh
child's connection weights from three donor genomes at the positions their
innovation numbers share, in the style of differential evolution.

Run from the repository root:
    uv run python -m examples.xor.neatdbm [--cpu | --gpu]

Artifacts are written to examples/xor/artifacts/neatdbm/:
    best_genome.json        - best genome of the run (JSON)
    best_genome.pkl         - best genome of the run (pickle)
    topology/final_best.svg - topology render of the final best genome
    tensorboard/            - TensorBoard event files
"""

from __future__ import annotations

from pathlib import Path

import torch

import polyneat as pn
from examples._experiment import ExperimentReport
from examples._run import run_example_main
from polyneat.evaluators.xor_evaluator import XORFitnessEvaluator

CONFIG_FILE_PATH = Path(__file__).parent / "neatdbm.yaml"
_ARTIFACTS_DIR = Path(__file__).parent / "artifacts" / "neatdbm"

_TARGET_FITNESS = 3.95
_MAX_GENERATION_NUMBER = 300


def run_experiment(
    device: torch.device | None = None,
    random_seed: int | None = None,
    artifacts_directory: Path | None = None,
) -> ExperimentReport:
    """Run the full XOR NEAT-DBM experiment once.

    Args:
        device: Phenotype evaluation device; ``None`` uses the yaml value.
        random_seed: Evolution seed override; ``None`` uses the yaml value.
        artifacts_directory: Where to write artifacts; ``None`` writes none.

    Returns:
        The run's fitness, generation count and runtime.
    """
    config = pn.NEATDBMConfig.load_from_yaml_file(CONFIG_FILE_PATH)
    algorithm = pn.NEATDBMAlgorithm.from_config(config, device_for_phenotype_computation=device)

    callbacks: list = [pn.ConsoleStatisticsLogger()]
    if artifacts_directory is not None:
        callbacks.append(pn.BestGenomePersister(output_directory=artifacts_directory))
        callbacks.append(pn.NetworkTopologyVisualizer(output_directory=artifacts_directory))
        callbacks.append(
            pn.TensorBoardLogger(
                log_directory=artifacts_directory / "tensorboard",
                run_label="xor-neatdbm",
            )
        )

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
    run_example_main(run_experiment, _ARTIFACTS_DIR)


if __name__ == "__main__":
    main()
