"""XOR baseline — PolyNEAT end-to-end demo.

Demonstrates the full PolyNEAT pipeline on the canonical XOR problem.
NEAT starts from minimal networks (inputs + bias → outputs) and evolves
both weights and topology until a network solves all four XOR patterns.

Run from the repository root:
    uv run python examples/xor_baseline.py [--cpu | --gpu]

Artifacts are written to examples/xor_artifacts/:
    best_genome_gen_<N>.json   — best genome per generation (JSON)
    best_genome_gen_<N>.pkl    — best genome per generation (pickle)
    topology_gen_<N>.png       — network topology visualizations
    tensorboard/               — TensorBoard event files
"""

from __future__ import annotations

from pathlib import Path

from _example_cli import parse_device_from_cli

import polyneat as pn
from polyneat.evaluators.xor_evaluator import XORFitnessEvaluator

_THIS_DIR = Path(__file__).parent
_ARTIFACTS_DIR = _THIS_DIR / "xor_artifacts"


def main() -> None:
    device = parse_device_from_cli()
    config = pn.NEATConfig.load_from_yaml_file(_THIS_DIR / "xor_baseline.yaml")

    algorithm = pn.NEATAlgorithm.from_config(config, device_for_phenotype_computation=device)

    runner = pn.EvolutionRunner(
        algorithm=algorithm,
        fitness_evaluator=XORFitnessEvaluator(),
        termination_criterion=pn.CompositeTermination(
            [
                pn.TargetFitnessTermination(target_fitness=3.95),
                pn.MaxGenerationsTermination(max_generations=300),
            ]
        ),
        callbacks=[
            pn.ConsoleStatisticsLogger(),
            pn.BestGenomePersister(output_directory=_ARTIFACTS_DIR),
            pn.NetworkTopologyVisualizer(output_directory=_ARTIFACTS_DIR),
            pn.TensorBoardLogger(log_directory=_ARTIFACTS_DIR / "tensorboard"),
        ],
        random_seed=config.random_seed,
    )

    result = runner.run_evolution()

    total_gens = len(result.full_generation_history)
    solved = result.best_fitness_ever_achieved >= 3.95

    print(f"\nTermination : {result.termination_reason}")
    print(f"Best fitness: {result.best_fitness_ever_achieved:.4f} / 4.0000")
    print(f"Generations : {total_gens}")
    print(f"Runtime     : {result.total_runtime_seconds:.1f}s")
    print(f"XOR solved  : {'YES' if solved else 'NO'}")


if __name__ == "__main__":
    main()
