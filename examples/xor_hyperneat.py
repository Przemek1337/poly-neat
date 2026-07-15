"""XOR via HyperNEAT - PolyNEAT end-to-end demo.

HyperNEAT evolves a CPPN that generates the connection weights of a fixed
layered substrate. The substrate (2 inputs -> 3 hidden -> 1 output) is the
network that actually computes XOR; evolution searches CPPNs, not substrate
weights directly.

Run from the repository root:
    uv run python examples/xor_hyperneat.py

Artifacts are written to examples/xor_hyperneat_artifacts/:
    best_genome_gen_<N>.json   - best CPPN genome per generation (JSON)
"""
from __future__ import annotations

from pathlib import Path

import polyneat as pn
from polyneat.evaluators.xor_evaluator import XORFitnessEvaluator

_THIS_DIR = Path(__file__).parent
_ARTIFACTS_DIR = _THIS_DIR / "xor_hyperneat_artifacts"


def main() -> None:
    config = pn.HyperNEATConfig.load_from_yaml_file(_THIS_DIR / "xor_hyperneat.yaml")

    algorithm = pn.HyperNEATAlgorithm.from_config(config)

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
