"""FS-NEAT on XOR with distractor inputs — automatic feature selection demo.

The task is the scenario from Whiteson et al. (2005): the four XOR patterns
are padded with pure-noise distractor columns, so only 2 of the 8 inputs carry
signal. FS-NEAT starts every genome with a *single* random input->output
connection instead of a fully connected layer; evolution has to discover both
the XOR topology and which inputs are worth connecting at all.

At the end the script reports which input features the best network actually
uses, so you can see the feature selection working.

Run from the repository root:
    uv run python examples/xor_fsneat.py [--cpu | --gpu]

Artifacts are written to examples/xor_fsneat_artifacts/ (best genome, topology
renders, TensorBoard event files under tensorboard/).
"""

from __future__ import annotations

from pathlib import Path

from _example_cli import parse_device_from_cli

import polyneat as pn
from polyneat.evaluators.xor_with_distractors_evaluator import XORWithDistractorsEvaluator

_THIS_DIR = Path(__file__).parent
_ARTIFACTS_DIR = _THIS_DIR / "xor_fsneat_artifacts"


def _input_node_ids_used_by(genome: pn.NEATGenome, number_of_inputs: int) -> list[int]:
    """Input node ids that feed the network through at least one enabled connection."""
    input_node_ids = set(range(number_of_inputs))
    enabled_source_node_ids = {
        connection.source_node_id for connection in genome.connection_genes if connection.is_enabled
    }
    return sorted(input_node_ids & enabled_source_node_ids)


def main() -> None:
    device = parse_device_from_cli()
    config = pn.NEATConfig.load_from_yaml_file(_THIS_DIR / "xor_fsneat.yaml")

    algorithm = pn.FSNEATAlgorithm.from_config(config, device_for_phenotype_computation=device)

    runner = pn.EvolutionRunner(
        algorithm=algorithm,
        fitness_evaluator=XORWithDistractorsEvaluator(),
        termination_criterion=pn.CompositeTermination(
            [
                pn.TargetFitnessTermination(target_fitness=3.95),
                pn.MaxGenerationsTermination(max_generations=500),
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

    best_genome = result.best_genome_ever_found
    assert isinstance(best_genome, pn.NEATGenome)
    used_inputs = _input_node_ids_used_by(best_genome, config.number_of_input_nodes)

    print(f"\nTermination   : {result.termination_reason}")
    print(f"Best fitness  : {result.best_fitness_ever_achieved:.4f} / 4.0000")
    print(f"Generations   : {len(result.full_generation_history)}")
    print(f"Runtime       : {result.total_runtime_seconds:.1f}s")
    print(f"XOR solved    : {'YES' if result.best_fitness_ever_achieved >= 3.95 else 'NO'}")
    print(
        f"Inputs used   : {used_inputs} "
        f"(relevant inputs are 0 and 1; 2..{config.number_of_input_nodes - 1} are noise)"
    )


if __name__ == "__main__":
    main()
