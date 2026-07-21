"""FS-NEAT on XOR with distractor inputs - automatic feature selection demo.

The task is the scenario from Whiteson et al. (2005): the four XOR patterns
are padded with pure-noise distractor columns, so only 2 of the 8 inputs carry
signal. FS-NEAT starts every genome with a *single* random input->output
connection instead of a fully connected layer; evolution has to discover both
the XOR topology and which inputs are worth connecting at all.

At the end the script reports which input features the best network uses,
making the feature selection visible.

Run from the repository root:
    uv run python -m examples.xor.fsneat [--cpu | --gpu]

Artifacts are written to examples/xor/artifacts/fsneat/:
    best_genome.json        - best genome of the run (JSON)
    best_genome.pkl         - best genome of the run (pickle)
    topology/final_best.svg - topology render of the final best genome
    tensorboard/            - TensorBoard event files
"""

from __future__ import annotations

from pathlib import Path

import torch

import polyneat as pn
from examples._example_cli import parse_device_from_cli
from examples._experiment import ExperimentReport, print_experiment_report
from polyneat.evaluators.xor_with_distractors_evaluator import XORWithDistractorsEvaluator

CONFIG_FILE_PATH = Path(__file__).parent / "fsneat.yaml"
_ARTIFACTS_DIR = Path(__file__).parent / "artifacts" / "fsneat"

_TARGET_FITNESS = 3.95
_MAX_GENERATION_NUMBER = 500


def _input_node_ids_used_by(genome: pn.NEATGenome, number_of_inputs: int) -> list[int]:
    """Input node ids that feed the network through at least one enabled connection."""
    input_node_ids = set(range(number_of_inputs))
    enabled_source_node_ids = {
        connection.source_node_id for connection in genome.connection_genes if connection.is_enabled
    }
    return sorted(input_node_ids & enabled_source_node_ids)


def run_experiment(
    device: torch.device | None = None,
    random_seed: int | None = None,
    artifacts_directory: Path | None = None,
) -> ExperimentReport:
    """Run the full XOR FS-NEAT experiment once.

    Args:
        device: Phenotype evaluation device; ``None`` uses the yaml value.
        random_seed: Evolution seed override; ``None`` uses the yaml value.
        artifacts_directory: Where to write artifacts; ``None`` writes none.

    Returns:
        The run's fitness, generation count and runtime.
    """
    config = pn.NEATConfig.load_from_yaml_file(CONFIG_FILE_PATH)
    algorithm = pn.FSNEATAlgorithm.from_config(config, device_for_phenotype_computation=device)

    callbacks: list = [pn.ConsoleStatisticsLogger()]
    if artifacts_directory is not None:
        callbacks.append(pn.BestGenomePersister(output_directory=artifacts_directory))
        callbacks.append(pn.NetworkTopologyVisualizer(output_directory=artifacts_directory))
        callbacks.append(pn.TensorBoardLogger(log_directory=artifacts_directory / "tensorboard"))

    runner = pn.EvolutionRunner(
        algorithm=algorithm,
        fitness_evaluator=XORWithDistractorsEvaluator(),
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

    best_genome = result.best_genome_ever_found
    assert isinstance(best_genome, pn.NEATGenome)
    used_inputs = _input_node_ids_used_by(best_genome, config.number_of_input_nodes)
    solved = result.best_fitness_ever_achieved >= _TARGET_FITNESS
    print(f"\nTermination : {result.termination_reason}")
    print(f"XOR solved  : {'YES' if solved else 'NO'}")
    print(
        f"Inputs used : {used_inputs} "
        f"(relevant inputs are 0 and 1; 2..{config.number_of_input_nodes - 1} are noise)"
    )
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
