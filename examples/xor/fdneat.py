"""FD-NEAT on XOR with distractor inputs - automatic feature *de*selection demo.

The counterpart of ``examples/xor/fsneat.py``: identical data, identical fitness
function, identical generation budget. FS-NEAT starts from a single random input
connection and adds the inputs that pay off; FD-NEAT starts fully connected and
deletes the ones that do not (Tan et al., 2012).

Both examples report ``number_of_connected_input_features`` computed by the same
library function, so the two strategies are directly comparable. The ideal
result is 2 - only the real XOR inputs left with a path to the output.

Run from the repository root:
    uv run python -m examples.xor.fdneat [--cpu | --gpu]

Artifacts are written to examples/xor/artifacts/fdneat/:
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
from polyneat.evaluators.xor_with_distractors_evaluator import XORWithDistractorsEvaluator
from polyneat.nn.topology_utilities import find_node_ids_with_enabled_path_to_any_target

CONFIG_FILE_PATH = Path(__file__).parent / "fdneat.yaml"
_ARTIFACTS_DIR = Path(__file__).parent / "artifacts" / "fdneat"

_TARGET_FITNESS = 3.95
_MAX_GENERATION_NUMBER = 500


def _input_node_ids_used_by(
    genome: pn.NEATGenome, number_of_inputs: int, number_of_outputs: int
) -> list[int]:
    """Input node ids with an enabled path to any output node.

    Reachability, not merely an outgoing connection: an input wired into a
    hidden node that leads nowhere is not a feature the network uses. The same
    library function backs ``examples/xor/fsneat.py``, so the two feature
    selection strategies are measured identically and can be compared.

    Node ids follow the generation-0 layout: inputs ``0 .. n-1``, bias ``n``,
    outputs ``n+1 .. n+number_of_outputs``.
    """
    return find_node_ids_with_enabled_path_to_any_target(
        candidate_source_node_ids=range(number_of_inputs),
        target_node_ids=range(
            number_of_inputs + 1, number_of_inputs + 1 + number_of_outputs
        ),
        enabled_directed_edges=[
            (connection.source_node_id, connection.target_node_id)
            for connection in genome.connection_genes
            if connection.is_enabled
        ],
    )


def run_experiment(
    device: torch.device | None = None,
    random_seed: int | None = None,
    artifacts_directory: Path | None = None,
) -> ExperimentReport:
    """Run the full XOR FD-NEAT experiment once.

    Args:
        device: Phenotype evaluation device; ``None`` uses the yaml value.
        random_seed: Evolution seed override; ``None`` uses the yaml value.
        artifacts_directory: Where to write artifacts; ``None`` writes none.

    Returns:
        The run's fitness, surviving input-feature count, generation count and
        runtime.
    """
    config = pn.FDNEATConfig.load_from_yaml_file(CONFIG_FILE_PATH)
    algorithm = pn.FDNEATAlgorithm.from_config(config, device_for_phenotype_computation=device)

    callbacks: list = [pn.ConsoleStatisticsLogger()]
    if artifacts_directory is not None:
        callbacks.append(pn.BestGenomePersister(output_directory=artifacts_directory))
        callbacks.append(pn.NetworkTopologyVisualizer(output_directory=artifacts_directory))
        callbacks.append(
            pn.TensorBoardLogger(
                log_directory=artifacts_directory / "tensorboard",
                run_label="xor-fdneat",
            )
        )

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
    used_inputs = _input_node_ids_used_by(
        best_genome, config.number_of_input_nodes, config.number_of_output_nodes
    )
    solved = result.best_fitness_ever_achieved >= _TARGET_FITNESS
    print(f"\nTermination : {result.termination_reason}")
    print(f"XOR solved  : {'YES' if solved else 'NO'}")
    print(
        f"Inputs used : {used_inputs} "
        f"(relevant inputs are 0 and 1; 2..{config.number_of_input_nodes - 1} are noise)"
    )
    print(
        f"Deselected  : {config.number_of_input_nodes - len(used_inputs)} of "
        f"{config.number_of_input_nodes} inputs"
    )
    return ExperimentReport(
        metric_values={
            "best_fitness": result.best_fitness_ever_achieved,
            "number_of_connected_input_features": float(len(used_inputs)),
        },
        number_of_generations=len(result.full_generation_history),
        runtime_seconds=result.total_runtime_seconds,
    )


def main() -> None:
    run_example_main(run_experiment, _ARTIFACTS_DIR)


if __name__ == "__main__":
    main()
