"""Visual discrimination with HyperNEAT - PolyNEAT end-to-end demo.

Reproduces Stanley, D'Ambrosio & Gauci (2009), Section 4: locate the center of
the larger of two black squares in a 2-D visual field.

Substrate: a state-space sandwich (paper figure 6c) - an R x R visual-field
input sheet wired directly to an R x R target-field output sheet, with no hidden
layer and both sheets sharing the coordinate square [-1, 1] x [-1, 1]. Output
nodes use identity activation, so each output is the weighted sum of its inputs
and the network's selection is the argmax output node.

Run from the repository root:
    uv run python -m examples.visual_discrimination.hyperneat [--cpu | --gpu]

Artifacts are written to examples/visual_discrimination/artifacts/hyperneat/:
    best_genome.json - best CPPN genome of the run (JSON)
    best_genome.pkl  - best CPPN genome of the run (pickle)
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import torch

import polyneat as pn
from examples._example_cli import parse_device_from_cli
from examples._experiment import ExperimentReport, print_experiment_report
from polyneat.core.neat.neat_algorithm import NEATAlgorithm
from polyneat.core.neat.neat_phenotype_decoder import NEATPhenotypeDecoder
from polyneat.evaluators.visual_discrimination_evaluator import (
    VisualDiscriminationFitnessEvaluator,
    generate_visual_discrimination_trials,
)

CONFIG_FILE_PATH = Path(__file__).parent / "hyperneat.yaml"
_ARTIFACTS_DIR = Path(__file__).parent / "artifacts" / "hyperneat"

# The paper evolves an 11 x 11 field (121 in, 121 out, 14,641 connections); this
# demo uses a smaller field for CPU-friendly runtime.
FIELD_SIDE = 7
BIG_OBJECT_SIDE = 3
SMALL_OBJECT_SIDE = 1
NUMBER_OF_TRIALS = 60
_TRIAL_SAMPLING_SEED = 20260713


def build_visual_discrimination_algorithm(
    config: pn.HyperNEATConfig,
    field_side: int,
    device_for_phenotype_computation: torch.device | None = None,
) -> NEATAlgorithm:
    """HyperNEAT assembled with the two-sheet visual-field substrate injected.

    Both sheets share the coordinate square, so nearby source and target
    coordinates correspond to spatially local connections. A
    ``device_for_phenotype_computation`` of ``None`` falls back to
    ``config.device_for_phenotype_evaluation``.
    """
    device = device_for_phenotype_computation or torch.device(
        config.device_for_phenotype_evaluation
    )
    base_algorithm = pn.HyperNEATAlgorithm.from_config(
        config, device_for_phenotype_computation=device
    )
    substrate = pn.build_grid_sandwich_substrate(
        input_grid_shape=(field_side, field_side),
        output_grid_shape=(field_side, field_side),
        coordinate_range_min=config.substrate_coordinate_range_min,
        coordinate_range_max=config.substrate_coordinate_range_max,
        include_bias_node=config.include_substrate_bias_node,
        output_sheet_shares_input_plane=True,
    )
    decoder = pn.HyperNEATPhenotypeDecoder(
        substrate=substrate,
        cppn_phenotype_decoder=NEATPhenotypeDecoder(device_for_computation=device),
        weight_expression_threshold=config.weight_expression_threshold,
        max_substrate_connection_weight_magnitude=(
            config.max_substrate_connection_weight_magnitude
        ),
        substrate_node_activation_function_name=config.substrate_node_activation_function,
        device_for_computation=device,
    )
    return dataclasses.replace(base_algorithm, _phenotype_decoder=decoder)


def run_experiment(
    device: torch.device | None = None,
    random_seed: int | None = None,
    artifacts_directory: Path | None = None,
) -> ExperimentReport:
    """Run the full visual discrimination HyperNEAT experiment once.

    Args:
        device: Phenotype evaluation device; ``None`` uses the yaml value.
        random_seed: Evolution seed override; ``None`` uses the yaml value.
            Trial generation always uses the module's sampling seed.
        artifacts_directory: Where to write artifacts; ``None`` writes none.

    Returns:
        The best mean reward (1.0 = always picks the exact big-object
        center; ~0.5 = no better than chance).
    """
    config = pn.HyperNEATConfig.load_from_yaml_file(CONFIG_FILE_PATH)

    trial_fields, true_rows, true_cols = generate_visual_discrimination_trials(
        field_side=FIELD_SIDE,
        big_object_side=BIG_OBJECT_SIDE,
        small_object_side=SMALL_OBJECT_SIDE,
        number_of_trials=NUMBER_OF_TRIALS,
        seed=_TRIAL_SAMPLING_SEED,
    )
    print(
        f"Visual discrimination: {FIELD_SIDE}x{FIELD_SIDE} field "
        f"({FIELD_SIDE * FIELD_SIDE} inputs -> {FIELD_SIDE * FIELD_SIDE} outputs, "
        f"{(FIELD_SIDE * FIELD_SIDE) ** 2} potential connections), "
        f"{NUMBER_OF_TRIALS} trials."
    )

    algorithm = build_visual_discrimination_algorithm(
        config, field_side=FIELD_SIDE, device_for_phenotype_computation=device
    )
    evaluator = VisualDiscriminationFitnessEvaluator(
        trial_fields, true_rows, true_cols, field_side=FIELD_SIDE
    )

    callbacks: list = [pn.ConsoleStatisticsLogger()]
    if artifacts_directory is not None:
        callbacks.append(pn.BestGenomePersister(output_directory=artifacts_directory))

    runner = pn.EvolutionRunner(
        algorithm=algorithm,
        fitness_evaluator=pn.ParallelFitnessEvaluatorWrapper(wrapped_evaluator=evaluator),
        termination_criterion=pn.CompositeTermination(
            [
                pn.TargetFitnessTermination(target_fitness=0.98),
                pn.MaxGenerationsTermination(max_generations=150),
                pn.FitnessStagnationTermination(stagnation_generations=40),
            ]
        ),
        callbacks=callbacks,
        random_seed=config.random_seed if random_seed is None else random_seed,
    )
    result = runner.run_evolution()

    print(f"\nTermination : {result.termination_reason}")
    print("(1.0 = always picks the exact big-object center; ~0.5 = no better than chance)")
    return ExperimentReport(
        metric_values={"best_mean_reward": result.best_fitness_ever_achieved},
        number_of_generations=len(result.full_generation_history),
        runtime_seconds=result.total_runtime_seconds,
    )


def main() -> None:
    """Evolve a HyperNEAT visual-discrimination solver and report the reward."""
    device = parse_device_from_cli()
    report = run_experiment(device=device, artifacts_directory=_ARTIFACTS_DIR)
    print_experiment_report(report)


if __name__ == "__main__":
    main()
