"""Visual Discrimination with HyperNEAT - the paper's flagship experiment.

Reproduces Stanley, D'Ambrosio & Gauci (2009), Section 4: locate the center of
the *larger* of two black squares in a 2-D visual field. This is the task that
demonstrates HyperNEAT's core strength - exploiting geometric regularity - and
where it decisively beats a direct encoding (their "P-NEAT").

Why it fits HyperNEAT (and MNIST digit-ID did not): the correct solution is the
SAME local motif repeated at every location - "strongly connect each input node
to the output nodes around the corresponding location, so an output accumulates
more activation the more adjacent loci feed into it" (paper Section 4.1). The
larger object has more pixels, so its center accumulates the most activation.
That motif is a translation-invariant receptive field: high weight when source
and target coordinates are close. A connective CPPN expresses exactly this as a
Gaussian of (x1-x2, y1-y2) - one regularity, reused across the whole grid. A
direct encoding (P-NEAT) must instead discover all R^4 weights independently.

Substrate: a state-space sandwich (paper figure 6c) - an R x R visual-field
input sheet wired directly to an R x R target-field output sheet, no hidden
layer, both sheets sharing the coordinate square [-1, 1] x [-1, 1]. Output nodes
use identity activation so each is the weighted sum of its inputs (the
"accumulation" the paper describes); the substrate's selection is the argmax
output node.

Run from the repository root:
    uv run python examples/visual_discrimination_hyperneat.py
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import torch

import polyneat as pn
from polyneat.core.neat.neat_algorithm import NEATAlgorithm
from polyneat.core.neat.neat_phenotype_decoder import NEATPhenotypeDecoder
from polyneat.evaluators.visual_discrimination_evaluator import (
    VisualDiscriminationFitnessEvaluator,
    generate_visual_discrimination_trials,
)

_THIS_DIR = Path(__file__).parent
_ARTIFACTS_DIR = _THIS_DIR / "visual_discrimination_artifacts"

# Paper evolves at 11 x 11 (121 in, 121 out, 14,641 connections). This demo uses
# a smaller field for CPU-friendly runtime; the whole point of HyperNEAT is that
# the SAME evolved CPPN scales to higher resolution without re-evolving.
FIELD_SIDE = 7
BIG_OBJECT_SIDE = 3
SMALL_OBJECT_SIDE = 1
NUMBER_OF_TRIALS = 60
_TRIAL_SAMPLING_SEED = 20260713


def build_visual_discrimination_algorithm(
    config: pn.HyperNEATConfig, field_side: int
) -> NEATAlgorithm:
    """HyperNEAT assembled with the two-sheet visual-field substrate injected.

    Both sheets share the coordinate square so "source near target" means spatial
    locality - the geometry the paper's receptive-field solution exploits.
    """
    base_algorithm = pn.HyperNEATAlgorithm.from_config(config)
    device = torch.device(config.device_for_phenotype_evaluation)
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


def main() -> None:
    """Evolve a HyperNEAT visual-discrimination solver and report accuracy."""
    config = pn.HyperNEATConfig.load_from_yaml_file(
        _THIS_DIR / "visual_discrimination_hyperneat.yaml"
    )

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

    algorithm = build_visual_discrimination_algorithm(config, field_side=FIELD_SIDE)
    evaluator = VisualDiscriminationFitnessEvaluator(
        trial_fields, true_rows, true_cols, field_side=FIELD_SIDE
    )

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
        callbacks=[
            pn.ConsoleStatisticsLogger(),
            pn.BestGenomePersister(output_directory=_ARTIFACTS_DIR),
        ],
        random_seed=config.random_seed,
    )

    result = runner.run_evolution()

    print(f"\nTermination        : {result.termination_reason}")
    print(f"Generations        : {len(result.full_generation_history)}")
    print(f"Runtime            : {result.total_runtime_seconds:.1f}s")
    print(f"Best mean reward   : {result.best_fitness_ever_achieved:.4f} / 1.0000")
    print("(1.0 = always picks the exact big-object center; ~0.5 = no better than chance)")


if __name__ == "__main__":
    main()
