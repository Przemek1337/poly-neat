"""Shared run logic for the two retina examples.

``hyperneat.py`` and ``leo.py`` must differ **only** in which algorithm they
evolve. Any divergence in the surrounding code - termination, callbacks, how the
metrics are computed - would quietly invalidate the comparison the benchmark
exists to make, so the whole run lives here once and each example supplies just
its config class, its algorithm class and its artifact directory.

References:
    Kashtan, N., & Alon, U. (2005). Spontaneous evolution of modularity and network
        motifs. *PNAS*, 102(39), 13773-13778. DOI: 10.1073/pnas.0503610102
    Verbancsics, P., & Stanley, K. O. (2011). Constraining Connectivity to Encourage
        Modularity in HyperNEAT. *GECCO '11*, pp. 1483-1490. DOI: 10.1145/2001576.2001776
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import torch

import polyneat as pn
from examples._experiment import ExperimentReport
from polyneat.algorithms.hyperneat.substrate_modularity import (
    count_cross_hemisphere_connections,
    count_cross_hemisphere_input_dependencies,
    count_expressed_connections,
)
from polyneat.evaluators.retina_evaluator import (
    MAXIMUM_RETINA_FITNESS,
    RetinaProblemEvaluator,
)

TARGET_FITNESS: float = 0.99 * MAXIMUM_RETINA_FITNESS

# Measured, not guessed. HyperNEAT-LEO sits on a plateau at ~208 for the first
# ~300 generations and only then climbs, reaching ~227 by generation 600 and
# ~230 by 1500. A 300-generation cap therefore cut the run off mid-plateau and
# understated LEO badly. The baseline is flat from the start (192.6 -> 199.3 over
# the same 1500 generations), so the budget costs it nothing.
MAX_GENERATION_NUMBER: int = 1500


class _SubstrateDecoder(Protocol):
    """The part of a HyperNEAT-family decoder this module needs."""

    @property
    def substrate(self): ...  # noqa: ANN201

    def decode_substrate_genome(self, genome): ...  # noqa: ANN001, ANN201


def run_retina_experiment(
    config_class,  # noqa: ANN001
    algorithm_class,  # noqa: ANN001
    config_file_path: Path,
    variant_label: str,
    device: torch.device | None = None,
    random_seed: int | None = None,
    artifacts_directory: Path | None = None,
    max_generations: int | None = None,
) -> ExperimentReport:
    """Evolve one HyperNEAT-family variant on the retina problem.

    Args:
        config_class: ``pn.HyperNEATConfig`` or ``pn.HyperNEATLEOConfig``.
        algorithm_class: The matching algorithm class.
        config_file_path: YAML config to load.
        variant_label: Name used in the printed summary.
        device: Phenotype evaluation device; ``None`` uses the yaml value.
        random_seed: Evolution seed override; ``None`` uses the yaml value.
        artifacts_directory: Where to write artifacts; ``None`` writes none.
        max_generations: Generation cap override, for fast smoke runs; ``None``
            uses :data:`MAX_GENERATION_NUMBER`.

    Returns:
        Fitness plus the two connection counts that make the variants
        comparable.
    """
    config = config_class.load_from_yaml_file(config_file_path)
    algorithm = algorithm_class.from_config(config, device_for_phenotype_computation=device)

    callbacks: list = [pn.ConsoleStatisticsLogger()]
    if artifacts_directory is not None:
        callbacks.append(pn.BestGenomePersister(output_directory=artifacts_directory))
        callbacks.append(pn.TensorBoardLogger(log_directory=artifacts_directory / "tensorboard"))

    runner = pn.EvolutionRunner(
        algorithm=algorithm,
        fitness_evaluator=RetinaProblemEvaluator(),
        termination_criterion=pn.CompositeTermination(
            [
                pn.TargetFitnessTermination(target_fitness=TARGET_FITNESS),
                pn.MaxGenerationsTermination(
                    max_generations=(
                        MAX_GENERATION_NUMBER if max_generations is None else max_generations
                    )
                ),
            ]
        ),
        callbacks=callbacks,
        random_seed=config.random_seed if random_seed is None else random_seed,
    )
    result = runner.run_evolution()

    decoder: _SubstrateDecoder = algorithm.phenotype_decoder
    decoded_genome = decoder.decode_substrate_genome(result.best_genome_ever_found)
    expressed_connection_count = count_expressed_connections(decoded_genome)
    crossing_connection_count = count_cross_hemisphere_connections(
        decoder.substrate, decoded_genome
    )
    crossing_fraction = (
        crossing_connection_count / expressed_connection_count
        if expressed_connection_count
        else 0.0
    )
    crossing_dependency_count = count_cross_hemisphere_input_dependencies(
        decoder.substrate, decoded_genome
    )

    print(f"\nVariant             : {variant_label}")
    print(f"Termination         : {result.termination_reason}")
    print(
        f"Best fitness        : {result.best_fitness_ever_achieved:.2f} "
        f"/ {MAXIMUM_RETINA_FITNESS:.0f}"
    )
    print(f"Expressed links     : {expressed_connection_count}")
    print(
        f"Crossing links      : {crossing_connection_count} "
        f"({crossing_fraction:.1%} of expressed) - wiring locality"
    )
    print(
        f"Crossing dependency : {crossing_dependency_count} / 8 "
        f"- functional modularity, 0 is perfectly modular"
    )
    return ExperimentReport(
        metric_values={
            "best_fitness": result.best_fitness_ever_achieved,
            "number_of_expressed_connections": float(expressed_connection_count),
            "number_of_cross_hemisphere_connections": float(crossing_connection_count),
            "fraction_of_cross_hemisphere_connections": crossing_fraction,
            "number_of_cross_hemisphere_input_dependencies": float(crossing_dependency_count),
        },
        number_of_generations=len(result.full_generation_history),
        runtime_seconds=result.total_runtime_seconds,
    )
