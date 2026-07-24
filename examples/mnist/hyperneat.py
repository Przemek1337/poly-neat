"""MNIST digit classification with HyperNEAT - PolyNEAT end-to-end demo.

HyperNEAT evolves a small CPPN that paints the connection weights of a fixed
substrate as a function of geometry. The substrate is a state-space sandwich
(Stanley, D'Ambrosio & Gauci, 2009, figure 6c): a ``GRID x GRID`` input sheet
whose nodes sit at their pixel coordinates, wired directly to a row of ten
output nodes with no hidden layer. With linear (``identity``) output activation
the substrate is a multinomial logistic regression whose 49x10 weight matrix is
a smooth function of pixel position.

The 28x28 images are average-pooled to ``GRID x GRID`` to keep the number of
queried connections small. Training uses the smooth softmax-likelihood fitness;
the best genome is reported as accuracy on train and a held-out test subset.

The dataset is downloaded once (standard Keras ``mnist.npz``) and cached under
``examples/mnist/data/``. Only numpy and torch are required.

Run from the repository root:
    uv run python -m examples.mnist.hyperneat [--cpu | --gpu]

Artifacts are written to examples/mnist/artifacts/hyperneat/:
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
from examples.mnist.dataset import load_mnist
from polyneat.core.neat.neat_algorithm import NEATAlgorithm
from polyneat.core.neat.neat_phenotype_decoder import NEATPhenotypeDecoder
from polyneat.evaluators.classification_accuracy_evaluator import (
    ClassificationAccuracyEvaluator,
)
from polyneat.evaluators.softmax_likelihood_evaluator import (
    SoftmaxLikelihoodFitnessEvaluator,
)

CONFIG_FILE_PATH = Path(__file__).parent / "hyperneat.yaml"
_ARTIFACTS_DIR = Path(__file__).parent / "artifacts" / "hyperneat"

GRID = 7
NUMBER_OF_CLASSES = 10

TRAINING_SUBSET_SIZE = 3000
TEST_SUBSET_SIZE = 2000

_SUBSET_SAMPLING_SEED = 12345


def build_mnist_hyperneat_algorithm(
    config: pn.HyperNEATConfig,
    grid_side: int,
    number_of_classes: int,
    device_for_phenotype_computation: torch.device | None = None,
) -> NEATAlgorithm:
    """Assemble a HyperNEAT algorithm whose decoder uses a 2-D image substrate.

    Builds the algorithm with ``HyperNEATAlgorithm.from_config`` and replaces its
    decoder with one over a two-sheet sandwich substrate: a ``grid_side x
    grid_side`` input pixel grid wired to a row of ``number_of_classes`` output
    nodes. ``output_sheet_shares_input_plane=False`` lifts the output row into a
    separate coordinate band, since the classes have no spatial relationship to
    the pixels. A ``device_for_phenotype_computation`` of ``None`` falls back to
    ``config.device_for_phenotype_evaluation``.
    """
    device = device_for_phenotype_computation or torch.device(
        config.device_for_phenotype_evaluation
    )
    base_algorithm = pn.HyperNEATAlgorithm.from_config(
        config, device_for_phenotype_computation=device
    )

    substrate = pn.build_grid_sandwich_substrate(
        input_grid_shape=(grid_side, grid_side),
        output_grid_shape=(1, number_of_classes),
        coordinate_range_min=config.substrate_coordinate_range_min,
        coordinate_range_max=config.substrate_coordinate_range_max,
        include_bias_node=config.include_substrate_bias_node,
        output_sheet_shares_input_plane=False,
    )
    grid_decoder = pn.HyperNEATPhenotypeDecoder(
        substrate=substrate,
        cppn_phenotype_decoder=NEATPhenotypeDecoder(device_for_computation=device),
        weight_expression_threshold=config.weight_expression_threshold,
        max_substrate_connection_weight_magnitude=(
            config.max_substrate_connection_weight_magnitude
        ),
        substrate_node_activation_function_name=config.substrate_node_activation_function,
        device_for_computation=device,
    )
    return dataclasses.replace(base_algorithm, _phenotype_decoder=grid_decoder)


def run_experiment(
    device: torch.device | None = None,
    random_seed: int | None = None,
    artifacts_directory: Path | None = None,
) -> ExperimentReport:
    """Run the full MNIST HyperNEAT experiment once.

    Args:
        device: Phenotype evaluation device; ``None`` uses the yaml value.
        random_seed: Evolution seed override; ``None`` uses the yaml value.
            The subset draw always uses the module's sampling seed, so
            repeats see the same data.
        artifacts_directory: Where to write artifacts; ``None`` writes none.

    Returns:
        Train and test accuracy of the best evolved network.
    """
    dataset = load_mnist(
        random_seed=_SUBSET_SAMPLING_SEED,
        grid_side=GRID,
        max_train_samples=TRAINING_SUBSET_SIZE,
        max_test_samples=TEST_SUBSET_SIZE,
    )
    print(
        f"MNIST loaded: {dataset.train_features.shape[0]} train / "
        f"{dataset.test_features.shape[0]} test "
        f"samples, {GRID}x{GRID} input grid, {NUMBER_OF_CLASSES} classes."
    )

    config = pn.HyperNEATConfig.load_from_yaml_file(CONFIG_FILE_PATH)

    algorithm = build_mnist_hyperneat_algorithm(
        config,
        grid_side=GRID,
        number_of_classes=NUMBER_OF_CLASSES,
        device_for_phenotype_computation=device,
    )

    training_evaluator = pn.ParallelFitnessEvaluatorWrapper(
        wrapped_evaluator=SoftmaxLikelihoodFitnessEvaluator(
            dataset.train_features, dataset.train_labels
        )
    )

    callbacks: list = [pn.ConsoleStatisticsLogger()]
    if artifacts_directory is not None:
        callbacks.append(pn.BestGenomePersister(output_directory=artifacts_directory))

    runner = pn.EvolutionRunner(
        algorithm=algorithm,
        fitness_evaluator=training_evaluator,
        termination_criterion=pn.CompositeTermination(
            [
                pn.TargetFitnessTermination(target_fitness=0.85),
                pn.MaxGenerationsTermination(max_generations=150),
                pn.FitnessStagnationTermination(stagnation_generations=40),
            ]
        ),
        callbacks=callbacks,
        random_seed=config.random_seed if random_seed is None else random_seed,
    )
    result = runner.run_evolution()

    best_phenotype = algorithm.phenotype_decoder.build_phenotype_from_genome(
        result.best_genome_ever_found
    )
    train_accuracy = ClassificationAccuracyEvaluator(
        dataset.train_features, dataset.train_labels
    ).evaluate_single_phenotype(best_phenotype)
    test_accuracy = ClassificationAccuracyEvaluator(
        dataset.test_features, dataset.test_labels
    ).evaluate_single_phenotype(best_phenotype)

    print(f"\nTermination            : {result.termination_reason}")
    print(f"Train fitness (softmax): {result.best_fitness_ever_achieved:.4f}")
    return ExperimentReport(
        metric_values={"train_accuracy": train_accuracy, "test_accuracy": test_accuracy},
        number_of_generations=len(result.full_generation_history),
        runtime_seconds=result.total_runtime_seconds,
    )


def main() -> None:
    """Evolve a HyperNEAT MNIST classifier and print training and test accuracy."""
    device = parse_device_from_cli()
    report = run_experiment(device=device, artifacts_directory=_ARTIFACTS_DIR)
    print_experiment_report(report)


if __name__ == "__main__":
    main()
