"""MNIST digit classification with NEAT - PolyNEAT end-to-end demo.

Evolves feed-forward networks that classify handwritten digits. Because vanilla
NEAT grows topology from a minimal fully-connected start, the raw 28x28 images
are average-pooled down to a small ``GRID x GRID`` grid to keep the search space
tractable; each pooled pixel becomes one input node and each of the ten digits
one output node. Training uses a smooth softmax-likelihood fitness (mean
probability of the correct class) for a dense selection signal, and the best
genome is finally reported as classification accuracy on train and a held-out
test subset.

The dataset is downloaded once (as the standard Keras ``mnist.npz``) and cached
under ``examples/mnist/data/``. Only numpy and torch are required.

Run from the repository root:
    uv run python -m examples.mnist.neat [--cpu | --gpu]

Artifacts are written to examples/mnist/artifacts/neat/:
    best_genome.json - best genome of the run (JSON)
    best_genome.pkl  - best genome of the run (pickle)
    tensorboard/     - TensorBoard event files
"""

from __future__ import annotations

from pathlib import Path

import torch

import polyneat as pn
from examples._experiment import ExperimentReport
from examples._run import run_example_main
from examples.mnist.dataset import load_mnist
from polyneat.evaluators.classification_accuracy_evaluator import (
    ClassificationAccuracyEvaluator,
)
from polyneat.evaluators.softmax_likelihood_evaluator import (
    SoftmaxLikelihoodFitnessEvaluator,
)

CONFIG_FILE_PATH = Path(__file__).parent / "neat.yaml"
_ARTIFACTS_DIR = Path(__file__).parent / "artifacts" / "neat"

GRID = 7

TRAINING_SUBSET_SIZE = 3000
TEST_SUBSET_SIZE = 2000

_SUBSET_SAMPLING_SEED = 12345


def _build_parallel_softmax_evaluator(
    features: torch.Tensor, labels: torch.Tensor
) -> pn.ParallelFitnessEvaluatorWrapper:
    """Build a thread-parallel smooth (softmax-likelihood) training evaluator."""
    return pn.ParallelFitnessEvaluatorWrapper(
        wrapped_evaluator=SoftmaxLikelihoodFitnessEvaluator(features, labels)
    )


def run_experiment(
    device: torch.device | None = None,
    random_seed: int | None = None,
    artifacts_directory: Path | None = None,
) -> ExperimentReport:
    """Run the full MNIST NEAT experiment once.

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
        f"samples, {dataset.number_of_features} inputs ({GRID}x{GRID}), "
        f"{dataset.number_of_classes} classes."
    )

    config = pn.NEATConfig.load_from_yaml_file(CONFIG_FILE_PATH)
    config.number_of_input_nodes = dataset.number_of_features
    config.number_of_output_nodes = dataset.number_of_classes

    algorithm = pn.NEATAlgorithm.from_config(config, device_for_phenotype_computation=device)

    training_evaluator = _build_parallel_softmax_evaluator(
        dataset.train_features, dataset.train_labels
    )

    callbacks: list = [pn.ConsoleStatisticsLogger()]
    if artifacts_directory is not None:
        callbacks.append(pn.BestGenomePersister(output_directory=artifacts_directory))
        callbacks.append(pn.TensorBoardLogger(log_directory=artifacts_directory / "tensorboard"))

    runner = pn.EvolutionRunner(
        algorithm=algorithm,
        fitness_evaluator=training_evaluator,
        termination_criterion=pn.CompositeTermination(
            [
                pn.TargetFitnessTermination(target_fitness=0.90),
                pn.MaxGenerationsTermination(max_generations=200),
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

    print(f"\nTermination        : {result.termination_reason}")
    print(f"Train fitness (nll): {result.best_fitness_ever_achieved:.4f}")
    return ExperimentReport(
        metric_values={"train_accuracy": train_accuracy, "test_accuracy": test_accuracy},
        number_of_generations=len(result.full_generation_history),
        runtime_seconds=result.total_runtime_seconds,
    )


def main() -> None:
    """Evolve an MNIST classifier and print training and test accuracy."""
    run_example_main(run_experiment, _ARTIFACTS_DIR)


if __name__ == "__main__":
    main()
