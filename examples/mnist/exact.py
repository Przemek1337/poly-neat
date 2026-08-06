"""MNIST digit classification with EXACT - PolyNEAT end-to-end demo.

Evolves the structure of small CNNs over full-resolution 28x28 MNIST images
(Desell 2017's benchmark task): every genome is trained by backpropagation
before evaluation, trained kernels are inherited epigenetically, and the
smooth softmax-likelihood fitness scores the trained networks. Fitness is
measured on the generalizability half of the held-out subset, never on the
training tensors (section V); the other half is a final test set evolution
never sees. Population and training budgets are far below the paper's
BOINC-scale run (population 50, 50 epochs, 12,500 CNNs per search) so the
example finishes on one machine; the best genome is finally reported as
classification accuracy on train, generalizability and test.

The dataset is downloaded once (as the standard Keras ``mnist.npz``) and
cached under ``examples/mnist/data/``.

Run from the repository root:
    uv run python -m examples.mnist.exact [--cpu | --gpu]

Artifacts are written to examples/mnist/artifacts/exact/:
    best_genome.json - best genome of the run (JSON)
    best_genome.pkl  - best genome of the run (pickle)
    tensorboard/     - TensorBoard event files
"""

from __future__ import annotations

from pathlib import Path

import torch

import polyneat as pn
from examples._example_cli import parse_device_from_cli
from examples._experiment import ExperimentReport, print_experiment_report
from examples.mnist.dataset import load_mnist
from polyneat.evaluators.classification_accuracy_evaluator import (
    ClassificationAccuracyEvaluator,
)
from polyneat.evaluators.softmax_likelihood_evaluator import (
    SoftmaxLikelihoodFitnessEvaluator,
)

CONFIG_FILE_PATH = Path(__file__).parent / "exact.yaml"
_ARTIFACTS_DIR = Path(__file__).parent / "artifacts" / "exact"

GRID = 28  # full resolution: EXACT evolves convolutions over the raw image

TRAINING_SUBSET_SIZE = 1500
TEST_SUBSET_SIZE = 1000

_SUBSET_SAMPLING_SEED = 12345


def run_experiment(
    device: torch.device | None = None,
    random_seed: int | None = None,
    artifacts_directory: Path | None = None,
) -> ExperimentReport:
    """Run the full MNIST EXACT experiment once.

    Args:
        device: Phenotype evaluation and training device; ``None`` uses the
            yaml value.
        random_seed: Evolution seed override; ``None`` uses the yaml value.
            The subset draw always uses the module's sampling seed, so
            repeats see the same data.
        artifacts_directory: Where to write artifacts; ``None`` writes none.

    Returns:
        Train, generalizability and test accuracy of the best evolved network.
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
        f"samples, {GRID}x{GRID} images, {dataset.number_of_classes} classes."
    )

    config = pn.EXACTConfig.load_from_yaml_file(CONFIG_FILE_PATH)
    config.number_of_output_nodes = dataset.number_of_classes
    config.input_image_height = GRID
    config.input_image_width = GRID

    algorithm = pn.EXACTAlgorithm.from_config(
        config, device_for_phenotype_computation=device
    )
    
    resolved_device = device or torch.device(config.device_for_phenotype_evaluation)
    backpropagation_trainer = pn.EXACTBackpropagationTrainer.from_config(
        config,
        training_features=dataset.train_features,
        training_labels=dataset.train_labels,
        device_for_computation=resolved_device,
    )
    algorithm.backpropagation_trainer = backpropagation_trainer

    # Section V: fitness comes from a held-out generalizability set; the
    # remaining test rows stay unseen by evolution entirely.
    number_of_generalizability_samples = dataset.test_features.shape[0] // 2
    generalizability_features = dataset.test_features[:number_of_generalizability_samples]
    generalizability_labels = dataset.test_labels[:number_of_generalizability_samples]
    final_test_features = dataset.test_features[number_of_generalizability_samples:]
    final_test_labels = dataset.test_labels[number_of_generalizability_samples:]

    generalizability_evaluator = SoftmaxLikelihoodFitnessEvaluator(
        generalizability_features, generalizability_labels
    )

    callbacks: list = [pn.ConsoleStatisticsLogger()]
    if artifacts_directory is not None:
        callbacks.append(pn.BestGenomePersister(output_directory=artifacts_directory))
        callbacks.append(
            pn.TensorBoardLogger(log_directory=artifacts_directory / "tensorboard")
        )

    runner = pn.EvolutionRunner(
        algorithm=algorithm,
        fitness_evaluator=generalizability_evaluator,
        termination_criterion=pn.CompositeTermination(
            [
                pn.TargetFitnessTermination(target_fitness=0.95),
                pn.MaxGenerationsTermination(max_generations=25),
                pn.FitnessStagnationTermination(stagnation_generations=10),
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
    generalizability_accuracy = ClassificationAccuracyEvaluator(
        generalizability_features, generalizability_labels
    ).evaluate_single_phenotype(best_phenotype)
    test_accuracy = ClassificationAccuracyEvaluator(
        final_test_features, final_test_labels
    ).evaluate_single_phenotype(best_phenotype)

    print(f"\nTermination            : {result.termination_reason}")
    print(f"Gen. fitness (softmax) : {result.best_fitness_ever_achieved:.4f}")
    return ExperimentReport(
        metric_values={
            "train_accuracy": train_accuracy,
            "generalizability_accuracy": generalizability_accuracy,
            "test_accuracy": test_accuracy,
        },
        number_of_generations=len(result.full_generation_history),
        runtime_seconds=result.total_runtime_seconds,
    )


def main() -> None:
    """Evolve an MNIST CNN classifier and print all three accuracies."""
    device = parse_device_from_cli()
    report = run_experiment(device=device, artifacts_directory=_ARTIFACTS_DIR)
    print_experiment_report(report)


if __name__ == "__main__":
    main()
