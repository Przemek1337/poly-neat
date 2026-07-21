"""Iris classification with NEAT-DBM - difference-based mutation demo.

Demonstrates NEAT-DBM (Stanovov, Akhmedova & Semenkin, 2021,
DOI 10.1088/1757-899X/1047/1/012075) on a multi-class classification problem,
the paper's primary experiment type (section 4, Tables 2 and 3). A single
network with one output per class is evolved and scored by a smooth
softmax-likelihood fitness - the mean probability the softmax assigns to the
correct class - which is the differentiable analogue of the paper's
categorical cross-entropy criterion. The difference-based mutation (section 3)
recombines each fresh child's connection weights from three donor genomes at
the positions their innovation numbers share, in the style of differential
evolution.

Following the paper, both training and test accuracy of the best network are
reported on a held-out split of the data.

The dataset is downloaded once from the UCI repository and cached under
examples/iris/data/. Only numpy and torch are required.

Run from the repository root:
    uv run python -m examples.iris.neatdbm [--cpu | --gpu]

Artifacts are written to examples/iris/artifacts/neatdbm/:
    best_genome.json - best genome of the run (JSON)
    best_genome.pkl  - best genome of the run (pickle)
    tensorboard/     - TensorBoard event files
"""

from __future__ import annotations

from pathlib import Path

import torch

import polyneat as pn
from examples._datasets import split_indices_into_train_and_test
from examples._example_cli import parse_device_from_cli
from examples._experiment import ExperimentReport, print_experiment_report
from examples.iris.dataset import load_iris_features_and_labels
from polyneat.evaluators.classification_accuracy_evaluator import (
    ClassificationAccuracyEvaluator,
)
from polyneat.evaluators.softmax_likelihood_evaluator import (
    SoftmaxLikelihoodFitnessEvaluator,
)

CONFIG_FILE_PATH = Path(__file__).parent / "neatdbm.yaml"
_ARTIFACTS_DIR = Path(__file__).parent / "artifacts" / "neatdbm"

_TRAIN_FRACTION = 0.66

_MAX_GENERATION_NUMBER = 100


def run_experiment(
    device: torch.device | None = None,
    random_seed: int | None = None,
    artifacts_directory: Path | None = None,
) -> ExperimentReport:
    """Run the full Iris NEAT-DBM experiment once.

    Args:
        device: Phenotype evaluation device; ``None`` uses the yaml value.
        random_seed: Evolution seed override; ``None`` uses the yaml value.
            The train/test split always uses the yaml seed, so repeats with
            different evolution seeds see the same split.
        artifacts_directory: Where to write artifacts; ``None`` writes none.

    Returns:
        Train and test accuracy of the best evolved network.
    """
    config = pn.NEATDBMConfig.load_from_yaml_file(CONFIG_FILE_PATH)
    features, labels = load_iris_features_and_labels()

    train_indices, test_indices = split_indices_into_train_and_test(
        number_of_samples=len(labels),
        train_fraction=_TRAIN_FRACTION,
        random_seed=config.random_seed,
    )

    algorithm = pn.NEATDBMAlgorithm.from_config(
        config, device_for_phenotype_computation=device
    )

    training_evaluator = SoftmaxLikelihoodFitnessEvaluator(
        input_features=features[train_indices],
        target_labels=labels[train_indices],
    )

    callbacks: list = [pn.ConsoleStatisticsLogger()]
    if artifacts_directory is not None:
        callbacks.append(pn.BestGenomePersister(output_directory=artifacts_directory))
        callbacks.append(pn.TensorBoardLogger(log_directory=artifacts_directory / "tensorboard"))

    runner = pn.EvolutionRunner(
        algorithm=algorithm,
        fitness_evaluator=training_evaluator,
        termination_criterion=pn.MaxGenerationsTermination(
            max_generations=_MAX_GENERATION_NUMBER
        ),
        callbacks=callbacks,
        random_seed=config.random_seed if random_seed is None else random_seed,
    )
    result = runner.run_evolution()

    best_phenotype = algorithm.phenotype_decoder.build_phenotype_from_genome(
        result.best_genome_ever_found
    )
    train_accuracy = ClassificationAccuracyEvaluator(
        input_features=features[train_indices], target_labels=labels[train_indices]
    ).evaluate_single_phenotype(best_phenotype)
    test_accuracy = ClassificationAccuracyEvaluator(
        input_features=features[test_indices], target_labels=labels[test_indices]
    ).evaluate_single_phenotype(best_phenotype)

    print(f"\nTermination : {result.termination_reason}")
    print(f"Train fitness (sm) : {result.best_fitness_ever_achieved:.4f}")
    return ExperimentReport(
        metric_values={"train_accuracy": train_accuracy, "test_accuracy": test_accuracy},
        number_of_generations=len(result.full_generation_history),
        runtime_seconds=result.total_runtime_seconds,
    )


def main() -> None:
    device = parse_device_from_cli()
    report = run_experiment(device=device, artifacts_directory=_ARTIFACTS_DIR)
    print_experiment_report(report)


if __name__ == "__main__":
    main()
