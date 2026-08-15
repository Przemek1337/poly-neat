"""Iris C-NEAT - container-based multi-class classification demo.

Demonstrates C-NEAT (Alfaham, Van Raemdonck & Mercelis, ACAI 2024,
DOI 10.1109/ACAI63924.2024.10899662) on the Iris dataset. Standard NEAT
genetics evolve the population, but each organism is scored only on
recognizing its assigned class label, and a container preserves
the best recognizer genome per class. The final classifier is the argmax over
the three container networks.

The dataset is downloaded once from the UCI repository and cached under
examples/iris/data/. Only numpy and torch are required.

Run from the repository root:
    uv run python -m examples.iris.cneat [--cpu | --gpu]

Artifacts are written to examples/iris/artifacts/cneat/:
    container.json - the final container (best genome + fitness per class)
    tensorboard/   - TensorBoard event files
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

import polyneat as pn
from examples._experiment import ExperimentReport
from examples._run import run_example_main
from examples.iris.dataset import load_iris
from polyneat.evaluators.classification_accuracy_evaluator import (
    ClassificationAccuracyEvaluator,
)
from polyneat.evaluators.multiclass_dataset_evaluator import (
    MulticlassDatasetFitnessEvaluator,
)

CONFIG_FILE_PATH = Path(__file__).parent / "cneat.yaml"
_ARTIFACTS_DIR = Path(__file__).parent / "artifacts" / "cneat"

_TRAIN_FRACTION = 0.66  # per the paper's larger-split scenario

_MAX_GENERATION_NUMBER = 99


def run_experiment(
    device: torch.device | None = None,
    random_seed: int | None = None,
    artifacts_directory: Path | None = None,
) -> ExperimentReport:
    """Run the full Iris C-NEAT experiment once.

    Args:
        device: Phenotype evaluation device; ``None`` uses the yaml value.
        random_seed: Evolution seed override; ``None`` uses the yaml value.
            The train/test split always uses the yaml seed, so repeats with
            different evolution seeds see the same split.
        artifacts_directory: Where to write artifacts; ``None`` writes none.

    Returns:
        Train and test accuracy of the final container ensemble.
    """
    config = pn.CNEATConfig.load_from_yaml_file(CONFIG_FILE_PATH)
    dataset = load_iris(
        train_fraction=_TRAIN_FRACTION, random_seed=config.random_seed
    )

    container = pn.ClassGenomeContainer(
        number_of_class_labels=config.number_of_class_labels
    )
    fitness_evaluator = MulticlassDatasetFitnessEvaluator(
        input_features=dataset.train_features,
        class_label_indices=dataset.train_labels,
        number_of_class_labels=config.number_of_class_labels,
    )
    algorithm = pn.CNEATAlgorithm.from_config(config, device_for_phenotype_computation=device)

    callbacks: list = [
        pn.ContainerProgressLogger(container),
        pn.ContainerUpdateCallback(container, fitness_evaluator),
    ]
    if artifacts_directory is not None:
        callbacks.append(
            pn.TensorBoardLogger(
                log_directory=artifacts_directory / "tensorboard",
                run_label="iris-cneat",
            )
        )

    runner = pn.EvolutionRunner(
        algorithm=algorithm,
        fitness_evaluator=fitness_evaluator,
        termination_criterion=pn.MaxGenerationsTermination(
            max_generations=_MAX_GENERATION_NUMBER
        ),
        callbacks=callbacks,
        random_seed=config.random_seed if random_seed is None else random_seed,
    )
    result = runner.run_evolution()

    ensemble = pn.ContainerEnsemblePhenotype.from_container(
        container, algorithm.phenotype_decoder
    )
    train_accuracy = ClassificationAccuracyEvaluator(
        input_features=dataset.train_features, target_labels=dataset.train_labels
    ).evaluate_single_phenotype(ensemble)
    test_accuracy = ClassificationAccuracyEvaluator(
        input_features=dataset.test_features, target_labels=dataset.test_labels
    ).evaluate_single_phenotype(ensemble)

    if artifacts_directory is not None:
        artifacts_directory.mkdir(parents=True, exist_ok=True)
        (artifacts_directory / "container.json").write_text(
            json.dumps(container.to_serializable_dict(), indent=2), encoding="utf-8"
        )

    print(f"\nTermination : {result.termination_reason}")
    for class_label_index in range(config.number_of_class_labels):
        best_fitness = container.best_fitness_for_class(class_label_index)
        print(f"Class {class_label_index} best per-class fitness: {best_fitness:.4f}")
    return ExperimentReport(
        metric_values={"train_accuracy": train_accuracy, "test_accuracy": test_accuracy},
        number_of_generations=len(result.full_generation_history),
        runtime_seconds=result.total_runtime_seconds,
    )


def main() -> None:
    run_example_main(run_experiment, _ARTIFACTS_DIR)


if __name__ == "__main__":
    main()
