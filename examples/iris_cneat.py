"""Iris C-NEAT — container-based multi-class classification demo.

Demonstrates C-NEAT (Alfaham, Van Raemdonck & Mercelis, ACAI 2024,
DOI 10.1109/ACAI63924.2024.10899662) on the Iris dataset. Standard NEAT
genetics evolve the population, but each organism is scored only on
recognizing its assigned class label, and a container preserves
the best recognizer genome per class. The final classifier is the argmax over
the three container networks.

The dataset is downloaded once from the UCI repository and cached under
examples/iris_data/ (gitignored). Only numpy and torch are required.

Run from the repository root:
    uv run python examples/iris_cneat.py [--cpu | --gpu]

Artifacts are written to examples/iris_cneat_artifacts/:
    container.json — the final container (best genome + fitness per class)
    tensorboard/   — TensorBoard event files
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import numpy as np
import torch
from _example_cli import parse_device_from_cli

import polyneat as pn
from polyneat.evaluators.classification_accuracy_evaluator import (
    ClassificationAccuracyEvaluator,
)
from polyneat.evaluators.multiclass_dataset_evaluator import (
    MulticlassDatasetFitnessEvaluator,
)

_THIS_DIR = Path(__file__).parent
_DATA_DIR = _THIS_DIR / "iris_data"
_ARTIFACTS_DIR = _THIS_DIR / "iris_cneat_artifacts"
_IRIS_DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
)

_CLASS_NAME_TO_INDEX = {"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2}
_TRAIN_FRACTION = 0.66  # per the paper's larger-split scenario

_MAX_GENERATION_NUMBER = 99


def _download_iris_if_missing() -> Path:
    """Download the UCI iris.data file to the cache dir if absent."""
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    data_path = _DATA_DIR / "iris.data"
    if not data_path.exists():
        print(f"Downloading Iris to {data_path} ...")
        urllib.request.urlretrieve(_IRIS_DATA_URL, data_path)
    return data_path


def load_iris_dataset() -> tuple[torch.Tensor, torch.Tensor]:
    """Return (features [150, 4] normalised to [0, 1], class label indices [150])."""
    data_path = _download_iris_if_missing()
    rows = [
        line.strip().split(",")
        for line in data_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    features = np.array([[float(value) for value in row[:4]] for row in rows], dtype=np.float32)
    labels = np.array([_CLASS_NAME_TO_INDEX[row[4]] for row in rows], dtype=np.int64)
    feature_minima = features.min(axis=0)
    feature_maxima = features.max(axis=0)
    features = (features - feature_minima) / (feature_maxima - feature_minima)
    return torch.from_numpy(features), torch.from_numpy(labels)


def main() -> None:
    device = parse_device_from_cli()
    config = pn.CNEATConfig.load_from_yaml_file(_THIS_DIR / "iris_cneat.yaml")
    features, labels = load_iris_dataset()

    split_rng = np.random.default_rng(config.random_seed)
    shuffled_indices = split_rng.permutation(len(labels))
    train_size = int(_TRAIN_FRACTION * len(labels))
    train_indices = shuffled_indices[:train_size]
    test_indices = shuffled_indices[train_size:]

    container = pn.ClassGenomeContainer(
        number_of_class_labels=config.number_of_class_labels
    )
    fitness_evaluator = MulticlassDatasetFitnessEvaluator(
        input_features=features[train_indices],
        class_label_indices=labels[train_indices],
        number_of_class_labels=config.number_of_class_labels,
    )
    algorithm = pn.CNEATAlgorithm.from_config(config, device_for_phenotype_computation=device)

    runner = pn.EvolutionRunner(
        algorithm=algorithm,
        fitness_evaluator=fitness_evaluator,
        termination_criterion=pn.MaxGenerationsTermination(
            max_generations=_MAX_GENERATION_NUMBER
        ),
        callbacks=[
            pn.ContainerProgressLogger(container),
            pn.ContainerUpdateCallback(container, fitness_evaluator),
            pn.TensorBoardLogger(log_directory=_ARTIFACTS_DIR / "tensorboard"),
        ],
        random_seed=config.random_seed,
    )

    result = runner.run_evolution()

    ensemble = pn.ContainerEnsemblePhenotype.from_container(
        container, algorithm.phenotype_decoder
    )
    train_accuracy = ClassificationAccuracyEvaluator(
        input_features=features[train_indices], target_labels=labels[train_indices]
    ).evaluate_single_phenotype(ensemble)
    test_accuracy = ClassificationAccuracyEvaluator(
        input_features=features[test_indices], target_labels=labels[test_indices]
    ).evaluate_single_phenotype(ensemble)

    _ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    (_ARTIFACTS_DIR / "container.json").write_text(
        json.dumps(container.to_serializable_dict(), indent=2), encoding="utf-8"
    )

    print(f"\nTermination    : {result.termination_reason}")
    print(f"Generations    : {len(result.full_generation_history)}")
    print(f"Runtime        : {result.total_runtime_seconds:.1f}s")
    print(f"Train accuracy : {train_accuracy:.3f}")
    print(f"Test accuracy  : {test_accuracy:.3f}")
    for class_label_index in range(config.number_of_class_labels):
        best_fitness = container.best_fitness_for_class(class_label_index)
        print(f"Class {class_label_index} best per-class fitness: {best_fitness:.4f}")


if __name__ == "__main__":
    main()
