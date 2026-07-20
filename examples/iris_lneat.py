"""Iris L-NEAT — NEAT with backpropagation learning, multi-class demo.

Demonstrates L-NEAT (Chen & Alahakoon, ICIA 2006, pp. 367-371) on the Iris
dataset. Divide and conquer: each of the 3 classes gets its own evolution run
of single-output recognizer networks; every ``learning_interval_generations``
the offspring that are not Type 1 undergo a backpropagation session on a
fixed learning subset, and the trained weights are inherited (Lamarckian).
The best recognizer per class is assembled into an argmax ensemble.

The dataset is downloaded once from the UCI repository and cached under
examples/iris_data/ (gitignored). Only numpy and torch are required.

Run from the repository root:
    uv run python examples/iris_lneat.py [--cpu | --gpu]

Artifacts are written to examples/iris_lneat_artifacts/:
    recognizers.json — best genome and fitness of every class label
    class_<k>/       — best genome of class k's run, JSON + pickle
    tensorboard/     — per-class TensorBoard event files
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np
import torch
from _example_cli import parse_device_from_cli

import polyneat as pn
from polyneat.evaluators.binary_recognizer_evaluator import (
    BinaryRecognizerFitnessEvaluator,
)
from polyneat.evaluators.classification_accuracy_evaluator import (
    ClassificationAccuracyEvaluator,
)
from polyneat.utils.artifact_serialization import save_as_json

_THIS_DIR = Path(__file__).parent
_DATA_DIR = _THIS_DIR / "iris_data"
_ARTIFACTS_DIR = _THIS_DIR / "iris_lneat_artifacts"
_IRIS_DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
)

_CLASS_NAME_TO_INDEX = {"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2}
_TRAIN_FRACTION = 0.66

_MAX_GENERATION_NUMBER_PER_CLASS = 49


def _download_iris_if_missing() -> Path:
    """Download the UCI iris.data file to the cache dir if absent."""
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    data_path = _DATA_DIR / "iris.data"
    if not data_path.exists():
        print(f"Downloading Iris to {data_path} ...")
        urllib.request.urlretrieve(_IRIS_DATA_URL, data_path)
    return data_path


def load_iris_dataset() -> tuple[np.ndarray, np.ndarray]:
    """Return (features [150, 4] normalized to [0, 1], class label indices [150])."""
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
    return features, labels


def _draw_stratified_learning_indices(
    train_indices: np.ndarray,
    labels: np.ndarray,
    number_of_class_labels: int,
    number_of_learning_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw the fixed learning subset with every class equally represented.

    Args:
        train_indices: Indices of the training split to draw from.
        labels: Class label index of every sample in the dataset.
        number_of_class_labels: Number of classes to spread the draw across.
        number_of_learning_samples: Total subset size (the paper's ``A``).
        rng: Source of randomness for the draw.

    Returns:
        Indices of the learning subset, holding
        ``number_of_learning_samples // number_of_class_labels`` samples of
        each class plus a remainder drawn from the classes in order.
    """
    samples_per_class = number_of_learning_samples // number_of_class_labels
    remainder = number_of_learning_samples - samples_per_class * number_of_class_labels
    drawn_indices: list[int] = []
    for class_label_index in range(number_of_class_labels):
        candidate_indices = train_indices[labels[train_indices] == class_label_index]
        draw_size = samples_per_class + (1 if class_label_index < remainder else 0)
        drawn_indices.extend(
            rng.choice(candidate_indices, size=draw_size, replace=False).tolist()
        )
    return np.array(sorted(drawn_indices), dtype=np.int64)


def main() -> None:
    device = parse_device_from_cli()
    config = pn.LNEATConfig.load_from_yaml_file(_THIS_DIR / "iris_lneat.yaml")
    features, labels = load_iris_dataset()

    split_rng = np.random.default_rng(config.random_seed)
    shuffled_indices = split_rng.permutation(len(labels))
    train_size = int(_TRAIN_FRACTION * len(labels))
    train_indices = shuffled_indices[:train_size]
    test_indices = shuffled_indices[train_size:]

    learning_indices = _draw_stratified_learning_indices(
        train_indices=train_indices,
        labels=labels,
        number_of_class_labels=config.number_of_class_labels,
        number_of_learning_samples=config.number_of_learning_samples,
        rng=split_rng,
    )

    train_features = torch.from_numpy(features[train_indices])
    train_labels = torch.from_numpy(labels[train_indices])
    test_features = torch.from_numpy(features[test_indices])
    test_labels = torch.from_numpy(labels[test_indices])
    learning_features = torch.from_numpy(features[learning_indices])
    learning_labels = torch.from_numpy(labels[learning_indices])

    best_recognizer_genomes: list[pn.Genome] = []
    best_recognizer_fitnesses: list[float] = []
    phenotype_decoder = None
    for class_label_index in range(config.number_of_class_labels):
        print(f"\n=== Evolving recognizer for class {class_label_index} ===")
        fitness_evaluator = BinaryRecognizerFitnessEvaluator(
            input_features=train_features,
            class_label_indices=train_labels,
            target_class_label_index=class_label_index,
            classification_threshold=config.classification_threshold,
        )
        backpropagation_trainer = pn.BackpropagationWeightTrainer(
            classification_features=train_features,
            classification_binary_targets=(
                train_labels == class_label_index
            ).float().unsqueeze(1),
            learning_sample_features=learning_features,
            learning_sample_binary_targets=(
                learning_labels == class_label_index
            ).float().unsqueeze(1),
            learning_rate=config.backpropagation_learning_rate,
            number_of_iterations=config.backpropagation_iterations_per_session,
            training_indicator=config.training_indicator,
            classification_threshold=config.classification_threshold,
            device_for_computation=device or torch.device(config.device_for_phenotype_evaluation),
        )
        algorithm = pn.LNEATAlgorithm.from_config(
            config, device_for_phenotype_computation=device
        )
        algorithm.backpropagation_trainer = backpropagation_trainer
        phenotype_decoder = algorithm.phenotype_decoder

        runner = pn.EvolutionRunner(
            algorithm=algorithm,
            fitness_evaluator=fitness_evaluator,
            termination_criterion=pn.MaxGenerationsTermination(
                max_generations=_MAX_GENERATION_NUMBER_PER_CLASS
            ),
            callbacks=[
                pn.ConsoleStatisticsLogger(),
                pn.BestGenomePersister(
                    output_directory=_ARTIFACTS_DIR / f"class_{class_label_index}"
                ),
                pn.TensorBoardLogger(
                    log_directory=_ARTIFACTS_DIR / "tensorboard" / f"class_{class_label_index}"
                ),
            ],
            random_seed=config.random_seed,
        )
        result = runner.run_evolution()
        best_recognizer_genomes.append(result.best_genome_ever_found)
        best_recognizer_fitnesses.append(result.best_fitness_ever_achieved)
        print(
            f"Class {class_label_index}: best fitness "
            f"{result.best_fitness_ever_achieved:.4f} "
            f"({result.termination_reason}, {result.total_runtime_seconds:.1f}s)"
        )

    assert phenotype_decoder is not None
    ensemble = pn.RecognizerEnsemblePhenotype.from_genomes(
        class_recognizer_genomes=best_recognizer_genomes,
        phenotype_decoder=phenotype_decoder,
    )
    train_accuracy = ClassificationAccuracyEvaluator(
        input_features=train_features, target_labels=train_labels
    ).evaluate_single_phenotype(ensemble)
    test_accuracy = ClassificationAccuracyEvaluator(
        input_features=test_features, target_labels=test_labels
    ).evaluate_single_phenotype(ensemble)

    _ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    recognizers_payload = {
        f"class_{class_label_index}": {
            "best_fitness": best_recognizer_fitnesses[class_label_index],
            "genome": best_recognizer_genomes[class_label_index].to_serializable_dict(),
        }
        for class_label_index in range(config.number_of_class_labels)
    }
    save_as_json(recognizers_payload, _ARTIFACTS_DIR / "recognizers.json")

    print(f"\nTrain accuracy : {train_accuracy:.3f}")
    print(f"Test accuracy  : {test_accuracy:.3f}")
    for class_label_index in range(config.number_of_class_labels):
        print(
            f"Class {class_label_index} best recognizer fitness: "
            f"{best_recognizer_fitnesses[class_label_index]:.4f}"
        )


if __name__ == "__main__":
    main()
