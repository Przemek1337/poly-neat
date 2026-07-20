"""Iris classification with NEAT-DBM — difference-based mutation demo.

Demonstrates NEAT-DBM (Stanovov, Akhmedova & Semenkin, 2021,
DOI 10.1088/1757-899X/1047/1/012075) on a multi-class classification problem,
the paper's primary experiment type (section 4, Tables 2 and 3). A single
network with one output per class is evolved and scored by a smooth
softmax-likelihood fitness — the mean probability the softmax assigns to the
correct class — which is the differentiable analogue of the paper's
categorical cross-entropy criterion. The difference-based mutation (section 3)
recombines each fresh child's connection weights from three donor genomes at
the positions their innovation numbers share, in the style of differential
evolution.

Following the paper, both training and test accuracy of the best network are
reported on a held-out split of the data.

The dataset is downloaded once from the UCI repository and cached under
examples/iris_data/ (gitignored). Only numpy and torch are required.

Run from the repository root:
    uv run python examples/iris_neatdbm.py [--cpu | --gpu]

Artifacts are written to examples/iris_neatdbm_artifacts/:
    best_genome_gen_<N>.json   — best genome per generation (JSON)
    best_genome_gen_<N>.pkl    — best genome per generation (pickle)
    tensorboard/               — TensorBoard event files
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np
import torch
from _example_cli import parse_device_from_cli

import polyneat as pn
from polyneat.evaluators.classification_accuracy_evaluator import (
    ClassificationAccuracyEvaluator,
)
from polyneat.evaluators.softmax_likelihood_evaluator import (
    SoftmaxLikelihoodFitnessEvaluator,
)

_THIS_DIR = Path(__file__).parent
_DATA_DIR = _THIS_DIR / "iris_data"
_ARTIFACTS_DIR = _THIS_DIR / "iris_neatdbm_artifacts"
_IRIS_DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
)

_CLASS_NAME_TO_INDEX = {"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2}
_TRAIN_FRACTION = 0.66  # matches the split of examples/iris_cneat.py

_MAX_GENERATION_NUMBER = 100


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
    config = pn.NEATDBMConfig.load_from_yaml_file(_THIS_DIR / "iris_neatdbm.yaml")
    features, labels = load_iris_dataset()

    split_rng = np.random.default_rng(config.random_seed)
    shuffled_indices = split_rng.permutation(len(labels))
    train_size = int(_TRAIN_FRACTION * len(labels))
    train_indices = shuffled_indices[:train_size]
    test_indices = shuffled_indices[train_size:]

    algorithm = pn.NEATDBMAlgorithm.from_config(
        config, device_for_phenotype_computation=device
    )

    training_evaluator = SoftmaxLikelihoodFitnessEvaluator(
        input_features=features[train_indices],
        target_labels=labels[train_indices],
    )

    runner = pn.EvolutionRunner(
        algorithm=algorithm,
        fitness_evaluator=training_evaluator,
        termination_criterion=pn.MaxGenerationsTermination(
            max_generations=_MAX_GENERATION_NUMBER
        ),
        callbacks=[
            pn.ConsoleStatisticsLogger(),
            pn.BestGenomePersister(output_directory=_ARTIFACTS_DIR),
            pn.TensorBoardLogger(log_directory=_ARTIFACTS_DIR / "tensorboard"),
        ],
        random_seed=config.random_seed,
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

    print(f"\nTermination        : {result.termination_reason}")
    print(f"Generations        : {len(result.full_generation_history)}")
    print(f"Runtime            : {result.total_runtime_seconds:.1f}s")
    print(f"Train fitness (sm) : {result.best_fitness_ever_achieved:.4f}")
    print(f"Train accuracy     : {train_accuracy:.3f}")
    print(f"Test accuracy      : {test_accuracy:.3f}")


if __name__ == "__main__":
    main()
