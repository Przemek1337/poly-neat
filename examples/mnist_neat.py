"""MNIST digit classification with NEAT — PolyNEAT end-to-end demo.

Evolves feed-forward networks that classify handwritten digits. Because vanilla
NEAT grows topology from a minimal fully-connected start, the raw 28x28 images
are average-pooled down to a small ``GRID x GRID`` grid to keep the search space
tractable; each pooled pixel becomes one input node and each of the ten digits
one output node. Training uses a smooth softmax-likelihood fitness (mean
probability of the correct class) for a dense selection signal, and the best
genome is finally reported as classification accuracy on train and a held-out
test subset.

The dataset is downloaded once (as the standard Keras ``mnist.npz``) and cached
under ``examples/mnist_data/``. Only numpy and torch are required.

Run from the repository root:
    uv run python examples/mnist_neat.py
"""
from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np
import torch

import polyneat as pn
from polyneat.evaluators.classification_accuracy_evaluator import (
    ClassificationAccuracyEvaluator,
)
from polyneat.evaluators.softmax_likelihood_evaluator import (
    SoftmaxLikelihoodFitnessEvaluator,
)

_THIS_DIR = Path(__file__).parent
_DATA_DIR = _THIS_DIR / "mnist_data"
_ARTIFACTS_DIR = _THIS_DIR / "mnist_artifacts"
_MNIST_NPZ_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"

GRID = 7
NUMBER_OF_CLASSES = 10

TRAINING_SUBSET_SIZE = 3000
TEST_SUBSET_SIZE = 2000

_SUBSET_SAMPLING_SEED = 12345


def _download_mnist_npz_if_missing() -> Path:
    """Download the Keras ``mnist.npz`` archive to the cache dir if absent."""
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    npz_path = _DATA_DIR / "mnist.npz"
    if not npz_path.exists():
        print(f"Downloading MNIST to {npz_path} ...")
        urllib.request.urlretrieve(_MNIST_NPZ_URL, npz_path)
    return npz_path


def _average_pool_images_to_grid(images_uint8: np.ndarray) -> np.ndarray:
    """Average-pool ``(N, 28, 28)`` uint8 images to flattened ``(N, GRID*GRID)`` floats.

    Pixels are scaled to ``[0, 1]`` and each output value is the mean of a
    ``block x block`` patch of the original image.
    """
    number_of_images = images_uint8.shape[0]
    block = 28 // GRID
    scaled = images_uint8.astype(np.float32) / 255.0
    pooled = scaled.reshape(number_of_images, GRID, block, GRID, block).mean(axis=(2, 4))
    return pooled.reshape(number_of_images, GRID * GRID)


def _sample_subset(
    features: np.ndarray,
    labels: np.ndarray,
    subset_size: int,
    rng: np.random.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw a random subset and return it as torch feature and label tensors."""
    subset_size = min(subset_size, features.shape[0])
    selected_indices = rng.choice(features.shape[0], size=subset_size, replace=False)
    feature_tensor = torch.from_numpy(features[selected_indices]).to(torch.float32)
    label_tensor = torch.from_numpy(labels[selected_indices]).to(torch.long)
    return feature_tensor, label_tensor


def _standardize_features(
    train_features: np.ndarray, test_features: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Zero-mean, unit-variance each feature using training statistics.

    Standardizing keeps pre-activation sums small so that the networks' outputs
    do not saturate, which is important for a usable softmax and for gradient in
    the fitness signal. Test features are scaled with the training mean and
    standard deviation to avoid leaking test statistics.
    """
    feature_mean = train_features.mean(axis=0, keepdims=True)
    feature_std = train_features.std(axis=0, keepdims=True) + 1e-6
    return (
        (train_features - feature_mean) / feature_std,
        (test_features - feature_mean) / feature_std,
    )


def _load_mnist_subsets() -> tuple[
    tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]
]:
    """Load MNIST and return ``(train_features, train_labels), (test_features, test_labels)``."""
    npz_path = _download_mnist_npz_if_missing()
    with np.load(npz_path) as mnist_data:
        train_images = _average_pool_images_to_grid(mnist_data["x_train"])
        train_labels = mnist_data["y_train"].astype(np.int64)
        test_images = _average_pool_images_to_grid(mnist_data["x_test"])
        test_labels = mnist_data["y_test"].astype(np.int64)

    train_images, test_images = _standardize_features(train_images, test_images)

    subset_rng = np.random.default_rng(_SUBSET_SAMPLING_SEED)
    training_subset = _sample_subset(
        train_images, train_labels, TRAINING_SUBSET_SIZE, subset_rng
    )
    test_subset = _sample_subset(test_images, test_labels, TEST_SUBSET_SIZE, subset_rng)
    return training_subset, test_subset


def _build_parallel_softmax_evaluator(
    features: torch.Tensor, labels: torch.Tensor
) -> pn.ParallelFitnessEvaluatorWrapper:
    """Build a thread-parallel smooth (softmax-likelihood) training evaluator."""
    return pn.ParallelFitnessEvaluatorWrapper(
        wrapped_evaluator=SoftmaxLikelihoodFitnessEvaluator(features, labels)
    )


def main() -> None:
    """Evolve an MNIST classifier and print training and test accuracy."""
    (train_features, train_labels), (test_features, test_labels) = _load_mnist_subsets()
    number_of_input_features = train_features.shape[1]
    print(
        f"MNIST loaded: {train_features.shape[0]} train / {test_features.shape[0]} test "
        f"samples, {number_of_input_features} inputs ({GRID}x{GRID}), "
        f"{NUMBER_OF_CLASSES} classes."
    )

    config = pn.NEATConfig.load_from_yaml_file(_THIS_DIR / "mnist_neat.yaml")
    config.number_of_input_nodes = number_of_input_features
    config.number_of_output_nodes = NUMBER_OF_CLASSES

    algorithm = pn.NEATAlgorithm.from_config(config)

    training_evaluator = _build_parallel_softmax_evaluator(train_features, train_labels)

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
        callbacks=[
            pn.ConsoleStatisticsLogger(),
            pn.BestGenomePersister(output_directory=_ARTIFACTS_DIR),
        ],
        random_seed=config.random_seed,
    )

    result = runner.run_evolution()

    best_phenotype = algorithm.phenotype_decoder.build_phenotype_from_genome(
        result.best_genome_ever_found
    )
    train_accuracy = ClassificationAccuracyEvaluator(
        train_features, train_labels
    ).evaluate_single_phenotype(best_phenotype)
    test_accuracy = ClassificationAccuracyEvaluator(
        test_features, test_labels
    ).evaluate_single_phenotype(best_phenotype)

    print(f"\nTermination        : {result.termination_reason}")
    print(f"Generations        : {len(result.full_generation_history)}")
    print(f"Runtime            : {result.total_runtime_seconds:.1f}s")
    print(f"Train fitness (nll): {result.best_fitness_ever_achieved:.4f}")
    print(f"Train accuracy     : {train_accuracy:.4f}")
    print(f"Test accuracy      : {test_accuracy:.4f}")


if __name__ == "__main__":
    main()
