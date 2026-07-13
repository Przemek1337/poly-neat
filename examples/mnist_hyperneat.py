"""MNIST digit classification with HyperNEAT - PolyNEAT end-to-end demo.

Where ``mnist_neat.py`` grows a network's topology directly, HyperNEAT instead
evolves a small CPPN that *paints* the connection weights of a fixed substrate
as a function of geometry. The substrate here is a state-space "sandwich"
(Stanley, D'Ambrosio & Gauci, 2009, figure 6c): a ``GRID x GRID`` input sheet
whose nodes sit at their pixel coordinates, wired directly to a row of ten
output nodes, with no hidden layer. The paper notes sandwich substrates have no
hidden nodes; with linear (``identity``) output activation the substrate is a
multinomial logistic regression whose 49x10 weight matrix is a smooth function
of pixel position -- so the CPPN can exploit the 2-D image geometry that vanilla
NEAT is blind to.

The 28x28 images are average-pooled to ``GRID x GRID`` (as in the NEAT demo) to
keep the number of queried connections small. Training uses the smooth
softmax-likelihood fitness; the best genome is reported as accuracy on train and
a held-out test subset.

The dataset is downloaded once (standard Keras ``mnist.npz``) and cached under
``examples/mnist_data/``. Only numpy and torch are required.

Run from the repository root:
    uv run python examples/mnist_hyperneat.py
"""
from __future__ import annotations

import dataclasses
import urllib.request
from pathlib import Path

import numpy as np
import torch

import polyneat as pn
from polyneat.algorithms.neat.neat_algorithm import NEATAlgorithm
from polyneat.algorithms.neat.neat_phenotype_decoder import NEATPhenotypeDecoder
from polyneat.evaluators.classification_accuracy_evaluator import (
    ClassificationAccuracyEvaluator,
)
from polyneat.evaluators.softmax_likelihood_evaluator import (
    SoftmaxLikelihoodFitnessEvaluator,
)

_THIS_DIR = Path(__file__).parent
_DATA_DIR = _THIS_DIR / "mnist_data"
_ARTIFACTS_DIR = _THIS_DIR / "mnist_hyperneat_artifacts"
_MNIST_NPZ_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"

GRID = 7
NUMBER_OF_CLASSES = 10

TRAINING_SUBSET_SIZE = 3000
TEST_SUBSET_SIZE = 2000

_SUBSET_SAMPLING_SEED = 12345


def build_mnist_hyperneat_algorithm(
    config: pn.HyperNEATConfig,
    grid_side: int,
    number_of_classes: int,
) -> NEATAlgorithm:
    """Assemble a HyperNEAT algorithm whose decoder uses a 2-D image substrate.

    Starts from the standard ``make_hyperneat_algorithm`` (CPPN genetics + a
    fully-connected 4-input/1-output CPPN initial population), then swaps in a
    phenotype decoder over a two-sheet sandwich substrate: a ``grid_side x
    grid_side`` input pixel grid and a row of ``number_of_classes`` output nodes.
    ``output_sheet_shares_input_plane=False`` lifts the output row into a
    separate coordinate band, since digit classes have no spatial relationship to
    the pixels and the CPPN needs to tell input positions from output positions.
    """
    base_algorithm = pn.make_hyperneat_algorithm(config)

    device = torch.device(config.device_for_phenotype_evaluation)
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
    ``block x block`` patch. The row-major flatten matches the substrate's
    row-major input-node id assignment.
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
    """Zero-mean, unit-variance each feature using training statistics."""
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


def main() -> None:
    """Evolve a HyperNEAT MNIST classifier and print training and test accuracy."""
    (train_features, train_labels), (test_features, test_labels) = _load_mnist_subsets()
    print(
        f"MNIST loaded: {train_features.shape[0]} train / {test_features.shape[0]} test "
        f"samples, {GRID}x{GRID} input grid, {NUMBER_OF_CLASSES} classes."
    )

    config = pn.HyperNEATConfig.load_from_yaml_file(_THIS_DIR / "mnist_hyperneat.yaml")

    algorithm = build_mnist_hyperneat_algorithm(
        config, grid_side=GRID, number_of_classes=NUMBER_OF_CLASSES
    )

    training_evaluator = pn.ParallelFitnessEvaluatorWrapper(
        wrapped_evaluator=SoftmaxLikelihoodFitnessEvaluator(train_features, train_labels)
    )

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

    print(f"\nTermination           : {result.termination_reason}")
    print(f"Generations           : {len(result.full_generation_history)}")
    print(f"Runtime               : {result.total_runtime_seconds:.1f}s")
    print(f"Train fitness (softmax): {result.best_fitness_ever_achieved:.4f}")
    print(f"Train accuracy        : {train_accuracy:.4f}")
    print(f"Test accuracy         : {test_accuracy:.4f}")


if __name__ == "__main__":
    main()
