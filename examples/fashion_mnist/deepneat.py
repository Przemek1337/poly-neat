"""Fashion-MNIST classification with DeepNEAT - PolyNEAT end-to-end demo.

Evolves layer-level CNN topology over full-resolution 28x28 Fashion-MNIST
images (Miikkulainen et al. 2017's central idea: each genome node is a whole
layer with its own hyperparameters, edges carry tensors, and every genome is
trained from scratch by backpropagation during fitness evaluation - DeepNEAT
genomes carry no weights between generations, unlike EXACT's epigenetic
inheritance). Fashion-MNIST is shape-identical to MNIST (28x28 grayscale, 10
classes), so this example otherwise mirrors ``examples/mnist/deepneat.py``.
Fitness is measured on a validation slice carved out of the training rows;
the test split stays untouched by evolution and is used only for the final
report. Population and epoch budgets are far below a production-scale search
so the example finishes on one machine; the best genome is finally retrained
from scratch on train+validation and reported as classification accuracy on
the untouched test split.

The dataset is downloaded once (as four gzipped IDX archives) and cached
under ``examples/fashion_mnist/data/``.

Run from the repository root:
    uv run python -m examples.fashion_mnist.deepneat [--cpu | --gpu]

Artifacts are written to examples/fashion_mnist/artifacts/deepneat/:
    best_genome.json - best genome of the run (JSON)
    best_genome.pkl  - best genome of the run (pickle)
    tensorboard/     - TensorBoard event files
"""

from __future__ import annotations

from pathlib import Path

import torch

import polyneat as pn
from examples._datasets import split_indices_into_train_and_test
from examples._experiment import ExperimentReport
from examples._run import run_example_main

# Imported as `load_mnist` so the frozen examples smoke test, which swaps dataset
# loaders by attribute name and knows only `load_iris` / `load_mnist`, reaches this
# one too. Fashion-MNIST is shape-identical to MNIST (28x28, 10 classes).
from examples.fashion_mnist.dataset import load_fashion_mnist as load_mnist

CONFIG_FILE_PATH = Path(__file__).parent / "deepneat.yaml"
_ARTIFACTS_DIR = Path(__file__).parent / "artifacts" / "deepneat"

GRID = 28  # full resolution: DeepNEAT layers operate on the raw image

TRAINING_SUBSET_SIZE = 1500
TEST_SUBSET_SIZE = 1000

_SUBSET_SAMPLING_SEED = 12345
_VALIDATION_FRACTION = 0.2
_FINAL_TRAINING_EPOCHS = 20


def run_experiment(
    device: torch.device | None = None,
    random_seed: int | None = None,
    artifacts_directory: Path | None = None,
) -> ExperimentReport:
    """Run the full Fashion-MNIST DeepNEAT experiment once.

    Args:
        device: Phenotype evaluation and training device; ``None`` uses the
            yaml value.
        random_seed: Evolution seed override; ``None`` uses the yaml value.
            The subset draw and the train/validation split always use the
            module's sampling seed, so repeats see the same data.
        artifacts_directory: Where to write artifacts; ``None`` writes none.

    Returns:
        Validation accuracy, test accuracy, layer count and parameter count
        of the best evolved network.
    """
    dataset = load_mnist(
        random_seed=_SUBSET_SAMPLING_SEED,
        grid_side=GRID,
        max_train_samples=TRAINING_SUBSET_SIZE,
        max_test_samples=TEST_SUBSET_SIZE,
    )
    print(
        f"Fashion-MNIST loaded: {dataset.train_features.shape[0]} train / "
        f"{dataset.test_features.shape[0]} test "
        f"samples, {GRID}x{GRID} images, {dataset.number_of_classes} classes."
    )

    config = pn.DeepNEATConfig.load_from_yaml_file(CONFIG_FILE_PATH)
    config.number_of_classes = dataset.number_of_classes
    config.number_of_output_nodes = dataset.number_of_classes
    # Keep the config's declared input geometry tied to the grid this module
    # actually pools to, as examples/mnist/exact.py does, so the two cannot
    # drift apart when GRID is changed.
    config.input_image_height = GRID
    config.input_image_width = GRID

    algorithm = pn.DeepNEATAlgorithm.from_config(
        config, device_for_phenotype_computation=device
    )

    resolved_device = device or torch.device(config.device_for_phenotype_evaluation)
    resolved_random_seed = config.random_seed if random_seed is None else random_seed

    # Seed torch's global RNG here, not only inside the evaluator: the first
    # phenotypes are decoded and their weights initialised before any
    # per-phenotype seeding runs, so without this the reported architecture
    # and parameter count differ between two runs of the same seed.
    torch.manual_seed(resolved_random_seed)

    # Carve a fixed validation slice out of the training rows; fitness comes
    # from it, never from the test split, which stays untouched by evolution.
    train_indices, validation_indices = split_indices_into_train_and_test(
        number_of_samples=dataset.train_features.shape[0],
        train_fraction=1.0 - _VALIDATION_FRACTION,
        random_seed=_SUBSET_SAMPLING_SEED,
    )
    all_train_images = dataset.train_features.reshape(-1, 1, GRID, GRID)
    train_features = all_train_images[train_indices]
    train_labels = dataset.train_labels[train_indices]
    validation_features = all_train_images[validation_indices]
    validation_labels = dataset.train_labels[validation_indices]
    test_features = dataset.test_features.reshape(-1, 1, GRID, GRID)
    test_labels = dataset.test_labels

    fitness_evaluator = pn.TrainedNetworkAccuracyEvaluator(
        train_features=train_features,
        train_labels=train_labels,
        validation_features=validation_features,
        validation_labels=validation_labels,
        number_of_epochs=config.training_epochs_per_evaluation,
        learning_rate=config.training_learning_rate,
        batch_size=config.training_batch_size,
        device_for_computation=resolved_device,
        base_random_seed=resolved_random_seed,
        use_deterministic_algorithms=config.use_deterministic_training_algorithms,
    )

    callbacks: list = [pn.ConsoleStatisticsLogger()]
    if artifacts_directory is not None:
        callbacks.append(pn.BestGenomePersister(output_directory=artifacts_directory))
        callbacks.append(
            pn.TensorBoardLogger(
                log_directory=artifacts_directory / "tensorboard",
                run_label="fashion-mnist-deepneat",
            )
        )

    runner = pn.EvolutionRunner(
        algorithm=algorithm,
        fitness_evaluator=fitness_evaluator,
        termination_criterion=pn.MaxGenerationsTermination(max_generations=25),
        callbacks=callbacks,
        random_seed=resolved_random_seed,
    )
    result = runner.run_evolution()

    best_genome = result.best_genome_ever_found

    # Final report: retrain the best genome from scratch on train+validation
    # and score it on the untouched test split. Constructing a second
    # TrainedNetworkAccuracyEvaluator whose "validation" set is the test
    # split trains the network and immediately scores it on that split in
    # one call, rather than duplicating the training loop here.
    final_training_features = torch.cat([train_features, validation_features], dim=0)
    final_training_labels = torch.cat([train_labels, validation_labels], dim=0)
    final_phenotype = algorithm.phenotype_decoder.build_phenotype_from_genome(best_genome)
    final_evaluator = pn.TrainedNetworkAccuracyEvaluator(
        train_features=final_training_features,
        train_labels=final_training_labels,
        validation_features=test_features,
        validation_labels=test_labels,
        number_of_epochs=_FINAL_TRAINING_EPOCHS,
        learning_rate=config.training_learning_rate,
        batch_size=config.training_batch_size,
        device_for_computation=resolved_device,
        base_random_seed=resolved_random_seed,
        use_deterministic_algorithms=config.use_deterministic_training_algorithms,
    )
    (test_accuracy,) = final_evaluator.evaluate_batch_of_phenotypes([final_phenotype])

    # Both size metrics come off the same phenotype object, so they cannot
    # disagree about what counts as a layer; neither changes during training.
    number_of_layers = float(final_phenotype.number_of_layer_modules)
    total_parameter_count = float(final_phenotype.total_parameter_count)

    print(f"\nTermination            : {result.termination_reason}")
    print(f"Validation accuracy    : {result.best_fitness_ever_achieved:.4f}")
    return ExperimentReport(
        metric_values={
            "validation_accuracy": float(result.best_fitness_ever_achieved),
            "test_accuracy": float(test_accuracy),
            "number_of_layers": number_of_layers,
            "total_parameter_count": total_parameter_count,
        },
        number_of_generations=len(result.full_generation_history),
        runtime_seconds=result.total_runtime_seconds,
    )


def main() -> None:
    """Evolve a Fashion-MNIST DeepNEAT classifier and print all four metrics."""
    run_example_main(run_experiment, _ARTIFACTS_DIR)


if __name__ == "__main__":
    main()
