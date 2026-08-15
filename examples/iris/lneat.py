"""Iris L-NEAT - NEAT with backpropagation learning, multi-class demo.

Demonstrates L-NEAT (Chen & Alahakoon, ICIA 2006, pp. 367-371) on the Iris
dataset. Divide and conquer: each of the 3 classes gets its own evolution run
of single-output recognizer networks; every ``learning_interval_generations``
the offspring that are not Type 1 undergo a backpropagation session on a
fixed learning subset, and the trained weights are inherited (Lamarckian).
The best recognizer per class is assembled into an argmax ensemble.

The dataset is downloaded once from the UCI repository and cached under
examples/iris/data/. Only numpy and torch are required.

Run from the repository root:
    uv run python -m examples.iris.lneat [--cpu | --gpu]

Artifacts are written to examples/iris/artifacts/lneat/:
    recognizers.json - best genome and fitness of every class label
    class_<k>/       - best genome of class k's run, JSON + pickle
    tensorboard/     - per-class TensorBoard event files
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

import polyneat as pn
from examples._experiment import ExperimentReport
from examples._run import run_example_main
from examples.iris.dataset import load_iris
from polyneat.evaluators.binary_recognizer_evaluator import (
    BinaryRecognizerFitnessEvaluator,
)
from polyneat.evaluators.classification_accuracy_evaluator import (
    ClassificationAccuracyEvaluator,
)
from polyneat.utils.artifact_serialization import save_as_json

CONFIG_FILE_PATH = Path(__file__).parent / "lneat.yaml"
_ARTIFACTS_DIR = Path(__file__).parent / "artifacts" / "lneat"

_TRAIN_FRACTION = 0.66

_MAX_GENERATION_NUMBER_PER_CLASS = 49


def _draw_stratified_learning_positions(
    train_labels: torch.Tensor,
    number_of_class_labels: int,
    number_of_learning_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw the fixed learning subset with every class equally represented.

    Positions index into the training split (``train_features`` /
    ``train_labels``), not the full dataset, so the caller draws the learning
    subset straight from the split bundle.

    Args:
        train_labels: Class label index of every sample in the training split.
        number_of_class_labels: Number of classes to spread the draw across.
        number_of_learning_samples: Total subset size (the paper's ``A``).
        rng: Source of randomness for the draw.

    Returns:
        Positions into the training split, holding
        ``number_of_learning_samples // number_of_class_labels`` samples of
        each class plus a remainder drawn from the classes in order.
    """
    train_labels_array = train_labels.numpy()
    samples_per_class = number_of_learning_samples // number_of_class_labels
    remainder = number_of_learning_samples - samples_per_class * number_of_class_labels
    drawn_positions: list[int] = []
    for class_label_index in range(number_of_class_labels):
        candidate_positions = np.flatnonzero(train_labels_array == class_label_index)
        draw_size = samples_per_class + (1 if class_label_index < remainder else 0)
        drawn_positions.extend(
            rng.choice(candidate_positions, size=draw_size, replace=False).tolist()
        )
    return np.array(sorted(drawn_positions), dtype=np.int64)


def run_experiment(
    device: torch.device | None = None,
    random_seed: int | None = None,
    artifacts_directory: Path | None = None,
) -> ExperimentReport:
    """Run the full Iris L-NEAT experiment once (one evolution per class).

    Args:
        device: Phenotype evaluation device; ``None`` uses the yaml value.
        random_seed: Evolution seed override for every per-class run;
            ``None`` uses the yaml value. The train/test split and the fixed
            learning subset always use the yaml seed.
        artifacts_directory: Where to write artifacts; ``None`` writes none.

    Returns:
        Train and test accuracy of the recognizer ensemble;
        ``number_of_generations`` and ``runtime_seconds`` are totals across
        the per-class runs.
    """
    config = pn.LNEATConfig.load_from_yaml_file(CONFIG_FILE_PATH)
    evolution_seed = config.random_seed if random_seed is None else random_seed
    dataset = load_iris(
        train_fraction=_TRAIN_FRACTION, random_seed=config.random_seed
    )

    train_features = dataset.train_features
    train_labels = dataset.train_labels
    test_features = dataset.test_features
    test_labels = dataset.test_labels

    learning_positions = _draw_stratified_learning_positions(
        train_labels=train_labels,
        number_of_class_labels=config.number_of_class_labels,
        number_of_learning_samples=config.number_of_learning_samples,
        rng=np.random.default_rng(config.random_seed),
    )
    learning_features = train_features[learning_positions]
    learning_labels = train_labels[learning_positions]

    best_recognizer_genomes: list[pn.Genome] = []
    best_recognizer_fitnesses: list[float] = []
    phenotype_decoder = None
    total_number_of_generations = 0
    total_runtime_seconds = 0.0
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

        callbacks: list = [pn.ConsoleStatisticsLogger()]
        if artifacts_directory is not None:
            callbacks.append(
                pn.BestGenomePersister(
                    output_directory=artifacts_directory / f"class_{class_label_index}"
                )
            )
            callbacks.append(
                pn.TensorBoardLogger(
                    log_directory=artifacts_directory
                    / "tensorboard"
                    / f"class_{class_label_index}"
                )
            )

        runner = pn.EvolutionRunner(
            algorithm=algorithm,
            fitness_evaluator=fitness_evaluator,
            termination_criterion=pn.MaxGenerationsTermination(
                max_generations=_MAX_GENERATION_NUMBER_PER_CLASS
            ),
            callbacks=callbacks,
            random_seed=evolution_seed,
        )
        result = runner.run_evolution()
        best_recognizer_genomes.append(result.best_genome_ever_found)
        best_recognizer_fitnesses.append(result.best_fitness_ever_achieved)
        total_number_of_generations += len(result.full_generation_history)
        total_runtime_seconds += result.total_runtime_seconds
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

    if artifacts_directory is not None:
        artifacts_directory.mkdir(parents=True, exist_ok=True)
        recognizers_payload = {
            f"class_{class_label_index}": {
                "best_fitness": best_recognizer_fitnesses[class_label_index],
                "genome": best_recognizer_genomes[
                    class_label_index
                ].to_serializable_dict(),
            }
            for class_label_index in range(config.number_of_class_labels)
        }
        save_as_json(recognizers_payload, artifacts_directory / "recognizers.json")

    for class_label_index in range(config.number_of_class_labels):
        print(
            f"Class {class_label_index} best recognizer fitness: "
            f"{best_recognizer_fitnesses[class_label_index]:.4f}"
        )
    return ExperimentReport(
        metric_values={"train_accuracy": train_accuracy, "test_accuracy": test_accuracy},
        number_of_generations=total_number_of_generations,
        runtime_seconds=total_runtime_seconds,
    )


def main() -> None:
    run_example_main(run_experiment, _ARTIFACTS_DIR)


if __name__ == "__main__":
    main()
