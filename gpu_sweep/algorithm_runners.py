"""One runner per PolyNEAT algorithm, all reduced to the same ``CellOutcome``.

Each runner mirrors the corresponding script under ``examples/`` - same
evaluators, same assembly - with three deliberate differences: the config is
built in Python instead of YAML, the network's input and output widths come
from the dataset, and the budgets are tiny because the sweep asks only whether
a dataset runs on the GPU and whether fitness moves at all.
"""

from __future__ import annotations

import dataclasses
import time
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch

import polyneat as pn
from gpu_sweep.dataset_catalog import TabularDataset
from polyneat.core.generation_statistics import GenerationStatistics
from polyneat.evaluators.binary_recognizer_evaluator import BinaryRecognizerFitnessEvaluator
from polyneat.evaluators.classification_accuracy_evaluator import (
    ClassificationAccuracyEvaluator,
)
from polyneat.evaluators.multiclass_dataset_evaluator import MulticlassDatasetFitnessEvaluator
from polyneat.evaluators.softmax_likelihood_evaluator import SoftmaxLikelihoodFitnessEvaluator

MULTI_OUTPUT_ACTIVATION_SETTINGS: dict[str, object] = {
    "available_activation_functions": (
        "sigmoid",
        "steepened_sigmoid",
        "tanh",
        "relu",
        "identity",
    ),
    "default_activation_function_for_hidden_nodes": "tanh",
    "default_activation_function_for_output_nodes": "identity",
}
"""Logit-producing settings: the softmax fitness needs an unbounded output."""

RECOGNIZER_ACTIVATION_SETTINGS: dict[str, object] = {
    "default_activation_function_for_hidden_nodes": "steepened_sigmoid",
    "default_activation_function_for_output_nodes": "steepened_sigmoid",
}
"""C-NEAT and L-NEAT validate that outputs stay inside ``[0, 1]``."""


@dataclass(frozen=True)
class CellOutcome:
    """What one ``(dataset, algorithm)`` run produced."""

    metric_values: dict[str, float]
    generations_completed: int
    first_generation_best_fitness: float
    last_generation_best_fitness: float
    runtime_seconds: float
    phenotype_output_device: str


def summarize_generation_history(
    generation_history: Sequence[GenerationStatistics],
    *,
    runtime_seconds: float,
    metric_values: dict[str, float],
    phenotype_output_device: str,
) -> CellOutcome:
    """Reduce a run's per-generation statistics to the sweep's outcome record.

    Args:
        generation_history: Statistics in generation order.
        runtime_seconds: Wall-clock duration of the run.
        metric_values: Accuracies (or whatever the runner measured).
        phenotype_output_device: Device the best phenotype's forward pass
            produced its output on - the sweep's evidence that CUDA was used.

    Returns:
        The assembled :class:`CellOutcome`; the fitness fields are ``nan`` when
        the run produced no generations at all.
    """
    if not generation_history:
        return CellOutcome(
            metric_values=metric_values,
            generations_completed=0,
            first_generation_best_fitness=float("nan"),
            last_generation_best_fitness=float("nan"),
            runtime_seconds=runtime_seconds,
            phenotype_output_device=phenotype_output_device,
        )
    return CellOutcome(
        metric_values=metric_values,
        generations_completed=len(generation_history),
        first_generation_best_fitness=float(generation_history[0].best_fitness),
        last_generation_best_fitness=float(generation_history[-1].best_fitness),
        runtime_seconds=runtime_seconds,
        phenotype_output_device=phenotype_output_device,
    )


def phenotype_output_device_name(phenotype: object, features: torch.Tensor) -> str:
    """Run one forward pass and report which device its output landed on."""
    with torch.no_grad():
        output_tensor = phenotype.forward_pass(features[:1])
    return str(output_tensor.device)


def move_dataset_to_device(dataset: TabularDataset, device: torch.device) -> TabularDataset:
    """Copy a dataset's feature tensors onto ``device`` once, up front.

    Phenotypes call ``.to(device)`` on their input on every forward pass, so
    pre-moving the features turns a per-genome host-to-device copy into a no-op.
    Labels stay on the CPU: every evaluator indexes them there.
    """
    return dataclasses.replace(
        dataset,
        train_features=dataset.train_features.to(device),
        test_features=dataset.test_features.to(device),
    )


def _accuracy_metrics(dataset: TabularDataset, phenotype: object) -> dict[str, float]:
    """Train and test accuracy of one phenotype over the whole split."""
    return {
        "train_accuracy": ClassificationAccuracyEvaluator(
            input_features=dataset.train_features, target_labels=dataset.train_labels
        ).evaluate_single_phenotype(phenotype),
        "test_accuracy": ClassificationAccuracyEvaluator(
            input_features=dataset.test_features, target_labels=dataset.test_labels
        ).evaluate_single_phenotype(phenotype),
    }


ALGORITHM_RUNNERS: dict[str, object] = {}
"""Filled in by Tasks 5-8; maps an algorithm name to its runner function."""


def run_algorithm_on_dataset(
    algorithm_name: str,
    dataset: TabularDataset,
    *,
    device: torch.device,
    population_size: int,
    number_of_generations: int,
    random_seed: int,
) -> CellOutcome:
    """Run one algorithm against one dataset and report the outcome.

    Args:
        algorithm_name: Key into :data:`ALGORITHM_RUNNERS`.
        dataset: The split to evolve against.
        device: Device phenotypes are evaluated (and, for L-NEAT and EXACT,
            trained) on.
        population_size: Genomes per generation.
        number_of_generations: Generation budget for the run.
        random_seed: Seed for the evolution.

    Returns:
        The run's :class:`CellOutcome`.

    Raises:
        ValueError: If ``algorithm_name`` has no registered runner.
    """
    runner = ALGORITHM_RUNNERS.get(algorithm_name)
    if runner is None:
        raise ValueError(
            f"unknown algorithm {algorithm_name!r}; known: {sorted(ALGORITHM_RUNNERS)}"
        )
    dataset_on_device = move_dataset_to_device(dataset, device)
    return runner(
        dataset_on_device,
        device=device,
        population_size=population_size,
        number_of_generations=number_of_generations,
        random_seed=random_seed,
    )


def _build_multi_output_config(
    config_class: type,
    dataset: TabularDataset,
    *,
    population_size: int,
    random_seed: int,
    device: torch.device,
) -> object:
    """Config for the algorithms that evolve one network with one output per class."""
    return config_class(
        population_size=population_size,
        number_of_input_nodes=dataset.number_of_features,
        number_of_output_nodes=dataset.number_of_classes,
        random_seed=random_seed,
        device_for_phenotype_evaluation=str(device),
        **MULTI_OUTPUT_ACTIVATION_SETTINGS,
    )


def _run_softmax_family(
    algorithm: object,
    dataset: TabularDataset,
    *,
    number_of_generations: int,
    random_seed: int,
) -> CellOutcome:
    """Evolve ``algorithm`` under the softmax-likelihood fitness and score it.

    The shape shared by NEAT, FS-NEAT and NEAT-DBM: one network with one output
    per class, a smooth fitness so five generations can visibly move, and train
    and test accuracy of the best genome as the reported metrics.
    """
    fitness_evaluator = SoftmaxLikelihoodFitnessEvaluator(
        input_features=dataset.train_features, target_labels=dataset.train_labels
    )
    runner = pn.EvolutionRunner(
        algorithm=algorithm,
        fitness_evaluator=fitness_evaluator,
        termination_criterion=pn.MaxGenerationsTermination(
            max_generations=number_of_generations
        ),
        callbacks=[pn.ConsoleStatisticsLogger()],
        random_seed=random_seed,
    )
    start_time = time.perf_counter()
    result = runner.run_evolution()
    runtime_seconds = time.perf_counter() - start_time

    best_phenotype = algorithm.phenotype_decoder.build_phenotype_from_genome(
        result.best_genome_ever_found
    )
    return summarize_generation_history(
        result.full_generation_history,
        runtime_seconds=runtime_seconds,
        metric_values=_accuracy_metrics(dataset, best_phenotype),
        phenotype_output_device=phenotype_output_device_name(
            best_phenotype, dataset.train_features
        ),
    )


def run_neat(
    dataset: TabularDataset,
    *,
    device: torch.device,
    population_size: int,
    number_of_generations: int,
    random_seed: int,
) -> CellOutcome:
    """Vanilla NEAT: fully connected initial population, softmax fitness."""
    config = _build_multi_output_config(
        pn.NEATConfig,
        dataset,
        population_size=population_size,
        random_seed=random_seed,
        device=device,
    )
    algorithm = pn.NEATAlgorithm.from_config(config, device_for_phenotype_computation=device)
    return _run_softmax_family(
        algorithm,
        dataset,
        number_of_generations=number_of_generations,
        random_seed=random_seed,
    )


def run_fsneat(
    dataset: TabularDataset,
    *,
    device: torch.device,
    population_size: int,
    number_of_generations: int,
    random_seed: int,
) -> CellOutcome:
    """FS-NEAT: every genome starts from a single input->output connection.

    The cheapest algorithm here on the wide microarray sets, because the initial
    genomes carry one connection instead of ``n_features * n_classes`` of them.
    """
    config = _build_multi_output_config(
        pn.NEATConfig,
        dataset,
        population_size=population_size,
        random_seed=random_seed,
        device=device,
    )
    algorithm = pn.FSNEATAlgorithm.from_config(config, device_for_phenotype_computation=device)
    return _run_softmax_family(
        algorithm,
        dataset,
        number_of_generations=number_of_generations,
        random_seed=random_seed,
    )


def run_neatdbm(
    dataset: TabularDataset,
    *,
    device: torch.device,
    population_size: int,
    number_of_generations: int,
    random_seed: int,
) -> CellOutcome:
    """NEAT-DBM: NEAT plus difference-based weight mutation from three donors."""
    config = _build_multi_output_config(
        pn.NEATDBMConfig,
        dataset,
        population_size=population_size,
        random_seed=random_seed,
        device=device,
    )
    algorithm = pn.NEATDBMAlgorithm.from_config(
        config, device_for_phenotype_computation=device
    )
    return _run_softmax_family(
        algorithm,
        dataset,
        number_of_generations=number_of_generations,
        random_seed=random_seed,
    )


ALGORITHM_RUNNERS["neat"] = run_neat
ALGORITHM_RUNNERS["fsneat"] = run_fsneat
ALGORITHM_RUNNERS["neatdbm"] = run_neatdbm


def _build_recognizer_config(
    config_class: type,
    dataset: TabularDataset,
    *,
    population_size: int,
    random_seed: int,
    device: torch.device,
    **extra_settings: object,
) -> object:
    """Config for the algorithms that evolve one single-output recognizer per class."""
    return config_class(
        population_size=population_size,
        number_of_input_nodes=dataset.number_of_features,
        number_of_output_nodes=1,
        number_of_class_labels=dataset.number_of_classes,
        random_seed=random_seed,
        device_for_phenotype_evaluation=str(device),
        **RECOGNIZER_ACTIVATION_SETTINGS,
        **extra_settings,
    )


def run_cneat(
    dataset: TabularDataset,
    *,
    device: torch.device,
    population_size: int,
    number_of_generations: int,
    random_seed: int,
) -> CellOutcome:
    """C-NEAT: one population, a container holding the best recognizer per class.

    Mirrors ``examples/iris/cneat.py``: every organism is scored only on its own
    class, the container update callback keeps the best genome per class, and
    the reported classifier is the argmax ensemble over the container.
    """
    config = _build_recognizer_config(
        pn.CNEATConfig,
        dataset,
        population_size=population_size,
        random_seed=random_seed,
        device=device,
    )
    container = pn.ClassGenomeContainer(number_of_class_labels=dataset.number_of_classes)
    fitness_evaluator = MulticlassDatasetFitnessEvaluator(
        input_features=dataset.train_features,
        class_label_indices=dataset.train_labels,
        number_of_class_labels=dataset.number_of_classes,
    )
    algorithm = pn.CNEATAlgorithm.from_config(config, device_for_phenotype_computation=device)

    runner = pn.EvolutionRunner(
        algorithm=algorithm,
        fitness_evaluator=fitness_evaluator,
        termination_criterion=pn.MaxGenerationsTermination(
            max_generations=number_of_generations
        ),
        callbacks=[
            pn.ContainerProgressLogger(container),
            pn.ContainerUpdateCallback(container, fitness_evaluator),
        ],
        random_seed=random_seed,
    )
    start_time = time.perf_counter()
    result = runner.run_evolution()
    runtime_seconds = time.perf_counter() - start_time

    ensemble = pn.ContainerEnsemblePhenotype.from_container(
        container, algorithm.phenotype_decoder
    )
    return summarize_generation_history(
        result.full_generation_history,
        runtime_seconds=runtime_seconds,
        metric_values=_accuracy_metrics(dataset, ensemble),
        phenotype_output_device=phenotype_output_device_name(
            ensemble, dataset.train_features
        ),
    )


def draw_stratified_learning_positions(
    train_labels: torch.Tensor,
    *,
    number_of_class_labels: int,
    number_of_learning_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw L-NEAT's fixed backpropagation subset with every class represented.

    Unlike ``examples/iris/lneat.py``, the per-class draw is clamped to what the
    class actually has: several of the paper's datasets are small and skewed
    enough that an even split would ask for more rows than exist.

    Args:
        train_labels: Class index of every row in the training split.
        number_of_class_labels: Classes to spread the draw across.
        number_of_learning_samples: Requested subset size.
        rng: Source of randomness for the draw.

    Returns:
        Sorted positions into the training split.
    """
    train_labels_array = train_labels.cpu().numpy()
    samples_per_class = max(1, number_of_learning_samples // number_of_class_labels)
    drawn_positions: list[int] = []
    for class_label_index in range(number_of_class_labels):
        candidate_positions = np.flatnonzero(train_labels_array == class_label_index)
        draw_size = min(samples_per_class, len(candidate_positions))
        if draw_size == 0:
            continue
        drawn_positions.extend(
            rng.choice(candidate_positions, size=draw_size, replace=False).tolist()
        )
    return np.array(sorted(drawn_positions), dtype=np.int64)


def run_lneat(
    dataset: TabularDataset,
    *,
    device: torch.device,
    population_size: int,
    number_of_generations: int,
    random_seed: int,
) -> CellOutcome:
    """L-NEAT: one evolution per class, with Lamarckian backpropagation sessions.

    Mirrors ``examples/iris/lneat.py``. Cost scales with the class count, so the
    reported runtime and generation count are totals across the per-class runs
    and the fitness fields are averaged over them.
    """
    config = _build_recognizer_config(
        pn.LNEATConfig,
        dataset,
        population_size=population_size,
        random_seed=random_seed,
        device=device,
        learning_interval_generations=2,
        number_of_learning_samples=4 * dataset.number_of_classes,
    )
    learning_positions = draw_stratified_learning_positions(
        dataset.train_labels,
        number_of_class_labels=dataset.number_of_classes,
        number_of_learning_samples=config.number_of_learning_samples,
        rng=np.random.default_rng(random_seed),
    )
    learning_features = dataset.train_features[learning_positions]
    learning_labels = dataset.train_labels[learning_positions]

    best_recognizer_genomes: list[pn.Genome] = []
    phenotype_decoder = None
    total_generations = 0
    total_runtime_seconds = 0.0
    first_generation_best_fitnesses: list[float] = []
    last_generation_best_fitnesses: list[float] = []

    for class_label_index in range(dataset.number_of_classes):
        print(f"=== L-NEAT recognizer for class {class_label_index} ===")
        fitness_evaluator = BinaryRecognizerFitnessEvaluator(
            input_features=dataset.train_features,
            class_label_indices=dataset.train_labels,
            target_class_label_index=class_label_index,
            classification_threshold=config.classification_threshold,
        )
        backpropagation_trainer = pn.BackpropagationWeightTrainer(
            classification_features=dataset.train_features,
            classification_binary_targets=(
                dataset.train_labels == class_label_index
            ).float().unsqueeze(1),
            learning_sample_features=learning_features,
            learning_sample_binary_targets=(
                learning_labels == class_label_index
            ).float().unsqueeze(1),
            learning_rate=config.backpropagation_learning_rate,
            number_of_iterations=config.backpropagation_iterations_per_session,
            training_indicator=config.training_indicator,
            classification_threshold=config.classification_threshold,
            device_for_computation=device,
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
                max_generations=number_of_generations
            ),
            callbacks=[pn.ConsoleStatisticsLogger()],
            random_seed=random_seed,
        )
        start_time = time.perf_counter()
        result = runner.run_evolution()
        total_runtime_seconds += time.perf_counter() - start_time
        total_generations += len(result.full_generation_history)
        if result.full_generation_history:
            first_generation_best_fitnesses.append(
                float(result.full_generation_history[0].best_fitness)
            )
            last_generation_best_fitnesses.append(
                float(result.full_generation_history[-1].best_fitness)
            )
        best_recognizer_genomes.append(result.best_genome_ever_found)

    assert phenotype_decoder is not None
    ensemble = pn.RecognizerEnsemblePhenotype.from_genomes(
        class_recognizer_genomes=best_recognizer_genomes,
        phenotype_decoder=phenotype_decoder,
    )
    return CellOutcome(
        metric_values=_accuracy_metrics(dataset, ensemble),
        generations_completed=total_generations,
        first_generation_best_fitness=(
            float(np.mean(first_generation_best_fitnesses))
            if first_generation_best_fitnesses
            else float("nan")
        ),
        last_generation_best_fitness=(
            float(np.mean(last_generation_best_fitnesses))
            if last_generation_best_fitnesses
            else float("nan")
        ),
        runtime_seconds=total_runtime_seconds,
        phenotype_output_device=phenotype_output_device_name(
            ensemble, dataset.train_features
        ),
    )


ALGORITHM_RUNNERS["cneat"] = run_cneat
ALGORITHM_RUNNERS["lneat"] = run_lneat


def run_hyperneat(
    dataset: TabularDataset,
    *,
    device: torch.device,
    population_size: int,
    number_of_generations: int,
    random_seed: int,
) -> CellOutcome:
    """HyperNEAT: a CPPN paints the weights of a 1-D input -> output substrate.

    The CPPN's own input and output widths are fixed (four coordinates, one
    weight), so only the substrate sizes come from the dataset. Identity
    substrate activation keeps the outputs logits for the softmax fitness.
    """
    config = pn.HyperNEATConfig(
        population_size=population_size,
        random_seed=random_seed,
        device_for_phenotype_evaluation=str(device),
        substrate_input_layer_size=dataset.number_of_features,
        substrate_hidden_layer_sizes=(),
        substrate_output_layer_size=dataset.number_of_classes,
        substrate_node_activation_function="identity",
        weight_expression_threshold=0.1,
        max_substrate_connection_weight_magnitude=3.0,
        available_activation_functions=(
            "sigmoid",
            "gaussian",
            "sine",
            "absolute_value",
            "identity",
        ),
        default_activation_function_for_hidden_nodes="sigmoid",
        default_activation_function_for_output_nodes="identity",
    )
    algorithm = pn.HyperNEATAlgorithm.from_config(
        config, device_for_phenotype_computation=device
    )
    return _run_softmax_family(
        algorithm,
        dataset,
        number_of_generations=number_of_generations,
        random_seed=random_seed,
    )


ALGORITHM_RUNNERS["hyperneat"] = run_hyperneat


def run_exact(
    dataset: TabularDataset,
    *,
    device: torch.device,
    population_size: int,
    number_of_generations: int,
    random_seed: int,
) -> CellOutcome:
    """EXACT: evolved CNN topologies, each trained by backpropagation.

    Mirrors ``examples/mnist/exact.py``, with one substantive difference: a
    tabular row is presented as a ``1 x number_of_features`` image, so EXACT's
    convolutions run along the feature axis. Fitness comes from a held-out
    generalizability half of the test split (paper section V); the other half
    stays unseen by evolution.
    """
    config = pn.EXACTConfig(
        population_size=population_size,
        number_of_input_nodes=1,
        number_of_output_nodes=dataset.number_of_classes,
        random_seed=random_seed,
        device_for_phenotype_evaluation=str(device),
        input_image_height=1,
        input_image_width=dataset.number_of_features,
        number_of_training_epochs_per_genome=2,
        training_batch_size=32,
        use_simplex_hyperparameter_optimization=False,
    )
    algorithm = pn.EXACTAlgorithm.from_config(config, device_for_phenotype_computation=device)
    algorithm.backpropagation_trainer = pn.EXACTBackpropagationTrainer.from_config(
        config,
        training_features=dataset.train_features,
        training_labels=dataset.train_labels,
        device_for_computation=device,
    )

    number_of_generalizability_samples = max(1, dataset.test_features.shape[0] // 2)
    generalizability_features = dataset.test_features[:number_of_generalizability_samples]
    generalizability_labels = dataset.test_labels[:number_of_generalizability_samples]
    final_test_features = dataset.test_features[number_of_generalizability_samples:]
    final_test_labels = dataset.test_labels[number_of_generalizability_samples:]

    runner = pn.EvolutionRunner(
        algorithm=algorithm,
        fitness_evaluator=SoftmaxLikelihoodFitnessEvaluator(
            input_features=generalizability_features, target_labels=generalizability_labels
        ),
        termination_criterion=pn.MaxGenerationsTermination(
            max_generations=number_of_generations
        ),
        callbacks=[pn.ConsoleStatisticsLogger()],
        random_seed=random_seed,
    )
    start_time = time.perf_counter()
    result = runner.run_evolution()
    runtime_seconds = time.perf_counter() - start_time

    best_phenotype = algorithm.phenotype_decoder.build_phenotype_from_genome(
        result.best_genome_ever_found
    )
    metric_values = {
        "train_accuracy": ClassificationAccuracyEvaluator(
            input_features=dataset.train_features, target_labels=dataset.train_labels
        ).evaluate_single_phenotype(best_phenotype),
        "generalizability_accuracy": ClassificationAccuracyEvaluator(
            input_features=generalizability_features, target_labels=generalizability_labels
        ).evaluate_single_phenotype(best_phenotype),
    }
    metric_values["test_accuracy"] = (
        ClassificationAccuracyEvaluator(
            input_features=final_test_features, target_labels=final_test_labels
        ).evaluate_single_phenotype(best_phenotype)
        if final_test_features.shape[0] > 0
        else metric_values["generalizability_accuracy"]
    )
    return summarize_generation_history(
        result.full_generation_history,
        runtime_seconds=runtime_seconds,
        metric_values=metric_values,
        phenotype_output_device=phenotype_output_device_name(
            best_phenotype, dataset.train_features
        ),
    )


ALGORITHM_RUNNERS["exact"] = run_exact
