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
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import torch

import polyneat as pn
from gpu_sweep.dataset_catalog import TabularDataset
from gpu_sweep.fitness import OneHotMeanSquaredErrorFitnessEvaluator
from gpu_sweep.metrics import evaluate_phenotype_metrics
from gpu_sweep.output_sums import count_tied_maxima, evaluate_networks_with_output_sums
from polyneat.core.generation_statistics import GenerationStatistics
from polyneat.evaluators.binary_recognizer_evaluator import BinaryRecognizerFitnessEvaluator
from polyneat.evaluators.multiclass_dataset_evaluator import MulticlassDatasetFitnessEvaluator

RECOGNIZER_ACTIVATION_SETTINGS: dict[str, object] = {
    "default_activation_function_for_hidden_nodes": "steepened_sigmoid",
    "default_activation_function_for_output_nodes": "steepened_sigmoid",
}
"""C-NEAT and L-NEAT validate that outputs stay inside ``[0, 1]``."""


@dataclass(frozen=True)
class CellOutcome:
    """What one run of one ``(dataset, algorithm)`` pair produced.

    Attributes:
        metric_values: Scalar scores - ``train_accuracy``, ``test_accuracy``,
            ``train_macro_f1``, ``test_macro_f1``.
        per_class_f1_scores: Test-split F1 of every class, in class order.
        generation_best_fitnesses: Best fitness of each generation, in order.
            This is the convergence curve the analysis pass plots.
        generations_completed: Length of that curve.
        first_generation_best_fitness: First entry, or ``nan`` for an empty run.
        last_generation_best_fitness: Last entry, or ``nan`` for an empty run.
        runtime_seconds: Wall-clock duration of the evolution.
        phenotype_output_device: Device an actual forward-pass output landed on.
        named_genomes: Genomes worth drawing, keyed by a filename-safe label -
            ``{"best": genome}`` for the single-network algorithms, one entry
            per class for the C-NEAT and L-NEAT ensembles.
        structure_notes: Free-form facts for the topology description, such as
            HyperNEAT's substrate layer sizes.
        generation_species_counts: Species count of each generation, in order
            (``None`` entries where the algorithm reports none). For L-NEAT a
            list per class, since every class has its own evolution.
        per_patient_outputs: Per-split arrays for every patient - positions,
            labels, output sums before the activation, outputs, predictions and
            a tie flag - keyed ``"<split>_<field>"``. See
            :func:`record_per_patient_outputs`.
        tie_counts: ``{"train": n, "test": n}``, patients whose largest output
            occurs in more than one column.
        output_sums_match_forward_pass: Whether the activations recomputed
            alongside the sums equal the library phenotype's outputs exactly on
            both splits - the check that the stored sums belong to the scored
            network.
    """

    metric_values: dict[str, float]
    per_class_f1_scores: list[float]
    generation_best_fitnesses: list[float]
    generations_completed: int
    first_generation_best_fitness: float
    last_generation_best_fitness: float
    runtime_seconds: float
    phenotype_output_device: str
    named_genomes: dict[str, object]
    structure_notes: dict[str, object]
    generation_species_counts: list = dataclasses.field(default_factory=list)
    per_patient_outputs: dict[str, np.ndarray] = dataclasses.field(default_factory=dict)
    tie_counts: dict[str, int] = dataclasses.field(default_factory=dict)
    output_sums_match_forward_pass: bool | None = None


@dataclass(frozen=True)
class PredictionRecord:
    """Per-patient outputs of the scored classifier, plus their summary checks."""

    per_patient_outputs: dict[str, np.ndarray]
    tie_counts: dict[str, int]
    output_sums_match_forward_pass: bool

    def as_outcome_fields(self) -> dict[str, object]:
        """The matching :class:`CellOutcome` keyword arguments."""
        return {
            "per_patient_outputs": self.per_patient_outputs,
            "tie_counts": self.tie_counts,
            "output_sums_match_forward_pass": self.output_sums_match_forward_pass,
        }


class LastGenerationBestTracker(pn.BaseEvolutionCallback):
    """Remember the highest-fitness genome of the most recently evaluated generation.

    The run's reported network is the best genome of the whole run, which may
    have appeared many generations before the end. This keeps, in memory only,
    the best genome of the last evaluated population, so it can be stored next
    to the reported one when the two differ. Ties are broken towards the first
    genome, as the runner itself does.
    """

    def __init__(self) -> None:
        self.genome: object | None = None
        self.fitness: float | None = None
        self.generation_number: int | None = None

    def on_population_evaluated(self, context, population, fitnesses) -> None:  # noqa: ANN001
        best_fitness = max(fitnesses)
        self.genome = population.genomes[fitnesses.index(best_fitness)]
        self.fitness = float(best_fitness)
        self.generation_number = int(population.generation_number)


def same_network(first_genome: object, second_genome: object) -> bool:
    """Whether two network genomes have the same nodes and the same connections."""

    def signature(genome: object) -> tuple:
        return (
            tuple(
                (node.node_id, node.node_type, getattr(node, "activation_function_name", None))
                for node in genome.node_genes
            ),
            tuple(
                (
                    connection.source_node_id,
                    connection.target_node_id,
                    float(connection.weight),
                    bool(connection.is_enabled),
                )
                for connection in genome.connection_genes
            ),
        )

    return signature(first_genome) == signature(second_genome)


def add_last_generation_network(
    named_genomes: dict[str, object],
    structure_notes: dict[str, object],
    *,
    label: str,
    best_network_genome: object,
    tracker: LastGenerationBestTracker,
    to_network: Callable[[object], object] | None = None,
) -> None:
    """Add the last generation's best network to ``named_genomes`` if it differs.

    ``structure_notes`` records the last generation's number and fitness and
    whether its best network equals the reported one, so a missing
    ``<label>_last_generation`` file is explained rather than ambiguous.
    """
    if tracker.genome is None:
        return
    last_network_genome = to_network(tracker.genome) if to_network is not None else tracker.genome
    is_same = same_network(last_network_genome, best_network_genome)
    structure_notes[f"{label}_last_generation"] = {
        "generation_number": tracker.generation_number,
        "best_fitness": tracker.fitness,
        "same_as_reported_network": is_same,
    }
    if not is_same:
        named_genomes[f"{label}_last_generation"] = last_network_genome


def species_counts_of(generation_history: Sequence[GenerationStatistics]) -> list[int | None]:
    """Species count of every generation, in order."""
    return [statistics.number_of_species for statistics in generation_history]


def build_cell_outcome(
    generation_history: Sequence[GenerationStatistics],
    *,
    runtime_seconds: float,
    metric_values: dict[str, float],
    per_class_f1_scores: list[float],
    phenotype_output_device: str,
    named_genomes: dict[str, object],
    structure_notes: dict[str, object],
    prediction_record: PredictionRecord | None = None,
) -> CellOutcome:
    """Reduce a run's per-generation statistics to one :class:`CellOutcome`.

    Args:
        generation_history: Statistics in generation order.
        runtime_seconds: Wall-clock duration of the run.
        metric_values: Accuracy and macro-F1 on both splits.
        per_class_f1_scores: Test-split F1 of every class.
        phenotype_output_device: Device the best phenotype's forward pass
            produced its output on - the sweep's evidence that CUDA was used.
        named_genomes: Genomes to draw, keyed by a filename-safe label.
        structure_notes: Extra structural facts for the topology description.
        prediction_record: Per-patient outputs and tie counts from
            :func:`record_per_patient_outputs`, when available.

    Returns:
        The assembled outcome; the fitness scalars are ``nan`` and the curve is
        empty when the run produced no generations at all.
    """
    curve = [float(statistics.best_fitness) for statistics in generation_history]
    return CellOutcome(
        metric_values=metric_values,
        per_class_f1_scores=per_class_f1_scores,
        generation_best_fitnesses=curve,
        generations_completed=len(curve),
        first_generation_best_fitness=curve[0] if curve else float("nan"),
        last_generation_best_fitness=curve[-1] if curve else float("nan"),
        runtime_seconds=runtime_seconds,
        phenotype_output_device=phenotype_output_device,
        named_genomes=named_genomes,
        structure_notes=structure_notes,
        generation_species_counts=species_counts_of(generation_history),
        **(prediction_record.as_outcome_fields() if prediction_record is not None else {}),
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


def record_per_patient_outputs(
    dataset: TabularDataset, phenotype: object, network_genomes: Sequence[object]
) -> PredictionRecord:
    """Store, for every patient of both splits, what the classifier produced.

    Per split (``train`` and ``test``) the arrays are: ``positions`` (row
    positions in the loaded dataset, ``-1`` when unknown), ``labels``,
    ``output_sums`` (the sums the output activation is applied to),
    ``outputs`` (the library phenotype's activations), ``predictions`` (the
    argmax of ``outputs`` - the rule :mod:`gpu_sweep.metrics` scores with) and
    ``tied`` (whether the row's largest output occurs more than once).

    The prediction rule itself is unchanged; the sums are stored so that a
    different rule can be applied after the sweep without evolving again.

    Args:
        dataset: The split the classifier was evolved against.
        phenotype: The scored phenotype (a single network or an ensemble).
        network_genomes: The phenotype's network genomes, in class order - the
            single genome, the decoded HyperNEAT substrate, or one recognizer
            per class.

    Returns:
        The arrays, the tie counts, and whether the recomputed activations
        equal the phenotype's outputs bit for bit on both splits.
    """
    per_patient_outputs: dict[str, np.ndarray] = {}
    tie_counts: dict[str, int] = {}
    all_outputs_match = True
    for split_name, features, labels, positions in (
        ("train", dataset.train_features, dataset.train_labels, dataset.train_positions),
        ("test", dataset.test_features, dataset.test_labels, dataset.test_positions),
    ):
        with torch.no_grad():
            library_outputs = phenotype.forward_pass(features).cpu()
        recomputed_outputs, output_sums = evaluate_networks_with_output_sums(
            network_genomes, features, features.device
        )
        all_outputs_match = all_outputs_match and bool(
            torch.equal(recomputed_outputs.cpu(), library_outputs)
        )
        row_maxima = library_outputs.max(dim=1, keepdim=True).values
        per_patient_outputs[f"{split_name}_positions"] = (
            np.asarray(positions, dtype=np.int64)
            if positions is not None
            else np.full(len(labels), -1, dtype=np.int64)
        )
        per_patient_outputs[f"{split_name}_labels"] = labels.cpu().numpy().astype(np.int64)
        per_patient_outputs[f"{split_name}_output_sums"] = output_sums.cpu().numpy()
        per_patient_outputs[f"{split_name}_outputs"] = library_outputs.numpy()
        per_patient_outputs[f"{split_name}_predictions"] = (
            library_outputs.argmax(dim=1).numpy().astype(np.int64)
        )
        per_patient_outputs[f"{split_name}_tied"] = (
            (library_outputs == row_maxima).sum(dim=1) > 1
        ).numpy()
        tie_counts[split_name] = count_tied_maxima(library_outputs)
    return PredictionRecord(
        per_patient_outputs=per_patient_outputs,
        tie_counts=tie_counts,
        output_sums_match_forward_pass=all_outputs_match,
    )


def _split_metrics(
    dataset: TabularDataset, phenotype: object
) -> tuple[dict[str, float], list[float]]:
    """Accuracy and macro-F1 on both splits, plus the test per-class F1 vector.

    Args:
        dataset: The split the phenotype was evolved against.
        phenotype: Phenotype to score.

    Returns:
        ``(metric_values, per_class_f1_scores)`` where ``metric_values`` holds
        ``train_accuracy``, ``train_macro_f1``, ``test_accuracy`` and
        ``test_macro_f1``.
    """
    train_metrics = evaluate_phenotype_metrics(
        phenotype, dataset.train_features, dataset.train_labels, dataset.number_of_classes
    )
    test_metrics = evaluate_phenotype_metrics(
        phenotype, dataset.test_features, dataset.test_labels, dataset.number_of_classes
    )
    metric_values = {
        "train_accuracy": float(train_metrics["accuracy"]),
        "train_macro_f1": float(train_metrics["macro_f1"]),
        "test_accuracy": float(test_metrics["accuracy"]),
        "test_macro_f1": float(test_metrics["macro_f1"]),
    }
    return metric_values, list(test_metrics["per_class_f1"])


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
    """Config for the algorithms that evolve one network with one output per class.

    Everything except the dataset's shape, the run's budget and the device is
    left at the library default, which is the NEAT paper's value - including
    ``steepened_sigmoid`` on both hidden and output nodes. The bounded output
    that implies is what the one-hot ``1 - MSE`` fitness expects.
    """
    return config_class(
        population_size=population_size,
        number_of_input_nodes=dataset.number_of_features,
        number_of_output_nodes=dataset.number_of_classes,
        random_seed=random_seed,
        device_for_phenotype_evaluation=str(device),
    )


def _run_single_network_family(
    algorithm: object,
    dataset: TabularDataset,
    *,
    number_of_generations: int,
    random_seed: int,
    structure_notes: dict[str, object] | None = None,
    genome_to_draw: Callable[[object], object] | None = None,
) -> CellOutcome:
    """Evolve one network with one output per class, scored by one-hot MSE.

    The shape shared by NEAT, FS-NEAT, NEAT-DBM and HyperNEAT. Fitness is
    ``1 - MSE`` against the one-hot target, which suits the paper-default
    bounded output activation and puts these four on the same ``[0, 1]``
    fitness scale as C-NEAT and L-NEAT.
    """
    fitness_evaluator = OneHotMeanSquaredErrorFitnessEvaluator(
        input_features=dataset.train_features,
        target_labels=dataset.train_labels,
        number_of_classes=dataset.number_of_classes,
    )
    last_generation_tracker = LastGenerationBestTracker()
    runner = pn.EvolutionRunner(
        algorithm=algorithm,
        fitness_evaluator=fitness_evaluator,
        termination_criterion=pn.MaxGenerationsTermination(
            max_generations=number_of_generations
        ),
        callbacks=[pn.ConsoleStatisticsLogger(), last_generation_tracker],
        random_seed=random_seed,
    )
    start_time = time.perf_counter()
    result = runner.run_evolution()
    runtime_seconds = time.perf_counter() - start_time

    best_phenotype = algorithm.phenotype_decoder.build_phenotype_from_genome(
        result.best_genome_ever_found
    )
    metric_values, per_class_f1_scores = _split_metrics(dataset, best_phenotype)
    scored_network_genome = (
        genome_to_draw(result.best_genome_ever_found)
        if genome_to_draw is not None
        else result.best_genome_ever_found
    )
    prediction_record = record_per_patient_outputs(
        dataset, best_phenotype, [scored_network_genome]
    )
    named_genomes: dict[str, object] = {"best": scored_network_genome}
    notes = dict(structure_notes or {})
    add_last_generation_network(
        named_genomes,
        notes,
        label="best",
        best_network_genome=scored_network_genome,
        tracker=last_generation_tracker,
        to_network=genome_to_draw,
    )
    return build_cell_outcome(
        result.full_generation_history,
        runtime_seconds=runtime_seconds,
        metric_values=metric_values,
        per_class_f1_scores=per_class_f1_scores,
        phenotype_output_device=phenotype_output_device_name(
            best_phenotype, dataset.train_features
        ),
        named_genomes=named_genomes,
        structure_notes=notes,
        prediction_record=prediction_record,
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
    return _run_single_network_family(
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
    return _run_single_network_family(
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
    return _run_single_network_family(
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
    metric_values, per_class_f1_scores = _split_metrics(dataset, ensemble)
    named_genomes = {
        f"class_{class_label_index}": container.best_genome_for_class(class_label_index)
        for class_label_index in range(dataset.number_of_classes)
    }
    prediction_record = record_per_patient_outputs(
        dataset,
        ensemble,
        [
            container.best_genome_for_class(class_label_index)
            for class_label_index in range(dataset.number_of_classes)
        ],
    )
    return build_cell_outcome(
        result.full_generation_history,
        runtime_seconds=runtime_seconds,
        metric_values=metric_values,
        per_class_f1_scores=per_class_f1_scores,
        phenotype_output_device=phenotype_output_device_name(
            ensemble, dataset.train_features
        ),
        named_genomes={
            name: genome for name, genome in named_genomes.items() if genome is not None
        },
        structure_notes={
            "genome_kind": "one single-output recognizer per class, combined by argmax"
        },
        prediction_record=prediction_record,
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
    per_class_generation_curves: list[list[float]] = []
    per_class_species_counts: list[list[int | None]] = []
    last_generation_trackers = [
        LastGenerationBestTracker() for _ in range(dataset.number_of_classes)
    ]

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
            callbacks=[pn.ConsoleStatisticsLogger(), last_generation_trackers[class_label_index]],
            random_seed=random_seed,
        )
        start_time = time.perf_counter()
        result = runner.run_evolution()
        total_runtime_seconds += time.perf_counter() - start_time
        total_generations += len(result.full_generation_history)
        per_class_generation_curves.append(
            [float(statistics.best_fitness) for statistics in result.full_generation_history]
        )
        per_class_species_counts.append(species_counts_of(result.full_generation_history))
        best_recognizer_genomes.append(result.best_genome_ever_found)

    assert phenotype_decoder is not None
    ensemble = pn.RecognizerEnsemblePhenotype.from_genomes(
        class_recognizer_genomes=best_recognizer_genomes,
        phenotype_decoder=phenotype_decoder,
    )
    metric_values, per_class_f1_scores = _split_metrics(dataset, ensemble)
    prediction_record = record_per_patient_outputs(dataset, ensemble, best_recognizer_genomes)
    averaged_curve_length = (
        min(len(curve) for curve in per_class_generation_curves)
        if per_class_generation_curves
        else 0
    )
    averaged_curve = [
        float(np.mean([curve[generation_index] for curve in per_class_generation_curves]))
        for generation_index in range(averaged_curve_length)
    ]
    named_genomes: dict[str, object] = {
        f"class_{class_label_index}": genome
        for class_label_index, genome in enumerate(best_recognizer_genomes)
    }
    structure_notes: dict[str, object] = {
        "genome_kind": "one backpropagation-trained recognizer per class, argmax ensemble",
        "total_generations_across_classes": total_generations,
        "note": (
            "the convergence curve is the per-generation mean across the "
            "per-class evolutions, which all share one generation budget; "
            "generations_completed is that curve's length, while "
            "total_generations_across_classes is the summed search effort"
        ),
    }
    for class_label_index, genome in enumerate(best_recognizer_genomes):
        add_last_generation_network(
            named_genomes,
            structure_notes,
            label=f"class_{class_label_index}",
            best_network_genome=genome,
            tracker=last_generation_trackers[class_label_index],
        )
    return CellOutcome(
        metric_values=metric_values,
        per_class_f1_scores=per_class_f1_scores,
        generation_best_fitnesses=averaged_curve,
        # Must equal len(averaged_curve): every consumer treats
        # generations_completed as that curve's length, and the shared test
        # helper asserts it. The total across the per-class evolutions is a
        # different quantity and lives in structure_notes instead.
        generations_completed=len(averaged_curve),
        first_generation_best_fitness=(
            averaged_curve[0] if averaged_curve else float("nan")
        ),
        last_generation_best_fitness=(
            averaged_curve[-1] if averaged_curve else float("nan")
        ),
        runtime_seconds=total_runtime_seconds,
        phenotype_output_device=phenotype_output_device_name(
            ensemble, dataset.train_features
        ),
        named_genomes=named_genomes,
        structure_notes=structure_notes,
        generation_species_counts=per_class_species_counts,
        **prediction_record.as_outcome_fields(),
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

    Only the substrate geometry comes from the dataset. Every other field -
    the CPPN function set, the weight expression threshold, the substrate's
    bounded activation - is left at the library default, which for this config
    is already the value published by Stanley, D'Ambrosio & Gauci (2009).
    """
    # DEVIATION, stated deliberately: substrate_hidden_layer_sizes defaults to
    # (3,) in HyperNEATConfig, and this sets it to (). Neither is a published
    # value - Stanley, D'Ambrosio & Gauci build a task-specific substrate per
    # experiment, and a fixed 3-node hidden layer is as arbitrary for a
    # 30-feature medical dataset as no hidden layer is. The empty choice matches
    # the two-sheet "state-space sandwich" of the paper's figure 6c, which is
    # what examples/mnist/hyperneat.py also uses, and it keeps the substrate a
    # direct feature-to-class mapping so the drawn picture is readable. Change
    # it here if you want a hidden sheet; it is the one HyperNEAT knob this
    # sweep does not leave at its default.
    config = pn.HyperNEATConfig(
        population_size=population_size,
        random_seed=random_seed,
        device_for_phenotype_evaluation=str(device),
        substrate_input_layer_size=dataset.number_of_features,
        substrate_hidden_layer_sizes=(),
        substrate_output_layer_size=dataset.number_of_classes,
    )
    algorithm = pn.HyperNEATAlgorithm.from_config(
        config, device_for_phenotype_computation=device
    )
    return _run_single_network_family(
        algorithm,
        dataset,
        number_of_generations=number_of_generations,
        random_seed=random_seed,
        structure_notes={
            "genome_kind": "substrate expressed by the evolved CPPN",
            "substrate_input_layer_size": dataset.number_of_features,
            "substrate_hidden_layer_sizes": [],
            "substrate_output_layer_size": dataset.number_of_classes,
            "substrate_node_activation_function": config.substrate_node_activation_function,
            "weight_expression_threshold": config.weight_expression_threshold,
        },
        genome_to_draw=algorithm.phenotype_decoder.decode_substrate_genome,
    )


ALGORITHM_RUNNERS["hyperneat"] = run_hyperneat
