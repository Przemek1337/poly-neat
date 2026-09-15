"""The four search methods this benchmark compares, as protocol-shaped callables.

Each function here builds one method's search and returns the candidate it
selected. Everything downstream - freezing, threshold selection, the official
test - is identical for all of them and lives in ``_protocol``.

The four are chosen to separate two questions that are easy to conflate. A
neuroevolution result is only interesting if it beats searching the same space
at random, so the random search shares DeepNEAT's decoder, genes, constraints,
fitness, budget and candidate training, and differs only in having no selection
pressure and no inheritance. And it is only interesting if it beats a
hand-designed network, so the fixed CNN runs through the same preprocessing,
class weights and recipe.

Declaring "the same space" is not the same as sampling it identically: the
random search draws a structural-mutation count from a stated distribution,
which is not the distribution DeepNEAT's population reaches after selection.
The report has to say so.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import cast

import torch

from examples.pediatric_pneumonia._protocol import SearchContext, SelectedCandidate
from polyneat.algorithms.deepneat.deepneat_algorithm import DeepNEATAlgorithm
from polyneat.algorithms.deepneat.deepneat_genome import DeepNEATGenome
from polyneat.algorithms.deepneat.torch_layer_stack_phenotype import (
    TorchLayerStackPhenotype,
)
from polyneat.algorithms.exact.exact_algorithm import EXACTAlgorithm
from polyneat.algorithms.exact.exact_backpropagation_trainer import (
    EXACTBackpropagationTrainer,
)
from polyneat.algorithms.exact.exact_genome import EXACTGenome
from polyneat.algorithms.exact.exact_image_input_adapter import FlattenedImageInputAdapter
from polyneat.algorithms.exact.exact_parameter_reset import reset_genome_for_fresh_training
from polyneat.algorithms.exact.torch_convolutional_phenotype import (
    TorchConvolutionalPhenotype,
)
from polyneat.configs.deepneat.deepneat_config import DeepNEATConfig
from polyneat.configs.exact.exact_config import EXACTConfig
from polyneat.core.neat.neat_genome import NEATGenome
from polyneat.evaluators.binary_auroc_evaluator import (
    PretrainedBinaryAurocEvaluator,
    TrainedBinaryAurocEvaluator,
    ValidationSplit,
)
from polyneat.logging_utils.custom_logger import get_logger
from polyneat.nn.fixed_convolutional_network import (
    FixedConvolutionalNetwork,
    FixedConvolutionalNetworkConfig,
)
from polyneat.nn.pretrained_image_classifier import (
    PretrainedClassifierConfig,
    PretrainedResNetClassifier,
)
from polyneat.runner.evaluation_record import EvaluationRecord
from polyneat.runner.resumable_search import (
    run_resumable_candidates,
    run_resumable_evolution,
)
from polyneat.runner.search_session import SearchSession
from polyneat.training.random_streams import (
    RandomStreamRole,
    create_numpy_generator,
    derive_stream_seed,
)

logger = get_logger(__name__)


class SelectedCandidateMismatchError(RuntimeError):
    """Raised when the kept weights do not reproduce the selected fitness.

    The search reports one number and the checkpoint has to be the model that
    produced it. If re-scoring the restored model disagrees, something has
    substituted a different network for the winner, and reporting either number
    would be wrong.
    """


def _validation_split(context: SearchContext) -> ValidationSplit:
    """The fitness split, in the shape the evaluators expect."""
    return ValidationSplit(
        images=context.search_validation.images,
        labels=context.search_validation.labels,
        example_ids=context.search_validation.example_ids,
        group_ids=context.search_validation.group_ids,
        split_name=context.search_validation.split_name,
    )


def _count_parameters(model) -> int:
    """Trainable parameters of a model, or zero when it has none."""
    parameters = getattr(model, "parameters", None)
    if not callable(parameters):
        return 0
    return sum(int(parameter.numel()) for parameter in parameters())


def _build_trained_evaluator(
    context: SearchContext, stage: str = "track_a"
) -> TrainedBinaryAurocEvaluator:
    """The evaluator every weightless-genome method shares."""
    return TrainedBinaryAurocEvaluator(
        train_images=context.train.images,
        train_labels=context.train.labels,
        trainer=context.build_trainer(),
        root_seed=context.root_seed,
        validation=_validation_split(context),
        preprocessor=context.preprocessor,
        device_for_computation=context.device_for_computation,
        inference_batch_size=context.inference_batch_size,
        maximum_phenotype_parameters=context.maximum_phenotype_parameters,
        should_stop=None if context.budget is None else context.budget.should_stop,
        stage=stage,
    )


def _restore_winning_weights(
    evaluator: TrainedBinaryAurocEvaluator,
    model: TorchLayerStackPhenotype,
    *,
    tolerance: float = 1e-6,
) -> float:
    """Load the winner's snapshot and check it reproduces the selected fitness.

    Decoding the winning genome again gives a network with fresh weights, which
    for a weightless genome never earned the selected score at all. So the
    snapshot taken during evaluation is loaded back, and then verified: if the
    restored model scores something else, the search and the checkpoint
    disagree about which candidate won and the run stops.

    Returns:
        The selected fitness, now known to be a real number produced by the
        model the caller holds.

    Raises:
        SelectedCandidateMismatchError: If no snapshot was kept, or the
            restored model does not reproduce the selected fitness.
    """
    if evaluator.best_model_state is None or evaluator.best_fitness is None:
        raise SelectedCandidateMismatchError(
            "the search kept no trained weights for its selected candidate"
        )
    model.load_state_dict(evaluator.best_model_state)
    restored_fitness = evaluator.score_phenotype(model, "restored_winner")
    if abs(restored_fitness - evaluator.best_fitness) > tolerance:
        raise SelectedCandidateMismatchError(
            f"the restored winner scores {restored_fitness:.6f} but the search selected on "
            f"{evaluator.best_fitness:.6f}; the checkpoint is not the model that won"
        )
    return float(evaluator.best_fitness)


def make_deepneat_search(
    config: DeepNEATConfig, *, number_of_generations: int
) -> Callable[[SearchContext], SelectedCandidate]:
    """Build a DeepNEAT search over layer graphs and their hyperparameters.

    DeepNEAT genomes carry no weights, so every evaluation trains a fresh
    network and the selected candidate has to be kept as a snapshot rather than
    rebuilt from its genome.
    """

    def search(context: SearchContext) -> SelectedCandidate:
        search_started_at = time.perf_counter()
        algorithm = DeepNEATAlgorithm.from_config(
            config, device_for_phenotype_computation=context.device_for_computation
        )
        evaluator = _build_trained_evaluator(context)
        best, _fitness, generations = run_resumable_evolution(
            algorithm,
            evaluator,
            session=context.session or SearchSession(None, binding={}, budget=context.budget),
            genome_from_dict=DeepNEATGenome.from_serializable_dict,
            seed=context.root_seed,
            number_of_generations=number_of_generations,
        )
        best_genome = cast(DeepNEATGenome, best)
        track_a_model = cast(
            TorchLayerStackPhenotype,
            algorithm.phenotype_decoder.build_phenotype_from_genome(best_genome),
        )
        selection_fitness = _restore_winning_weights(evaluator, track_a_model)
        return SelectedCandidate(
            track_a_model=track_a_model,
            rebuild_model=lambda: cast(
                TorchLayerStackPhenotype,
                algorithm.phenotype_decoder.build_phenotype_from_genome(best_genome),
            ),
            genome_kind=type(best_genome).__name__,
            genome_payload=best_genome.to_serializable_dict(),
            selection_fitness=selection_fitness,
            evaluation_records=evaluator.evaluation_records,
            number_of_generations=generations,
            search_seconds=time.perf_counter() - search_started_at,
            parameter_count=_count_parameters(track_a_model),
        )

    return search


def make_random_search(
    config: DeepNEATConfig,
    *,
    number_of_candidates: int,
    minimum_structural_mutations: int,
    maximum_structural_mutations: int,
) -> Callable[[SearchContext], SelectedCandidate]:
    """Build the mandatory control: the same space, sampled without selection.

    Every candidate is drawn independently from the minimal genome by applying
    a number of structural mutations drawn from a stated uniform range. No
    fitness enters the generation of any candidate, nothing is inherited, and
    the decoder, gene ranges, constraints, fitness, candidate training and
    budget are the ones DeepNEAT uses.

    The sampling distribution is written down here and frozen in the pilot,
    because "the same space" does not imply the same sampling distribution and
    the report has to state the difference.

    Args:
        config: The DeepNEAT configuration whose space is being sampled.
        number_of_candidates: How many independent candidates to draw.
        minimum_structural_mutations: Lower end of the structural-mutation
            count, inclusive.
        maximum_structural_mutations: Upper end, inclusive.
    """
    if minimum_structural_mutations < 0 or maximum_structural_mutations < (
        minimum_structural_mutations
    ):
        raise ValueError(
            "structural mutation range must satisfy 0 <= minimum <= maximum, got "
            f"({minimum_structural_mutations}, {maximum_structural_mutations})"
        )

    def search(context: SearchContext) -> SelectedCandidate:
        search_started_at = time.perf_counter()
        algorithm = DeepNEATAlgorithm.from_config(
            config, device_for_phenotype_computation=context.device_for_computation
        )
        evaluator = _build_trained_evaluator(context)
        sampling_rng = create_numpy_generator(
            role=RandomStreamRole.EVOLUTION,
            root_seed=context.root_seed,
            evaluation_id="random_search",
        )

        def draw() -> DeepNEATGenome:
            population = algorithm.create_initial_population(sampling_rng)
            genome = cast(
                NEATGenome,
                population.genomes[int(sampling_rng.integers(0, len(population.genomes)))],
            )
            number_of_mutations = int(
                sampling_rng.integers(
                    minimum_structural_mutations, maximum_structural_mutations + 1
                )
            )
            for _ in range(number_of_mutations):
                genome = algorithm.mutation.apply_to_genome(
                    genome, sampling_rng, algorithm.innovation_tracker
                )
            return cast(DeepNEATGenome, genome)

        best, _fitness = run_resumable_candidates(
            algorithm,
            evaluator,
            session=context.session or SearchSession(None, binding={}, budget=context.budget),
            genome_from_dict=DeepNEATGenome.from_serializable_dict,
            rng=sampling_rng,
            draw=draw,
            count=number_of_candidates,
        )
        best_genome = cast(DeepNEATGenome, best)
        track_a_model = cast(
            TorchLayerStackPhenotype,
            algorithm.phenotype_decoder.build_phenotype_from_genome(best_genome),
        )
        selection_fitness = _restore_winning_weights(evaluator, track_a_model)
        return SelectedCandidate(
            track_a_model=track_a_model,
            rebuild_model=lambda: cast(
                TorchLayerStackPhenotype,
                algorithm.phenotype_decoder.build_phenotype_from_genome(best_genome),
            ),
            genome_kind=type(best_genome).__name__,
            genome_payload=best_genome.to_serializable_dict(),
            selection_fitness=selection_fitness,
            evaluation_records=evaluator.evaluation_records,
            number_of_generations=1,
            search_seconds=time.perf_counter() - search_started_at,
            parameter_count=_count_parameters(track_a_model),
        )

    return search


def _evaluate_baseline(context: SearchContext, evaluator, model) -> None:
    """Resume completed baseline evaluations; incomplete training rolls back."""
    session = context.session
    if session is not None and session.state is not None:
        if session.state["kind"] != "baseline":
            raise ValueError("wrong checkpoint kind")
        evaluator.load_state_dict(session.state["evaluator"])
        if session.state["done"]:
            model.load_state_dict(evaluator.best_model_state)
            model.eval()
            return
    if session is not None:
        session.commit({"kind": "baseline", "done": False, "evaluator": evaluator.state_dict()})
    evaluator.evaluate_candidate(model, "gen0/cand0")
    if session is not None:
        session.commit({"kind": "baseline", "done": True, "evaluator": evaluator.state_dict()})


def make_fixed_cnn_baseline(
    architecture: FixedConvolutionalNetworkConfig,
) -> Callable[[SearchContext], SelectedCandidate]:
    """Build the hand-designed baseline as a search that considers one candidate.

    It is shaped like a search so it reaches the same freezing, threshold
    selection and test evaluation as everything else. There is no search cost
    to report, but if several hand-designed configurations are compared, the
    cost of trying them is baseline tuning and belongs in the report.
    """

    def search(context: SearchContext) -> SelectedCandidate:
        search_started_at = time.perf_counter()
        evaluator = _build_trained_evaluator(context)
        model = FixedConvolutionalNetwork(architecture)
        _evaluate_baseline(context, evaluator, model)
        if evaluator.best_fitness is None:
            raise SelectedCandidateMismatchError("the fixed CNN baseline could not be evaluated")
        return SelectedCandidate(
            track_a_model=model,
            rebuild_model=lambda: FixedConvolutionalNetwork(architecture),
            genome_kind="FixedConvolutionalNetworkConfig",
            genome_payload={
                "architecture": architecture.describe(),
                "convolution_channels": list(architecture.convolution_channels),
                "kernel_size": architecture.kernel_size,
                "uses_batch_normalization": architecture.uses_batch_normalization,
                "dropout_probability": architecture.dropout_probability,
            },
            selection_fitness=float(evaluator.best_fitness),
            evaluation_records=evaluator.evaluation_records,
            number_of_generations=1,
            search_seconds=time.perf_counter() - search_started_at,
            parameter_count=model.total_parameter_count,
        )

    return search


def make_exact_search(
    config: EXACTConfig, *, number_of_generations: int, image_side: int
) -> Callable[[SearchContext], SelectedCandidate]:
    """Build an EXACT search over convolutional topologies with inherited kernels.

    EXACT trains between generations and writes the trained kernels back into
    the genotype, so its evaluator only scores. Its trainer is given the shared
    class weights, the shared preprocessing as a batch transform, an explicit
    batch-order stream and the search deadline - none of which change its
    published learning rules, all of which the benchmark needs applied
    consistently across methods.

    Track B resets the selected genome first: clearing the kernels alone leaves
    the evolved batch-normalization state behind, and an already-trained genome
    is skipped by the trainer entirely.
    """

    def search(context: SearchContext) -> SelectedCandidate:
        search_started_at = time.perf_counter()
        algorithm = EXACTAlgorithm.from_config(
            config, device_for_phenotype_computation=context.device_for_computation
        )
        flattened_training_features = context.train.images.reshape(
            context.train.images.shape[0], -1
        ).to(torch.float32)
        trainer = EXACTBackpropagationTrainer.from_config(
            config,
            training_features=flattened_training_features,
            training_labels=context.train.labels,
            device_for_computation=context.device_for_computation,
        )
        # One stream each for the whole run, so every training session EXACT
        # performs continues the same augmentation and batch-order sequences
        # rather than restarting them per generation.
        augmentation_generator = torch.Generator()
        augmentation_generator.manual_seed(
            derive_stream_seed(
                role=RandomStreamRole.AUGMENTATION, root_seed=context.root_seed, track="A"
            )
        )
        batch_order_generator = torch.Generator()
        batch_order_generator.manual_seed(
            derive_stream_seed(
                role=RandomStreamRole.BATCH_ORDER, root_seed=context.root_seed, track="A"
            )
        )
        trainer.configure_supervised_extras(
            class_weights=context.class_weights,
            batch_transform=lambda batch: context.preprocessor.apply(
                batch, training=True, generator=augmentation_generator
            ),
            batch_order_generator=batch_order_generator,
            should_stop=None if context.budget is None else context.budget.should_stop,
        )
        # In this sequential runner train immediately before scoring a candidate.
        # This preserves inheritance while allowing a completed model to survive
        # a deadline inside the initial population, not only after its last member.

        evaluator = PretrainedBinaryAurocEvaluator(
            validation=_validation_split(context),
            preprocessor=context.preprocessor,
            device_for_computation=context.device_for_computation,
            inference_batch_size=context.inference_batch_size,
            maximum_phenotype_parameters=context.maximum_phenotype_parameters,
            should_stop=None if context.budget is None else context.budget.should_stop,
        )

        def export_streams() -> dict:
            return {
                "augmentation": augmentation_generator.get_state(),
                "batch_order": batch_order_generator.get_state(),
            }

        def import_streams(state: dict) -> None:
            augmentation_generator.set_state(state["augmentation"])
            batch_order_generator.set_state(state["batch_order"])

        best, selection_fitness, generations = run_resumable_evolution(
            algorithm,
            _AdaptedEvaluator(evaluator),
            session=context.session or SearchSession(None, binding={}, budget=context.budget),
            genome_from_dict=EXACTGenome.from_serializable_dict,
            seed=context.root_seed,
            number_of_generations=number_of_generations,
            export_training_state=export_streams,
            import_training_state=import_streams,
            prepare_genome=lambda genome: trainer.train_genome(cast(EXACTGenome, genome)),
        )
        best_genome = cast(EXACTGenome, best)
        track_a_model = FlattenedImageInputAdapter(
            cast(
                TorchConvolutionalPhenotype,
                algorithm.phenotype_decoder.build_phenotype_from_genome(best_genome),
            )
        )
        reset_genome = reset_genome_for_fresh_training(best_genome)
        return SelectedCandidate(
            track_a_model=track_a_model,
            rebuild_model=lambda: FlattenedImageInputAdapter(
                cast(
                    TorchConvolutionalPhenotype,
                    algorithm.phenotype_decoder.build_phenotype_from_genome(reset_genome),
                )
            ),
            genome_kind=type(best_genome).__name__,
            genome_payload=best_genome.to_serializable_dict(),
            selection_fitness=selection_fitness,
            evaluation_records=evaluator.evaluation_records,
            number_of_generations=generations,
            search_seconds=time.perf_counter() - search_started_at,
            parameter_count=_count_parameters(track_a_model),
        )

    return search


class _AdaptedEvaluator:
    """Wraps every phenotype in the image adapter before it is scored.

    The runner decodes genomes with the algorithm's own decoder, which for
    EXACT produces a phenotype expecting flat rows. The shared inference path
    hands out spatial batches. Adapting here keeps both sides unchanged.
    """

    def __init__(self, inner: PretrainedBinaryAurocEvaluator) -> None:
        self._inner = inner

    def evaluate_batch_of_phenotypes(self, phenotypes: list) -> list[float]:
        """Adapt each phenotype, then delegate to the wrapped evaluator."""
        return self._inner.evaluate_batch_of_phenotypes(
            [FlattenedImageInputAdapter(phenotype) for phenotype in phenotypes]
        )

    def evaluate_candidate(self, phenotype, evaluation_id: str) -> EvaluationRecord:
        return self._inner.evaluate_candidate(FlattenedImageInputAdapter(phenotype), evaluation_id)

    def state_dict(self) -> dict:
        return self._inner.state_dict()

    def load_state_dict(self, state: dict) -> None:
        self._inner.load_state_dict(state)

    @property
    def evaluation_records(self) -> tuple[EvaluationRecord, ...]:
        """Records of the wrapped evaluator."""
        return self._inner.evaluation_records


def make_transfer_learning_baseline(
    architecture: PretrainedClassifierConfig,
) -> Callable[[SearchContext], SelectedCandidate]:
    """Build the pretrained ResNet-18 baseline as a one-candidate search.

    Reported separately from the neuroevolution comparison, and deliberately
    not given a track B: retraining it from scratch would delete the pretrained
    features that are the entire reason it is here. Its budget is also not
    comparable with a search that started from nothing, because the cost of the
    original ImageNet pretraining is counted nowhere.

    It brings its own normalization, so the profile that uses it sets the
    shared preprocessing to identity standardization; applying dataset
    statistics first and ImageNet statistics second would hand the backbone an
    input distribution it was never calibrated for.
    """

    def search(context: SearchContext) -> SelectedCandidate:
        search_started_at = time.perf_counter()
        evaluator = _build_trained_evaluator(context, stage="transfer_learning")
        model = PretrainedResNetClassifier(architecture)
        _evaluate_baseline(context, evaluator, model)
        if evaluator.best_fitness is None:
            failure = evaluator.evaluation_records[0].failure_reason
            raise SelectedCandidateMismatchError(
                f"the transfer-learning baseline could not be evaluated: {failure}"
            )
        return SelectedCandidate(
            track_a_model=model,
            rebuild_model=lambda: PretrainedResNetClassifier(architecture),
            genome_kind="PretrainedClassifierConfig",
            genome_payload={
                "weights_identifier": architecture.weights_identifier,
                "weights_file_name": model.weights_file_name,
                "unfreezing_policy": str(architecture.unfreezing_policy),
                "uses_imagenet_normalization": architecture.uses_imagenet_normalization,
            },
            selection_fitness=float(evaluator.best_fitness),
            evaluation_records=evaluator.evaluation_records,
            number_of_generations=1,
            search_seconds=time.perf_counter() - search_started_at,
            parameter_count=model.trainable_parameter_count,
        )

    return search
