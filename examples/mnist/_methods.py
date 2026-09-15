"""The two search methods the MNIST benchmark compares, as protocol-shaped callables.

DeepNEAT and EXACT learn in different places, and the difference is the whole
point of comparing them, so it is kept visible rather than hidden behind a flag.
DeepNEAT genomes carry no weights: every evaluation trains a fresh network and
the selected candidate is kept as a snapshot. EXACT is Lamarckian: it trains
between generations and writes the trained kernels back into the genotype, so its
evaluator only scores, and its selected genome is reset before track B so that
retraining really starts from scratch.

Both reach the same downstream protocol - track A snapshot, track B retrain, the
held-out test - which lives in :mod:`examples.mnist._protocol`.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import cast

import torch

from examples.mnist._protocol import SearchContext, SelectedCandidate
from polyneat.algorithms.deepneat.deepneat_algorithm import DeepNEATAlgorithm
from polyneat.algorithms.deepneat.deepneat_genome import DeepNEATGenome
from polyneat.algorithms.deepneat.torch_layer_stack_phenotype import TorchLayerStackPhenotype
from polyneat.algorithms.exact.exact_algorithm import EXACTAlgorithm
from polyneat.algorithms.exact.exact_backpropagation_trainer import EXACTBackpropagationTrainer
from polyneat.algorithms.exact.exact_genome import EXACTGenome
from polyneat.algorithms.exact.exact_image_input_adapter import FlattenedImageInputAdapter
from polyneat.algorithms.exact.exact_parameter_reset import reset_genome_for_fresh_training
from polyneat.algorithms.exact.torch_convolutional_phenotype import TorchConvolutionalPhenotype
from polyneat.configs.deepneat.deepneat_config import DeepNEATConfig
from polyneat.configs.exact.exact_config import EXACTConfig
from polyneat.evaluators.multiclass_accuracy_evaluator import (
    PretrainedMulticlassAccuracyEvaluator,
    TrainedMulticlassAccuracyEvaluator,
)
from polyneat.logging_utils.custom_logger import get_logger
from polyneat.runner.evaluation_record import EvaluationRecord
from polyneat.runner.resumable_search import run_resumable_evolution
from polyneat.runner.search_session import SearchSession
from polyneat.training.random_streams import RandomStreamRole, derive_stream_seed

logger = get_logger(__name__)


class SelectedCandidateMismatchError(RuntimeError):
    """Raised when the kept weights do not reproduce the selected fitness."""


def _count_parameters(model) -> int:
    """Trainable parameters of a model, or zero when it has none."""
    parameters = getattr(model, "parameters", None)
    if not callable(parameters):
        return 0
    return sum(int(parameter.numel()) for parameter in parameters())


def _build_trained_evaluator(context: SearchContext) -> TrainedMulticlassAccuracyEvaluator:
    """The accuracy evaluator every weightless-genome method shares."""
    return TrainedMulticlassAccuracyEvaluator(
        train_images=context.train.images,
        train_labels=context.train.labels,
        trainer=context.build_trainer(),
        root_seed=context.root_seed,
        validation=context.accuracy_split(),
        preprocessor=context.preprocessor,
        device_for_computation=context.device_for_computation,
        inference_batch_size=context.inference_batch_size,
        maximum_phenotype_parameters=context.maximum_phenotype_parameters,
        should_stop=None if context.budget is None else context.budget.should_stop,
    )


def _restore_winning_weights(
    evaluator: TrainedMulticlassAccuracyEvaluator,
    model: TorchLayerStackPhenotype,
    *,
    tolerance: float = 1e-6,
) -> float:
    """Load the winner's snapshot and check it reproduces the selected fitness."""
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
    """Build a DeepNEAT search over layer graphs and their hyperparameters."""

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


def make_exact_search(
    config: EXACTConfig, *, number_of_generations: int
) -> Callable[[SearchContext], SelectedCandidate]:
    """Build an EXACT search over convolutional topologies with inherited kernels."""

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

        evaluator = PretrainedMulticlassAccuracyEvaluator(
            validation=context.accuracy_split(),
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

    The runner decodes genomes with the algorithm's own decoder, which for EXACT
    produces a phenotype expecting flat rows. The shared inference path hands out
    spatial batches. Adapting here keeps both sides unchanged.
    """

    def __init__(self, inner: PretrainedMulticlassAccuracyEvaluator) -> None:
        self._inner = inner

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
