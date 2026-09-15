"""DeepNEAT for the CIFAR-10 experiment, as a protocol-shaped callable.

CIFAR runs DeepNEAT only: EXACT is single-channel by construction and CIFAR is
a colour task, so the two-method comparison lives on MNIST instead. DeepNEAT
genomes carry no weights, so every evaluation trains a fresh network and the
selected candidate is kept as a snapshot; track B then rebuilds that topology
with fresh weights and retrains it under the shared recipe, all downstream in
:mod:`examples._benchmark.multiclass_protocol`.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import cast

from examples.cifar10._protocol import SearchContext, SelectedCandidate
from polyneat.algorithms.deepneat.deepneat_algorithm import DeepNEATAlgorithm
from polyneat.algorithms.deepneat.deepneat_genome import DeepNEATGenome
from polyneat.algorithms.deepneat.torch_layer_stack_phenotype import TorchLayerStackPhenotype
from polyneat.configs.deepneat.deepneat_config import DeepNEATConfig
from polyneat.evaluators.multiclass_accuracy_evaluator import TrainedMulticlassAccuracyEvaluator
from polyneat.logging_utils.custom_logger import get_logger
from polyneat.runner.resumable_search import run_resumable_evolution
from polyneat.runner.search_session import SearchSession

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
