"""A dataset-agnostic multi-class benchmark controller: one fixed split, two tracks.

The digit (MNIST) and object (CIFAR) benchmarks run the same protocol on
different pixels: a fixed train/validation split carved from the official
training set by a seed, a search whose fitness is validation accuracy, a track A
that keeps the exact network the search selected and a track B that retrains that
topology from fresh weights under one shared recipe, and an official test set
loaded only at the end and only outside pilot mode. None of that depends on how
many channels an image has or where the pixels came from.

So it lives here once, parameterised by a loader and an environment function.
Each family binds those and its geometry (side, channels, class count) and gets
the whole protocol without copying it. What stays in a family is only what is
genuinely its own: which dataset to load and which libraries to pin.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import torch
from torch import nn

from examples._benchmark.execution import ExecutionLockError, ExecutionOptions, canonical_digest
from examples._experiment import ExperimentReport
from polyneat.evaluators.multiclass_accuracy_evaluator import (
    AccuracyValidationSplit,
    validation_top1_accuracy,
)
from polyneat.logging_utils.custom_logger import get_logger
from polyneat.runner.evaluation_record import EvaluationRecord, count_by_status
from polyneat.runner.run_checkpoint import build_run_binding
from polyneat.runner.search_session import SearchSession
from polyneat.runner.wall_clock_budget import WallClockBudget
from polyneat.training.class_weights import compute_balanced_class_weights
from polyneat.training.image_preprocessing import (
    ImageAugmentationConfig,
    ImagePreprocessingConfig,
    ImagePreprocessor,
)
from polyneat.training.model_checkpoint import capture_model_checkpoint
from polyneat.training.parameter_initialization import initialize_module_parameters
from polyneat.training.random_streams import TrainingRandomStreams
from polyneat.training.supervised_trainer import SupervisedTrainer
from polyneat.training.trainable_model import TrainableModel
from polyneat.training.training_recipe import TrainingRecipe

logger = get_logger(__name__)

TRACK_A = "track_a"
TRACK_B = "track_b"

# A loader returns the official split as (train_features, train_labels,
# test_features, test_labels), each a CPU tensor; features are flat rows.
DatasetLoader = Callable[
    ["MulticlassBenchmarkSettings"], tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
]
EnvironmentFunction = Callable[[torch.device], dict]


@dataclass(frozen=True)
class Split:
    """One split as tensors, with the ids needed for provenance.

    Attributes:
        images: ``(n, channels, side, side)`` ``uint8`` tensor on the CPU.
        labels: ``(n,)`` long tensor of class indices.
        example_ids: Stable ids aligned with the rows above.
        split_name: Which split this is, carried so a mix-up is visible.
    """

    images: torch.Tensor
    labels: torch.Tensor
    example_ids: tuple[str, ...]
    split_name: str


@dataclass(frozen=True)
class MulticlassBenchmarkSettings:
    """Everything a multi-class benchmark run needs that is not the search itself.

    Attributes:
        protocol_id: Identifier of this protocol version.
        dataset_release: Identifier of the dataset used, recorded in the report.
        dataset_license: License of that dataset.
        image_side: Square side every image is brought to.
        channels: Colour channels of the images (1 for MNIST, 3 for CIFAR).
        number_of_classes: Class count of the task.
        split_seed: Seed of the fixed train/validation split and subset draw.
        validation_fraction: Fraction of the official training set held out.
        search_seed: Root seed of the search and of track A training.
        retraining_seeds: Track B retraining seeds for the selected topology.
        candidate_recipe: How each search candidate is trained in track A.
        track_b_recipe: The one shared recipe every topology is retrained under.
        inference_batch_size: Rows per forward pass during scoring.
        maximum_phenotype_parameters: Candidates above this are rejected.
        search_budget_seconds: Active wall-clock allowance of the search.
        uses_augmentation: Whether the training augmentation runs at all.
        maximum_training_samples: Cap on official training rows (smoke subsets).
        maximum_test_samples: Cap on official test rows.
        cache_directory: Where the dataset cache lives.
        execution: Mode, resume and lock policy for this run.
        profile_payload: The whole profile yaml, recorded and lock-checked.
        device_for_computation: The one device the run uses.
    """

    protocol_id: str
    dataset_release: str
    dataset_license: str
    image_side: int
    channels: int
    number_of_classes: int
    split_seed: int
    validation_fraction: float
    search_seed: int
    retraining_seeds: tuple[int, ...]
    candidate_recipe: TrainingRecipe
    track_b_recipe: TrainingRecipe
    inference_batch_size: int = 64
    maximum_phenotype_parameters: int | None = None
    search_budget_seconds: float | None = None
    uses_augmentation: bool = True
    maximum_training_samples: int | None = None
    maximum_test_samples: int | None = None
    cache_directory: Path | None = None
    execution: ExecutionOptions = field(default_factory=ExecutionOptions)
    profile_payload: dict = field(default_factory=dict)
    device_for_computation: torch.device = field(default_factory=lambda: torch.device("cpu"))


@dataclass(frozen=True)
class PreparedData:
    """The fixed split, materialised as tensors for the stages that may see it."""

    dataset_identity_sha256: str
    train: Split
    validation: Split
    official_test: Split


@dataclass(frozen=True)
class SearchContext:
    """What a search method is given, and nothing beyond it."""

    train: Split
    search_validation: Split
    preprocessor: ImagePreprocessor
    class_weights: torch.Tensor
    device_for_computation: torch.device
    root_seed: int
    candidate_recipe: TrainingRecipe
    inference_batch_size: int
    maximum_phenotype_parameters: int | None
    budget: WallClockBudget | None
    session: SearchSession | None = None

    def build_trainer(self) -> SupervisedTrainer:
        """A trainer wired to this stage's preprocessing, weights and budget."""
        return SupervisedTrainer(
            recipe=self.candidate_recipe,
            preprocessor=self.preprocessor,
            device_for_computation=self.device_for_computation,
            class_weights=self.class_weights,
            should_stop=None if self.budget is None else self.budget.should_stop,
        )

    def accuracy_split(self) -> AccuracyValidationSplit:
        """The validation split in the shape the accuracy evaluator expects."""
        return AccuracyValidationSplit(
            images=self.search_validation.images,
            labels=self.search_validation.labels,
            split_name=self.search_validation.split_name,
        )


@dataclass(frozen=True)
class SelectedCandidate:
    """The candidate a search selected, and how to rebuild it from scratch."""

    track_a_model: TrainableModel
    rebuild_model: Callable[[], TrainableModel]
    genome_kind: str
    genome_payload: dict
    selection_fitness: float
    evaluation_records: tuple[EvaluationRecord, ...]
    number_of_generations: int
    search_seconds: float
    parameter_count: int


SearchCallable = Callable[[SearchContext], SelectedCandidate]


def _dataset_identity(settings: MulticlassBenchmarkSettings, prepared: dict) -> str:
    """Bind actual data and split, independently of editable provenance labels."""
    return canonical_digest(
        {
            "identity_schema": "content-v2",
            "image_side": settings.image_side,
            "channels": settings.channels,
            "number_of_classes": settings.number_of_classes,
            "split_seed": settings.split_seed,
            "validation_fraction": settings.validation_fraction,
            "maximum_training_samples": settings.maximum_training_samples,
            "maximum_test_samples": settings.maximum_test_samples,
            **prepared,
        }
    )


def _tensor_digest(tensor: torch.Tensor) -> str:
    """Hash typed, ordered CPU tensor contents without copying a large byte string."""
    array = tensor.detach().cpu().contiguous().numpy()
    digest = hashlib.sha256()
    digest.update(canonical_digest({"shape": list(array.shape), "dtype": array.dtype.str}).encode())
    digest.update(array.data.cast("B"))
    return digest.hexdigest()


def to_uint8_images(features: torch.Tensor, *, channels: int, image_side: int) -> torch.Tensor:
    """Reshape flat rows to ``(n, channels, side, side)`` uint8 in ``[0, 255]``.

    Loaders hand back flat rows scaled to ``[0, 255]``; the shared preprocessing
    expects unpreprocessed uint8 images, the same currency as the pneumonia
    cache, so the geometry and the type are restored here once.
    """
    rows = features.reshape(features.shape[0], channels, image_side, image_side)
    return rows.clamp(0.0, 255.0).round().to(torch.uint8)


def prepare_data(
    settings: MulticlassBenchmarkSettings, *, load_dataset: DatasetLoader
) -> PreparedData:
    """Load the dataset, carve the fixed validation split, hold out the test set."""
    train_features, train_labels, test_features, test_labels = load_dataset(settings)
    train_images = to_uint8_images(
        train_features, channels=settings.channels, image_side=settings.image_side
    )
    test_images = to_uint8_images(
        test_features, channels=settings.channels, image_side=settings.image_side
    )

    generator = torch.Generator().manual_seed(settings.split_seed)
    permutation = torch.randperm(train_images.shape[0], generator=generator)
    validation_size = int(round(settings.validation_fraction * train_images.shape[0]))
    if not 0 < validation_size < train_images.shape[0]:
        raise ValueError("validation_fraction must carve a non-empty train and validation split")
    validation_indices = permutation[:validation_size]
    train_indices = permutation[validation_size:]

    train_split = Split(
        images=train_images[train_indices],
        labels=train_labels[train_indices],
        example_ids=tuple(f"train-{int(index)}" for index in train_indices),
        split_name="train",
    )
    validation_split = Split(
        images=train_images[validation_indices],
        labels=train_labels[validation_indices],
        example_ids=tuple(f"train-{int(index)}" for index in validation_indices),
        split_name="validation",
    )
    test_split = Split(
        images=test_images,
        labels=test_labels,
        example_ids=tuple(f"test-{index}" for index in range(test_images.shape[0])),
        split_name="official_test",
    )
    identity = _dataset_identity(
        settings,
        {
            "train_rows": int(train_split.images.shape[0]),
            "validation_rows": int(validation_split.images.shape[0]),
            "test_rows": int(test_split.images.shape[0]),
            "official_train_images": _tensor_digest(train_images),
            "official_train_labels": _tensor_digest(train_labels),
            "official_test_images": _tensor_digest(test_images),
            "official_test_labels": _tensor_digest(test_labels),
            "train_indices": _tensor_digest(train_indices),
            "validation_indices": _tensor_digest(validation_indices),
        },
    )
    return PreparedData(
        dataset_identity_sha256=identity,
        train=train_split,
        validation=validation_split,
        official_test=test_split,
    )


def build_preprocessor(
    settings: MulticlassBenchmarkSettings, training_images: torch.Tensor
) -> ImagePreprocessor:
    """Fit one preprocessor on the training images of one stage."""
    preprocessor = ImagePreprocessor(
        ImagePreprocessingConfig(target_side=settings.image_side),
        ImageAugmentationConfig() if settings.uses_augmentation else None,
    )
    preprocessor.fit_standardization(training_images)
    return preprocessor


def effective_configuration(settings: MulticlassBenchmarkSettings, method_name: str) -> dict:
    """The configuration this run actually used, after every override."""
    return {
        "method": method_name,
        "mode": settings.execution.mode,
        "protocol_id": settings.protocol_id,
        "dataset_release": settings.dataset_release,
        "dataset_license": settings.dataset_license,
        "image_side": settings.image_side,
        "channels": settings.channels,
        "number_of_classes": settings.number_of_classes,
        "split_seed": settings.split_seed,
        "validation_fraction": settings.validation_fraction,
        "search_seed": settings.search_seed,
        "retraining_seeds": list(settings.retraining_seeds),
        "candidate_recipe": settings.candidate_recipe.to_serializable_dict(),
        "track_b_recipe": settings.track_b_recipe.to_serializable_dict(),
        "inference_batch_size": settings.inference_batch_size,
        "maximum_phenotype_parameters": settings.maximum_phenotype_parameters,
        "search_budget_seconds": settings.search_budget_seconds,
        "uses_augmentation": settings.uses_augmentation,
        "maximum_training_samples": settings.maximum_training_samples,
        "maximum_test_samples": settings.maximum_test_samples,
        "device_for_computation": str(settings.device_for_computation),
        "profile_payload": settings.profile_payload,
    }


def run_search_stage(
    settings: MulticlassBenchmarkSettings,
    data: PreparedData,
    search: SearchCallable,
    *,
    environment: EnvironmentFunction,
    artifacts_directory: Path | None = None,
    method_name: str = "unspecified",
    lock_sha256: str = "unlocked",
) -> tuple[SelectedCandidate, ImagePreprocessor]:
    """Run one method's search and return what it selected, with its preprocessing."""
    preprocessor = build_preprocessor(settings, data.train.images)
    budget = (
        None
        if settings.search_budget_seconds is None
        else WallClockBudget(settings.search_budget_seconds)
    )
    session = SearchSession(
        None if artifacts_directory is None else artifacts_directory / "search",
        binding=build_run_binding(
            manifest_sha256=data.dataset_identity_sha256,
            protocol_lock_sha256=lock_sha256,
            effective_configuration={
                **effective_configuration(settings, method_name),
                "environment": environment(settings.device_for_computation),
            },
        ),
        budget=budget,
        resume=settings.execution.resume,
        lost_work_seconds=settings.execution.lost_work_seconds,
    )
    context = SearchContext(
        train=data.train,
        search_validation=data.validation,
        preprocessor=preprocessor,
        class_weights=compute_balanced_class_weights(data.train.labels, settings.number_of_classes),
        device_for_computation=settings.device_for_computation,
        root_seed=settings.search_seed,
        candidate_recipe=settings.candidate_recipe,
        inference_batch_size=settings.inference_batch_size,
        maximum_phenotype_parameters=settings.maximum_phenotype_parameters,
        budget=budget,
        session=session,
    )
    if not settings.execution.resume:
        torch.manual_seed(settings.search_seed)
    with session:
        selected = search(context)
    if budget is not None:
        selected = replace(selected, search_seconds=budget.consumed_seconds)
    logger.info(
        "Search selected a candidate with validation accuracy %.4f after %d generations",
        selected.selection_fitness,
        selected.number_of_generations,
    )
    return selected, preprocessor


def retrain_topology_for_track_b(
    settings: MulticlassBenchmarkSettings,
    data: PreparedData,
    selected: SelectedCandidate,
    retraining_seed: int,
) -> tuple[TrainableModel, ImagePreprocessor, dict]:
    """Retrain the selected topology from scratch under the shared recipe."""
    combined_images = torch.cat([data.train.images, data.validation.images])
    combined_labels = torch.cat([data.train.labels, data.validation.labels])

    preprocessor = build_preprocessor(settings, combined_images)
    streams = TrainingRandomStreams.derive(
        root_seed=retraining_seed, evaluation_id="selected_topology", track=TRACK_B
    )
    model = selected.rebuild_model()
    if not isinstance(model, nn.Module):
        raise TypeError(
            "track B applies one shared initialization scheme, which needs a torch module; "
            f"the selected candidate rebuilt as {type(model).__name__}"
        )
    initialize_module_parameters(model, streams.parameter_initialization)

    trainer = SupervisedTrainer(
        recipe=settings.track_b_recipe,
        preprocessor=preprocessor,
        device_for_computation=settings.device_for_computation,
        class_weights=compute_balanced_class_weights(combined_labels, settings.number_of_classes),
    )
    training_result = trainer.train(
        model,
        images=combined_images,
        labels=combined_labels,
        batch_order_generator=streams.batch_order,
        augmentation_generator=streams.augmentation,
    )
    return model, preprocessor, training_result.to_serializable_dict()


@dataclass(frozen=True)
class FrozenModel:
    """One model that is finished changing, with its own preprocessing."""

    model_id: str
    track: str
    model: TrainableModel
    preprocessor: ImagePreprocessor
    checkpoint_sha256: str
    details: dict = field(default_factory=dict)


def freeze_model(
    settings: MulticlassBenchmarkSettings,
    *,
    model: TrainableModel,
    preprocessor: ImagePreprocessor,
    model_id: str,
    track: str,
    genome_kind: str,
    genome_payload: dict,
    metadata: dict,
    artifacts_directory: Path | None,
) -> FrozenModel:
    """Snapshot one model so the reported result is a network that cannot change."""
    checkpoint = capture_model_checkpoint(
        model,
        model_id=model_id,
        stage=track,
        genome_kind=genome_kind,
        genome_payload=genome_payload,
        preprocessing_state=preprocessor.state_dict(),
        recipe=settings.track_b_recipe.to_serializable_dict()
        if track == TRACK_B
        else settings.candidate_recipe.to_serializable_dict(),
        metadata=metadata,
    )
    checkpoint_sha256 = checkpoint.compute_sha256()
    if artifacts_directory is not None:
        checkpoint.write_file(artifacts_directory / "checkpoints" / f"{model_id}.pt")
    return FrozenModel(
        model_id=model_id,
        track=track,
        model=model,
        preprocessor=preprocessor,
        checkpoint_sha256=checkpoint_sha256,
        details=metadata,
    )


def evaluate_on_official_test(
    settings: MulticlassBenchmarkSettings, data: PreparedData, frozen_models: list[FrozenModel]
) -> dict[str, dict]:
    """Score every frozen model on the official test set, once, at the end."""
    results: dict[str, dict] = {}
    for frozen in frozen_models:
        accuracy = validation_top1_accuracy(
            frozen.model,
            images=data.official_test.images,
            labels=data.official_test.labels,
            preprocessor=frozen.preprocessor,
            batch_size=settings.inference_batch_size,
            device_for_computation=settings.device_for_computation,
        )
        results[frozen.model_id] = {
            "track": frozen.track,
            "checkpoint_sha256": frozen.checkpoint_sha256,
            "test_accuracy": accuracy,
            "test_error": 1.0 - accuracy,
            "details": frozen.details,
        }
    return results


def run_multiclass_protocol(
    settings: MulticlassBenchmarkSettings,
    search: SearchCallable,
    *,
    method_name: str,
    load_dataset: DatasetLoader,
    environment: EnvironmentFunction,
    artifacts_directory: Path | None = None,
    lock_sha256: str = "unlocked",
) -> ExperimentReport:
    """Run one method through the whole protocol and report both tracks.

    In ``pilot`` mode the official test is never scored: a pilot exists to
    measure cost and feasibility before the protocol is frozen, and looking at
    the test would spend the one measurement the full run is for.
    """
    experiment_started_at = time.perf_counter()
    if settings.execution.mode == "full" and (
        artifacts_directory is None or lock_sha256 == "unlocked"
    ):
        raise ExecutionLockError("full requires an artifacts directory and a validated lock")
    data = prepare_data(settings, load_dataset=load_dataset)
    if artifacts_directory is not None:
        artifacts_directory.mkdir(parents=True, exist_ok=True)
        report_path = artifacts_directory / "run_report.json"
        if report_path.exists():
            if not settings.execution.resume:
                raise ExecutionLockError("completed run exists; use resume or a new directory")
            previous = json.loads(report_path.read_text(encoding="utf-8"))
            if (
                previous.get("effective_configuration")
                != effective_configuration(settings, method_name)
                or previous.get("dataset_identity_sha256") != data.dataset_identity_sha256
                or previous.get("environment") != environment(settings.device_for_computation)
                or previous.get("protocol_lock_sha256") != lock_sha256
            ):
                raise ExecutionLockError(
                    "completed run differs in configuration, data, environment or lock"
                )
            summary = previous.get("summary", {})
            if "runtime_seconds" not in summary or "number_of_generations" not in summary:
                raise ExecutionLockError(
                    "legacy completed report cannot be resumed; use a new directory"
                )
            return ExperimentReport(**summary)

    selected, track_a_preprocessor = run_search_stage(
        settings,
        data,
        search,
        environment=environment,
        artifacts_directory=artifacts_directory,
        method_name=method_name,
        lock_sha256=lock_sha256,
    )

    frozen_models = [
        freeze_model(
            settings,
            model=selected.track_a_model,
            preprocessor=track_a_preprocessor,
            model_id=f"{method_name}_track_a_seed{settings.search_seed}",
            track=TRACK_A,
            genome_kind=selected.genome_kind,
            genome_payload=selected.genome_payload,
            metadata={
                "selection_fitness": selected.selection_fitness,
                "parameter_count": selected.parameter_count,
                "method": method_name,
            },
            artifacts_directory=artifacts_directory,
        )
    ]
    for retraining_seed in settings.retraining_seeds:
        retrained_model, retrained_preprocessor, training_summary = retrain_topology_for_track_b(
            settings, data, selected, retraining_seed
        )
        frozen_models.append(
            freeze_model(
                settings,
                model=retrained_model,
                preprocessor=retrained_preprocessor,
                model_id=f"{method_name}_track_b_seed{retraining_seed}",
                track=TRACK_B,
                genome_kind=selected.genome_kind,
                genome_payload=selected.genome_payload,
                metadata={
                    "retraining_seed": retraining_seed,
                    "training": training_summary,
                    "method": method_name,
                },
                artifacts_directory=artifacts_directory,
            )
        )

    test_results = (
        {}
        if settings.execution.mode == "pilot"
        else evaluate_on_official_test(settings, data, frozen_models)
    )
    return _build_report(
        settings,
        data,
        selected,
        frozen_models,
        test_results,
        method_name=method_name,
        environment=environment,
        artifacts_directory=artifacts_directory,
        experiment_started_at=experiment_started_at,
        lock_sha256=lock_sha256,
    )


def _build_report(
    settings: MulticlassBenchmarkSettings,
    data: PreparedData,
    selected: SelectedCandidate,
    frozen_models: list[FrozenModel],
    test_results: dict[str, dict],
    *,
    method_name: str,
    environment: EnvironmentFunction,
    artifacts_directory: Path | None,
    experiment_started_at: float,
    lock_sha256: str,
) -> ExperimentReport:
    """Assemble the report from the selection and the (optional) test results."""
    metric_values: dict[str, float] = {
        "validation_accuracy": float(selected.selection_fitness),
        "selected_parameter_count": float(selected.parameter_count),
        "search_runtime_seconds": float(selected.search_seconds),
    }
    for model_id, result in test_results.items():
        prefix = f"{result['track']}_{model_id.rsplit('_', 1)[-1]}"
        metric_values[f"{prefix}_test_accuracy"] = float(result["test_accuracy"])
        metric_values[f"{prefix}_test_error"] = float(result["test_error"])

    track_b_accuracies = [
        result["test_accuracy"] for result in test_results.values() if result["track"] == TRACK_B
    ]
    if track_b_accuracies:
        metric_values["track_b_mean_test_accuracy"] = sum(track_b_accuracies) / len(
            track_b_accuracies
        )

    evaluation_status_counts = count_by_status(list(selected.evaluation_records))
    number_of_evaluations = max(len(selected.evaluation_records), 1)
    metric_values["failed_evaluation_fraction"] = (
        sum(count for status, count in evaluation_status_counts.items() if status != "succeeded")
        / number_of_evaluations
    )

    configuration = effective_configuration(settings, method_name)
    report = ExperimentReport(
        metric_values=metric_values,
        number_of_generations=selected.number_of_generations,
        runtime_seconds=time.perf_counter() - experiment_started_at,
        effective_configuration=configuration,
        artifact_paths={}
        if artifacts_directory is None
        else {
            "run_report": (artifacts_directory / "run_report.json").as_posix(),
            "checkpoints": (artifacts_directory / "checkpoints").as_posix(),
        },
    )
    if artifacts_directory is not None:
        _write_run_report(
            artifacts_directory,
            {
                "method": method_name,
                "mode": settings.execution.mode,
                "protocol_id": settings.protocol_id,
                "dataset_identity_sha256": data.dataset_identity_sha256,
                "protocol_lock_sha256": lock_sha256,
                "environment": environment(settings.device_for_computation),
                "evaluation_status_counts": evaluation_status_counts,
                "evaluation_records": [
                    record.to_serializable_dict() for record in selected.evaluation_records
                ],
                "frozen_models": {
                    frozen.model_id: {
                        "track": frozen.track,
                        "checkpoint_sha256": frozen.checkpoint_sha256,
                        "details": frozen.details,
                    }
                    for frozen in frozen_models
                },
                "official_test": test_results,
                "summary": asdict(report),
                "effective_configuration": configuration,
            },
        )

    return report


def _write_run_report(artifacts_directory: Path, payload: dict) -> None:
    """Write the run report next to the checkpoints."""
    artifacts_directory.mkdir(parents=True, exist_ok=True)
    temporary = artifacts_directory / "run_report.partial"
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    temporary.replace(artifacts_directory / "run_report.json")
