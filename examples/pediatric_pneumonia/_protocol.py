"""The stage controller: audit, search, both tracks, threshold, test.

This is where the benchmark protocol lives, and deliberately not where any
algorithm lives. The controller knows the order of the stages and what each one
is allowed to see; a method plugs in as one callable that runs a search and
hands back the candidate it selected. DeepNEAT, EXACT, the random search over
DeepNEAT's space and the fixed CNN all reach the same threshold selection and
the same test evaluation through this file.

Stage permissions are enforced by what is passed where, not by convention:

* the search receives train and search_validation and nothing else;
* track B refits preprocessing and class weights on train+search_validation,
  because that is what it trains on;
* threshold_validation is loaded only after a checkpoint is frozen, and only to
  pick its threshold;
* the official test split is loaded last, after every model and threshold is
  already fixed, and never reaches a training or selection call.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path

import torch
from torch import nn

from examples._experiment import ExperimentReport
from examples.pediatric_pneumonia._dataset_audit import (
    audit_pediatric_pneumonia_dataset,
)
from examples.pediatric_pneumonia._dataset_manifest import (
    OFFICIAL_TEST_SPLIT,
    SEARCH_VALIDATION_SPLIT,
    THRESHOLD_VALIDATION_SPLIT,
    TRAIN_SPLIT,
    DatasetManifest,
    build_manifest_from_audit,
)
from examples.pediatric_pneumonia._execution import (
    ExecutionOptions,
    execution_environment,
    validate_execution_lock,
)
from examples.pediatric_pneumonia._protocol_lock import ProtocolLockError
from examples.pediatric_pneumonia.dataset import SplitTensors, load_split_tensors
from polyneat.evaluators.binary_classification_metrics import (
    REFERENCE_THRESHOLD,
    compute_binary_metrics,
)
from polyneat.evaluators.binary_inference import predict_binary
from polyneat.evaluators.binary_predictions import NonFinitePredictionError
from polyneat.evaluators.bootstrap_confidence_intervals import (
    BootstrapConfig,
    bootstrap_metric_confidence_intervals,
)
from polyneat.evaluators.decision_threshold import (
    compute_threshold_binding,
    select_threshold_by_youden_j,
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
NUMBER_OF_CLASSES = 2


@dataclass(frozen=True)
class BenchmarkSettings:
    """Everything the protocol needs that is not the search method itself.

    Attributes:
        data_directory: Directory the archive was extracted into.
        protocol_id: Identifier of this protocol version.
        dataset_release: Identifier of the archive actually used.
        dataset_license: License of that release.
        image_side: Square side every image is brought to.
        split_seed: Seed of the grouped development split.
        search_seed: Root seed of the search and of track A training.
        retraining_seeds: Track B retraining seeds applied to the selected
            topology, reported separately and aggregated within it.
        bootstrap_seed: Seed of the test bootstrap.
        candidate_recipe: How each search candidate is trained in track A.
        track_b_recipe: The one shared recipe every selected topology is
            retrained under.
        inference_batch_size: Rows per forward pass during scoring.
        bootstrap: Frozen bootstrap settings.
        maximum_phenotype_parameters: Candidates above this are rejected.
        search_budget_seconds: Active wall-clock allowance of the search.
            ``None`` leaves the search bounded only by its generation count.
        device_for_computation: The one device every training session and every
            forward pass of this run happens on. Chosen explicitly rather than
            probed, so a run that was meant for the GPU fails loudly instead of
            quietly producing CPU timings under a wall-clock budget.
        uses_augmentation: Whether the training augmentation runs at all.
        uses_identity_standardization: Skip fitting dataset statistics and hand
            the model images in ``[0, 1]``. Set only for a model that brings its
            own normalization, such as the pretrained transfer-learning
            baseline, and recorded in the effective configuration either way.
        cache_directory: Where decoded splits are cached.
    """

    data_directory: Path
    protocol_id: str
    dataset_release: str
    dataset_license: str
    image_side: int
    split_seed: int
    search_seed: int
    retraining_seeds: tuple[int, ...]
    bootstrap_seed: int
    candidate_recipe: TrainingRecipe
    track_b_recipe: TrainingRecipe
    inference_batch_size: int = 32
    bootstrap: BootstrapConfig = field(default_factory=BootstrapConfig)
    maximum_phenotype_parameters: int | None = None
    search_budget_seconds: float | None = None
    device_for_computation: torch.device = field(default_factory=lambda: torch.device("cpu"))
    uses_augmentation: bool = True
    uses_identity_standardization: bool = False
    cache_directory: Path | None = None
    execution: ExecutionOptions = field(default_factory=ExecutionOptions)
    profile_payload: dict = field(default_factory=dict)


@dataclass(frozen=True)
class PreparedData:
    """The frozen split, materialised as tensors for the stages that may see it.

    Attributes:
        manifest: The frozen split every method and seed shares.
        manifest_sha256: Its digest, bound into thresholds and checkpoints.
        train: The training split.
        search_validation: The split fitness and model selection use.
        blocking_findings: Audit findings that must be resolved before a full
            series and pilots. Synthetic smoke runs may proceed and report them.
    """

    manifest: DatasetManifest
    manifest_sha256: str
    train: SplitTensors
    search_validation: SplitTensors
    blocking_findings: tuple[str, ...]


@dataclass(frozen=True)
class SearchContext:
    """What a search method is given, and nothing beyond it.

    Attributes:
        train: Training split for candidate training.
        search_validation: Split every fitness is measured on.
        preprocessor: Preprocessing fitted on train alone.
        class_weights: Class weights of the training split.
        device_for_computation: Device to run on.
        root_seed: Root of every random stream this search derives.
        candidate_recipe: How to train one candidate.
        inference_batch_size: Rows per forward pass when scoring.
        maximum_phenotype_parameters: Candidate parameter budget.
        budget: Wall-clock allowance, already started, or ``None``.
    """

    train: SplitTensors
    search_validation: SplitTensors
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


@dataclass(frozen=True)
class SelectedCandidate:
    """The candidate a search selected, and how to rebuild it from scratch.

    Attributes:
        track_a_model: The exact model that earned the selected fitness, with
            its trained weights. Not a model rebuilt from the genome.
        rebuild_model: Builds the same topology with untrained parameters, for
            track B. Called once per retraining seed.
        genome_kind: Class name of the genome behind the topology.
        genome_payload: The genome as serializable data.
        selection_fitness: Its AUROC on search_validation.
        evaluation_records: Every evaluation the search performed.
        number_of_generations: Generations the search completed.
        search_seconds: Active wall-clock time the search consumed.
        parameter_count: Trainable parameters of the selected topology.
    """

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


def prepare_data(settings: BenchmarkSettings) -> PreparedData:
    """Audit the archive, freeze the split, and load what the search may see.

    The official test split is deliberately not loaded here. It is read once,
    at the end, by :func:`evaluate_on_official_test`, after every model and
    every threshold is already fixed.
    """
    audit = audit_pediatric_pneumonia_dataset(
        settings.data_directory, expected_counts=None, compute_similarity_report=False
    )
    manifest = build_manifest_from_audit(
        audit,
        protocol_id=settings.protocol_id,
        dataset_release=settings.dataset_release,
        dataset_license=settings.dataset_license,
        split_seed=settings.split_seed,
    )
    manifest_sha256 = manifest.compute_sha256()
    if manifest.blocking_findings and settings.execution.mode != "smoke":
        raise ProtocolLockError(
            "Dataset audit blocks this run: "
            + "; ".join(finding.message for finding in manifest.blocking_findings)
        )

    def load(split_name: str) -> SplitTensors:
        return load_split_tensors(
            manifest,
            data_directory=settings.data_directory,
            split_name=split_name,
            target_side=settings.image_side,
            manifest_sha256=manifest_sha256,
            cache_directory=settings.cache_directory,
        )

    return PreparedData(
        manifest=manifest,
        manifest_sha256=manifest_sha256,
        train=load(TRAIN_SPLIT),
        search_validation=load(SEARCH_VALIDATION_SPLIT),
        blocking_findings=tuple(
            f"{finding.kind}: {finding.message}" for finding in manifest.blocking_findings
        ),
    )


def build_preprocessor(
    settings: BenchmarkSettings, training_images: torch.Tensor
) -> ImagePreprocessor:
    """Fit one preprocessor on the training images of one stage.

    Track A calls this with train; track B calls it again with
    train+search_validation. Each stage owns its own statistics, and neither
    ever sees the threshold split or the official test set.
    """
    preprocessor = ImagePreprocessor(
        ImagePreprocessingConfig(target_side=settings.image_side),
        ImageAugmentationConfig() if settings.uses_augmentation else None,
    )
    if settings.uses_identity_standardization:
        preprocessor.use_identity_standardization()
    else:
        preprocessor.fit_standardization(training_images)
    return preprocessor


def run_search_stage(
    settings: BenchmarkSettings,
    data: PreparedData,
    search: SearchCallable,
    *,
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
            manifest_sha256=data.manifest_sha256,
            protocol_lock_sha256=lock_sha256,
            effective_configuration={
                **_effective_configuration(settings, method_name),
                "environment": execution_environment(settings.device_for_computation),
            },
        ),
        budget=budget,
        resume=settings.execution.resume,
        lost_work_seconds=settings.execution.lost_work_seconds,
    )
    context = SearchContext(
        train=data.train,
        search_validation=data.search_validation,
        preprocessor=preprocessor,
        class_weights=compute_balanced_class_weights(data.train.labels, NUMBER_OF_CLASSES),
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
        "Search selected a candidate with search_validation AUROC %.4f after %d generations",
        selected.selection_fitness,
        selected.number_of_generations,
    )
    return selected, preprocessor


def retrain_topology_for_track_b(
    settings: BenchmarkSettings,
    data: PreparedData,
    selected: SelectedCandidate,
    retraining_seed: int,
) -> tuple[TrainableModel, ImagePreprocessor, dict]:
    """Retrain the selected topology from scratch under the shared recipe.

    Track B trains on train+search_validation, so it refits both the
    preprocessing statistics and the class weights on that union. The evolved
    learning recipe is not consulted: the whole point of this track is that
    every topology is trained the same way.
    """
    combined_images = torch.cat([data.train.images, data.search_validation.images])
    combined_labels = torch.cat([data.train.labels, data.search_validation.labels])

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
        class_weights=compute_balanced_class_weights(combined_labels, NUMBER_OF_CLASSES),
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
    """One model that is finished changing, with its own preprocessing and threshold.

    Attributes:
        model_id: Identifier used in prediction files and thresholds.
        track: ``track_a`` or ``track_b``.
        model: The frozen model.
        preprocessor: Its own fitted preprocessing.
        checkpoint_sha256: Digest of its snapshot.
        threshold: Threshold chosen for it on the threshold split.
        details: Extras recorded alongside it.
    """

    model_id: str
    track: str
    model: TrainableModel
    preprocessor: ImagePreprocessor
    checkpoint_sha256: str
    threshold: float
    details: dict = field(default_factory=dict)


def freeze_and_calibrate(
    settings: BenchmarkSettings,
    data: PreparedData,
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
    """Snapshot one model, then pick its threshold on the threshold split.

    The order matters and is enforced here: the checkpoint is taken *first*, so
    the threshold is bound to a model that can no longer change. The threshold
    split is loaded at this point and used for nothing else - it never reaches
    training, early stopping, architecture selection or recipe choice, and no
    model or seed is chosen by comparing scores on it.
    """
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

    threshold_split = load_split_tensors(
        data.manifest,
        data_directory=settings.data_directory,
        split_name=THRESHOLD_VALIDATION_SPLIT,
        target_side=settings.image_side,
        manifest_sha256=data.manifest_sha256,
        cache_directory=settings.cache_directory,
    )
    threshold_predictions = predict_binary(
        model,
        images=threshold_split.images,
        labels=threshold_split.labels,
        example_ids=threshold_split.example_ids,
        group_ids=threshold_split.group_ids,
        preprocessor=preprocessor,
        batch_size=settings.inference_batch_size,
        device_for_computation=settings.device_for_computation,
        model_id=model_id,
        stage=track,
        split_name=THRESHOLD_VALIDATION_SPLIT,
    )
    selected_threshold = select_threshold_by_youden_j(
        threshold_predictions.labels,
        threshold_predictions.positive_class_probabilities,
        model_id=model_id,
        split_name=THRESHOLD_VALIDATION_SPLIT,
        binding_sha256=compute_threshold_binding(
            checkpoint_sha256=checkpoint_sha256,
            preprocessing_state=preprocessor.state_dict(),
            manifest_sha256=data.manifest_sha256,
        ),
    )
    if artifacts_directory is not None:
        threshold_predictions.write_json_file(
            artifacts_directory / "predictions" / f"{model_id}_threshold_validation.json"
        )

    return FrozenModel(
        model_id=model_id,
        track=track,
        model=model,
        preprocessor=preprocessor,
        checkpoint_sha256=checkpoint_sha256,
        threshold=selected_threshold.threshold,
        details={"threshold": selected_threshold.to_serializable_dict(), **metadata},
    )


def evaluate_on_official_test(
    settings: BenchmarkSettings,
    data: PreparedData,
    frozen_models: list[FrozenModel],
    artifacts_directory: Path | None,
) -> dict[str, dict]:
    """Score every frozen model on the official test set, once, at the end.

    Nothing about the run may change after this point. Every model is frozen,
    every threshold is already chosen and bound, and the metrics at the
    reference threshold of 0.5 are reported next to the chosen one because both
    were decided in advance.
    """
    test_split = load_split_tensors(
        data.manifest,
        data_directory=settings.data_directory,
        split_name=OFFICIAL_TEST_SPLIT,
        target_side=settings.image_side,
        manifest_sha256=data.manifest_sha256,
        cache_directory=settings.cache_directory,
    )
    resample_groups = data.manifest.patient_independence_established

    results: dict[str, dict] = {}
    for frozen in frozen_models:
        predictions = predict_binary(
            frozen.model,
            images=test_split.images,
            labels=test_split.labels,
            example_ids=test_split.example_ids,
            group_ids=test_split.group_ids,
            preprocessor=frozen.preprocessor,
            batch_size=settings.inference_batch_size,
            device_for_computation=settings.device_for_computation,
            model_id=frozen.model_id,
            stage=frozen.track,
            split_name=OFFICIAL_TEST_SPLIT,
        )
        if artifacts_directory is not None:
            predictions.write_json_file(
                artifacts_directory / "predictions" / f"{frozen.model_id}_official_test.json"
            )
        results[frozen.model_id] = {
            "track": frozen.track,
            "checkpoint_sha256": frozen.checkpoint_sha256,
            "metrics_at_selected_threshold": compute_binary_metrics(
                predictions.labels,
                predictions.positive_class_probabilities,
                frozen.threshold,
            ).to_serializable_dict(),
            "metrics_at_reference_threshold": compute_binary_metrics(
                predictions.labels,
                predictions.positive_class_probabilities,
                REFERENCE_THRESHOLD,
            ).to_serializable_dict(),
            "confidence_intervals": {
                metric_name: interval.to_serializable_dict()
                for metric_name, interval in bootstrap_metric_confidence_intervals(
                    predictions,
                    threshold=frozen.threshold,
                    config=settings.bootstrap,
                    random_seed=settings.bootstrap_seed,
                    resample_groups=resample_groups,
                ).items()
            },
            "details": frozen.details,
        }
    return results


def run_pneumonia_protocol(
    settings: BenchmarkSettings,
    search: SearchCallable,
    *,
    method_name: str,
    artifacts_directory: Path | None = None,
) -> ExperimentReport:
    """Run one method through the whole protocol and report both tracks.

    Args:
        settings: Everything frozen before the run.
        search: The method's search, receiving only train and
            search_validation.
        method_name: Label recorded in the report and in artifact names.
        artifacts_directory: Where checkpoints, predictions and the manifest
            are written. ``None`` runs without artifacts, which is fine for a
            test but not for a result seed.

    Returns:
        An :class:`~examples._experiment.ExperimentReport` whose metrics are
        the test AUROC of both tracks, and whose extra fields carry the
        statuses, the artifact paths and the effective configuration.
    """
    experiment_started_at = time.perf_counter()
    if settings.execution.mode == "full" and artifacts_directory is None:
        raise ProtocolLockError("full requires an artifacts directory")
    data = prepare_data(settings)
    lock_sha256 = "unlocked"
    if settings.execution.mode == "full":
        assert settings.execution.protocol_lock_path is not None
        lock_sha256 = validate_execution_lock(
            settings.execution.protocol_lock_path,
            method=method_name,
            profile=settings.profile_payload,
            search_seed=settings.search_seed,
            device=settings.device_for_computation,
            manifest=data.manifest,
        )
    if settings.execution.resume and artifacts_directory is not None:
        report_path = artifacts_directory / "run_report.json"
        if report_path.exists():
            previous = json.loads(report_path.read_text(encoding="utf-8"))
            if (
                previous["effective_configuration"]
                != _effective_configuration(settings, method_name)
                or previous["manifest_sha256"] != data.manifest_sha256
                or previous.get("environment")
                != execution_environment(settings.device_for_computation)
            ):
                raise ProtocolLockError("completed run does not match the requested execution")
            if "summary" in previous:
                return ExperimentReport(**previous["summary"])
    if artifacts_directory is not None:
        artifacts_directory.mkdir(parents=True, exist_ok=True)
        data.manifest.write_json_file(artifacts_directory / "manifest.json")
    if data.blocking_findings:
        logger.warning(
            "The audit reported %d blocking findings; a full result series must resolve them "
            "before running: %s",
            len(data.blocking_findings),
            "; ".join(data.blocking_findings[:3]),
        )

    selected, track_a_preprocessor = run_search_stage(
        settings,
        data,
        search,
        artifacts_directory=artifacts_directory,
        method_name=method_name,
        lock_sha256=lock_sha256,
    )

    if settings.execution.mode == "pilot":
        # Pilot can inspect only search validation. Neither threshold nor test
        # tensors are loaded, and no threshold-dependent decisions are made.
        if artifacts_directory is not None:
            capture_model_checkpoint(
                selected.track_a_model,
                model_id=f"{method_name}_pilot",
                stage="pilot",
                genome_kind=selected.genome_kind,
                genome_payload=selected.genome_payload,
                preprocessing_state=track_a_preprocessor.state_dict(),
            ).write_file(artifacts_directory / "checkpoints" / "pilot_selected.pt")
        return _build_report(
            settings,
            data,
            selected,
            [],
            {},
            method_name=method_name,
            artifacts_directory=artifacts_directory,
            experiment_started_at=experiment_started_at,
        )

    frozen_models = [
        freeze_and_calibrate(
            settings,
            data,
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
    failed_retrainings: list[dict] = []
    for retraining_seed in settings.retraining_seeds:
        model_id = f"{method_name}_track_b_seed{retraining_seed}"
        retrained_model, retrained_preprocessor, training_summary = retrain_topology_for_track_b(
            settings, data, selected, retraining_seed
        )
        try:
            frozen_models.append(
                freeze_and_calibrate(
                    settings,
                    data,
                    model=retrained_model,
                    preprocessor=retrained_preprocessor,
                    model_id=model_id,
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
        except NonFinitePredictionError as error:
            # A retraining that diverges under the shared recipe is a result about
            # the topology, not a reason to discard the run: it is recorded as a
            # failed retraining, and track A and the other retrainings are still
            # scored. Dropping it silently would bias track B towards survivors.
            logger.warning(
                "Track B retraining %s diverged and is recorded as failed: %s", model_id, error
            )
            failed_retrainings.append(
                {
                    "model_id": model_id,
                    "retraining_seed": retraining_seed,
                    "reason": str(error),
                    "training": training_summary,
                }
            )

    test_results = evaluate_on_official_test(settings, data, frozen_models, artifacts_directory)
    return _build_report(
        settings,
        data,
        selected,
        frozen_models,
        test_results,
        method_name=method_name,
        artifacts_directory=artifacts_directory,
        experiment_started_at=experiment_started_at,
        failed_retrainings=failed_retrainings,
    )


def _build_report(
    settings: BenchmarkSettings,
    data: PreparedData,
    selected: SelectedCandidate,
    frozen_models: list[FrozenModel],
    test_results: dict[str, dict],
    *,
    method_name: str,
    artifacts_directory: Path | None,
    experiment_started_at: float,
    failed_retrainings: list[dict] | None = None,
) -> ExperimentReport:
    """Assemble the report, keeping undefined metrics out of the scalar table."""
    failed_retrainings = [] if failed_retrainings is None else failed_retrainings
    metric_values: dict[str, float] = {
        "search_validation_auroc": float(selected.selection_fitness),
        "selected_parameter_count": float(selected.parameter_count),
        "search_runtime_seconds": float(selected.search_seconds),
    }
    undefined_metrics: dict[str, str] = {}

    for model_id, result in test_results.items():
        metrics = result["metrics_at_selected_threshold"]
        prefix = f"{result['track']}_{model_id.rsplit('_', 1)[-1]}"
        for metric_name in ("auroc", "average_precision", "sensitivity", "specificity"):
            value = metrics[metric_name]
            if value is None:
                undefined_metrics[f"{prefix}_test_{metric_name}"] = metrics[
                    "undefined_reasons"
                ].get(metric_name, "undefined")
            else:
                metric_values[f"{prefix}_test_{metric_name}"] = float(value)

    track_b_aurocs = [
        result["metrics_at_selected_threshold"]["auroc"]
        for result in test_results.values()
        if result["track"] == TRACK_B
        and result["metrics_at_selected_threshold"]["auroc"] is not None
    ]
    if track_b_aurocs:
        # Retrainings of one topology are aggregated within that topology; they
        # are repeats of one search, not independent searches.
        metric_values["track_b_mean_test_auroc"] = sum(track_b_aurocs) / len(track_b_aurocs)
    if frozen_models:
        # Reported next to the mean so a mean over fewer retrainings is visible.
        metric_values["track_b_failed_retraining_count"] = float(len(failed_retrainings))

    evaluation_status_counts = count_by_status(list(selected.evaluation_records))
    number_of_evaluations = max(len(selected.evaluation_records), 1)
    metric_values["failed_evaluation_fraction"] = (
        sum(count for status, count in evaluation_status_counts.items() if status != "succeeded")
        / number_of_evaluations
    )

    if artifacts_directory is not None:
        _write_run_report(
            artifacts_directory,
            {
                "method": method_name,
                "protocol_id": settings.protocol_id,
                "manifest_sha256": data.manifest_sha256,
                "blocking_audit_findings": list(data.blocking_findings),
                "patient_independence_established": (
                    data.manifest.patient_independence_established
                ),
                "evaluation_status_counts": evaluation_status_counts,
                "evaluation_records": [
                    record.to_serializable_dict() for record in selected.evaluation_records
                ],
                "frozen_models": {
                    frozen.model_id: {
                        "track": frozen.track,
                        "checkpoint_sha256": frozen.checkpoint_sha256,
                        "threshold": frozen.threshold,
                        "details": frozen.details,
                    }
                    for frozen in frozen_models
                },
                "official_test": test_results,
                "track_b_failed_retrainings": failed_retrainings,
                "effective_configuration": _effective_configuration(settings, method_name),
                "environment": execution_environment(settings.device_for_computation),
                "summary": {
                    "metric_values": metric_values,
                    "number_of_generations": selected.number_of_generations,
                    "runtime_seconds": time.perf_counter() - experiment_started_at,
                    "effective_configuration": _effective_configuration(settings, method_name),
                    "undefined_metrics": undefined_metrics,
                    "artifact_paths": {"run_report": str(artifacts_directory / "run_report.json")},
                },
            },
        )

    return ExperimentReport(
        metric_values=metric_values,
        number_of_generations=selected.number_of_generations,
        runtime_seconds=time.perf_counter() - experiment_started_at,
        effective_configuration=_effective_configuration(settings, method_name),
        artifact_paths=(
            {}
            if artifacts_directory is None
            else {
                "manifest": (artifacts_directory / "manifest.json").as_posix(),
                "run_report": (artifacts_directory / "run_report.json").as_posix(),
                "checkpoints": (artifacts_directory / "checkpoints").as_posix(),
                "predictions": (artifacts_directory / "predictions").as_posix(),
            }
        ),
        undefined_metrics=undefined_metrics,
    )


def _effective_configuration(settings: BenchmarkSettings, method_name: str) -> dict:
    """The configuration this run actually used, after every override."""
    return {
        "method": method_name,
        "mode": settings.execution.mode,
        "profile_payload": settings.profile_payload,
        "protocol_id": settings.protocol_id,
        "dataset_release": settings.dataset_release,
        "dataset_license": settings.dataset_license,
        "image_side": settings.image_side,
        "split_seed": settings.split_seed,
        "search_seed": settings.search_seed,
        "retraining_seeds": list(settings.retraining_seeds),
        "bootstrap_seed": settings.bootstrap_seed,
        "candidate_recipe": settings.candidate_recipe.to_serializable_dict(),
        "track_b_recipe": settings.track_b_recipe.to_serializable_dict(),
        "inference_batch_size": settings.inference_batch_size,
        "bootstrap_replicates": settings.bootstrap.number_of_replicates,
        "maximum_bootstrap_attempts": settings.bootstrap.maximum_attempts,
        "confidence_level": settings.bootstrap.confidence_level,
        "maximum_phenotype_parameters": settings.maximum_phenotype_parameters,
        "search_budget_seconds": settings.search_budget_seconds,
        "device_for_computation": str(settings.device_for_computation),
        "uses_augmentation": settings.uses_augmentation,
        "uses_identity_standardization": settings.uses_identity_standardization,
    }


def _write_run_report(artifacts_directory: Path, payload: dict) -> None:
    """Write the run report next to the checkpoints and predictions."""
    artifacts_directory.mkdir(parents=True, exist_ok=True)
    temporary = artifacts_directory / "run_report.json.partial"
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    temporary.replace(artifacts_directory / "run_report.json")
