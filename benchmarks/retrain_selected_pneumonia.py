"""Retrain the topologies a finished pneumonia series selected, under a stabilized track B recipe.

Invoke as a module from the repository root, on the machine that holds the data::

    uv run --no-sync python -m benchmarks.retrain_selected_pneumonia --gpu \\
        --series-directory benchmarks/results/pneumonia/series-v2 \\
        --data-directory examples/pediatric_pneumonia/data/kaggle

In the frozen series, track B rebuilt each selected topology, initialized it
with the shared He-normal scheme and trained it with plain SGD (momentum 0.9,
no warm-up, no gradient clipping). Deep DeepNEAT graphs without batch
normalization, with dropout and max-pooled concatenations, start from logits
an order of magnitude larger under that scheme than under the initialization
they were evolved with; their first updates are then large enough that the
weights overflow to NaN or every rectifier dies and the network outputs a
constant. That is a property of the recipe meeting those graphs, not of what
the search found.

This module repeats track B for the selected topologies with the same split,
preprocessing, class weights, random streams, epochs, schedule, threshold
selection, official test and bootstrap as the series, and changes only the
optimizer step:

- the global gradient norm is clipped to ``--gradient-clipping-max-norm``;
- the learning rate rises linearly from zero over ``--warmup-steps`` steps,
  then follows the series' schedule unchanged;
- a step with a non-finite gradient norm stops the retraining at once and
  records it as diverged, instead of training on for the remaining epochs.

The initialization stays the shared track B scheme unless
``--initialization search`` asks for the one the search trained candidates
with (PyTorch defaults times the evolved weight-scaling genes).

Each run of the series gets ``<output-directory>/<method>/seed_<N>/`` with the
retrained checkpoints, the threshold and test predictions and a
``retrain_report.json``. Track A is not touched: it is the model the search
selected and its result in the series stands.

This is an analysis made after the official test of the series was seen, so
it has to be reported as such, next to the series rather than in place of it.
It sits outside the digested implementation and never writes into the series
directories, so it cannot invalidate the frozen series.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

import torch
from torch import nn

from examples.pediatric_pneumonia._execution import ExecutionOptions
from examples.pediatric_pneumonia._profiles import _recipe_from_payload
from examples.pediatric_pneumonia._protocol import (
    NUMBER_OF_CLASSES,
    TRACK_B,
    BenchmarkSettings,
    FrozenModel,
    PreparedData,
    build_preprocessor,
    evaluate_on_official_test,
    freeze_and_calibrate,
    prepare_data,
)
from polyneat.algorithms.deepneat.deepneat_algorithm import DeepNEATAlgorithm
from polyneat.algorithms.deepneat.deepneat_genome import DeepNEATGenome
from polyneat.configs.deepneat.deepneat_config import DeepNEATConfig
from polyneat.evaluators.binary_predictions import NonFinitePredictionError
from polyneat.evaluators.bootstrap_confidence_intervals import BootstrapConfig
from polyneat.logging_utils.custom_logger import get_logger
from polyneat.nn.fixed_convolutional_network import (
    FixedConvolutionalNetwork,
    FixedConvolutionalNetworkConfig,
)
from polyneat.training.class_weights import compute_balanced_class_weights
from polyneat.training.model_checkpoint import ModelCheckpoint
from polyneat.training.parameter_initialization import initialize_module_parameters
from polyneat.training.random_streams import TrainingRandomStreams
from polyneat.training.supervised_trainer import SupervisedTrainer
from polyneat.training.training_recipe import TrainingRecipe

logger = get_logger(__name__)

SUPPORTED_METHODS = ("deepneat", "random_search", "fixed_cnn")
SHARED_INITIALIZATION = "shared"
SEARCH_INITIALIZATION = "search"


class TrainingDivergedError(RuntimeError):
    """Raised when a step's gradient norm is no longer a finite number."""

    def __init__(self, optimizer_step: int, gradient_norm: float) -> None:
        super().__init__(f"gradient norm became {gradient_norm} at optimizer step {optimizer_step}")
        self.optimizer_step = optimizer_step
        self.gradient_norm = gradient_norm


class StabilizedSGD(torch.optim.SGD):
    """SGD that clips the global gradient norm and warms the learning rate up.

    The trainer sets each parameter group's learning rate at the start of every
    epoch; the warm-up factor is applied on top of that inside :meth:`step` and
    removed again afterwards, so the schedule itself is left as it was.

    Args:
        parameters: Parameters to optimize.
        gradient_clipping_max_norm: Largest global gradient norm a step may
            apply; ``None`` measures the norm without clipping.
        warmup_steps: Steps over which the learning rate rises linearly from
            ``1 / warmup_steps`` of its scheduled value to all of it; 0 disables.
        **sgd_arguments: Passed to :class:`torch.optim.SGD`.
    """

    def __init__(
        self,
        parameters,
        *,
        gradient_clipping_max_norm: float | None,
        warmup_steps: int,
        **sgd_arguments,
    ) -> None:
        super().__init__(parameters, **sgd_arguments)
        self._gradient_clipping_max_norm = gradient_clipping_max_norm
        self._warmup_steps = warmup_steps
        self.completed_steps = 0

    @torch.no_grad()
    def step(self, closure=None):
        """Clip, check, warm up and step.

        Raises:
            TrainingDivergedError: If the global gradient norm is not finite.
        """
        parameters = [
            parameter
            for group in self.param_groups
            for parameter in group["params"]
            if parameter.grad is not None
        ]
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            parameters,
            math.inf
            if self._gradient_clipping_max_norm is None
            else self._gradient_clipping_max_norm,
        )
        if not math.isfinite(float(gradient_norm)):
            raise TrainingDivergedError(self.completed_steps, float(gradient_norm))
        warmup_factor = (
            min(1.0, (self.completed_steps + 1) / self._warmup_steps)
            if self._warmup_steps > 0
            else 1.0
        )
        scheduled_rates = [group["lr"] for group in self.param_groups]
        for group in self.param_groups:
            group["lr"] *= warmup_factor
        loss = super().step(closure)
        for group, scheduled_rate in zip(self.param_groups, scheduled_rates, strict=True):
            group["lr"] = scheduled_rate
        self.completed_steps += 1
        return loss


@dataclass(frozen=True)
class StabilizedTrainingRecipe(TrainingRecipe):
    """The series' track B recipe with a clipped, warmed-up SGD step.

    Attributes:
        gradient_clipping_max_norm: See :class:`StabilizedSGD`.
        warmup_steps: See :class:`StabilizedSGD`.
        initialization: ``shared`` (track B's He-normal scheme) or ``search``.
    """

    gradient_clipping_max_norm: float | None = 1.0
    warmup_steps: int = 0
    initialization: str = SHARED_INITIALIZATION

    def build_optimizer(self, parameters: list[torch.nn.Parameter]) -> torch.optim.Optimizer:
        """A fresh :class:`StabilizedSGD`; the series' recipes are all SGD."""
        return StabilizedSGD(
            parameters,
            gradient_clipping_max_norm=self.gradient_clipping_max_norm,
            warmup_steps=self.warmup_steps,
            lr=self.learning_rate,
            momentum=self.momentum,
            nesterov=self.uses_nesterov_momentum and self.momentum > 0.0,
            weight_decay=self.weight_decay,
        )

    def to_serializable_dict(self) -> dict:
        """The base recipe plus the stabilization, for the report."""
        return {
            **super().to_serializable_dict(),
            "gradient_clipping_max_norm": self.gradient_clipping_max_norm,
            "warmup_steps": self.warmup_steps,
            "initialization": self.initialization,
        }


def settings_from_report(
    report: dict, *, data_directory: Path, device: torch.device
) -> BenchmarkSettings:
    """Rebuild the series' settings from the configuration its report recorded."""
    configuration = report["effective_configuration"]
    profile = configuration["profile_payload"]
    protocol = profile["protocol"]
    return BenchmarkSettings(
        data_directory=data_directory,
        protocol_id=configuration["protocol_id"],
        dataset_release=configuration["dataset_release"],
        dataset_license=configuration["dataset_license"],
        image_side=int(configuration["image_side"]),
        split_seed=int(configuration["split_seed"]),
        search_seed=int(configuration["search_seed"]),
        retraining_seeds=tuple(int(seed) for seed in configuration["retraining_seeds"]),
        bootstrap_seed=int(configuration["bootstrap_seed"]),
        candidate_recipe=_recipe_from_payload(profile["candidate_recipe"]),
        track_b_recipe=_recipe_from_payload(profile["track_b_recipe"]),
        inference_batch_size=int(configuration["inference_batch_size"]),
        bootstrap=BootstrapConfig(
            number_of_replicates=int(configuration["bootstrap_replicates"]),
            maximum_attempts=int(configuration["maximum_bootstrap_attempts"]),
            confidence_level=float(configuration["confidence_level"]),
        ),
        maximum_phenotype_parameters=configuration["maximum_phenotype_parameters"],
        device_for_computation=device,
        uses_augmentation=bool(protocol.get("uses_augmentation", True)),
        uses_identity_standardization=bool(protocol.get("uses_identity_standardization", False)),
        cache_directory=data_directory.parent / "cache",
        execution=ExecutionOptions(mode="pilot"),
        profile_payload=profile,
    )


def build_rebuilder(
    method: str, checkpoint: ModelCheckpoint, profile: dict, device: torch.device
) -> Callable[[], nn.Module]:
    """How the series rebuilt this method's selected topology with fresh parameters."""
    if checkpoint.genome_kind == "DeepNEATGenome":
        decoder = DeepNEATAlgorithm.from_config(
            DeepNEATConfig.from_dict(profile["algorithm"]),
            device_for_phenotype_computation=device,
        ).phenotype_decoder
        genome = DeepNEATGenome.from_serializable_dict(checkpoint.genome_payload)
        return lambda: decoder.build_phenotype_from_genome(genome)
    if checkpoint.genome_kind == "FixedConvolutionalNetworkConfig":
        architecture = profile["architecture"]
        config = FixedConvolutionalNetworkConfig(
            input_channels=1,
            number_of_classes=NUMBER_OF_CLASSES,
            convolution_channels=tuple(architecture["convolution_channels"]),
            kernel_size=int(architecture["kernel_size"]),
            uses_batch_normalization=bool(architecture["uses_batch_normalization"]),
            dropout_probability=float(architecture["dropout_probability"]),
        )
        return lambda: FixedConvolutionalNetwork(config)
    raise ValueError(f"{method}: cannot rebuild a {checkpoint.genome_kind}")


def _initialize(model: nn.Module, recipe: StabilizedTrainingRecipe, streams) -> None:
    if recipe.initialization == SHARED_INITIALIZATION:
        initialize_module_parameters(model, streams.parameter_initialization)
        return
    # The search's scheme: every layer's own reset, then the evolved scaling.
    seed = int(torch.randint(0, 2**62, (1,), generator=streams.parameter_initialization))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        reinitialize = getattr(model, "reinitialize_parameters", None)
        if callable(reinitialize):
            reinitialize()
        else:
            for module in model.modules():
                if module is not model and callable(getattr(module, "reset_parameters", None)):
                    module.reset_parameters()


def retrain_once(
    settings: BenchmarkSettings,
    data: PreparedData,
    rebuild: Callable[[], nn.Module],
    retraining_seed: int,
) -> tuple[nn.Module, object, dict]:
    """One track B retraining, as the series does it, with the stabilized step.

    Raises:
        TrainingDivergedError: If the gradient norm stops being finite.
    """
    combined_images = torch.cat([data.train.images, data.search_validation.images])
    combined_labels = torch.cat([data.train.labels, data.search_validation.labels])
    preprocessor = build_preprocessor(settings, combined_images)
    streams = TrainingRandomStreams.derive(
        root_seed=retraining_seed, evaluation_id="selected_topology", track=TRACK_B
    )
    model = rebuild()
    recipe = settings.track_b_recipe
    assert isinstance(recipe, StabilizedTrainingRecipe)
    _initialize(model, recipe, streams)
    trainer = SupervisedTrainer(
        recipe=recipe,
        preprocessor=preprocessor,
        device_for_computation=settings.device_for_computation,
        class_weights=compute_balanced_class_weights(combined_labels, NUMBER_OF_CLASSES),
    )
    result = trainer.train(
        model,
        images=combined_images,
        labels=combined_labels,
        batch_order_generator=streams.batch_order,
        augmentation_generator=streams.augmentation,
    )
    return model, preprocessor, result.to_serializable_dict()


def _parameter_count(model: nn.Module) -> int:
    return sum(int(parameter.numel()) for parameter in model.parameters())


def retrain_run(
    run_directory: Path,
    *,
    method: str,
    data_directory: Path,
    device: torch.device,
    output_directory: Path,
    stabilization: dict,
    data_cache: dict,
) -> dict:
    """Retrain one run's selected topology for every retraining seed of the series."""
    report = json.loads((run_directory / "run_report.json").read_text(encoding="utf-8"))
    settings = settings_from_report(report, data_directory=data_directory, device=device)
    settings = replace(
        settings,
        track_b_recipe=StabilizedTrainingRecipe(
            **{
                name: getattr(settings.track_b_recipe, name)
                for name in TrainingRecipe.__dataclass_fields__
            },
            **stabilization,
        ),
    )
    cache_key = (settings.protocol_id, settings.split_seed, settings.image_side)
    if cache_key not in data_cache:
        data_cache[cache_key] = prepare_data(settings)
    data = data_cache[cache_key]
    if data.manifest_sha256 != report["manifest_sha256"]:
        raise RuntimeError(
            f"{run_directory}: the archive gives manifest {data.manifest_sha256[:12]}, the series "
            f"used {report['manifest_sha256'][:12]}; this is not the same split"
        )

    track_a_id = next(
        model_id
        for model_id, frozen in report["frozen_models"].items()
        if frozen["track"] == "track_a"
    )
    checkpoint, checkpoint_sha256 = ModelCheckpoint.read_file(
        run_directory / "checkpoints" / f"{track_a_id}.pt"
    )
    rebuild = build_rebuilder(method, checkpoint, settings.profile_payload, device)
    expected_parameters = report["frozen_models"][track_a_id]["details"]["parameter_count"]
    if _parameter_count(rebuild()) != expected_parameters:
        raise RuntimeError(
            f"{run_directory}: rebuilt topology has {_parameter_count(rebuild())} parameters, "
            f"the series selected one with {expected_parameters}"
        )

    seed = run_directory.name.removeprefix("seed_")
    artifacts_directory = output_directory / method / run_directory.name
    frozen_models: list[FrozenModel] = []
    failed: list[dict] = []
    for retraining_seed in settings.retraining_seeds:
        model_id = f"{method}_track_b_stable_seed{retraining_seed}"
        try:
            model, preprocessor, training = retrain_once(settings, data, rebuild, retraining_seed)
            frozen_models.append(
                freeze_and_calibrate(
                    settings,
                    data,
                    model=model,
                    preprocessor=preprocessor,
                    model_id=model_id,
                    track=TRACK_B,
                    genome_kind=checkpoint.genome_kind,
                    genome_payload=checkpoint.genome_payload,
                    metadata={
                        "retraining_seed": retraining_seed,
                        "training": training,
                        "method": method,
                    },
                    artifacts_directory=artifacts_directory,
                )
            )
        except (TrainingDivergedError, NonFinitePredictionError) as error:
            logger.warning("%s seed %s: %s diverged: %s", method, seed, model_id, error)
            failed.append(
                {"model_id": model_id, "retraining_seed": retraining_seed, "reason": str(error)}
            )

    official_test = evaluate_on_official_test(settings, data, frozen_models, artifacts_directory)
    summary = {
        "method": method,
        "search_seed": int(seed),
        "source_run": str(run_directory),
        "source_checkpoint_sha256": checkpoint_sha256,
        "manifest_sha256": data.manifest_sha256,
        "track_b_recipe": settings.track_b_recipe.to_serializable_dict(),
        "series_track_b_failed_retrainings": len(report.get("track_b_failed_retrainings", [])),
        "series_track_b_test_auroc": [
            entry["metrics_at_reference_threshold"]["auroc"]
            for entry in report["official_test"].values()
            if entry["track"] == "track_b"
        ],
        "frozen_models": {
            frozen.model_id: {"threshold": frozen.threshold, "details": frozen.details}
            for frozen in frozen_models
        },
        "official_test": official_test,
        "failed_retrainings": failed,
    }
    artifacts_directory.mkdir(parents=True, exist_ok=True)
    (artifacts_directory / "retrain_report.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    return summary


def _mean(values: list[float]) -> str:
    return f"{sum(values) / len(values):.4f}" if values else "-"


def summary_table(summaries: list[dict]) -> str:
    """Series track B against the stabilized retraining, one row per run."""
    lines = [
        "| method | seed | series track B AUROC (mean) | series failed | stabilized AUROC "
        "per retraining | stabilized mean | stabilized failed |",
        "|---|---|---|---|---|---|---|",
    ]
    for summary in summaries:
        aurocs = [
            entry["metrics_at_reference_threshold"]["auroc"]
            for entry in summary["official_test"].values()
            if entry["metrics_at_reference_threshold"]["auroc"] is not None
        ]
        lines.append(
            f"| {summary['method']} | {summary['search_seed']} "
            f"| {_mean(summary['series_track_b_test_auroc'])} "
            f"| {summary['series_track_b_failed_retrainings']} "
            f"| {', '.join(f'{value:.4f}' for value in aurocs) or '-'} "
            f"| {_mean(aurocs)} | {len(summary['failed_retrainings'])} |"
        )
    return "\n".join(lines)


def main(argument_list: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--series-directory", type=Path, required=True)
    parser.add_argument("--data-directory", type=Path, required=True)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=None,
        help="default: <series-directory>-stabilized-track-b",
    )
    parser.add_argument("--methods", nargs="+", default=["deepneat"], choices=SUPPORTED_METHODS)
    parser.add_argument("--seeds", nargs="+", type=int, default=None, help="default: every run")
    parser.add_argument("--gradient-clipping-max-norm", type=float, default=1.0)
    parser.add_argument("--warmup-steps", type=int, default=70)
    parser.add_argument(
        "--initialization",
        choices=(SHARED_INITIALIZATION, SEARCH_INITIALIZATION),
        default=SHARED_INITIALIZATION,
    )
    device_group = parser.add_mutually_exclusive_group()
    device_group.add_argument("--gpu", action="store_true")
    device_group.add_argument("--cpu", action="store_true")
    arguments = parser.parse_args(argument_list)

    if arguments.gpu and not torch.cuda.is_available():
        raise SystemExit("--gpu was requested but CUDA is not available")
    device = torch.device("cuda" if arguments.gpu else "cpu")
    series_directory = arguments.series_directory
    output_directory = arguments.output_directory or series_directory.with_name(
        f"{series_directory.name}-stabilized-track-b"
    )
    stabilization = {
        "gradient_clipping_max_norm": arguments.gradient_clipping_max_norm,
        "warmup_steps": arguments.warmup_steps,
        "initialization": arguments.initialization,
    }

    summaries = []
    data_cache: dict = {}
    for method in arguments.methods:
        for run_directory in sorted((series_directory / "runs" / method).glob("seed_*")):
            if arguments.seeds and int(run_directory.name.removeprefix("seed_")) not in set(
                arguments.seeds
            ):
                continue
            logger.info("Retraining %s %s", method, run_directory.name)
            summaries.append(
                retrain_run(
                    run_directory,
                    method=method,
                    data_directory=arguments.data_directory,
                    device=device,
                    output_directory=output_directory,
                    stabilization=stabilization,
                    data_cache=data_cache,
                )
            )

    table = summary_table(summaries)
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "summary.md").write_text(table + "\n", encoding="utf-8")
    logger.info("Stabilized track B against the series:\n%s", table)
    logger.info("Written to %s", output_directory)


if __name__ == "__main__":
    main()
