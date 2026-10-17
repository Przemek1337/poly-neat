"""Export what the pneumonia notebook needs from the machine that holds the radiographs.

Invoke as a module from the repository root, on the server::

    uv run --no-sync python -m benchmarks.export_pneumonia_notebook_data --gpu \\
        --series-directory benchmarks/results/pneumonia/series-v2 \\
        --data-directory examples/pediatric_pneumonia/data/kaggle

and copy ``<output-directory>`` (default
``benchmarks/results/pneumonia/notebook-data``) to the workstation. It writes:

- ``official_test_<side>.npz`` - the official test split exactly as the
  protocol scored it (padded, resized, ``uint8``), so saliency maps can be
  computed on the images the models saw;
- ``raw/<relative path>`` - the original JPEGs of a few examples drawn per
  category with a fixed seed, for the dataset and preprocessing figures;
- ``learning_curves/<model id>_search_seed<N>.json`` - per-epoch loss,
  accuracy and AUROC of every stabilized track B retraining;
- ``export_report.json`` - what was written, from which manifest, and how
  closely each recorded retraining reproduced the checkpoint it stands for.

The trainer keeps no per-epoch history, so the curves come from running each
stabilized retraining once more with the same recipe and seeds, measuring the
model at every epoch boundary. The measurement draws from no random stream and
runs in evaluation mode, so it does not change the trajectory; whether the
GPU reproduced the stored weights is recorded rather than assumed, because the
series ran without deterministic algorithms. ``train`` in the curves is the
track B training set (train + search_validation) without augmentation or
dropout; ``validation`` is threshold_validation, which no model was trained or
selected on.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from sklearn.metrics import roc_auc_score
from torch import nn

from benchmarks.pneumonia_notebook import read_predictions, select_examples
from benchmarks.retrain_selected_pneumonia import (
    StabilizedTrainingRecipe,
    build_rebuilder,
    retrain_once,
    settings_from_report,
)
from examples.pediatric_pneumonia._protocol import (
    NUMBER_OF_CLASSES,
    build_preprocessor,
    prepare_data,
)
from examples.pediatric_pneumonia.dataset import load_split_tensors
from polyneat.evaluators.binary_inference import predict_binary_logits
from polyneat.logging_utils.custom_logger import get_logger
from polyneat.training.class_weights import compute_balanced_class_weights
from polyneat.training.image_preprocessing import ImagePreprocessor
from polyneat.training.model_checkpoint import ModelCheckpoint
from polyneat.training.training_recipe import TrainingRecipe

logger = get_logger(__name__)

OFFICIAL_TEST_SPLIT = "official_test"
THRESHOLD_VALIDATION_SPLIT = "threshold_validation"
GALLERY_SPLIT = "train"
GALLERY_SEED = 20251013
GALLERY_COUNTS = {"NORMAL": 4, "bacteria": 2, "virus": 2}


@dataclass(frozen=True)
class EpochObservingRecipe(StabilizedTrainingRecipe):
    """A stabilized recipe that reports every epoch boundary to an observer.

    The trainer asks the recipe for the learning rate once, at the start of
    every epoch, so that call is the epoch boundary: ``epoch_observer(e)`` runs
    after ``e`` epochs have completed and before the next one starts. The
    schedule itself is the parent's, unchanged.

    Attributes:
        epoch_observer: Called with the number of completed epochs.
    """

    epoch_observer: Callable[[int], None] | None = field(default=None, compare=False, repr=False)

    def learning_rate_for_epoch(self, epoch_index: int) -> float:
        """Notify the observer, then return the parent's scheduled rate."""
        if self.epoch_observer is not None:
            self.epoch_observer(epoch_index)
        return super().learning_rate_for_epoch(epoch_index)


@dataclass
class CurveSplit:
    """One split the recorder measures: images, labels and the name in the curves."""

    name: str
    images: torch.Tensor
    labels: torch.Tensor


class LearningCurveRecorder:
    """Measures a model on fixed splits whenever an epoch boundary is reported.

    Measurement runs in evaluation mode under ``no_grad`` and puts the model
    back into training mode afterwards. It draws no random numbers, so the
    training run it observes follows the same trajectory as one without it.
    """

    def __init__(
        self,
        *,
        splits: list[CurveSplit],
        preprocessor: ImagePreprocessor,
        class_weights: torch.Tensor,
        device: torch.device,
        batch_size: int,
    ) -> None:
        self._splits = splits
        self._preprocessor = preprocessor
        self._class_weights = class_weights.to(torch.float64)
        self._device = device
        self._batch_size = batch_size
        self.model: nn.Module | None = None
        self.epochs: list[dict] = []

    def remember(self, build: Callable[[], nn.Module]) -> Callable[[], nn.Module]:
        """Wrap a model builder so the recorder sees the model it builds."""

        def build_and_remember() -> nn.Module:
            self.model = build()
            return self.model

        return build_and_remember

    def observe(self, completed_epochs: int) -> None:
        """Measure every split after ``completed_epochs`` epochs."""
        if self.model is None:
            raise RuntimeError("an epoch boundary was reported before the model was built")
        was_training = self.model.training
        row: dict = {"completed_epochs": completed_epochs}
        for split in self._splits:
            logits = predict_binary_logits(
                self.model,
                images=split.images,
                preprocessor=self._preprocessor,
                batch_size=self._batch_size,
                device_for_computation=self._device,
            )
            labels = split.labels.to(torch.long)
            probabilities = torch.softmax(logits, dim=1)[:, 1].numpy()
            row[split.name] = {
                "loss": float(functional.cross_entropy(logits, labels, weight=self._class_weights)),
                "accuracy": float(np.mean((probabilities >= 0.5) == labels.numpy())),
                "auroc": float(roc_auc_score(labels.numpy(), probabilities)),
            }
        self.epochs.append(row)
        self.model.train(was_training)


def _max_weight_difference(model: nn.Module, stored_state: dict) -> float:
    current = model.state_dict()
    return max(
        float(
            (current[name].detach().cpu().to(torch.float64) - tensor.to(torch.float64)).abs().max()
        )
        for name, tensor in stored_state.items()
        if tensor.numel()
    )


def record_learning_curves(
    run_directory: Path,
    stabilized_run_directory: Path,
    *,
    data_directory: Path,
    device: torch.device,
    output_directory: Path,
    data_cache: dict,
) -> list[dict]:
    """Re-run every stabilized retraining of one search seed with a recorder attached."""
    report = json.loads((run_directory / "run_report.json").read_text(encoding="utf-8"))
    stabilized = json.loads(
        (stabilized_run_directory / "retrain_report.json").read_text(encoding="utf-8")
    )
    settings = settings_from_report(report, data_directory=data_directory, device=device)
    stabilization = {
        name: stabilized["track_b_recipe"][name]
        for name in ("gradient_clipping_max_norm", "warmup_steps", "initialization")
    }
    base_recipe = {
        name: getattr(settings.track_b_recipe, name) for name in TrainingRecipe.__dataclass_fields__
    }
    cache_key = (settings.protocol_id, settings.split_seed, settings.image_side)
    if cache_key not in data_cache:
        data_cache[cache_key] = prepare_data(settings)
    data = data_cache[cache_key]
    if data.manifest_sha256 != stabilized["manifest_sha256"]:
        raise RuntimeError(f"{run_directory}: the data give another manifest than the retraining")

    track_a_id = next(
        model_id
        for model_id, frozen in report["frozen_models"].items()
        if frozen["track"] == "track_a"
    )
    source, _ = ModelCheckpoint.read_file(run_directory / "checkpoints" / f"{track_a_id}.pt")
    rebuild = build_rebuilder(stabilized["method"], source, settings.profile_payload, device)
    combined_images = torch.cat([data.train.images, data.search_validation.images])
    combined_labels = torch.cat([data.train.labels, data.search_validation.labels])
    threshold_split = load_split_tensors(
        data.manifest,
        data_directory=settings.data_directory,
        split_name=THRESHOLD_VALIDATION_SPLIT,
        target_side=settings.image_side,
        manifest_sha256=data.manifest_sha256,
        cache_directory=settings.cache_directory,
    )
    test_split = load_split_tensors(
        data.manifest,
        data_directory=settings.data_directory,
        split_name=OFFICIAL_TEST_SPLIT,
        target_side=settings.image_side,
        manifest_sha256=data.manifest_sha256,
        cache_directory=settings.cache_directory,
    )

    written = []
    for model_id, frozen in sorted(stabilized["frozen_models"].items()):
        retraining_seed = int(frozen["details"]["retraining_seed"])
        recorder = LearningCurveRecorder(
            splits=[
                CurveSplit("train", combined_images, combined_labels),
                CurveSplit("validation", threshold_split.images, threshold_split.labels),
            ],
            preprocessor=build_preprocessor(settings, combined_images),
            class_weights=compute_balanced_class_weights(combined_labels, NUMBER_OF_CLASSES),
            device=device,
            batch_size=settings.inference_batch_size,
        )
        recipe = EpochObservingRecipe(
            **base_recipe, **stabilization, epoch_observer=recorder.observe
        )
        started = time.perf_counter()
        model, preprocessor, training = retrain_once(
            replace(settings, track_b_recipe=recipe),
            data,
            recorder.remember(rebuild),
            retraining_seed,
        )
        recorder.observe(recipe.number_of_epochs)
        seconds = time.perf_counter() - started

        stored_checkpoint, _ = ModelCheckpoint.read_file(
            stabilized_run_directory / "checkpoints" / f"{model_id}.pt"
        )
        stored_predictions = read_predictions(
            stabilized_run_directory / "predictions" / f"{model_id}_official_test.json"
        )
        logits = predict_binary_logits(
            model,
            images=test_split.images,
            preprocessor=preprocessor,
            batch_size=settings.inference_batch_size,
            device_for_computation=device,
        )
        rerun_probabilities = torch.softmax(logits, dim=1)[:, 1].numpy()
        if tuple(test_split.example_ids) != stored_predictions.example_ids:
            raise RuntimeError(f"{model_id}: the test split is not in the stored order")
        reproduction = {
            "max_abs_probability_difference": float(
                np.max(np.abs(rerun_probabilities - stored_predictions.probabilities))
            ),
            "max_abs_weight_difference": _max_weight_difference(
                model, stored_checkpoint.model_state
            ),
            "stored_test_auroc": float(
                roc_auc_score(stored_predictions.labels, stored_predictions.probabilities)
            ),
            "rerun_test_auroc": float(
                roc_auc_score(stored_predictions.labels, rerun_probabilities)
            ),
        }
        payload = {
            "model_id": model_id,
            "method": stabilized["method"],
            "search_seed": int(stabilized["search_seed"]),
            "retraining_seed": retraining_seed,
            "manifest_sha256": data.manifest_sha256,
            "recipe": recipe.to_serializable_dict(),
            "splits": {
                "train": "train + search_validation, no augmentation, evaluation mode",
                "validation": "threshold_validation, evaluation mode",
            },
            "loss": "class-weighted cross-entropy with the training class weights",
            "epochs": recorder.epochs,
            "training": training,
            "reproduction": reproduction,
            "wall_clock_seconds": seconds,
        }
        path = (
            output_directory
            / "learning_curves"
            / f"{model_id}_search_seed{stabilized['search_seed']}.json"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        logger.info(
            "%s seed %s: curves written, test AUROC stored %.4f rerun %.4f, max |dp| %.2e",
            model_id,
            stabilized["search_seed"],
            reproduction["stored_test_auroc"],
            reproduction["rerun_test_auroc"],
            reproduction["max_abs_probability_difference"],
        )
        written.append({"path": str(path), **reproduction})
    return written


def export_images(settings, data, output_directory: Path) -> dict:
    """Write the official test split as scored and copy the gallery's original files."""
    test_split = load_split_tensors(
        data.manifest,
        data_directory=settings.data_directory,
        split_name=OFFICIAL_TEST_SPLIT,
        target_side=settings.image_side,
        manifest_sha256=data.manifest_sha256,
        cache_directory=settings.cache_directory,
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    test_path = output_directory / f"{OFFICIAL_TEST_SPLIT}_{settings.image_side}.npz"
    np.savez_compressed(
        test_path,
        images=test_split.images.numpy(),
        labels=test_split.labels.numpy(),
        example_ids=np.array(test_split.example_ids),
        group_ids=np.array(test_split.group_ids),
        manifest_sha256=np.array(data.manifest_sha256),
    )

    entries = [
        {
            "example_id": entry.example_id,
            "label": entry.label,
            "target_split": entry.target_split,
            "relative_path": entry.relative_path,
        }
        for entry in data.manifest.entries
    ]
    gallery = select_examples(
        entries, split=GALLERY_SPLIT, counts=GALLERY_COUNTS, seed=GALLERY_SEED
    )
    archive_root = settings.data_directory / data.manifest.archive_root_relative_path
    for entry in gallery:
        destination = output_directory / "raw" / entry["relative_path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(archive_root / entry["relative_path"], destination)
    return {
        "official_test": {"path": str(test_path), "rows": len(test_split)},
        "gallery": {"split": GALLERY_SPLIT, "seed": GALLERY_SEED, "examples": gallery},
    }


def main(argument_list: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--series-directory", type=Path, required=True)
    parser.add_argument("--data-directory", type=Path, required=True)
    parser.add_argument(
        "--stabilized-directory",
        type=Path,
        default=None,
        help="default: <series-directory>-stabilized-track-b",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=None,
        help="default: notebook-data next to the series directory",
    )
    parser.add_argument("--method", default="deepneat")
    parser.add_argument("--seeds", nargs="+", type=int, default=None, help="default: every run")
    parser.add_argument("--skip-curves", action="store_true", help="export the images only")
    device_group = parser.add_mutually_exclusive_group()
    device_group.add_argument("--gpu", action="store_true")
    device_group.add_argument("--cpu", action="store_true")
    arguments = parser.parse_args(argument_list)

    if arguments.gpu and not torch.cuda.is_available():
        raise SystemExit("--gpu was requested but CUDA is not available")
    device = torch.device("cuda" if arguments.gpu else "cpu")
    series_directory = arguments.series_directory
    stabilized_directory = arguments.stabilized_directory or series_directory.with_name(
        f"{series_directory.name}-stabilized-track-b"
    )
    output_directory = arguments.output_directory or series_directory.with_name("notebook-data")

    run_directories = [
        run_directory
        for run_directory in sorted((series_directory / "runs" / arguments.method).glob("seed_*"))
        if not arguments.seeds or int(run_directory.name.removeprefix("seed_")) in arguments.seeds
    ]
    if not run_directories:
        raise SystemExit(f"no {arguments.method} runs selected under {series_directory}")

    data_cache: dict = {}
    first_report = json.loads((run_directories[0] / "run_report.json").read_text(encoding="utf-8"))
    settings = settings_from_report(
        first_report, data_directory=arguments.data_directory, device=device
    )
    data = prepare_data(settings)
    if data.manifest_sha256 != first_report["manifest_sha256"]:
        raise SystemExit("the data directory gives another manifest than the series used")
    data_cache[(settings.protocol_id, settings.split_seed, settings.image_side)] = data
    export = {
        "manifest_sha256": data.manifest_sha256,
        "series_directory": str(series_directory),
        "stabilized_directory": str(stabilized_directory),
        **export_images(settings, data, output_directory),
        "learning_curves": [],
    }
    if not arguments.skip_curves:
        for run_directory in run_directories:
            logger.info("Recording learning curves for %s %s", arguments.method, run_directory.name)
            export["learning_curves"].extend(
                record_learning_curves(
                    run_directory,
                    stabilized_directory / arguments.method / run_directory.name,
                    data_directory=arguments.data_directory,
                    device=device,
                    output_directory=output_directory,
                    data_cache=data_cache,
                )
            )
    (output_directory / "export_report.json").write_text(
        json.dumps(export, indent=2, default=str) + "\n", encoding="utf-8"
    )
    logger.info("Notebook data written to %s", output_directory)


if __name__ == "__main__":
    main()
