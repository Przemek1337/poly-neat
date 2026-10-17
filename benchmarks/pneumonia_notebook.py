"""Metrics, saliency maps and example images behind the pediatric pneumonia notebook.

Imported by ``benchmarks/rtg_deepneat.ipynb`` and by
:mod:`benchmarks.export_pneumonia_notebook_data`. It reads frozen checkpoints
and prediction files and never trains, writes a series or touches a lock, so it
cannot change a reported number; it sits outside the digested implementation.

Two saliency methods are provided so the same images can be explained twice,
by mechanisms that share no assumptions:

- Grad-CAM (Selvaraju et al., 2017, *Grad-CAM: Visual Explanations from Deep
  Networks via Gradient-based Localization*, ICCV) weights the feature maps
  that feed the classifier by the spatial mean of their gradient;
- occlusion sensitivity (Zeiler and Fergus, 2014, *Visualizing and
  Understanding Convolutional Networks*, ECCV) hides one patch at a time and
  measures how much the decision weakens, using forward passes only.

Both explain the logit margin ``z_c - z_other`` of the explained class ``c``
rather than its softmax probability: for two classes the probability is a
sigmoid of that margin, and a confident prediction saturates the sigmoid, which
would leave occlusion with nothing to measure.

Not every DeepNEAT graph is a chain. Several feature maps of different
resolutions may be max-pooled, concatenated and flattened into the classifier,
and one input may even skip every convolution. Grad-CAM is therefore computed
on each convolutional map feeding the flatten, at that map's own resolution,
and the maps are added after upsampling, each weighted by its number of
positions so that every branch contributes in the same units (a first-order
share of the logit margin). A direct skip from the image is not a feature map
and is left out; the notebook says so where it applies.
"""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn

from benchmarks.describe_winners import ArchitectureDescription, render_matplotlib
from benchmarks.retrain_selected_pneumonia import build_rebuilder
from polyneat.training.image_preprocessing import (
    ImageAugmentationConfig,
    ImagePreprocessingConfig,
    ImagePreprocessor,
    apply_affine_and_contrast_augmentation,
    pad_to_square,
    resize_images,
)
from polyneat.training.model_checkpoint import ModelCheckpoint
from polyneat.training.parameter_initialization import initialize_module_parameters

CLASS_NAMES = ("NORMAL", "PNEUMONIA")
POSITIVE_CLASS = 1
NORMAL_CATEGORY = "NORMAL"
BACTERIAL_CATEGORY = "bacteria"
VIRAL_CATEGORY = "virus"


@dataclass(frozen=True)
class PredictionSet:
    """One model's scores on one split, as the protocol exported them.

    Attributes:
        model_id: Model the scores belong to.
        split_name: ``threshold_validation`` or ``official_test``.
        example_ids: Manifest ids, aligned with the arrays below.
        labels: ``0`` for NORMAL, ``1`` for PNEUMONIA.
        probabilities: Positive-class probability of every row.
    """

    model_id: str
    split_name: str
    example_ids: tuple[str, ...]
    labels: np.ndarray
    probabilities: np.ndarray


def read_predictions(path: Path) -> PredictionSet:
    """Read a ``*_official_test.json`` or ``*_threshold_validation.json`` file."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload["records"]
    return PredictionSet(
        model_id=payload["model_id"],
        split_name=payload["split_name"],
        example_ids=tuple(record["example_id"] for record in records),
        labels=np.array([int(record["label"]) for record in records], dtype=np.int64),
        probabilities=np.array(
            [float(record["positive_class_probability"]) for record in records], dtype=np.float64
        ),
    )


def _ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def extended_metrics(
    labels: np.ndarray, probabilities: np.ndarray, threshold: float
) -> dict[str, float | int | None]:
    """Every metric of Salamon and Ksiazek (2025, *J. Clin. Med.* 14, 7212), plus a few more.

    A row is positive when ``probability >= threshold``, the protocol's
    convention. A ratio whose denominator is zero is ``None`` rather than 0.

    Args:
        labels: ``(n,)`` ground truth, ``1`` = PNEUMONIA.
        probabilities: ``(n,)`` positive-class probabilities.
        threshold: Decision threshold.

    Returns:
        Confusion counts, accuracy, sensitivity, specificity, PPV, NPV, FPR,
        FNR, F1, MCC (Matthews, 1975), balanced accuracy, ROC AUC, average
        precision (the PR AUC summary scikit-learn computes) and the Brier
        score (Brier, 1950).
    """
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    predicted = probabilities >= threshold
    true_positives = int(np.sum(predicted & (labels == 1)))
    true_negatives = int(np.sum(~predicted & (labels == 0)))
    false_positives = int(np.sum(predicted & (labels == 0)))
    false_negatives = int(np.sum(~predicted & (labels == 1)))
    sensitivity = _ratio(true_positives, true_positives + false_negatives)
    specificity = _ratio(true_negatives, true_negatives + false_positives)
    mcc_denominator = math.sqrt(
        (true_positives + false_positives)
        * (true_positives + false_negatives)
        * (true_negatives + false_positives)
        * (true_negatives + false_negatives)
    )
    return {
        "threshold": float(threshold),
        "true_positives": true_positives,
        "true_negatives": true_negatives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "accuracy": (true_positives + true_negatives) / len(labels),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": _ratio(true_positives, true_positives + false_positives),
        "negative_predictive_value": _ratio(true_negatives, true_negatives + false_negatives),
        "false_positive_rate": _ratio(false_positives, false_positives + true_negatives),
        "false_negative_rate": _ratio(false_negatives, false_negatives + true_positives),
        "f1": _ratio(2 * true_positives, 2 * true_positives + false_positives + false_negatives),
        "mcc": _ratio(
            true_positives * true_negatives - false_positives * false_negatives, mcc_denominator
        ),
        "balanced_accuracy": None
        if sensitivity is None or specificity is None
        else (sensitivity + specificity) / 2,
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "pr_auc": float(average_precision_score(labels, probabilities)),
        "brier": float(np.mean((probabilities - labels) ** 2)),
    }


def calibration_bins(
    labels: np.ndarray, probabilities: np.ndarray, number_of_bins: int = 10
) -> list[dict[str, float | int]]:
    """Equal-width reliability-diagram bins: mean score against observed positive rate.

    Empty bins are left out rather than plotted at zero.
    """
    edges = np.linspace(0.0, 1.0, number_of_bins + 1)
    bin_indices = np.clip(np.digitize(probabilities, edges[1:-1]), 0, number_of_bins - 1)
    bins = []
    for index in range(number_of_bins):
        members = bin_indices == index
        if not np.any(members):
            continue
        bins.append(
            {
                "lower": float(edges[index]),
                "upper": float(edges[index + 1]),
                "count": int(np.sum(members)),
                "mean_probability": float(np.mean(probabilities[members])),
                "positive_rate": float(np.mean(labels[members])),
            }
        )
    return bins


@dataclass(frozen=True)
class LoadedModel:
    """A frozen checkpoint rebuilt into a runnable model with its own preprocessing.

    Attributes:
        model_id: Identifier the checkpoint was written under.
        model: The network in evaluation mode, on the CPU.
        preprocessor: The preprocessing fitted for this model, restored.
        checkpoint: The verified checkpoint it came from.
    """

    model_id: str
    model: nn.Module
    preprocessor: ImagePreprocessor
    checkpoint: ModelCheckpoint


def restore_preprocessor(preprocessing_state: dict) -> ImagePreprocessor:
    """An evaluation-only preprocessor carrying a checkpoint's fitted statistics."""
    preprocessor = ImagePreprocessor(
        ImagePreprocessingConfig(
            target_side=int(preprocessing_state["target_side"]),
            pad_value=float(preprocessing_state["pad_value"]),
            use_antialias=bool(preprocessing_state["use_antialias"]),
            standardization_epsilon=float(preprocessing_state["standardization_epsilon"]),
        )
    )
    preprocessor.load_state_dict(preprocessing_state)
    return preprocessor


def load_checkpoint_model(checkpoint_path: Path, profile_payload: dict) -> LoadedModel:
    """Rebuild the topology of a checkpoint and load its weights and preprocessing.

    Args:
        checkpoint_path: A track A or track B ``.pt`` file; its digest is
            verified on reading.
        profile_payload: ``effective_configuration.profile_payload`` of the run
            the checkpoint belongs to, which carries the decoder settings.

    Raises:
        RuntimeError: If the rebuilt network does not have the parameter count
            the checkpoint's tensors imply.
    """
    checkpoint, _ = ModelCheckpoint.read_file(checkpoint_path)
    model = build_rebuilder(
        checkpoint.genome_kind, checkpoint, profile_payload, torch.device("cpu")
    )()
    checkpoint.restore_into(model)
    stored = sum(
        int(tensor.numel())
        for name, tensor in checkpoint.model_state.items()
        if not name.endswith(("running_mean", "running_var", "num_batches_tracked"))
    )
    rebuilt = sum(int(parameter.numel()) for parameter in model.parameters())
    if stored != rebuilt:
        raise RuntimeError(
            f"{checkpoint_path}: rebuilt model has {rebuilt} parameters, the checkpoint {stored}"
        )
    return LoadedModel(
        model_id=checkpoint.model_id,
        model=model,
        preprocessor=restore_preprocessor(checkpoint.preprocessing_state),
        checkpoint=checkpoint,
    )


def positive_probabilities(
    loaded: LoadedModel, images: torch.Tensor, batch_size: int = 64
) -> np.ndarray:
    """Positive-class probabilities of ``uint8`` images, the way the protocol scored them."""
    collected = []
    loaded.model.eval()
    with torch.no_grad():
        for start in range(0, int(images.shape[0]), batch_size):
            batch = loaded.preprocessor.apply(images[start : start + batch_size], training=False)
            logits = loaded.model.forward_pass(batch).to(torch.float64)
            collected.append(torch.softmax(logits, dim=1)[:, POSITIVE_CLASS])
    return torch.cat(collected).numpy()


def classifier_feature_sources(model: nn.Module) -> list[int]:
    """Convolutional nodes whose maps are flattened into a dense or output layer.

    These are the last feature maps before the decision, the layer Grad-CAM is
    defined on. In a DeepNEAT graph there can be several of them.
    """
    sources: set[int] = set()
    for target, strategy in model._merge_strategy_by_node_id.items():
        if not strategy.flatten_inputs:
            continue
        for source in model._incoming_source_ids_by_target.get(target, []):
            if source == model._input_node_id:
                continue
            module = model._layer_modules_by_node_id[str(source)]
            if isinstance(module, nn.Sequential) and isinstance(module[0], nn.Conv2d):
                sources.add(source)
    return sorted(sources)


def image_skips_into_classifier(model: nn.Module) -> bool:
    """Whether the raw image is also flattened straight into the classifier."""
    return any(
        strategy.flatten_inputs
        and model._input_node_id in model._incoming_source_ids_by_target.get(target, [])
        for target, strategy in model._merge_strategy_by_node_id.items()
    )


@dataclass(frozen=True)
class SaliencyMap:
    """One explanation of one decision.

    Attributes:
        heatmap: ``(height, width)`` map in ``[0, 1]``, the input's resolution;
            only evidence *for* the explained class, negative parts clipped.
        signed: The same map before clipping and scaling, in logit units per
            pixel for occlusion and in arbitrary units for Grad-CAM.
        explained_class: Class whose logit margin was explained.
        positive_probability: The model's PNEUMONIA probability for the image.
    """

    heatmap: np.ndarray
    signed: np.ndarray
    explained_class: int
    positive_probability: float


def _margin(logits: torch.Tensor, explained_class: int) -> torch.Tensor:
    return logits[:, explained_class] - logits[:, 1 - explained_class]


def _normalized_positive(signed: torch.Tensor) -> np.ndarray:
    positive = signed.clamp(min=0.0)
    peak = float(positive.max())
    return (positive / peak if peak > 0.0 else positive).numpy()


def grad_cam(
    model: nn.Module, image: torch.Tensor, explained_class: int | None = None
) -> SaliencyMap:
    """Grad-CAM over every convolutional map that feeds the classifier.

    Args:
        model: A DeepNEAT phenotype.
        image: One preprocessed image, ``(1, 1, height, width)``.
        explained_class: Class to explain; the predicted class when ``None``.

    Raises:
        ValueError: If no convolutional map reaches the classifier, as in a
            dense-only network.
    """
    sources = classifier_feature_sources(model)
    if not sources:
        raise ValueError("no convolutional feature map feeds the classifier; Grad-CAM needs one")
    activations: dict[int, torch.Tensor] = {}
    gradients: dict[int, torch.Tensor] = {}

    def capture(node_id: int):
        def hook(_module, _inputs, output: torch.Tensor) -> None:
            activations[node_id] = output

            def keep_gradient(gradient: torch.Tensor) -> None:
                gradients[node_id] = gradient

            output.register_hook(keep_gradient)

        return hook

    handles = [
        model._layer_modules_by_node_id[str(node_id)].register_forward_hook(capture(node_id))
        for node_id in sources
    ]
    model.eval()
    try:
        with torch.enable_grad():
            logits = model.forward_pass(image)
            if explained_class is None:
                explained_class = int(logits.argmax(dim=1)[0])
            model.zero_grad(set_to_none=True)
            _margin(logits, explained_class)[0].backward()
    finally:
        for handle in handles:
            handle.remove()
        model.zero_grad(set_to_none=True)

    side = tuple(image.shape[-2:])
    combined = torch.zeros(side)
    for node_id in sources:
        feature_maps = activations[node_id][0].detach()
        channel_weights = gradients[node_id][0].mean(dim=(1, 2))
        local = (channel_weights[:, None, None] * feature_maps).sum(dim=0)
        upsampled = functional.interpolate(
            local[None, None], size=side, mode="bilinear", align_corners=False
        )[0, 0]
        combined += local.numel() * upsampled
    probability = float(torch.softmax(logits.detach().to(torch.float64), dim=1)[0, POSITIVE_CLASS])
    return SaliencyMap(
        heatmap=_normalized_positive(combined),
        signed=combined.numpy(),
        explained_class=explained_class,
        positive_probability=probability,
    )


def occlusion_sensitivity(
    model: nn.Module,
    image: torch.Tensor,
    explained_class: int | None = None,
    *,
    patch_size: int = 16,
    stride: int = 4,
    baseline: float = 0.0,
    batch_size: int = 64,
) -> SaliencyMap:
    """How much the logit margin drops when each patch is hidden.

    Args:
        model: Any model whose ``forward_pass`` returns two logits.
        image: One preprocessed image, ``(1, channels, height, width)``.
        explained_class: Class to explain; the predicted class when ``None``.
        patch_size: Side of the square that is hidden.
        stride: Step between patch positions; the last row and column are
            always included so the border is covered.
        baseline: Value written into the patch. 0 in standardized units is the
            mean intensity of the training images.
        batch_size: Occluded copies per forward pass.

    Returns:
        The mean drop over every patch covering a pixel; a positive value means
        hiding that region weakened the decision.
    """
    model.eval()
    height, width = image.shape[-2:]
    with torch.no_grad():
        logits = model.forward_pass(image)
        if explained_class is None:
            explained_class = int(logits.argmax(dim=1)[0])
        reference = float(_margin(logits, explained_class)[0])

        def starts(length: int) -> list[int]:
            positions = list(range(0, length - patch_size + 1, stride))
            if positions[-1] != length - patch_size:
                positions.append(length - patch_size)
            return positions

        corners = [(top, left) for top in starts(height) for left in starts(width)]
        drop_sum = torch.zeros(height, width, dtype=torch.float64)
        coverage = torch.zeros(height, width, dtype=torch.float64)
        for start in range(0, len(corners), batch_size):
            chunk = corners[start : start + batch_size]
            occluded = image.repeat(len(chunk), 1, 1, 1)
            for row, (top, left) in enumerate(chunk):
                occluded[row, :, top : top + patch_size, left : left + patch_size] = baseline
            margins = _margin(model.forward_pass(occluded), explained_class).to(torch.float64)
            for row, (top, left) in enumerate(chunk):
                window = (slice(top, top + patch_size), slice(left, left + patch_size))
                drop_sum[window] += reference - float(margins[row])
                coverage[window] += 1.0
    signed = (drop_sum / coverage).to(torch.float32)
    probability = float(torch.softmax(logits.to(torch.float64), dim=1)[0, POSITIVE_CLASS])
    return SaliencyMap(
        heatmap=_normalized_positive(signed),
        signed=signed.numpy(),
        explained_class=explained_class,
        positive_probability=probability,
    )


def top_fraction_overlap(first: np.ndarray, second: np.ndarray, fraction: float = 0.2) -> float:
    """Intersection over union of the ``fraction`` most salient pixels of two maps."""
    count = max(1, int(round(fraction * first.size)))
    first_top = np.zeros(first.size, dtype=bool)
    second_top = np.zeros(second.size, dtype=bool)
    first_top[np.argsort(first.ravel(), kind="stable")[-count:]] = True
    second_top[np.argsort(second.ravel(), kind="stable")[-count:]] = True
    return float(np.sum(first_top & second_top) / np.sum(first_top | second_top))


def randomized_copy(model: nn.Module, seed: int) -> nn.Module:
    """The same topology with fresh track B initial weights, for a randomization check.

    Adebayo et al. (2018, *Sanity Checks for Saliency Maps*, NeurIPS): a
    saliency method that draws the same map for a trained and an untrained
    network explains the input, not the model.
    """
    randomized = copy.deepcopy(model)
    initialize_module_parameters(randomized, torch.Generator().manual_seed(seed))
    randomized.eval()
    return randomized


def preprocessing_stages(
    pixels: np.ndarray, preprocessor: ImagePreprocessor
) -> dict[str, np.ndarray]:
    """Every intermediate image of the evaluation path, for one decoded radiograph.

    Mirrors the cache (pad, antialiased resize, round to ``uint8``) and then
    the preprocessor (scale to ``[0, 1]``, standardize with fitted statistics).

    Args:
        pixels: ``(height, width)`` ``uint8`` grayscale image as decoded.
        preprocessor: A fitted preprocessor; its statistics are used as-is.
    """
    config = preprocessor.config
    as_batch = torch.from_numpy(pixels.copy()).to(torch.float32)[None, None]
    padded = pad_to_square(as_batch, pad_value=config.pad_value)
    resized = resize_images(padded, config.target_side, use_antialias=config.use_antialias)
    cached = resized.clamp(0.0, 255.0).round().to(torch.uint8)
    scaled = preprocessor.geometrically_normalize(cached)
    return {
        "original": pixels,
        "padded": padded[0, 0].numpy(),
        "resized": cached[0, 0].numpy(),
        "scaled": scaled[0, 0].numpy(),
        "standardized": preprocessor.standardize(scaled)[0, 0].numpy(),
    }


def augmentation_draws(image: np.ndarray, number_of_draws: int, seed: int) -> list[np.ndarray]:
    """Training-time variants of one ``[0, 1]`` image under the frozen augmentation ranges."""
    generator = torch.Generator().manual_seed(seed)
    batch = torch.from_numpy(np.repeat(image[None, None], number_of_draws, axis=0).copy())
    augmented = apply_affine_and_contrast_augmentation(
        batch.to(torch.float32), ImageAugmentationConfig(), generator
    )
    return [augmented[index, 0].numpy() for index in range(number_of_draws)]


def example_category(example_id: str, label: int) -> str:
    """``NORMAL``, ``bacteria`` or ``virus``; the Kermany file names carry the subtype."""
    if label == 0:
        return NORMAL_CATEGORY
    name = example_id.rsplit("/", 1)[-1].lower()
    if BACTERIAL_CATEGORY in name:
        return BACTERIAL_CATEGORY
    if VIRAL_CATEGORY in name:
        return VIRAL_CATEGORY
    return "pneumonia"


def select_examples(
    entries: list[dict], *, split: str, counts: dict[str, int], seed: int
) -> list[dict]:
    """Draw examples per category without looking at any model's output.

    Args:
        entries: Manifest-style records with ``example_id``, ``label`` and
            ``target_split``.
        split: Which split to draw from.
        counts: How many to draw per :func:`example_category`.
        seed: Seed of the draw, fixed in the notebook before any image is seen.

    Returns:
        The drawn records, category by category, each in id order.

    Raises:
        ValueError: If a category has fewer examples than requested.
    """
    generator = np.random.default_rng(seed)
    selected = []
    for category, count in counts.items():
        pool = sorted(
            (
                entry
                for entry in entries
                if entry["target_split"] == split
                and example_category(entry["example_id"], int(entry["label"])) == category
            ),
            key=lambda entry: entry["example_id"],
        )
        if len(pool) < count:
            raise ValueError(f"{split}/{category} has {len(pool)} examples, {count} requested")
        chosen = sorted(generator.choice(len(pool), size=count, replace=False))
        selected.extend(pool[index] for index in chosen)
    return selected


def render_architecture(description: ArchitectureDescription, title: str, stem: Path) -> Path:
    """Draw a network with :func:`benchmarks.describe_winners.render_matplotlib`.

    That renderer switches matplotlib to the Agg backend; the backend the
    caller had (the notebook's inline one) is put back afterwards.

    Returns:
        Path of the PNG that was written.
    """
    import matplotlib

    backend = matplotlib.get_backend()
    try:
        render_matplotlib(description, title, stem)
    finally:
        matplotlib.use(backend)
    return stem.with_suffix(".png")
