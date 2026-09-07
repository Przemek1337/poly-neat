"""Turning the frozen manifest into tensors, once, with a reusable cache.

Decoding five thousand JPEGs is the expensive part of every run, and doing it
differently in two places is how two runs stop being comparable. This module
therefore does it once per (manifest, split, resolution) and caches the result
as ``uint8`` on the CPU.

The cached images are already padded to a square and resized to the target
side, which is exactly the deterministic prefix of the shared image path. Those
two steps are idempotent - padding a square image and resizing to the size it
already has are both no-ops - so a cached batch can be handed to
:class:`~polyneat.training.image_preprocessing.ImagePreprocessor` without the
path running twice.

The cache is keyed by the manifest digest, so a manifest change invalidates it
instead of silently mixing two splits. Nothing here reads or returns a split
the caller did not ask for.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from examples.pediatric_pneumonia._dataset_manifest import DatasetManifest
from examples.pediatric_pneumonia._image_digests import load_grayscale_image
from polyneat.logging_utils.custom_logger import get_logger
from polyneat.training.image_preprocessing import pad_to_square, resize_images

logger = get_logger(__name__)

CACHE_SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class SplitTensors:
    """One split as tensors, with the ids needed to export predictions.

    Attributes:
        images: ``(n, 1, side, side)`` ``uint8`` tensor on the CPU.
        labels: ``(n,)`` long tensor of class indices.
        example_ids: Stable manifest ids, aligned with the rows above.
        group_ids: Split group of each row, for the grouped bootstrap.
        split_name: Which split this is, carried so a mix-up is visible.
    """

    images: torch.Tensor
    labels: torch.Tensor
    example_ids: tuple[str, ...]
    group_ids: tuple[str, ...]
    split_name: str

    def __post_init__(self) -> None:
        counts = {
            "images": int(self.images.shape[0]),
            "labels": int(self.labels.shape[0]),
            "example_ids": len(self.example_ids),
            "group_ids": len(self.group_ids),
        }
        if len(set(counts.values())) != 1:
            raise ValueError(f"split {self.split_name!r} has misaligned columns: {counts}")

    def __len__(self) -> int:
        return int(self.images.shape[0])

    @property
    def class_counts(self) -> dict[int, int]:
        """How many examples each class contributes to this split."""
        unique_labels, counts = torch.unique(self.labels, return_counts=True)
        return {int(label): int(count) for label, count in zip(unique_labels, counts, strict=True)}


def _cache_file_path(
    cache_directory: Path, manifest_sha256: str, split_name: str, target_side: int
) -> Path:
    """Cache file for one (manifest, split, resolution) triple."""
    return cache_directory / f"{manifest_sha256[:16]}_{split_name}_{target_side}.npz"


def _decode_and_resize_split(
    manifest: DatasetManifest,
    archive_root: Path,
    split_name: str,
    target_side: int,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...], tuple[str, ...]]:
    """Decode one split and bring every image to ``target_side`` squares."""
    entries = manifest.entries_of_split(split_name)
    if not entries:
        raise ValueError(f"split {split_name!r} is empty in this manifest")

    resized_images = np.empty((len(entries), 1, target_side, target_side), dtype=np.uint8)
    labels = np.empty(len(entries), dtype=np.int64)
    for row_index, entry in enumerate(entries):
        image_path = archive_root / entry.relative_path
        grayscale_pixels = load_grayscale_image(image_path)
        # PIL hands back a read-only view; copy so torch owns writable memory.
        as_batch = torch.from_numpy(grayscale_pixels.copy()).to(torch.float32)[None, None, :, :]
        squared = pad_to_square(as_batch, pad_value=0.0)
        resized = resize_images(squared, target_side, use_antialias=True)
        resized_images[row_index, 0] = (
            resized.clamp(0.0, 255.0).round().to(torch.uint8)[0, 0].numpy()
        )
        labels[row_index] = entry.label

    logger.info(
        "Decoded split %s: %d images at %dx%d", split_name, len(entries), target_side, target_side
    )
    return (
        resized_images,
        labels,
        tuple(entry.example_id for entry in entries),
        tuple(entry.split_group_id for entry in entries),
    )


def load_split_tensors(
    manifest: DatasetManifest,
    *,
    data_directory: Path,
    split_name: str,
    target_side: int,
    manifest_sha256: str | None = None,
    cache_directory: Path | None = None,
) -> SplitTensors:
    """Load one split of the frozen manifest as CPU tensors.

    Args:
        manifest: The frozen split. Only the requested split is read.
        data_directory: Directory the archive was extracted into.
        split_name: One of the protocol's four splits.
        target_side: Square side every image is brought to. Must match the
            resolution the preprocessor is configured for.
        manifest_sha256: Digest of the manifest, used as the cache key.
            Computed when omitted.
        cache_directory: Where decoded splits are cached. ``None`` disables
            caching and decodes every time.

    Returns:
        The split as :class:`SplitTensors`.

    Raises:
        ValueError: If the split is empty in this manifest.
    """
    archive_root = data_directory / manifest.archive_root_relative_path
    resolved_digest = manifest_sha256 or manifest.compute_sha256()

    cache_path = (
        None
        if cache_directory is None
        else _cache_file_path(cache_directory, resolved_digest, split_name, target_side)
    )
    if cache_path is not None and cache_path.is_file():
        cached = np.load(cache_path, allow_pickle=False)
        if str(cached["schema_version"]) == CACHE_SCHEMA_VERSION:
            logger.info("Loaded split %s from cache %s", split_name, cache_path.name)
            return SplitTensors(
                images=torch.from_numpy(cached["images"]),
                labels=torch.from_numpy(cached["labels"]),
                example_ids=tuple(str(value) for value in cached["example_ids"]),
                group_ids=tuple(str(value) for value in cached["group_ids"]),
                split_name=split_name,
            )
        logger.warning(
            "Ignoring cache %s written by schema version %s",
            cache_path.name,
            cached["schema_version"],
        )

    images, labels, example_ids, group_ids = _decode_and_resize_split(
        manifest, archive_root, split_name, target_side
    )
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            schema_version=np.array(CACHE_SCHEMA_VERSION),
            images=images,
            labels=labels,
            example_ids=np.array(example_ids),
            group_ids=np.array(group_ids),
        )
        logger.info("Cached split %s to %s", split_name, cache_path.name)

    return SplitTensors(
        images=torch.from_numpy(images),
        labels=torch.from_numpy(labels),
        example_ids=example_ids,
        group_ids=group_ids,
        split_name=split_name,
    )
