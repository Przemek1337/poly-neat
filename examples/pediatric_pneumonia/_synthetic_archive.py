"""A small synthetic stand-in for the pediatric chest X-ray archive.

Development and smoke runs need an archive with the right *shape* - the same
directory nesting, the same file-naming conventions, rectangular grayscale
JPEGs of varying size, two classes with a learnable difference - without
downloading five thousand real radiographs or putting patient images in a test
fixture.

The images are deliberately not radiographs. Pneumonia cases carry a bright
blob on a smooth background so a tiny network can reach an AUROC above chance
in a few epochs, which is what a smoke run needs to prove the plumbing works.
Nothing produced here may appear in a results table.

Patient numbering restarts in the official test directory, exactly as the real
release numbers its collections independently, so the audit exercises its
per-pool identifier scope rather than a convenient fiction.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from polyneat.logging_utils.custom_logger import get_logger

logger = get_logger(__name__)

_JPEG_QUALITY = 92


@dataclass(frozen=True)
class SyntheticArchiveSpec:
    """How much synthetic data to write, and in what shape.

    Attributes:
        train_normal_patients: NORMAL series written into ``train``.
        train_pneumonia_patients: PNEUMONIA patients written into ``train``.
        validation_normal_patients: NORMAL series written into ``val``.
        validation_pneumonia_patients: PNEUMONIA patients written into ``val``.
        test_normal_patients: NORMAL series written into ``test``.
        test_pneumonia_patients: PNEUMONIA patients written into ``test``.
        images_per_patient: Images each patient or series contributes, so the
            grouped split has something to keep together.
        minimum_image_side: Smallest side length drawn per image.
        maximum_image_side: Largest side length drawn per image.
        nesting: Directory levels inserted between the destination and the
            split directories, e.g. ``("chest_xray",)``.
    """

    train_normal_patients: int = 12
    train_pneumonia_patients: int = 18
    validation_normal_patients: int = 3
    validation_pneumonia_patients: int = 3
    test_normal_patients: int = 5
    test_pneumonia_patients: int = 7
    images_per_patient: int = 2
    minimum_image_side: int = 40
    maximum_image_side: int = 64
    nesting: tuple[str, ...] = ("chest_xray",)


def _draw_image(
    rng: np.random.Generator, *, is_pneumonia: bool, spec: SyntheticArchiveSpec
) -> np.ndarray:
    """Draw one grayscale image with a class-dependent bright region."""
    height = int(rng.integers(spec.minimum_image_side, spec.maximum_image_side + 1))
    width = int(rng.integers(spec.minimum_image_side, spec.maximum_image_side + 1))
    # Per-image ramp endpoints and a few random blobs keep the fixture from
    # collapsing into near-identical images, which would make every pair look
    # like a perceptual duplicate and drown the similarity report in noise.
    row_ramp = np.linspace(
        rng.uniform(0.05, 0.35), rng.uniform(0.45, 0.75), height, dtype=np.float32
    )[:, None]
    column_ramp = np.linspace(
        rng.uniform(0.05, 0.30), rng.uniform(0.35, 0.60), width, dtype=np.float32
    )[None, :]
    image = 0.5 * (row_ramp + column_ramp)
    image += rng.normal(loc=0.0, scale=0.06, size=(height, width)).astype(np.float32)

    row_grid, column_grid = np.mgrid[0:height, 0:width]
    for _ in range(int(rng.integers(3, 7))):
        blob_row = rng.uniform(0.0, 1.0) * height
        blob_column = rng.uniform(0.0, 1.0) * width
        blob_radius = rng.uniform(0.08, 0.25) * min(height, width)
        blob_amplitude = rng.uniform(-0.25, 0.25)
        squared_distance = (row_grid - blob_row) ** 2 + (column_grid - blob_column) ** 2
        image += blob_amplitude * np.exp(
            -squared_distance / (2.0 * blob_radius**2)
        ).astype(np.float32)

    if is_pneumonia:
        centre_row = rng.uniform(0.3, 0.7) * height
        centre_column = rng.uniform(0.3, 0.7) * width
        radius = 0.18 * min(height, width)
        squared_distance = (row_grid - centre_row) ** 2 + (column_grid - centre_column) ** 2
        image += 0.45 * np.exp(-squared_distance / (2.0 * radius**2)).astype(np.float32)

    return (np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)


def _write_series(
    directory: Path,
    rng: np.random.Generator,
    spec: SyntheticArchiveSpec,
    *,
    is_pneumonia: bool,
    number_of_patients: int,
    first_patient_number: int,
    normal_collection_prefix: str,
) -> int:
    """Write one class directory and return the next free patient number."""
    directory.mkdir(parents=True, exist_ok=True)
    patient_number = first_patient_number
    for _ in range(number_of_patients):
        for image_index in range(1, spec.images_per_patient + 1):
            if is_pneumonia:
                pathogen = "bacteria" if patient_number % 2 == 0 else "virus"
                file_name = f"person{patient_number}_{pathogen}_{image_index}.jpeg"
            else:
                file_name = (
                    f"{normal_collection_prefix}-{patient_number:04d}"
                    f"-{image_index:04d}.jpeg"
                )
            pixels = _draw_image(rng, is_pneumonia=is_pneumonia, spec=spec)
            Image.fromarray(pixels, mode="L").save(
                directory / file_name, format="JPEG", quality=_JPEG_QUALITY
            )
        patient_number += 1
    return patient_number


def write_synthetic_archive(
    destination_directory: Path,
    *,
    random_seed: int = 20260905,
    spec: SyntheticArchiveSpec | None = None,
) -> Path:
    """Write a synthetic archive that the real loader and audit accept.

    Args:
        destination_directory: Directory to create the archive under. It is
            created if missing; existing files with the same names are
            overwritten.
        random_seed: Seed for image content and sizes, so a fixture is stable.
        spec: Size and shape of the archive. Defaults to a tiny one suitable
            for tests.

    Returns:
        The directory that should be passed to the loader, i.e.
        ``destination_directory`` itself - the nesting from ``spec`` is created
        underneath it.
    """
    resolved_spec = spec or SyntheticArchiveSpec()
    rng = np.random.default_rng(random_seed)
    archive_root = destination_directory.joinpath(*resolved_spec.nesting)

    next_development_patient = 1
    next_development_patient = _write_series(
        archive_root / "train" / "PNEUMONIA",
        rng,
        resolved_spec,
        is_pneumonia=True,
        number_of_patients=resolved_spec.train_pneumonia_patients,
        first_patient_number=next_development_patient,
        normal_collection_prefix="",
    )
    _write_series(
        archive_root / "val" / "PNEUMONIA",
        rng,
        resolved_spec,
        is_pneumonia=True,
        number_of_patients=resolved_spec.validation_pneumonia_patients,
        first_patient_number=next_development_patient,
        normal_collection_prefix="",
    )
    _write_series(
        archive_root / "train" / "NORMAL",
        rng,
        resolved_spec,
        is_pneumonia=False,
        number_of_patients=resolved_spec.train_normal_patients,
        first_patient_number=1,
        normal_collection_prefix="IM",
    )
    _write_series(
        archive_root / "val" / "NORMAL",
        rng,
        resolved_spec,
        is_pneumonia=False,
        number_of_patients=resolved_spec.validation_normal_patients,
        first_patient_number=1,
        normal_collection_prefix="NORMAL2-IM",
    )
    # The official test directory restarts both counters, mirroring the real
    # release: person 1 here is not person 1 in train.
    _write_series(
        archive_root / "test" / "PNEUMONIA",
        rng,
        resolved_spec,
        is_pneumonia=True,
        number_of_patients=resolved_spec.test_pneumonia_patients,
        first_patient_number=1,
        normal_collection_prefix="",
    )
    _write_series(
        archive_root / "test" / "NORMAL",
        rng,
        resolved_spec,
        is_pneumonia=False,
        number_of_patients=resolved_spec.test_normal_patients,
        first_patient_number=1,
        normal_collection_prefix="IM",
    )
    logger.info("Synthetic pediatric pneumonia archive written to %s", archive_root)
    return destination_directory
