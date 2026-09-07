"""Digests used by the chest X-ray audit: file, pixel and perceptual.

Three different questions need three different digests:

* ``file_sha256`` answers "is this byte-for-byte the same file", which is what a
  download-integrity check needs.
* ``pixel_sha256`` answers "does this decode to the same image", which survives
  re-encoding and is what a duplicate check needs. Dimensions are hashed in
  front of the pixels so two different shapes can never collide by accident.
* ``perceptual_hash`` answers "does this *look* like that", which is a signal to
  review, never a proof. The protocol forbids deleting an image on a perceptual
  match alone.

Decoding is pinned to Pillow in grayscale mode; the audit records the Pillow
version so a later run can tell whether a digest change came from the data or
from the decoder.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from PIL import Image

# Frozen for the whole benchmark: two images whose perceptual hashes differ in
# at most this many bits are *reported* as similar. Raising it later changes the
# audit report and therefore requires a new protocol id.
DEFAULT_PERCEPTUAL_HASH_HAMMING_THRESHOLD = 5

# Difference hash side: an 8x8 grid of column-to-column comparisons taken from a
# 9x8 downsample gives the usual 64-bit fingerprint.
_PERCEPTUAL_HASH_SIDE = 8

_FILE_READ_CHUNK_BYTES = 1 << 20


class CorruptImageError(RuntimeError):
    """Raised when a file under a class directory cannot be decoded as an image."""


def compute_file_sha256(file_path: Path) -> str:
    """Hash the raw bytes of ``file_path``.

    Args:
        file_path: File to read. Read in chunks so a large archive member does
            not have to fit in memory.

    Returns:
        Lowercase hexadecimal SHA-256 digest.
    """
    digest = hashlib.sha256()
    with file_path.open("rb") as binary_file:
        while chunk := binary_file.read(_FILE_READ_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def load_grayscale_image(file_path: Path) -> np.ndarray:
    """Decode ``file_path`` to a single-channel ``uint8`` array.

    Args:
        file_path: Image file to decode.

    Returns:
        ``(height, width)`` ``uint8`` array.

    Raises:
        CorruptImageError: If the file cannot be opened or decoded. The audit
            turns this into a recorded finding rather than skipping the file
            silently.
    """
    try:
        with Image.open(file_path) as opened_image:
            grayscale_image = opened_image.convert("L")
            return np.asarray(grayscale_image, dtype=np.uint8)
    except Exception as decode_error:  # noqa: BLE001 - re-raised with context below
        raise CorruptImageError(f"cannot decode {file_path} as an image") from decode_error


def compute_pixel_sha256(grayscale_pixels: np.ndarray) -> str:
    """Hash decoded pixels together with their shape.

    Args:
        grayscale_pixels: ``(height, width)`` ``uint8`` array from
            :func:`load_grayscale_image`.

    Returns:
        Lowercase hexadecimal SHA-256 digest of ``"<height>x<width>x1"`` followed
        by the raw pixel bytes in C order.

    Raises:
        ValueError: If the array is not a 2-D ``uint8`` array.
    """
    if grayscale_pixels.ndim != 2 or grayscale_pixels.dtype != np.uint8:
        raise ValueError(
            "compute_pixel_sha256 expects a 2-D uint8 grayscale array, got "
            f"ndim={grayscale_pixels.ndim} dtype={grayscale_pixels.dtype}"
        )
    height, width = grayscale_pixels.shape
    digest = hashlib.sha256()
    digest.update(f"{height}x{width}x1".encode())
    digest.update(np.ascontiguousarray(grayscale_pixels).tobytes())
    return digest.hexdigest()


def compute_perceptual_hash(grayscale_pixels: np.ndarray) -> int:
    """Compute a 64-bit difference hash of an image.

    The image is resized to ``9x8`` and each pixel is compared with its right
    neighbour, giving 64 bits that survive rescaling and mild re-encoding.

    Args:
        grayscale_pixels: ``(height, width)`` ``uint8`` array.

    Returns:
        The fingerprint as a 64-bit unsigned integer, most significant bit
        first in row-major order.

    Raises:
        ValueError: If the array is empty.
    """
    if grayscale_pixels.size == 0:
        raise ValueError("compute_perceptual_hash received an empty image")
    resized_image = Image.fromarray(grayscale_pixels, mode="L").resize(
        (_PERCEPTUAL_HASH_SIDE + 1, _PERCEPTUAL_HASH_SIDE),
        resample=Image.Resampling.BILINEAR,
    )
    resized_pixels = np.asarray(resized_image, dtype=np.int16)
    is_brighter_than_right_neighbour = resized_pixels[:, 1:] > resized_pixels[:, :-1]

    fingerprint = 0
    for bit_value in is_brighter_than_right_neighbour.reshape(-1):
        fingerprint = (fingerprint << 1) | int(bit_value)
    return fingerprint


def perceptual_hash_hamming_distance(first_hash: int, second_hash: int) -> int:
    """Number of differing bits between two perceptual hashes.

    Args:
        first_hash: Fingerprint from :func:`compute_perceptual_hash`.
        second_hash: The other fingerprint.

    Returns:
        Hamming distance in ``[0, 64]``. Smaller means more similar.
    """
    return int((first_hash ^ second_hash).bit_count())


def describe_decoder_version() -> str:
    """Return the pinned decoder identity recorded in the audit report."""
    from PIL import __version__ as pillow_version

    return f"pillow=={pillow_version}"
