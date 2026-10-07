"""Fetching and parsing the raw dataset files the GPU sweep runs on.

Self-contained on purpose: the sweep is a throwaway harness, so it carries its
own copies of these primitives rather than importing anything from ``examples``.
Formats covered are exactly the ones the paper's datasets ship in - delimited
text, ARFF, and little-endian MATLAB 5 MAT-files.
"""

from __future__ import annotations

import struct
import urllib.request
import zipfile
import zlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MISSING_VALUE_MARKERS: tuple[str, ...] = ("?", "", "NA", "na", "nan", "NaN")
FEATURE_SCALING_CHOICES: tuple[str, ...] = ("none", "min_max", "standardize")


def download_file_if_missing(source_url: str, destination_path: Path) -> Path:
    """Download ``source_url`` to ``destination_path`` unless it is already there.

    Args:
        source_url: URL to fetch on a cache miss.
        destination_path: Where the file is cached; parents are created.

    Returns:
        ``destination_path``, for chaining into a read.
    """
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    if not destination_path.exists():
        print(f"Downloading {source_url} -> {destination_path}")
        urllib.request.urlretrieve(source_url, destination_path)
    return destination_path


def extract_zip_member_if_missing(archive_path: Path, member_name: str) -> Path:
    """Extract one member of a zip archive next to it unless it is already there.

    Args:
        archive_path: Zip archive to read on a cache miss.
        member_name: Name of the member inside the archive.

    Returns:
        Path of the extracted member, for chaining into a read.
    """
    member_path = archive_path.parent / member_name
    if not member_path.exists():
        with zipfile.ZipFile(archive_path) as archive:
            member_path.write_bytes(archive.read(member_name))
    return member_path


def read_delimited_rows(
    file_path: Path,
    *,
    delimiter: str | None = ",",
    skip_header_rows: int = 0,
) -> list[list[str]]:
    """Read a plain-text table into a list of stripped string cells.

    Blank lines are dropped before ``skip_header_rows`` is applied, so a file
    with a trailing newline needs no special casing at the call site.

    Args:
        file_path: File to read; decoded as UTF-8 with replacement.
        delimiter: Column separator. ``None`` splits on runs of whitespace,
            which is what the Statlog and thyroid files need.
        skip_header_rows: Number of leading non-blank rows to discard.

    Returns:
        One list of cell strings per data row.
    """
    lines = [
        line
        for line in file_path.read_text(encoding="utf-8", errors="replace").splitlines()
        if line.strip()
    ]
    return [
        [cell.strip() for cell in line.split(delimiter)]
        for line in lines[skip_header_rows:]
    ]


def read_arff_data_rows(file_path: Path, *, delimiter: str = ",") -> list[list[str]]:
    """Read the ``@data`` section of an ARFF file as string cells.

    The attribute declarations are skipped rather than interpreted: every column
    the catalog reads is numeric and the catalog states the layout itself.

    Args:
        file_path: ARFF file to read.
        delimiter: Column separator inside the data section.

    Returns:
        One list of cell strings per data row, comments excluded.

    Raises:
        ValueError: If the file holds no ``@data`` marker.
    """
    lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
    data_section_start: int | None = None
    for line_index, line in enumerate(lines):
        if line.strip().lower().startswith("@data"):
            data_section_start = line_index + 1
            break
    if data_section_start is None:
        raise ValueError(f"{file_path} holds no @data section")
    return [
        [cell.strip() for cell in line.split(delimiter)]
        for line in lines[data_section_start:]
        if line.strip() and not line.lstrip().startswith("%")
    ]


def _fill_missing_cells_with_column_mean(features: np.ndarray) -> np.ndarray:
    """Replace every NaN with the mean of the finite values in its column."""
    missing_mask = np.isnan(features)
    if not missing_mask.any():
        return features
    column_means = np.nanmean(features, axis=0)
    column_means = np.where(np.isnan(column_means), 0.0, column_means)
    filled = features.copy()
    filled[missing_mask] = column_means[np.nonzero(missing_mask)[1]]
    return filled.astype(np.float32)


def rows_to_features_and_labels(
    rows: list[list[str]],
    *,
    label_column_index: int,
    feature_column_indices: Sequence[int],
    label_value_to_index: dict[str, int],
    categorical_value_maps: dict[int, dict[str, float]] | None = None,
    missing_value_handling: str = "drop_row",
    missing_value_markers: tuple[str, ...] = MISSING_VALUE_MARKERS,
) -> tuple[np.ndarray, np.ndarray]:
    """Turn string rows into a numeric feature matrix and a label index vector.

    Label values are mapped explicitly rather than inferred, because the raw
    files disagree about what a class looks like - ``M``/``B``, ``1``/``2``,
    ``0``..``4`` - and sorting string labels would order ``"10"`` before ``"2"``.

    Args:
        rows: Rows of stripped cell strings.
        label_column_index: Column holding the class label.
        feature_column_indices: Columns to keep as features, in output order.
        label_value_to_index: Maps every raw label string to a class index.
        categorical_value_maps: Optional ``{column_index: {cell: value}}`` for
            non-numeric feature columns such as ILPD's gender.
        missing_value_handling: ``"drop_row"`` discards rows holding a missing
            marker; ``"column_mean"`` fills them with the column mean.
        missing_value_markers: Cell values that count as missing.

    Returns:
        ``(features, labels)`` as a ``[n_samples, n_features]`` float32 array
        and an ``[n_samples]`` int64 array.

    Raises:
        ValueError: If a label is not in ``label_value_to_index``, or
            ``missing_value_handling`` is not one of the two supported values.
    """
    if missing_value_handling not in ("drop_row", "column_mean"):
        raise ValueError(
            f"missing_value_handling must be 'drop_row' or 'column_mean', "
            f"got {missing_value_handling!r}"
        )
    categorical_value_maps = categorical_value_maps or {}

    parsed_feature_rows: list[list[float]] = []
    parsed_label_indices: list[int] = []
    for row in rows:
        label_cell = row[label_column_index]
        if label_cell not in label_value_to_index:
            raise ValueError(
                f"unmapped label {label_cell!r}; known labels: {sorted(label_value_to_index)}"
            )
        feature_values: list[float] = []
        for column_index in feature_column_indices:
            cell = row[column_index]
            value_map = categorical_value_maps.get(column_index)
            if value_map is not None:
                feature_values.append(float(value_map[cell]))
            elif cell in missing_value_markers:
                feature_values.append(float("nan"))
            else:
                feature_values.append(float(cell))
        parsed_feature_rows.append(feature_values)
        parsed_label_indices.append(label_value_to_index[label_cell])

    features = np.array(parsed_feature_rows, dtype=np.float32)
    labels = np.array(parsed_label_indices, dtype=np.int64)

    if missing_value_handling == "drop_row":
        complete_row_mask = ~np.isnan(features).any(axis=1)
        return features[complete_row_mask], labels[complete_row_mask]
    return _fill_missing_cells_with_column_mean(features), labels


@dataclass(frozen=True)
class FeatureScaling:
    """Column scaling fitted on the training rows only, applicable to any rows.

    ``apply`` computes ``(features - offsets) / divisors`` and then sets every
    column in ``constant_feature_indices`` to zero. A column that is constant
    across the training rows carries no training signal, and dividing a
    test-row deviation by a near-zero spread would blow it up; zeroing keeps
    the network's input width unchanged while making the column inert.

    Attributes:
        method: One of :data:`FEATURE_SCALING_CHOICES`.
        offsets: ``[n_features]`` values subtracted from every column.
        divisors: ``[n_features]`` values every column is divided by.
        constant_feature_indices: Columns constant across the training rows;
            zeroed by ``apply`` unless ``method`` is ``"none"``.
    """

    method: str
    offsets: np.ndarray
    divisors: np.ndarray
    constant_feature_indices: tuple[int, ...]

    def apply(self, features: np.ndarray) -> np.ndarray:
        """Scale ``features`` with the fitted statistics; returns float32."""
        scaled = (np.asarray(features, dtype=np.float64) - self.offsets) / self.divisors
        if self.method != "none" and self.constant_feature_indices:
            scaled[:, list(self.constant_feature_indices)] = 0.0
        return scaled.astype(np.float32)

    def to_serializable_dict(self) -> dict[str, object]:
        """JSON-safe copy of the fitted statistics, for the per-run split record."""
        return {
            "method": self.method,
            "offsets": [float(value) for value in self.offsets],
            "divisors": [float(value) for value in self.divisors],
            "constant_feature_indices": list(self.constant_feature_indices),
        }


def fit_feature_scaling(train_features: np.ndarray, feature_scaling: str) -> FeatureScaling:
    """Fit a column scaling on the training rows only.

    ``"standardize"`` uses the training mean and population standard deviation;
    ``"min_max"`` the training minimum and span; ``"none"`` is the identity.
    Columns whose training values are all equal are reported in
    ``constant_feature_indices`` and get a divisor of 1.

    Args:
        train_features: ``[n_train, n_features]`` float array - training rows only.
        feature_scaling: One of :data:`FEATURE_SCALING_CHOICES`.

    Returns:
        The fitted :class:`FeatureScaling`.

    Raises:
        ValueError: If ``feature_scaling`` is not a known choice.
    """
    if feature_scaling not in FEATURE_SCALING_CHOICES:
        raise ValueError(
            f"feature_scaling must be one of {FEATURE_SCALING_CHOICES}, got {feature_scaling!r}"
        )
    train = np.asarray(train_features, dtype=np.float64)
    constant_mask = train.max(axis=0) == train.min(axis=0)
    constant_feature_indices = tuple(int(index) for index in np.flatnonzero(constant_mask))
    number_of_features = train.shape[1]
    if feature_scaling == "none":
        offsets = np.zeros(number_of_features)
        divisors = np.ones(number_of_features)
    elif feature_scaling == "min_max":
        offsets = train.min(axis=0)
        divisors = np.where(constant_mask, 1.0, train.max(axis=0) - train.min(axis=0))
    else:
        offsets = train.mean(axis=0)
        divisors = np.where(constant_mask, 1.0, train.std(axis=0))
    return FeatureScaling(
        method=feature_scaling,
        offsets=offsets,
        divisors=divisors,
        constant_feature_indices=constant_feature_indices,
    )


def min_max_normalize_columns(features: np.ndarray) -> np.ndarray:
    """Scale every column onto ``[0, 1]`` over all rows, leaving constant columns at zero.

    Whole-matrix scaling, kept for one-off comparisons and for re-scoring the
    networks of the first sweep. The catalog fits its scaling on the training
    rows instead - see :func:`fit_feature_scaling`.
    """
    column_minima = features.min(axis=0, keepdims=True)
    column_maxima = features.max(axis=0, keepdims=True)
    column_spans = np.where(
        column_maxima == column_minima, 1.0, column_maxima - column_minima
    )
    return ((features - column_minima) / column_spans).astype(np.float32)


def standardize_columns(features: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-variance every column over the whole dataset.

    This is the scaling the first sweep used, statistics taken over all rows
    before the split - which lets the test rows leak into the scaling. It is
    kept only to re-score that sweep's stored networks on the inputs they were
    evolved against; the catalog now uses :func:`fit_feature_scaling` on the
    training rows. Constant columns come out at zero: their spread is zero, and
    the ``1e-6`` added to the standard deviation keeps the division finite.
    """
    column_means = features.mean(axis=0, keepdims=True)
    column_standard_deviations = features.std(axis=0, keepdims=True) + 1e-6
    return ((features - column_means) / column_standard_deviations).astype(np.float32)


def scale_feature_columns(features: np.ndarray, feature_scaling: str) -> np.ndarray:
    """Apply the caller's chosen column scaling, computed over all rows.

    Whole-matrix scaling, as in the first sweep; see :func:`standardize_columns`
    for why the catalog no longer uses it.

    Args:
        features: ``[n_samples, n_features]`` float array.
        feature_scaling: One of :data:`FEATURE_SCALING_CHOICES`.

    Returns:
        A float32 array of the same shape.

    Raises:
        ValueError: If ``feature_scaling`` is not a known choice.
    """
    if feature_scaling == "none":
        return features.astype(np.float32)
    if feature_scaling == "min_max":
        return min_max_normalize_columns(features)
    if feature_scaling == "standardize":
        return standardize_columns(features)
    raise ValueError(
        f"feature_scaling must be one of {FEATURE_SCALING_CHOICES}, got {feature_scaling!r}"
    )


_MATLAB_HEADER_SIZE = 128
_MATLAB_MATRIX_TYPE = 14
_MATLAB_COMPRESSED_TYPE = 15
_MATLAB_NUMPY_DTYPE_BY_TYPE_CODE = {
    1: "i1",
    2: "u1",
    3: "i2",
    4: "u2",
    5: "i4",
    6: "u4",
    7: "f4",
    9: "f8",
    12: "i8",
    13: "u8",
}


def _read_matlab_element(buffer: bytes, offset: int) -> tuple[int, bytes, int]:
    """Read one MAT-file data element as ``(type_code, payload, next_offset)``.

    Handles both tag forms. The small-data form packs the byte count into the
    high half of the first word and always spans exactly 8 bytes; the regular
    form pads its payload up to an 8-byte boundary - except for compressed
    elements, which are written without padding.
    """
    first_word = struct.unpack_from("<I", buffer, offset)[0]
    small_form_byte_count = first_word >> 16
    if small_form_byte_count:
        type_code = first_word & 0xFFFF
        payload = buffer[offset + 4 : offset + 4 + small_form_byte_count]
        return type_code, payload, offset + 8

    type_code = first_word
    byte_count = struct.unpack_from("<I", buffer, offset + 4)[0]
    payload = buffer[offset + 8 : offset + 8 + byte_count]
    if type_code == _MATLAB_COMPRESSED_TYPE:
        return type_code, payload, offset + 8 + byte_count
    return type_code, payload, offset + 8 + byte_count + (-byte_count) % 8


def read_matlab_v5_arrays(file_path: Path) -> dict[str, np.ndarray]:
    """Read the numeric arrays out of a little-endian MATLAB 5 MAT-file.

    Written by hand rather than delegated to ``scipy.io.loadmat`` so the three
    microarray files do not drag scipy into the dependency set. Scope is exactly
    what those files contain: top-level real numeric matrices, optionally
    zlib-compressed. Cell arrays, structs, complex values and MAT-file v7.3
    (HDF5) are out of scope and are skipped.

    Args:
        file_path: ``.mat`` file to read.

    Returns:
        ``{variable_name: array}``; arrays keep their stored shape and dtype.

    Raises:
        ValueError: If the file is shorter than a MAT-file header.
    """
    buffer = file_path.read_bytes()
    if len(buffer) < _MATLAB_HEADER_SIZE:
        raise ValueError(f"{file_path} is not a MATLAB 5 MAT-file: header is truncated")

    named_arrays: dict[str, np.ndarray] = {}
    offset = _MATLAB_HEADER_SIZE
    while offset < len(buffer):
        type_code, payload, offset = _read_matlab_element(buffer, offset)
        if type_code == _MATLAB_COMPRESSED_TYPE:
            type_code, payload, _ = _read_matlab_element(zlib.decompress(payload), 0)
        if type_code != _MATLAB_MATRIX_TYPE:
            continue

        payload_offset = 0
        _, _, payload_offset = _read_matlab_element(payload, payload_offset)
        _, dimensions_bytes, payload_offset = _read_matlab_element(payload, payload_offset)
        _, name_bytes, payload_offset = _read_matlab_element(payload, payload_offset)
        value_type_code, value_bytes, payload_offset = _read_matlab_element(
            payload, payload_offset
        )
        if value_type_code not in _MATLAB_NUMPY_DTYPE_BY_TYPE_CODE:
            continue

        dimensions = np.frombuffer(dimensions_bytes, dtype="<i4")
        numpy_dtype = "<" + _MATLAB_NUMPY_DTYPE_BY_TYPE_CODE[value_type_code]
        named_arrays[name_bytes.decode("ascii")] = np.frombuffer(
            value_bytes, dtype=numpy_dtype
        ).reshape(dimensions, order="F")
    return named_arrays
