"""Parsing primitives of the GPU sweep, exercised on fixture files only."""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np
import pytest

from gpu_sweep.raw_parsing import (
    read_arff_data_rows,
    read_delimited_rows,
    read_matlab_v5_arrays,
    rows_to_features_and_labels,
    scale_feature_columns,
)


def test_read_delimited_rows_drops_blank_lines_then_skips_headers(tmp_path: Path) -> None:
    data_file = tmp_path / "table.csv"
    data_file.write_text("head,er\n\n1,2\n3,4\n\n", encoding="utf-8")

    rows = read_delimited_rows(data_file, skip_header_rows=1)

    assert rows == [["1", "2"], ["3", "4"]]


def test_read_delimited_rows_splits_on_whitespace_when_delimiter_is_none(
    tmp_path: Path,
) -> None:
    data_file = tmp_path / "table.dat"
    data_file.write_text("70.0  1.0   2.0\n67.0 0.0  1.0\n", encoding="utf-8")

    rows = read_delimited_rows(data_file, delimiter=None)

    assert rows == [["70.0", "1.0", "2.0"], ["67.0", "0.0", "1.0"]]


def test_read_arff_data_rows_starts_after_the_data_marker(tmp_path: Path) -> None:
    data_file = tmp_path / "features.arff"
    data_file.write_text(
        "@relation messidor\n@attribute a numeric\n@data\n% comment\n1,0\n2,1\n",
        encoding="utf-8",
    )

    rows = read_arff_data_rows(data_file)

    assert rows == [["1", "0"], ["2", "1"]]


def test_rows_to_features_and_labels_drops_rows_holding_a_missing_marker() -> None:
    rows = [["1", "2", "M"], ["?", "4", "B"], ["5", "6", "M"]]

    features, labels = rows_to_features_and_labels(
        rows,
        label_column_index=2,
        feature_column_indices=range(0, 2),
        label_value_to_index={"M": 1, "B": 0},
        missing_value_handling="drop_row",
    )

    assert features.tolist() == [[1.0, 2.0], [5.0, 6.0]]
    assert labels.tolist() == [1, 1]


def test_rows_to_features_and_labels_fills_missing_cells_with_the_column_mean() -> None:
    rows = [["1", "2", "2"], ["?", "4", "4"], ["5", "6", "2"]]

    features, labels = rows_to_features_and_labels(
        rows,
        label_column_index=2,
        feature_column_indices=range(0, 2),
        label_value_to_index={"2": 0, "4": 1},
        missing_value_handling="column_mean",
    )

    assert features.tolist() == [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    assert labels.tolist() == [0, 1, 0]


def test_rows_to_features_and_labels_maps_categorical_feature_cells() -> None:
    rows = [["40", "Male", "1"], ["50", "Female", "2"]]

    features, _ = rows_to_features_and_labels(
        rows,
        label_column_index=2,
        feature_column_indices=range(0, 2),
        label_value_to_index={"1": 0, "2": 1},
        categorical_value_maps={1: {"Male": 1.0, "Female": 0.0}},
    )

    assert features.tolist() == [[40.0, 1.0], [50.0, 0.0]]


def test_rows_to_features_and_labels_rejects_an_unmapped_label() -> None:
    with pytest.raises(ValueError, match="unmapped label"):
        rows_to_features_and_labels(
            [["1", "Z"]],
            label_column_index=1,
            feature_column_indices=range(0, 1),
            label_value_to_index={"A": 0},
        )


def test_scale_feature_columns_min_max_leaves_a_constant_column_at_zero() -> None:
    features = np.array([[0.0, 7.0], [10.0, 7.0]], dtype=np.float32)

    scaled = scale_feature_columns(features, "min_max")

    assert scaled.tolist() == [[0.0, 0.0], [1.0, 0.0]]


def test_scale_feature_columns_standardize_centres_every_column() -> None:
    features = np.array([[1.0], [3.0]], dtype=np.float32)

    scaled = scale_feature_columns(features, "standardize")

    assert scaled.mean() == pytest.approx(0.0, abs=1e-5)


def _matlab_matrix_element(name: str, array: np.ndarray) -> bytes:
    """Encode one real numeric MAT-file matrix element (regular tag form)."""

    def tagged(type_code: int, payload: bytes) -> bytes:
        padding = b"\x00" * ((-len(payload)) % 8)
        return struct.pack("<II", type_code, len(payload)) + payload + padding

    flags = tagged(6, struct.pack("<II", 6, 0))  # uint32 array flags, class 6 = double
    dimensions = tagged(5, struct.pack("<ii", *array.shape))  # int32 dimensions
    name_bytes = tagged(1, name.encode("ascii"))  # int8 name
    values = tagged(9, array.astype("<f8").tobytes(order="F"))  # double payload
    body = flags + dimensions + name_bytes + values
    return struct.pack("<II", 14, len(body)) + body


def test_read_matlab_v5_arrays_reads_compressed_and_plain_elements(tmp_path: Path) -> None:
    feature_matrix = np.array([[1.0, 2.0], [3.0, 4.0]])
    label_vector = np.array([[-1.0], [1.0]])
    plain_element = _matlab_matrix_element("X", feature_matrix)
    compressed_payload = zlib.compress(_matlab_matrix_element("Y", label_vector))
    compressed_element = (
        struct.pack("<II", 15, len(compressed_payload)) + compressed_payload
    )
    mat_file = tmp_path / "tiny.mat"
    mat_file.write_bytes(b"\x00" * 128 + plain_element + compressed_element)

    named_arrays = read_matlab_v5_arrays(mat_file)

    assert named_arrays["X"].tolist() == [[1.0, 2.0], [3.0, 4.0]]
    assert named_arrays["Y"].reshape(-1).tolist() == [-1.0, 1.0]
