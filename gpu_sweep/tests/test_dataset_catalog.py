"""Catalog behaviour, exercised on fixture files - never on the network."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from gpu_sweep import dataset_catalog
from gpu_sweep.dataset_catalog import (
    DATASET_SPECS,
    load_features_and_labels,
    load_tabular_dataset,
    stratified_train_test_positions,
)

EXPECTED_DATASET_KEYS = (
    "diagnostic",
    "breast_original",
    "prognostic",
    "coimbra",
    "retinopathy",
    "dermatology",
    "ilpd",
    "lymphography",
    "parkinson_s",
    "spect",
    "cleveland",
    "heart_ew",
    "hepatitis",
    "saheart",
    "spectf_heart",
    "thyroid",
    "pima_diabetes",
    "leukemia",
    "colon",
    "prostate_ge",
)


def test_catalog_holds_the_twenty_distinct_paper_datasets() -> None:
    assert tuple(DATASET_SPECS) == EXPECTED_DATASET_KEYS


def test_catalog_excludes_the_duplicate_and_unfetchable_datasets() -> None:
    excluded = {"breast_ew", "heart", "parkinson_c", "covid19"}

    assert excluded.isdisjoint(DATASET_SPECS)


def test_every_spec_states_at_least_one_raw_file_and_two_classes() -> None:
    for dataset_key, spec in DATASET_SPECS.items():
        assert spec.raw_files, dataset_key
        assert spec.number_of_classes >= 2, dataset_key


def test_every_spec_standardizes_its_features() -> None:
    assert {spec.feature_scaling for spec in DATASET_SPECS.values()} == {"standardize"}


def test_load_features_and_labels_parses_a_delimited_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = tmp_path / "wdbc.data"
    fixture.write_text(
        "1,M," + ",".join(["1.0"] * 30) + "\n" + "2,B," + ",".join(["2.0"] * 30) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        dataset_catalog, "download_file_if_missing", lambda url, path: fixture
    )

    features, labels = load_features_and_labels(
        DATASET_SPECS["diagnostic"], cache_root=tmp_path
    )

    assert features.shape == (2, 30)
    assert labels.tolist() == [1, 0]


def test_load_features_and_labels_parses_a_matlab_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        dataset_catalog, "download_file_if_missing", lambda url, path: tmp_path / "x.mat"
    )
    monkeypatch.setattr(
        dataset_catalog,
        "read_matlab_v5_arrays",
        lambda path: {
            "X": np.array([[1.0, 2.0], [3.0, 4.0]]),
            "Y": np.array([[-1.0], [1.0]]),
        },
    )

    features, labels = load_features_and_labels(
        DATASET_SPECS["colon"], cache_root=tmp_path
    )

    assert features.shape == (2, 2)
    assert labels.tolist() == [0, 1]


def test_stratified_positions_keep_every_class_in_both_halves() -> None:
    labels = np.array([0] * 50 + [1] * 6, dtype=np.int64)

    train_positions, test_positions = stratified_train_test_positions(
        labels, train_fraction=0.66, random_seed=0
    )

    assert set(labels[train_positions]) == {0, 1}
    assert set(labels[test_positions]) == {0, 1}
    assert sorted([*train_positions, *test_positions]) == list(range(56))


def test_load_tabular_dataset_returns_a_disjoint_standardized_split(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = [
        f"{index},{'M' if index % 2 else 'B'}," + ",".join([str(float(index))] * 30)
        for index in range(20)
    ]
    fixture = tmp_path / "wdbc.data"
    fixture.write_text("\n".join(rows) + "\n", encoding="utf-8")
    monkeypatch.setattr(
        dataset_catalog, "download_file_if_missing", lambda url, path: fixture
    )

    dataset = load_tabular_dataset(
        DATASET_SPECS["diagnostic"],
        cache_root=tmp_path,
        train_fraction=0.66,
        random_seed=42,
    )

    assert dataset.number_of_features == 30
    assert dataset.number_of_classes == 2
    assert dataset.train_features.shape[0] + dataset.test_features.shape[0] == 20
    # The scaling is fitted on the training rows only: the training half
    # carries zero mean and unit variance per column, the test half need not.
    train_column_means = dataset.train_features.mean(dim=0)
    train_column_deviations = dataset.train_features.std(dim=0, unbiased=False)
    assert torch.allclose(train_column_means, torch.zeros(30), atol=1e-5)
    assert torch.allclose(train_column_deviations, torch.ones(30), atol=1e-4)
    assert sorted([*dataset.train_positions, *dataset.test_positions]) == list(range(20))
    assert dataset.feature_scaling.constant_feature_indices == ()


def _write_diagnostic_fixture(tmp_path: Path, feature_rows: list[list[float]]) -> Path:
    rows = [
        f"{index},{'M' if index % 2 else 'B'}," + ",".join(str(value) for value in features)
        for index, features in enumerate(feature_rows)
    ]
    fixture = tmp_path / "wdbc.data"
    fixture.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return fixture


def test_test_rows_do_not_influence_the_scaling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base_rows = [[float(index)] * 30 for index in range(20)]
    fixture = _write_diagnostic_fixture(tmp_path, base_rows)
    monkeypatch.setattr(dataset_catalog, "download_file_if_missing", lambda url, path: fixture)
    reference = load_tabular_dataset(
        DATASET_SPECS["diagnostic"], cache_root=tmp_path, random_seed=42
    )

    # Change only the test rows' values; the training half must scale the same.
    shifted_rows = [list(row) for row in base_rows]
    for position in reference.test_positions:
        shifted_rows[int(position)] = [value + 1000.0 for value in shifted_rows[int(position)]]
    fixture = _write_diagnostic_fixture(tmp_path, shifted_rows)
    shifted = load_tabular_dataset(
        DATASET_SPECS["diagnostic"], cache_root=tmp_path, random_seed=42
    )

    assert torch.equal(shifted.train_features, reference.train_features)
    assert np.array_equal(shifted.test_positions, reference.test_positions)


def test_a_column_constant_in_training_is_zero_in_both_halves(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    feature_rows = [[float(index)] * 30 for index in range(20)]
    probe = _write_diagnostic_fixture(tmp_path, feature_rows)
    monkeypatch.setattr(dataset_catalog, "download_file_if_missing", lambda url, path: probe)
    train_positions = load_tabular_dataset(
        DATASET_SPECS["diagnostic"], cache_root=tmp_path, random_seed=42
    ).train_positions
    for position, row in enumerate(feature_rows):
        # Column 0: constant on the training rows, different on the test rows.
        row[0] = 5.0 if position in set(train_positions.tolist()) else 9.0
    fixture = _write_diagnostic_fixture(tmp_path, feature_rows)
    monkeypatch.setattr(dataset_catalog, "download_file_if_missing", lambda url, path: fixture)

    dataset = load_tabular_dataset(
        DATASET_SPECS["diagnostic"], cache_root=tmp_path, random_seed=42
    )

    assert dataset.feature_scaling.constant_feature_indices == (0,)
    assert torch.all(dataset.train_features[:, 0] == 0.0)
    assert torch.all(dataset.test_features[:, 0] == 0.0)
    assert torch.isfinite(dataset.test_features).all()


def test_different_split_seeds_give_different_splits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _write_diagnostic_fixture(tmp_path, [[float(index)] * 30 for index in range(40)])
    monkeypatch.setattr(dataset_catalog, "download_file_if_missing", lambda url, path: fixture)

    first = load_tabular_dataset(DATASET_SPECS["diagnostic"], cache_root=tmp_path, random_seed=1042)
    again = load_tabular_dataset(DATASET_SPECS["diagnostic"], cache_root=tmp_path, random_seed=1042)
    other = load_tabular_dataset(DATASET_SPECS["diagnostic"], cache_root=tmp_path, random_seed=1043)

    assert np.array_equal(first.test_positions, again.test_positions)
    assert not np.array_equal(first.test_positions, other.test_positions)
    assert len(first.test_positions) == len(other.test_positions)
