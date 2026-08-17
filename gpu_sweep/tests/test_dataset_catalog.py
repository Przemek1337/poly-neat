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
    # Standardisation runs over all rows before the split, so the two halves
    # together - not either half alone - carry zero mean and unit variance.
    all_features = torch.cat([dataset.train_features, dataset.test_features])
    assert float(all_features.mean()) == pytest.approx(0.0, abs=1e-4)
    assert float(all_features.std(unbiased=False)) == pytest.approx(1.0, abs=1e-3)
