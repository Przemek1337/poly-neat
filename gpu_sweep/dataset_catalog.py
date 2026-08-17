"""The paper's tabular medical datasets, each fetched from an anonymous URL.

Twenty of the paper's twenty-four entries are here. BreastEW is dropped as a
duplicate of Diagnostic (the same WDBC file; the paper's 596 is a typo for 569)
and the Kaggle Heart set as a duplicate of Cleveland. ParkinsonC (a .rar inside
a UCI zip) and COVID-19 (Kaggle, authenticated) have no anonymous download and
are out of scope for an automatic sweep.

Where a KEEL source is unreachable (sci2s.ugr.es fails TLS verification), the
dataset is taken from its original upstream instead.

Every dataset is standardised per column - zero mean, unit variance - before
the split. One scaling for all twenty keeps runtimes and fitness curves
comparable across the sweep, and it is the scaling the wide microarray sets
need anyway: their expression values span several orders of magnitude, which
min-max would squash into a narrow band near zero. Statistics are taken over
all rows, before the split, so the test half contributes to the column mean and
standard deviation. That is a mild leak; it is deliberate here, because the
sweep measures whether a dataset runs on the GPU and whether fitness moves, not
generalisation quality.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from gpu_sweep.raw_parsing import (
    download_file_if_missing,
    read_arff_data_rows,
    read_delimited_rows,
    read_matlab_v5_arrays,
    rows_to_features_and_labels,
    scale_feature_columns,
)

DEFAULT_CACHE_ROOT = Path(__file__).resolve().parent.parent / "gpu_sweep_data"

_UCI = "https://archive.ics.uci.edu/ml/machine-learning-databases"
_SKFEATURE = "https://raw.githubusercontent.com/jundongl/scikit-feature/master/skfeature/data"


@dataclass(frozen=True)
class TabularDatasetSpec:
    """Everything needed to turn one dataset's raw files into a numeric matrix.

    Attributes:
        dataset_key: Catalog key, also the cache subdirectory name.
        human_name: Name as the paper's table spells it.
        reader: ``"delimited"``, ``"whitespace"``, ``"arff"`` or ``"matlab"``.
        raw_files: ``(file_name, source_url)`` pairs; several files are read in
            order and concatenated, which is how SPECT, SPECTF and thyroid ship.
        number_of_classes: Class count after label mapping.
        feature_scaling: ``"standardize"`` everywhere - see the module
            docstring for why the sweep does not mix scalings.
        expected_shape: ``(n_samples, n_features)`` this plan verified.
        delimiter/skip_header_rows/label_column_index/feature_column_indices/
        label_value_to_index/categorical_value_maps/missing_value_handling:
            Text-reader arguments; unused by the MATLAB reader.
        matlab_label_value_to_index: Maps the raw ``Y`` values of a MAT-file to
            class indices; unused by the text readers.
    """

    dataset_key: str
    human_name: str
    reader: str
    raw_files: tuple[tuple[str, str], ...]
    number_of_classes: int
    feature_scaling: str
    expected_shape: tuple[int, int]
    delimiter: str | None = ","
    skip_header_rows: int = 0
    label_column_index: int = 0
    feature_column_indices: Sequence[int] = ()
    label_value_to_index: dict[str, int] = field(default_factory=dict)
    categorical_value_maps: dict[int, dict[str, float]] = field(default_factory=dict)
    missing_value_handling: str = "drop_row"
    matlab_label_value_to_index: dict[int, int] = field(default_factory=dict)


@dataclass(frozen=True)
class TabularDataset:
    """A stratified train/test split of one dataset, ready for a fitness evaluator."""

    dataset_key: str
    train_features: torch.Tensor  # [n_train, n_features] float32
    train_labels: torch.Tensor  # [n_train] long
    test_features: torch.Tensor  # [n_test, n_features] float32
    test_labels: torch.Tensor  # [n_test] long
    number_of_classes: int

    @property
    def number_of_features(self) -> int:
        """Length of one feature vector (the network's input width)."""
        return int(self.train_features.shape[1])

    @property
    def number_of_samples(self) -> int:
        """Rows across both halves of the split."""
        return int(self.train_features.shape[0] + self.test_features.shape[0])


DATASET_SPECS: dict[str, TabularDatasetSpec] = {
    "diagnostic": TabularDatasetSpec(
        dataset_key="diagnostic",
        human_name="Diagnostic (WDBC)",
        reader="delimited",
        raw_files=(("wdbc.data", f"{_UCI}/breast-cancer-wisconsin/wdbc.data"),),
        number_of_classes=2,
        feature_scaling="standardize",
        expected_shape=(569, 30),
        label_column_index=1,
        feature_column_indices=range(2, 32),
        label_value_to_index={"B": 0, "M": 1},
        missing_value_handling="drop_row",
    ),
    "breast_original": TabularDatasetSpec(
        dataset_key="breast_original",
        human_name="BreastOriginal",
        reader="delimited",
        raw_files=(
            (
                "breast-cancer-wisconsin.data",
                f"{_UCI}/breast-cancer-wisconsin/breast-cancer-wisconsin.data",
            ),
        ),
        number_of_classes=2,
        feature_scaling="standardize",
        expected_shape=(699, 9),
        label_column_index=10,
        feature_column_indices=range(1, 10),
        label_value_to_index={"2": 0, "4": 1},
        missing_value_handling="column_mean",
    ),
    "prognostic": TabularDatasetSpec(
        dataset_key="prognostic",
        human_name="Prognostic (WPBC)",
        reader="delimited",
        raw_files=(("wpbc.data", f"{_UCI}/breast-cancer-wisconsin/wpbc.data"),),
        number_of_classes=2,
        feature_scaling="standardize",
        expected_shape=(194, 33),
        label_column_index=1,
        feature_column_indices=range(2, 35),
        label_value_to_index={"N": 0, "R": 1},
        missing_value_handling="drop_row",
    ),
    "coimbra": TabularDatasetSpec(
        dataset_key="coimbra",
        human_name="Coimbra",
        reader="delimited",
        raw_files=(("dataR2.csv", f"{_UCI}/00451/dataR2.csv"),),
        number_of_classes=2,
        feature_scaling="standardize",
        expected_shape=(116, 9),
        skip_header_rows=1,
        label_column_index=9,
        feature_column_indices=range(0, 9),
        label_value_to_index={"1": 0, "2": 1},
        missing_value_handling="drop_row",
    ),
    "retinopathy": TabularDatasetSpec(
        dataset_key="retinopathy",
        human_name="Retinopathy",
        reader="arff",
        raw_files=(("messidor_features.arff", f"{_UCI}/00329/messidor_features.arff"),),
        number_of_classes=2,
        feature_scaling="standardize",
        expected_shape=(1151, 19),
        label_column_index=19,
        feature_column_indices=range(0, 19),
        label_value_to_index={"0": 0, "1": 1},
        missing_value_handling="drop_row",
    ),
    "dermatology": TabularDatasetSpec(
        dataset_key="dermatology",
        human_name="Dermatology",
        reader="delimited",
        raw_files=(("dermatology.data", f"{_UCI}/dermatology/dermatology.data"),),
        number_of_classes=6,
        feature_scaling="standardize",
        expected_shape=(366, 34),
        label_column_index=34,
        feature_column_indices=range(0, 34),
        label_value_to_index={"1": 0, "2": 1, "3": 2, "4": 3, "5": 4, "6": 5},
        missing_value_handling="column_mean",
    ),
    "ilpd": TabularDatasetSpec(
        dataset_key="ilpd",
        human_name="ILPD",
        reader="delimited",
        raw_files=(
            (
                "ilpd.csv",
                f"{_UCI}/00225/Indian%20Liver%20Patient%20Dataset%20(ILPD).csv",
            ),
        ),
        number_of_classes=2,
        feature_scaling="standardize",
        expected_shape=(583, 10),
        label_column_index=10,
        feature_column_indices=range(0, 10),
        label_value_to_index={"1": 0, "2": 1},
        categorical_value_maps={1: {"Female": 0.0, "Male": 1.0}},
        missing_value_handling="column_mean",
    ),
    "lymphography": TabularDatasetSpec(
        dataset_key="lymphography",
        human_name="Lymphography",
        reader="delimited",
        raw_files=(("lymphography.data", f"{_UCI}/lymphography/lymphography.data"),),
        number_of_classes=4,
        feature_scaling="standardize",
        expected_shape=(148, 18),
        label_column_index=0,
        feature_column_indices=range(1, 19),
        label_value_to_index={"1": 0, "2": 1, "3": 2, "4": 3},
        missing_value_handling="drop_row",
    ),
    "parkinson_s": TabularDatasetSpec(
        dataset_key="parkinson_s",
        human_name="ParkinsonS",
        reader="delimited",
        raw_files=(("parkinsons.data", f"{_UCI}/parkinsons/parkinsons.data"),),
        number_of_classes=2,
        feature_scaling="standardize",
        expected_shape=(195, 22),
        skip_header_rows=1,
        label_column_index=17,
        feature_column_indices=[*range(1, 17), *range(18, 24)],
        label_value_to_index={"0": 0, "1": 1},
        missing_value_handling="drop_row",
    ),
    "spect": TabularDatasetSpec(
        dataset_key="spect",
        human_name="SPECT",
        reader="delimited",
        raw_files=(
            ("SPECT.train", f"{_UCI}/spect/SPECT.train"),
            ("SPECT.test", f"{_UCI}/spect/SPECT.test"),
        ),
        number_of_classes=2,
        feature_scaling="standardize",
        expected_shape=(267, 22),
        label_column_index=0,
        feature_column_indices=range(1, 23),
        label_value_to_index={"0": 0, "1": 1},
        missing_value_handling="drop_row",
    ),
    "cleveland": TabularDatasetSpec(
        dataset_key="cleveland",
        human_name="Cleveland",
        reader="delimited",
        raw_files=(
            ("processed.cleveland.data", f"{_UCI}/heart-disease/processed.cleveland.data"),
        ),
        number_of_classes=5,
        feature_scaling="standardize",
        expected_shape=(297, 13),
        label_column_index=13,
        feature_column_indices=range(0, 13),
        label_value_to_index={"0": 0, "1": 1, "2": 2, "3": 3, "4": 4},
        missing_value_handling="drop_row",
    ),
    "heart_ew": TabularDatasetSpec(
        dataset_key="heart_ew",
        human_name="HeartEW",
        reader="whitespace",
        raw_files=(("heart.dat", f"{_UCI}/statlog/heart/heart.dat"),),
        number_of_classes=2,
        feature_scaling="standardize",
        expected_shape=(270, 13),
        delimiter=None,
        label_column_index=13,
        feature_column_indices=range(0, 13),
        label_value_to_index={"1": 0, "2": 1},
        missing_value_handling="drop_row",
    ),
    "hepatitis": TabularDatasetSpec(
        dataset_key="hepatitis",
        human_name="Hepatitis",
        reader="delimited",
        raw_files=(("hepatitis.data", f"{_UCI}/hepatitis/hepatitis.data"),),
        number_of_classes=2,
        feature_scaling="standardize",
        expected_shape=(80, 19),
        label_column_index=0,
        feature_column_indices=range(1, 20),
        label_value_to_index={"1": 0, "2": 1},
        missing_value_handling="drop_row",
    ),
    "saheart": TabularDatasetSpec(
        dataset_key="saheart",
        human_name="Saheart",
        reader="delimited",
        raw_files=(
            ("SAheart.data", "https://hastie.su.domains/ElemStatLearn/datasets/SAheart.data"),
        ),
        number_of_classes=2,
        feature_scaling="standardize",
        expected_shape=(462, 9),
        skip_header_rows=1,
        label_column_index=10,
        feature_column_indices=range(1, 10),
        label_value_to_index={"0": 0, "1": 1},
        categorical_value_maps={5: {"Absent": 0.0, "Present": 1.0}},
        missing_value_handling="drop_row",
    ),
    "spectf_heart": TabularDatasetSpec(
        dataset_key="spectf_heart",
        human_name="Spectfheart",
        reader="delimited",
        raw_files=(
            ("SPECTF.train", f"{_UCI}/spect/SPECTF.train"),
            ("SPECTF.test", f"{_UCI}/spect/SPECTF.test"),
        ),
        number_of_classes=2,
        feature_scaling="standardize",
        expected_shape=(267, 44),
        label_column_index=0,
        feature_column_indices=range(1, 45),
        label_value_to_index={"0": 0, "1": 1},
        missing_value_handling="drop_row",
    ),
    "thyroid": TabularDatasetSpec(
        dataset_key="thyroid",
        human_name="Thyroid",
        reader="whitespace",
        raw_files=(
            ("ann-train.data", f"{_UCI}/thyroid-disease/ann-train.data"),
            ("ann-test.data", f"{_UCI}/thyroid-disease/ann-test.data"),
        ),
        number_of_classes=3,
        feature_scaling="standardize",
        expected_shape=(7200, 21),
        delimiter=None,
        label_column_index=21,
        feature_column_indices=range(0, 21),
        label_value_to_index={"1": 0, "2": 1, "3": 2},
        missing_value_handling="drop_row",
    ),
    "pima_diabetes": TabularDatasetSpec(
        dataset_key="pima_diabetes",
        human_name="PimaDiabetes",
        reader="delimited",
        raw_files=(
            (
                "pima-indians-diabetes.csv",
                "https://raw.githubusercontent.com/jbrownlee/Datasets/master/"
                "pima-indians-diabetes.csv",
            ),
        ),
        number_of_classes=2,
        feature_scaling="standardize",
        expected_shape=(768, 8),
        label_column_index=8,
        feature_column_indices=range(0, 8),
        label_value_to_index={"0": 0, "1": 1},
        missing_value_handling="drop_row",
    ),
    "leukemia": TabularDatasetSpec(
        dataset_key="leukemia",
        human_name="Leukemia",
        reader="matlab",
        raw_files=(("leukemia.mat", f"{_SKFEATURE}/leukemia.mat"),),
        number_of_classes=2,
        feature_scaling="standardize",
        expected_shape=(72, 7070),
        matlab_label_value_to_index={-1: 0, 1: 1},
    ),
    "colon": TabularDatasetSpec(
        dataset_key="colon",
        human_name="Colon",
        reader="matlab",
        raw_files=(("colon.mat", f"{_SKFEATURE}/colon.mat"),),
        number_of_classes=2,
        feature_scaling="standardize",
        expected_shape=(62, 2000),
        matlab_label_value_to_index={-1: 0, 1: 1},
    ),
    "prostate_ge": TabularDatasetSpec(
        dataset_key="prostate_ge",
        human_name="ProstateGE",
        reader="matlab",
        raw_files=(("Prostate-GE.mat", f"{_SKFEATURE}/Prostate-GE.mat"),),
        number_of_classes=2,
        feature_scaling="standardize",
        expected_shape=(102, 5966),
        matlab_label_value_to_index={1: 0, 2: 1},
    ),
}


def load_features_and_labels(
    spec: TabularDatasetSpec, *, cache_root: Path
) -> tuple[torch.Tensor, torch.Tensor]:
    """Download (once) and parse one dataset into scaled features and labels.

    Args:
        spec: Catalog entry describing the files and their layout.
        cache_root: Directory the raw files are cached under, one
            subdirectory per dataset key.

    Returns:
        ``(features, labels)`` as a ``[n_samples, n_features]`` float32 tensor
        and an ``[n_samples]`` long tensor.
    """
    dataset_directory = cache_root / spec.dataset_key

    if spec.reader == "matlab":
        data_path = download_file_if_missing(
            spec.raw_files[0][1], dataset_directory / spec.raw_files[0][0]
        )
        named_arrays = read_matlab_v5_arrays(data_path)
        features = scale_feature_columns(
            named_arrays["X"].astype(np.float32), spec.feature_scaling
        )
        labels = np.array(
            [
                spec.matlab_label_value_to_index[int(raw_label)]
                for raw_label in named_arrays["Y"].reshape(-1)
            ],
            dtype=np.int64,
        )
        return torch.from_numpy(features), torch.from_numpy(labels)

    rows: list[list[str]] = []
    for file_name, source_url in spec.raw_files:
        data_path = download_file_if_missing(source_url, dataset_directory / file_name)
        if spec.reader == "arff":
            rows.extend(read_arff_data_rows(data_path))
        else:
            rows.extend(
                read_delimited_rows(
                    data_path,
                    delimiter=spec.delimiter,
                    skip_header_rows=spec.skip_header_rows,
                )
            )

    features, labels = rows_to_features_and_labels(
        rows,
        label_column_index=spec.label_column_index,
        feature_column_indices=spec.feature_column_indices,
        label_value_to_index=spec.label_value_to_index,
        categorical_value_maps=spec.categorical_value_maps or None,
        missing_value_handling=spec.missing_value_handling,
    )
    return (
        torch.from_numpy(scale_feature_columns(features, spec.feature_scaling)),
        torch.from_numpy(labels),
    )


def stratified_train_test_positions(
    labels: np.ndarray, *, train_fraction: float, random_seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Split row positions per class so both halves see every class.

    A plain shuffle-split can leave a rare class out of the training half -
    Cleveland's rarest class is 13 of 297 rows - and the per-class evaluators
    C-NEAT and L-NEAT use raise ``ValueError`` when that happens. Splitting
    within each class removes the failure mode entirely.

    Args:
        labels: ``[n_samples]`` array of class indices.
        train_fraction: Share of each class assigned to training.
        random_seed: Seed for the per-class shuffle.

    Returns:
        ``(train_positions, test_positions)`` as disjoint sorted int arrays
        whose union covers every row.
    """
    split_rng = np.random.default_rng(random_seed)
    train_positions: list[int] = []
    test_positions: list[int] = []
    for class_index in np.unique(labels):
        positions_of_class = np.flatnonzero(labels == class_index)
        split_rng.shuffle(positions_of_class)
        train_size = int(round(train_fraction * len(positions_of_class)))
        train_size = max(1, min(len(positions_of_class) - 1, train_size))
        train_positions.extend(positions_of_class[:train_size].tolist())
        test_positions.extend(positions_of_class[train_size:].tolist())
    return (
        np.array(sorted(train_positions), dtype=np.int64),
        np.array(sorted(test_positions), dtype=np.int64),
    )


def load_tabular_dataset(
    spec: TabularDatasetSpec,
    *,
    cache_root: Path = DEFAULT_CACHE_ROOT,
    train_fraction: float = 0.66,
    random_seed: int = 42,
) -> TabularDataset:
    """Load one dataset and cut it into a stratified train/test split.

    Args:
        spec: Catalog entry to load.
        cache_root: Raw file cache directory.
        train_fraction: Share of each class assigned to training.
        random_seed: Seed for the split.

    Returns:
        The assembled :class:`TabularDataset`.
    """
    features, labels = load_features_and_labels(spec, cache_root=cache_root)
    train_positions, test_positions = stratified_train_test_positions(
        labels.numpy(), train_fraction=train_fraction, random_seed=random_seed
    )
    return TabularDataset(
        dataset_key=spec.dataset_key,
        train_features=features[train_positions],
        train_labels=labels[train_positions],
        test_features=features[test_positions],
        test_labels=labels[test_positions],
        number_of_classes=spec.number_of_classes,
    )
