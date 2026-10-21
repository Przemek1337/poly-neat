"""Cross-dataset primitives shared by the PolyNEAT example scripts.

Only what every dataset needs lives here: downloading and caching a raw file,
drawing a reproducible train/test split, and the shared bundle type every
classification example returns. Dataset-specific parsing lives next to the
examples that use it (``examples/<dataset>/dataset.py``).
"""

from __future__ import annotations

import urllib.request
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


def download_file_if_missing(source_url: str, destination_path: Path) -> Path:
    """Download ``source_url`` to ``destination_path`` unless it is already there.

    Args:
        source_url: URL to fetch on a cache miss.
        destination_path: Where the file is cached. Parent directories are
            created if needed.

    Returns:
        ``destination_path``, for chaining into a read.
    """
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    if not destination_path.exists():
        print(f"Downloading {source_url} to {destination_path} ...")
        urllib.request.urlretrieve(source_url, destination_path)
    return destination_path


def split_indices_into_train_and_test(
    number_of_samples: int,
    train_fraction: float,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Shuffle ``range(number_of_samples)`` and cut it into a train and test part.

    Indices are returned rather than the split data itself because some callers
    need to draw further subsets from within the training split.

    Args:
        number_of_samples: Total number of samples in the dataset.
        train_fraction: Share of samples assigned to the training split.
        random_seed: Seed for the shuffle, so that runs are reproducible.

    Returns:
        ``(train_indices, test_indices)`` as disjoint integer arrays whose union
        covers every sample index.
    """
    split_rng = np.random.default_rng(random_seed)
    shuffled_indices = split_rng.permutation(number_of_samples)
    train_size = int(train_fraction * number_of_samples)
    return shuffled_indices[:train_size], shuffled_indices[train_size:]


@dataclass(frozen=True)
class ClassificationDataset:
    """A train/test split of a classification dataset as one named bundle.

    Every classification example (iris, mnist) returns this exact type from its
    top-level loader, so their call sites read identically: reach for
    ``.train_features`` / ``.train_labels`` / ``.test_features`` /
    ``.test_labels`` rather than unpacking a positional tuple whose order every
    caller has to remember.
    """

    train_features: torch.Tensor  # [n_train, n_features] float32
    train_labels: torch.Tensor  # [n_train] long
    test_features: torch.Tensor  # [n_test, n_features] float32
    test_labels: torch.Tensor  # [n_test] long
    number_of_classes: int

    @property
    def number_of_features(self) -> int:
        """Length of one feature vector (the network's input width)."""
        return int(self.train_features.shape[1])


def build_dataset_from_official_splits(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    test_features: torch.Tensor,
    test_labels: torch.Tensor,
    *,
    random_seed: int,
    max_train_samples: int | None = None,
    max_test_samples: int | None = None,
    number_of_classes: int | None = None,
) -> ClassificationDataset:
    """Keep an upstream dataset's official split and optionally subsample it.

    Unlike :func:`split_features_and_labels`, this helper never mixes official
    test rows back into the training pool.  It is the correct entry point for
    benchmark datasets such as MNIST, Fashion-MNIST and CIFAR-10, whose
    distributions already provide a canonical held-out test set.
    """
    if train_features.shape[0] != train_labels.shape[0]:
        raise ValueError("train features/labels sample counts do not match")
    if test_features.shape[0] != test_labels.shape[0]:
        raise ValueError("test features/labels sample counts do not match")

    subset_rng = np.random.default_rng(random_seed)
    train_indices = np.arange(train_labels.shape[0])
    test_indices = np.arange(test_labels.shape[0])
    if max_train_samples is not None and len(train_indices) > max_train_samples:
        train_indices = subset_rng.choice(
            train_indices, size=max_train_samples, replace=False
        )
    if max_test_samples is not None and len(test_indices) > max_test_samples:
        test_indices = subset_rng.choice(test_indices, size=max_test_samples, replace=False)

    if number_of_classes is None:
        all_labels = torch.cat([train_labels, test_labels])
        number_of_classes = int(all_labels.max()) + 1

    return ClassificationDataset(
        train_features=train_features[train_indices].to(torch.float32),
        train_labels=train_labels[train_indices].to(torch.long),
        test_features=test_features[test_indices].to(torch.float32),
        test_labels=test_labels[test_indices].to(torch.long),
        number_of_classes=number_of_classes,
    )


def standardize_feature_splits_from_training_statistics(
    train_features: torch.Tensor,
    *other_feature_splits: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """Fit element-wise standardization on train and transform every split.

    The batch dimension is the only reduced dimension.  Consequently this is
    usable for both flattened ``[N, F]`` inputs and image tensors
    ``[N, C, H, W]``.  Validation and test values never influence the fitted
    mean or standard deviation.
    """
    if train_features.shape[0] < 1:
        raise ValueError("cannot fit standardization on an empty training split")
    train_features = train_features.to(torch.float32)
    training_mean = train_features.mean(dim=0, keepdim=True)
    training_std = train_features.std(dim=0, unbiased=False, keepdim=True) + 1e-6
    return tuple(
        (feature_split.to(torch.float32) - training_mean) / training_std
        for feature_split in (train_features, *other_feature_splits)
    )


def split_features_and_labels(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    train_fraction: float,
    random_seed: int,
    max_train_samples: int | None = None,
    max_test_samples: int | None = None,
    number_of_classes: int | None = None,
) -> ClassificationDataset:
    """Shuffle-split ``(features, labels)`` into a :class:`ClassificationDataset`.

    The single seam every classification example splits through, so grid size,
    split fraction and subset caps are all tweakable knobs on one call.

    Args:
        features: ``[n_samples, n_features]`` float tensor.
        labels: ``[n_samples]`` long tensor of class label indices.
        train_fraction: Share of samples assigned to the training split.
        random_seed: Seed for both the split and the optional subset draw, so
            repeats see the same data.
        max_train_samples: If set and smaller than the training split, randomly
            subsample the training split to this many rows (for tractability on
            large sets like MNIST). ``None`` keeps the whole split.
        max_test_samples: As above, for the test split.
        number_of_classes: Number of classes. ``None`` infers it as
            ``int(labels.max()) + 1`` over the full label tensor.

    Returns:
        The assembled :class:`ClassificationDataset`.
    """
    train_indices, test_indices = split_indices_into_train_and_test(
        number_of_samples=len(labels),
        train_fraction=train_fraction,
        random_seed=random_seed,
    )

    subset_rng = np.random.default_rng(random_seed)
    if max_train_samples is not None and len(train_indices) > max_train_samples:
        train_indices = subset_rng.choice(
            train_indices, size=max_train_samples, replace=False
        )
    if max_test_samples is not None and len(test_indices) > max_test_samples:
        test_indices = subset_rng.choice(
            test_indices, size=max_test_samples, replace=False
        )

    if number_of_classes is None:
        number_of_classes = int(labels.max()) + 1

    return ClassificationDataset(
        train_features=features[train_indices],
        train_labels=labels[train_indices],
        test_features=features[test_indices],
        test_labels=labels[test_indices],
        number_of_classes=number_of_classes,
    )


@dataclass(frozen=True)
class GroupedSplitResult:
    """A grouped, stratified split of a pool of examples.

    Attributes:
        split_of_group: Which split each group id was assigned to.
        counts_by_split_and_label: Realised ``(split, label) -> count``, so a
            caller can record the deviation from the requested fractions
            instead of assuming it hit them.
        requested_fractions: The fractions asked for, kept alongside the counts
            for the same reason.
    """

    split_of_group: dict[str, str]
    counts_by_split_and_label: dict[tuple[str, int], int]
    requested_fractions: dict[str, float]

    def split_of_each_example(self, group_ids: Sequence[str]) -> list[str]:
        """Expand the per-group assignment back to one split name per example."""
        return [self.split_of_group[group_id] for group_id in group_ids]

    def realised_fraction(self, split_name: str, total_number_of_examples: int) -> float:
        """Share of the pool that ended up in ``split_name``."""
        if total_number_of_examples <= 0:
            return 0.0
        assigned = sum(
            count
            for (split, _label), count in self.counts_by_split_and_label.items()
            if split == split_name
        )
        return assigned / total_number_of_examples


def split_groups_into_stratified_parts(
    *,
    group_ids: Sequence[str],
    labels: Sequence[int],
    split_fractions: Mapping[str, float],
    random_seed: int,
    require_every_label_in_every_split: bool = True,
) -> GroupedSplitResult:
    """Split a pool into named parts without ever separating a group.

    Group disjointness comes first, stratification second and the exact
    fractions last: whole groups are the unit of assignment, so the realised
    sizes can only approximate the requested ones. The realised counts are
    returned rather than assumed, and the caller is expected to record them.

    Groups are visited largest first, so the coarse-grained decisions happen
    while there is still room to balance them; ties are broken by a seeded
    shuffle, which is what makes the result depend on ``random_seed`` at all.
    Each group goes to the split that is furthest from meeting its target for
    the labels the group actually contains.

    Args:
        group_ids: One group id per example. Examples sharing an id are never
            separated.
        labels: One integer class label per example, aligned with ``group_ids``.
        split_fractions: Requested share per split name, e.g.
            ``{"train": 0.7, "search_validation": 0.15,
            "threshold_validation": 0.15}``. Values must be positive and sum to
            approximately one.
        random_seed: Seed for the tie-breaking shuffle.
        require_every_label_in_every_split: When ``True`` (the default), a
            split that ended up without one of the labels is repaired by moving
            the smallest eligible group into it, and a pool that cannot satisfy
            this raises instead of silently producing a single-class split.

    Returns:
        The :class:`GroupedSplitResult` with the per-group assignment and the
        realised counts.

    Raises:
        ValueError: If the inputs disagree in length, are empty, if the
            fractions are not positive or do not sum to one, or if every label
            cannot be placed in every split.
    """
    if len(group_ids) != len(labels):
        raise ValueError(
            f"group_ids has {len(group_ids)} entries but labels has {len(labels)}"
        )
    if not group_ids:
        raise ValueError("cannot split an empty pool")
    if not split_fractions:
        raise ValueError("split_fractions must name at least one split")
    if any(fraction <= 0.0 for fraction in split_fractions.values()):
        raise ValueError(f"every split fraction must be > 0, got {dict(split_fractions)}")
    fraction_total = sum(split_fractions.values())
    if abs(fraction_total - 1.0) > 1e-6:
        raise ValueError(f"split fractions must sum to 1.0, got {fraction_total}")

    label_counts_by_group: dict[str, Counter[int]] = defaultdict(Counter)
    for group_id, label in zip(group_ids, labels, strict=True):
        label_counts_by_group[group_id][int(label)] += 1

    all_labels = sorted({int(label) for label in labels})
    total_count_by_label = Counter(int(label) for label in labels)
    split_names = sorted(split_fractions)
    target_by_split_and_label = {
        (split_name, label): split_fractions[split_name] * total_count_by_label[label]
        for split_name in split_names
        for label in all_labels
    }

    shuffle_rng = np.random.default_rng(random_seed)
    shuffled_group_ids = list(label_counts_by_group)
    shuffle_rng.shuffle(shuffled_group_ids)
    ordered_group_ids = sorted(
        shuffled_group_ids,
        key=lambda group_id: -sum(label_counts_by_group[group_id].values()),
    )

    assigned_by_split_and_label: Counter[tuple[str, int]] = Counter()
    split_of_group: dict[str, str] = {}
    for group_id in ordered_group_ids:
        group_label_counts = label_counts_by_group[group_id]
        best_split_name = min(
            split_names,
            key=lambda split_name: (
                max(
                    (assigned_by_split_and_label[(split_name, label)] + count)
                    / max(target_by_split_and_label[(split_name, label)], 1e-9)
                    for label, count in group_label_counts.items()
                ),
                split_name,
            ),
        )
        split_of_group[group_id] = best_split_name
        for label, count in group_label_counts.items():
            assigned_by_split_and_label[(best_split_name, label)] += count

    if require_every_label_in_every_split:
        _repair_missing_labels(
            split_names=split_names,
            all_labels=all_labels,
            label_counts_by_group=label_counts_by_group,
            split_of_group=split_of_group,
            assigned_by_split_and_label=assigned_by_split_and_label,
        )

    return GroupedSplitResult(
        split_of_group=split_of_group,
        counts_by_split_and_label={
            (split_name, label): assigned_by_split_and_label[(split_name, label)]
            for split_name in split_names
            for label in all_labels
        },
        requested_fractions=dict(split_fractions),
    )


def _repair_missing_labels(
    *,
    split_names: Sequence[str],
    all_labels: Sequence[int],
    label_counts_by_group: Mapping[str, Counter[int]],
    split_of_group: dict[str, str],
    assigned_by_split_and_label: Counter[tuple[str, int]],
) -> None:
    """Move whole groups until every split holds every label.

    The smallest donor group is preferred so the repair disturbs the requested
    proportions as little as possible, and a donor is only taken from a split
    that keeps at least one example of that label afterwards.

    Raises:
        ValueError: If some split cannot be given a label without emptying it
            somewhere else. A single-class split would silently invalidate
            class weighting, threshold selection and every ranking metric, so
            this fails loudly instead.
    """
    for split_name in split_names:
        for label in all_labels:
            if assigned_by_split_and_label[(split_name, label)] > 0:
                continue
            donor_group_id = _find_donor_group(
                label=label,
                receiving_split_name=split_name,
                label_counts_by_group=label_counts_by_group,
                split_of_group=split_of_group,
                assigned_by_split_and_label=assigned_by_split_and_label,
            )
            if donor_group_id is None:
                raise ValueError(
                    f"cannot place label {label} into split {split_name!r} without leaving "
                    "another split without it; the pool has too few groups carrying that "
                    "label for the requested split layout"
                )
            donor_split_name = split_of_group[donor_group_id]
            for moved_label, count in label_counts_by_group[donor_group_id].items():
                assigned_by_split_and_label[(donor_split_name, moved_label)] -= count
                assigned_by_split_and_label[(split_name, moved_label)] += count
            split_of_group[donor_group_id] = split_name


def _find_donor_group(
    *,
    label: int,
    receiving_split_name: str,
    label_counts_by_group: Mapping[str, Counter[int]],
    split_of_group: Mapping[str, str],
    assigned_by_split_and_label: Counter[tuple[str, int]],
) -> str | None:
    """Smallest group carrying ``label`` whose move leaves its donor split valid."""
    eligible_group_ids = [
        group_id
        for group_id, counts in label_counts_by_group.items()
        if counts.get(label, 0) > 0
        and split_of_group[group_id] != receiving_split_name
        and assigned_by_split_and_label[(split_of_group[group_id], label)] > counts[label]
    ]
    if not eligible_group_ids:
        return None
    return min(
        eligible_group_ids,
        key=lambda group_id: (sum(label_counts_by_group[group_id].values()), group_id),
    )
