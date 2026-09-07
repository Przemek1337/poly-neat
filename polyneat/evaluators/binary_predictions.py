"""One frozen model's predictions on one split, with the ids to trace them.

Everything downstream - metrics, threshold selection, the bootstrap, the paired
comparison between two methods - works on this record rather than on a bare
array of scores. Carrying the example and group ids alongside the numbers is
what makes a grouped bootstrap and a paired comparison possible at all: two
methods can only be compared on the same drawn ids if both know which id each
row belongs to.

Both the raw logits and the derived probability are stored. The ranking score
is ``softmax(logits)[:, 1]``, the probability of the positive class, never
``argmax``: a hard label throws away the ordering that AUROC and average
precision are computed from.

A non-finite score is not a prediction. It means the model produced NaN or Inf,
which the protocol records as a failed evaluation, so it raises here instead of
being silently ranked at the bottom.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch


class PredictionValidationError(ValueError):
    """Raised when a prediction set is empty, misaligned or otherwise unusable."""


class NonFinitePredictionError(PredictionValidationError):
    """Raised when a model produced NaN or Inf.

    Separate from the other validation failures because the protocol treats it
    differently: it marks the evaluation failed and records the cost, rather
    than scoring the candidate zero and letting it compete.
    """


@dataclass(frozen=True)
class BinaryPredictions:
    """Scores of one model on one split, aligned with stable ids.

    Attributes:
        example_ids: Manifest ids, one per row.
        group_ids: Split group of each row, used by the grouped bootstrap.
        labels: Ground-truth class indices, ``0`` or ``1``.
        positive_class_probabilities: ``softmax(logits)[:, 1]`` per row.
        logits: The raw two-column output, kept so a checkpoint can be
            re-verified without re-deriving probabilities from a summary.
        model_id: Which model produced this, e.g. a checkpoint id.
        stage: Which stage produced it, e.g. ``track_a`` or ``track_b_seed2``.
        split_name: Which split it was computed on.
    """

    example_ids: tuple[str, ...]
    group_ids: tuple[str, ...]
    labels: torch.Tensor
    positive_class_probabilities: torch.Tensor
    logits: torch.Tensor
    model_id: str
    stage: str
    split_name: str

    def __post_init__(self) -> None:
        number_of_rows = len(self.example_ids)
        if number_of_rows == 0:
            raise PredictionValidationError(
                f"prediction set for model {self.model_id!r} on split {self.split_name!r} is empty"
            )
        lengths = {
            "example_ids": number_of_rows,
            "group_ids": len(self.group_ids),
            "labels": int(self.labels.shape[0]),
            "probabilities": int(self.positive_class_probabilities.shape[0]),
            "logits": int(self.logits.shape[0]),
        }
        if len(set(lengths.values())) != 1:
            raise PredictionValidationError(f"prediction columns are misaligned: {lengths}")
        if not torch.isfinite(self.positive_class_probabilities).all():
            raise NonFinitePredictionError(
                f"model {self.model_id!r} produced non-finite probabilities on split "
                f"{self.split_name!r}; this is a failed evaluation, not a score of zero"
            )
        if not torch.isfinite(self.logits).all():
            raise NonFinitePredictionError(
                f"model {self.model_id!r} produced non-finite logits on split "
                f"{self.split_name!r}; this is a failed evaluation, not a score of zero"
            )

    def __len__(self) -> int:
        return len(self.example_ids)

    @property
    def number_of_positives(self) -> int:
        """Rows whose ground-truth label is the positive class."""
        return int((self.labels == 1).sum())

    @property
    def number_of_negatives(self) -> int:
        """Rows whose ground-truth label is the negative class."""
        return int((self.labels == 0).sum())

    @property
    def contains_both_classes(self) -> bool:
        """Whether both classes occur, which every ranking metric requires."""
        return self.number_of_positives > 0 and self.number_of_negatives > 0

    def select_rows(self, row_indices: torch.Tensor) -> BinaryPredictions:
        """Return the same prediction set restricted to ``row_indices``.

        Used by the bootstrap, which resamples rows with replacement. The ids
        travel with the rows so a replicate can still be traced.
        """
        index_list = row_indices.tolist()
        return BinaryPredictions(
            example_ids=tuple(self.example_ids[index] for index in index_list),
            group_ids=tuple(self.group_ids[index] for index in index_list),
            labels=self.labels[row_indices],
            positive_class_probabilities=self.positive_class_probabilities[row_indices],
            logits=self.logits[row_indices],
            model_id=self.model_id,
            stage=self.stage,
            split_name=self.split_name,
        )

    def to_serializable_records(self) -> list[dict]:
        """Return one JSON-compatible record per row, in order."""
        return [
            {
                "example_id": example_id,
                "group_id": group_id,
                "label": int(label),
                "positive_class_probability": float(probability),
                "logit_negative": float(logit_pair[0]),
                "logit_positive": float(logit_pair[1]),
            }
            for example_id, group_id, label, probability, logit_pair in zip(
                self.example_ids,
                self.group_ids,
                self.labels.tolist(),
                self.positive_class_probabilities.tolist(),
                self.logits.tolist(),
                strict=True,
            )
        ]

    def write_json_file(self, predictions_path: Path) -> None:
        """Write the predictions and their provenance to ``predictions_path``."""
        predictions_path.parent.mkdir(parents=True, exist_ok=True)
        predictions_path.write_text(
            json.dumps(
                {
                    "model_id": self.model_id,
                    "stage": self.stage,
                    "split_name": self.split_name,
                    "records": self.to_serializable_records(),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )


def probabilities_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Return the positive-class probability of a two-column logit tensor.

    Args:
        logits: ``(n, 2)`` tensor of raw model outputs.

    Returns:
        ``(n,)`` tensor of ``softmax(logits)[:, 1]``.

    Raises:
        PredictionValidationError: If the tensor is not two-column. The
            protocol ranks by the positive-class probability of two logits, so
            a differently shaped output is a call error, not something to
            reinterpret.
    """
    if logits.ndim != 2 or logits.shape[1] != 2:
        raise PredictionValidationError(
            f"expected a two-column logit tensor, got shape {tuple(logits.shape)}"
        )
    return torch.softmax(logits.to(torch.float64), dim=1)[:, 1]


def build_binary_predictions(
    *,
    example_ids: tuple[str, ...],
    group_ids: tuple[str, ...],
    labels: torch.Tensor,
    logits: torch.Tensor,
    model_id: str,
    stage: str,
    split_name: str,
) -> BinaryPredictions:
    """Assemble a prediction set from raw logits, deriving the ranking score."""
    return BinaryPredictions(
        example_ids=example_ids,
        group_ids=group_ids,
        labels=labels.to(torch.long),
        positive_class_probabilities=probabilities_from_logits(logits),
        logits=logits.to(torch.float64),
        model_id=model_id,
        stage=stage,
        split_name=split_name,
    )
