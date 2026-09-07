"""A model snapshot that stops being affected by what happens next.

The protocol needs the *exact* model that achieved a selected fitness, not a
model that can be rebuilt from its genome. For DeepNEAT the difference is
total: its genome carries no weights at all, so decoding it again produces a
different network that never earned that score. For EXACT the genome does carry
kernels, but a later generation overwrites them.

So a checkpoint here is a deep copy taken at capture time, detached and moved to
the CPU, plus the preprocessing state the model was fitted with and enough
provenance to tell two of them apart. Capturing and restoring never run
training, and the digest covers the tensors themselves, which is what a
threshold is later bound to.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import torch

from polyneat.logging_utils.custom_logger import get_logger
from polyneat.training.trainable_model import TrainableModel

logger = get_logger(__name__)

CHECKPOINT_SCHEMA_VERSION = "1.0"


class CheckpointError(RuntimeError):
    """Raised when a checkpoint is unreadable or does not match its target."""


@dataclass(frozen=True)
class ModelCheckpoint:
    """One frozen model, its preprocessing and its provenance.

    Attributes:
        schema_version: Layout version of this record.
        model_id: Stable identifier, used in prediction files and thresholds.
        stage: Which stage produced it, e.g. ``track_a`` or ``track_b_seed2``.
        genome_kind: Class name of the genome it was decoded from.
        genome_payload: The genome as serializable data, so the topology can be
            inspected without the library that produced it.
        model_state: Deep-copied parameters and buffers, on the CPU.
        preprocessing_state: State of the preprocessing fitted for this model.
        recipe: The learning recipe used, when one applies.
        metadata: Free-form provenance, e.g. the fitness that selected it.
    """

    schema_version: str
    model_id: str
    stage: str
    genome_kind: str
    genome_payload: dict
    model_state: dict
    preprocessing_state: dict
    recipe: dict | None = None
    metadata: dict = field(default_factory=dict)

    def compute_sha256(self) -> str:
        """Digest covering the tensors, the preprocessing and the provenance.

        Tensor bytes are hashed together with their dtype and shape, in sorted
        key order, so the digest is stable across processes and machines and
        changes whenever a single weight does.
        """
        digest = hashlib.sha256()
        digest.update(
            json.dumps(
                {
                    "schema_version": self.schema_version,
                    "model_id": self.model_id,
                    "stage": self.stage,
                    "genome_kind": self.genome_kind,
                    "genome_payload": self.genome_payload,
                    "preprocessing_state": self.preprocessing_state,
                    "recipe": self.recipe,
                },
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8")
        )
        for parameter_name in sorted(self.model_state):
            value = self.model_state[parameter_name]
            digest.update(parameter_name.encode("utf-8"))
            if isinstance(value, torch.Tensor):
                digest.update(str(value.dtype).encode("utf-8"))
                digest.update(str(tuple(value.shape)).encode("utf-8"))
                digest.update(value.detach().cpu().contiguous().numpy().tobytes())
            else:
                digest.update(repr(value).encode("utf-8"))
        return digest.hexdigest()

    def write_file(self, checkpoint_path: Path) -> str:
        """Write the checkpoint atomically and return its digest.

        The file is written next to its destination and then moved into place,
        so an interrupted write cannot leave a half-written checkpoint that
        looks loadable.
        """
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_sha256 = self.compute_sha256()
        payload = {
            "schema_version": self.schema_version,
            "model_id": self.model_id,
            "stage": self.stage,
            "genome_kind": self.genome_kind,
            "genome_payload": self.genome_payload,
            "model_state": self.model_state,
            "preprocessing_state": self.preprocessing_state,
            "recipe": self.recipe,
            "metadata": self.metadata,
            "checkpoint_sha256": checkpoint_sha256,
        }
        temporary_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".partial")
        torch.save(payload, temporary_path)
        temporary_path.replace(checkpoint_path)
        logger.info(
            "Checkpoint %s written to %s (sha256 %s)",
            self.model_id,
            checkpoint_path,
            checkpoint_sha256[:12],
        )
        return checkpoint_sha256

    @classmethod
    def read_file(cls, checkpoint_path: Path) -> tuple[ModelCheckpoint, str]:
        """Read a checkpoint and verify the digest stored with it.

        Args:
            checkpoint_path: File written by :meth:`write_file`.

        Returns:
            ``(checkpoint, checkpoint_sha256)``.

        Raises:
            CheckpointError: If the file is missing, was written by another
                schema version, or its content no longer hashes to the stored
                digest.
        """
        if not checkpoint_path.is_file():
            raise CheckpointError(f"checkpoint {checkpoint_path} does not exist")
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        stored_version = payload.get("schema_version")
        if stored_version != CHECKPOINT_SCHEMA_VERSION:
            raise CheckpointError(
                f"checkpoint {checkpoint_path} has schema version {stored_version!r}, this build "
                f"reads {CHECKPOINT_SCHEMA_VERSION!r}"
            )
        stored_digest = payload.get("checkpoint_sha256")
        checkpoint = cls(
            schema_version=stored_version,
            model_id=payload["model_id"],
            stage=payload["stage"],
            genome_kind=payload["genome_kind"],
            genome_payload=payload["genome_payload"],
            model_state=payload["model_state"],
            preprocessing_state=payload["preprocessing_state"],
            recipe=payload.get("recipe"),
            metadata=payload.get("metadata", {}),
        )
        recomputed_digest = checkpoint.compute_sha256()
        if stored_digest != recomputed_digest:
            raise CheckpointError(
                f"checkpoint {checkpoint_path} was modified after it was written: stored digest "
                f"{str(stored_digest)[:12]} but its content hashes to {recomputed_digest[:12]}"
            )
        return checkpoint, recomputed_digest

    def restore_into(self, model: TrainableModel) -> None:
        """Load the snapshot into ``model``, leaving it in evaluation mode.

        Restoring never trains and never fits preprocessing. The preprocessing
        statistics are restored separately by their owner, from
        :attr:`preprocessing_state`.
        """
        model.load_state_dict(copy.deepcopy(self.model_state))
        model.eval()


def capture_model_checkpoint(
    model: TrainableModel,
    *,
    model_id: str,
    stage: str,
    genome_kind: str,
    genome_payload: dict,
    preprocessing_state: dict,
    recipe: dict | None = None,
    metadata: dict | None = None,
) -> ModelCheckpoint:
    """Take an independent snapshot of ``model`` as it is right now.

    Every tensor is detached, moved to the CPU and cloned, so training the
    model afterwards cannot change what the checkpoint holds. That independence
    is the whole point: track A has to keep the network that earned the fitness
    the search selected on, not a later version of it.

    Args:
        model: Model to snapshot. Left untouched.
        model_id: Stable identifier for this checkpoint.
        stage: Stage that produced it.
        genome_kind: Class name of the genome behind the model.
        genome_payload: The genome as serializable data.
        preprocessing_state: State of that model's fitted preprocessing.
        recipe: Learning recipe used, when one applies.
        metadata: Free-form provenance, e.g. the selection fitness.

    Returns:
        The independent :class:`ModelCheckpoint`.
    """
    snapshot_state = {
        parameter_name: (
            value.detach().cpu().clone()
            if isinstance(value, torch.Tensor)
            else copy.deepcopy(value)
        )
        for parameter_name, value in model.state_dict().items()
    }
    return ModelCheckpoint(
        schema_version=CHECKPOINT_SCHEMA_VERSION,
        model_id=model_id,
        stage=stage,
        genome_kind=genome_kind,
        genome_payload=copy.deepcopy(genome_payload),
        model_state=snapshot_state,
        preprocessing_state=copy.deepcopy(preprocessing_state),
        recipe=None if recipe is None else copy.deepcopy(recipe),
        metadata=dict(metadata or {}),
    )
