"""Checkpoints that can actually resume a search, and refuse when they cannot.

Saving the best genome is enough to report a result and nowhere near enough to
continue a run. Resuming needs the whole population, the species assignment and
the historical gene markings that numbered its genes, every random stream position,
the evaluations already completed and the budget already spent. A checkpoint
missing any of those would resume into a *different* search that happens to
start from the same genomes.

So this module records what it can reach generically and is explicit about the
rest: an algorithm exports its own evolution state through
:class:`SupportsEvolutionStateExport`, and a checkpoint without that state
reports itself unusable for resume rather than pretending. Reporting-only
checkpoints are still useful, and still written; they simply cannot be resumed
from.

A resume also refuses to start when the manifest, the protocol lock or the
effective configuration have changed, because continuing a run under different
settings produces a series that no artifact describes.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from polyneat.logging_utils.custom_logger import get_logger

logger = get_logger(__name__)

RUN_CHECKPOINT_SCHEMA_VERSION = "1.0"


class CheckpointResumeError(RuntimeError):
    """Raised when a checkpoint cannot be resumed from as-is."""


@runtime_checkable
class SupportsEvolutionStateExport(Protocol):
    """An algorithm that can hand over and take back its own search state.

    Species membership and the historical gene markings are algorithm-specific and
    not reachable from the generic loop, so an algorithm that wants to be
    resumable exports them itself. One that does not implement this can still
    be checkpointed for reporting; it just cannot be continued.
    """

    def export_evolution_state(self) -> dict:
        """Return species, gene-marking state and any other search state."""
        ...

    def import_evolution_state(self, state: dict) -> None:
        """Restore state produced by :meth:`export_evolution_state`."""
        ...


@dataclass(frozen=True)
class RunCheckpoint:
    """One point a run can be described from, and sometimes continued from.

    Attributes:
        schema_version: Layout version of this record.
        run_id: Identifier of the run.
        stage: Which stage produced it, e.g. ``search`` or ``track_b``.
        generation_number: Generation the checkpoint was taken at.
        genome_kind: Class name of the genomes in the population.
        population_payload: Every genome of the current population, serialized.
        algorithm_state: Species and gene-marking state, when the algorithm can
            export it. ``None`` marks the checkpoint report-only.
        random_generator_state: Position of the run's numpy generator.
        torch_stream_states: Positions of the named torch streams, if any.
        best_genome_payload: Best genome so far, serialized.
        best_fitness: Its fitness.
        best_model_reference: Checkpoint id of the weights that earned it, so a
            resumed run does not have to retrain the incumbent.
        evaluation_records: Every completed evaluation, serialized.
        budget_state: Consumed and total budget seconds.
        binding: Digests of the manifest, protocol lock and effective
            configuration this run belongs to.
        metadata: Free-form extras.
    """

    schema_version: str
    run_id: str
    stage: str
    generation_number: int
    genome_kind: str
    population_payload: list[dict]
    algorithm_state: dict | None
    random_generator_state: dict
    torch_stream_states: dict
    best_genome_payload: dict | None
    best_fitness: float | None
    best_model_reference: str | None
    evaluation_records: list[dict]
    budget_state: dict
    binding: dict
    metadata: dict = field(default_factory=dict)

    @property
    def is_resumable(self) -> bool:
        """Whether this checkpoint carries everything a resume needs."""
        return self.algorithm_state is not None and bool(self.population_payload)

    def to_serializable_dict(self) -> dict:
        """Return the checkpoint as JSON-compatible data, digest excluded."""
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "stage": self.stage,
            "generation_number": self.generation_number,
            "genome_kind": self.genome_kind,
            "population_payload": self.population_payload,
            "algorithm_state": self.algorithm_state,
            "random_generator_state": self.random_generator_state,
            "torch_stream_states": self.torch_stream_states,
            "best_genome_payload": self.best_genome_payload,
            "best_fitness": self.best_fitness,
            "best_model_reference": self.best_model_reference,
            "evaluation_records": self.evaluation_records,
            "budget_state": self.budget_state,
            "binding": self.binding,
            "metadata": self.metadata,
        }

    def compute_sha256(self) -> str:
        """Digest of the canonical encoding of this checkpoint."""
        canonical_json = json.dumps(
            self.to_serializable_dict(), sort_keys=True, separators=(",", ":"), default=str
        )
        return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()

    @classmethod
    def from_serializable_dict(cls, payload: dict) -> RunCheckpoint:
        """Rebuild a checkpoint from :meth:`to_serializable_dict` output."""
        return cls(
            schema_version=payload["schema_version"],
            run_id=payload["run_id"],
            stage=payload["stage"],
            generation_number=int(payload["generation_number"]),
            genome_kind=payload["genome_kind"],
            population_payload=list(payload["population_payload"]),
            algorithm_state=payload["algorithm_state"],
            random_generator_state=payload["random_generator_state"],
            torch_stream_states=payload.get("torch_stream_states", {}),
            best_genome_payload=payload["best_genome_payload"],
            best_fitness=payload["best_fitness"],
            best_model_reference=payload["best_model_reference"],
            evaluation_records=list(payload["evaluation_records"]),
            budget_state=payload["budget_state"],
            binding=payload["binding"],
            metadata=payload.get("metadata", {}),
        )


class RunCheckpointStore:
    """Writes and reads run checkpoints on the filesystem, atomically.

    Every write goes to a temporary file that is then moved into place, so a
    crash mid-write cannot leave a truncated checkpoint that loads but resumes
    into nonsense. Each checkpoint is kept under its generation number, and the
    latest one is found by that number rather than by file timestamp.
    """

    def __init__(self, directory: Path) -> None:
        """Store checkpoints under ``directory``, creating it if needed."""
        self._directory = directory
        self._directory.mkdir(parents=True, exist_ok=True)

    @property
    def directory(self) -> Path:
        """Where checkpoints are written."""
        return self._directory

    def write(self, checkpoint: RunCheckpoint) -> Path:
        """Write one checkpoint and return the path it landed at."""
        payload = checkpoint.to_serializable_dict()
        payload["checkpoint_sha256"] = checkpoint.compute_sha256()
        destination = self._directory / f"generation_{checkpoint.generation_number:06d}.json"
        temporary_path = destination.with_suffix(".json.partial")
        temporary_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
        )
        temporary_path.replace(destination)
        logger.info(
            "Run checkpoint for generation %d written to %s (resumable: %s)",
            checkpoint.generation_number,
            destination.name,
            checkpoint.is_resumable,
        )
        return destination

    def read_latest(self) -> RunCheckpoint | None:
        """Return the highest-numbered checkpoint, or ``None`` when there is none.

        Raises:
            CheckpointResumeError: If the newest file was modified after it was
                written, or was produced by another schema version.
        """
        checkpoint_paths = sorted(self._directory.glob("generation_*.json"))
        if not checkpoint_paths:
            return None
        return self.read(checkpoint_paths[-1])

    def read(self, checkpoint_path: Path) -> RunCheckpoint:
        """Read and verify one checkpoint file.

        Raises:
            CheckpointResumeError: On a missing file, an unknown schema version
                or a digest that no longer matches the content.
        """
        if not checkpoint_path.is_file():
            raise CheckpointResumeError(f"checkpoint {checkpoint_path} does not exist")
        payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        stored_digest = payload.pop("checkpoint_sha256", None)
        if payload.get("schema_version") != RUN_CHECKPOINT_SCHEMA_VERSION:
            raise CheckpointResumeError(
                f"checkpoint {checkpoint_path} has schema version "
                f"{payload.get('schema_version')!r}, this build reads "
                f"{RUN_CHECKPOINT_SCHEMA_VERSION!r}"
            )
        checkpoint = RunCheckpoint.from_serializable_dict(payload)
        if stored_digest != checkpoint.compute_sha256():
            raise CheckpointResumeError(
                f"checkpoint {checkpoint_path} was modified after it was written"
            )
        return checkpoint


def build_run_binding(
    *,
    manifest_sha256: str,
    protocol_lock_sha256: str,
    effective_configuration: dict[str, Any],
) -> dict:
    """Digest what a run must not change between segments.

    Args:
        manifest_sha256: Digest of the frozen split.
        protocol_lock_sha256: Digest of the protocol lock file.
        effective_configuration: The configuration after overrides were
            applied, not the original YAML - two runs differing only in an
            override are two different runs.

    Returns:
        The binding recorded in every checkpoint of the run.
    """
    configuration_json = json.dumps(
        effective_configuration, sort_keys=True, separators=(",", ":"), default=str
    )
    return {
        "manifest_sha256": manifest_sha256,
        "protocol_lock_sha256": protocol_lock_sha256,
        "effective_configuration_sha256": hashlib.sha256(
            configuration_json.encode("utf-8")
        ).hexdigest(),
    }


def verify_checkpoint_is_resumable(checkpoint: RunCheckpoint, *, expected_binding: dict) -> None:
    """Refuse a resume that would silently continue a different experiment.

    Args:
        checkpoint: Checkpoint a run is about to continue from.
        expected_binding: Binding of the run that wants to continue it, from
            :func:`build_run_binding`.

    Raises:
        CheckpointResumeError: If the checkpoint carries no algorithm state, or
            if any bound digest disagrees. Every disagreement is listed at
            once.
    """
    problems: list[str] = []
    if not checkpoint.is_resumable:
        problems.append(
            "the checkpoint carries no exported algorithm state, so species membership and "
            "gene numbering cannot be restored; saving only the best genome is not enough to "
            "resume"
        )
    for binding_key, expected_value in sorted(expected_binding.items()):
        recorded_value = checkpoint.binding.get(binding_key)
        if recorded_value != expected_value:
            problems.append(
                f"{binding_key} changed: checkpoint has {str(recorded_value)[:12]}, this run has "
                f"{str(expected_value)[:12]}"
            )
    if problems:
        raise CheckpointResumeError(
            f"cannot resume run {checkpoint.run_id} from generation "
            f"{checkpoint.generation_number}:\n  - " + "\n  - ".join(problems)
        )
    logger.info(
        "Resuming run %s from generation %d with %d completed evaluations",
        checkpoint.run_id,
        checkpoint.generation_number,
        len(checkpoint.evaluation_records),
    )
