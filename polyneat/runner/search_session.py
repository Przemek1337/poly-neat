"""Atomic, bound snapshots for sequential searches, independent of any algorithm.

Snapshots contain only exported state, never callables, datasets or live models.
Unfinished work rolls back to the last committed boundary, but its cost remains
charged. An unclean process death requires explicit accounting of lost time.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import uuid
from pathlib import Path

import torch

from polyneat.runner.run_checkpoint import CheckpointResumeError
from polyneat.runner.wall_clock_budget import WallClockBudget


class SearchSession:
    """Own persistence and budget accounting; the caller owns search semantics."""

    def __init__(
        self,
        directory: Path | None,
        *,
        binding: dict,
        budget: WallClockBudget | None,
        resume: bool = False,
        lost_work_seconds: float | None = None,
    ) -> None:
        self.directory = directory
        self.binding = binding
        self.budget = budget
        self.resume = resume
        self.lost_work_seconds = lost_work_seconds
        self.state: dict | None = None
        self._pointer: dict | None = None

    def __enter__(self) -> SearchSession:
        if self.resume and self.directory is None:
            raise CheckpointResumeError("resume requires an artifacts directory")
        if self.directory is not None:
            self.directory.mkdir(parents=True, exist_ok=True)
            pointer_path = self.directory / "latest.json"
            if pointer_path.exists():
                if not self.resume:
                    raise CheckpointResumeError(
                        "search already exists; use --resume or a new directory"
                    )
                self._pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
                if self._pointer["binding"] != self.binding:
                    raise CheckpointResumeError(
                        "manifest, protocol, environment or configuration changed"
                    )
                name = self._pointer["snapshot"]
                if Path(name).name != name:
                    raise CheckpointResumeError("invalid snapshot filename")
                snapshot_path = self.directory / name
                digest = hashlib.sha256(snapshot_path.read_bytes()).hexdigest()
                if digest != self._pointer["sha256"]:
                    raise CheckpointResumeError("search snapshot checksum mismatch")
                payload = torch.load(snapshot_path, map_location="cpu", weights_only=True)
                if payload["schema_version"] != 1 or payload["binding"] != self.binding:
                    raise CheckpointResumeError("unsupported or mismatched search snapshot")
                self.state = payload["state"]
                journal = json.loads((self.directory / "budget.json").read_text(encoding="utf-8"))
                if journal["binding"] != self.binding:
                    raise CheckpointResumeError("budget journal binding mismatch")
                lost = self.lost_work_seconds
                if journal["running"] and lost is None:
                    raise CheckpointResumeError(
                        "unclean shutdown: supply --lost-work-seconds from scheduler/logs; "
                        "time since the last accounting record must not be forgiven"
                    )
                if lost is not None and (not math.isfinite(lost) or lost < 0):
                    raise CheckpointResumeError("lost-work-seconds must be finite and nonnegative")
                if self.budget is not None:
                    budget_state = journal["budget"]
                    budget_state["consumed_seconds"] += lost or 0.0
                    self.budget.load_state_dict(budget_state)
                torch.random.set_rng_state(payload["torch_cpu"])
                cuda_states = payload["torch_cuda"]
                if cuda_states:
                    if (
                        not torch.cuda.is_available()
                        or len(cuda_states) != torch.cuda.device_count()
                    ):
                        raise CheckpointResumeError("CUDA RNG device count changed")
                    torch.cuda.set_rng_state_all(cuda_states)
                random.setstate(payload["python_random"])
            elif self.resume:
                raise CheckpointResumeError("no search snapshot to resume")
        if self.budget is not None:
            self.budget.start()
        self._write_journal(running=True)
        return self

    def _write_journal(self, *, running: bool) -> None:
        if self.directory is None:
            return
        payload = {
            "binding": self.binding,
            "running": running,
            "budget": None if self.budget is None else self.budget.state_dict(),
        }
        self._write_json(self.directory / "budget.json", payload)

    @staticmethod
    def _write_json(path: Path, payload: dict) -> None:
        temporary = path.with_suffix(".partial")
        temporary.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        temporary.replace(path)

    def commit(self, state: dict) -> None:
        """Commit an independent boundary, then discard the previous snapshot."""
        if self.directory is None:
            return
        name = f"state_{uuid.uuid4().hex}.pt"
        destination = self.directory / name
        temporary = destination.with_suffix(".partial")
        torch.save(
            {
                "schema_version": 1,
                "binding": self.binding,
                "state": state,
                "torch_cpu": torch.random.get_rng_state(),
                "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                "python_random": random.getstate(),
            },
            temporary,
        )
        temporary.replace(destination)
        pointer = {
            "binding": self.binding,
            "snapshot": name,
            "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        }
        self._write_json(self.directory / "latest.json", pointer)
        previous = self._pointer
        self._pointer = pointer
        self._write_journal(running=True)
        if previous is not None:
            (self.directory / previous["snapshot"]).unlink(missing_ok=True)

    def __exit__(self, exception_type, exception, traceback) -> None:
        # Include interrupted/repeated work, not just time of the last snapshot.
        self._write_journal(running=False)
        if self.budget is not None:
            self.budget.pause()
