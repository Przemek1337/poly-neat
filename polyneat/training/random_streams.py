"""Independent, role-named random streams derived from one recorded root seed.

Every source of randomness in the benchmark gets its own stream: the split, the
smoke subsample, evolution, parameter initialization, minibatch order,
augmentation, the final retraining and the bootstrap. They are derived by
hashing the role together with the root seed and a stable evaluation id, so:

* the data streams do not move when the model changes. Drawing a different
  number of values while initializing a bigger network cannot shift the
  minibatch order or the augmentation, which one shared global seed would;
* the assignment of seeds to evaluations is deterministic and reproducible
  from the recorded root seed alone;
* re-running one evaluation reproduces its stream without replaying every
  evaluation before it.

Seeding the global ``torch`` state is deliberately not enough on its own and is
not what this module does. Callers pass the returned generators explicitly into
the operations that consume them.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import torch

# Seeds are drawn from a 63-bit space: wide enough that role collisions are not
# a practical concern, and still a positive value every consumer accepts.
_SEED_MODULUS = 2**63


class RandomStreamRole(StrEnum):
    """The named sources of randomness the protocol keeps apart.

    Attributes:
        SPLIT: The grouped development split. Drawn once per manifest.
        SMOKE_SAMPLING: Subsampling used by smoke profiles only.
        EVOLUTION: The search itself: mutation, crossover, parent selection.
        PARAMETER_INITIALIZATION: Fresh weights for one trained candidate.
        BATCH_ORDER: Minibatch order within one training session.
        AUGMENTATION: The affine and contrast draws of one training session.
        FINAL_TRAINING: Track B retraining of a selected topology.
        BOOTSTRAP: Test-set resampling for confidence intervals.
    """

    SPLIT = "split"
    SMOKE_SAMPLING = "smoke_sampling"
    EVOLUTION = "evolution"
    PARAMETER_INITIALIZATION = "parameter_initialization"
    BATCH_ORDER = "batch_order"
    AUGMENTATION = "augmentation"
    FINAL_TRAINING = "final_training"
    BOOTSTRAP = "bootstrap"


def derive_stream_seed(
    *,
    role: RandomStreamRole | str,
    root_seed: int,
    evaluation_id: str = "",
    track: str = "",
) -> int:
    """Derive one stream seed from the root seed and what it is for.

    The derivation is a hash rather than an offset, so two roles cannot land on
    the same stream because their offsets happened to differ by the number of
    draws taken elsewhere.

    Args:
        role: What the stream is for.
        root_seed: The run's recorded root seed.
        evaluation_id: Stable identifier of the candidate or model, so
            re-running one evaluation reproduces its stream on its own.
        track: ``"A"`` or ``"B"``, so the two tracks of one selected topology
            never share an initialization or a batch order.

    Returns:
        A non-negative 63-bit seed.
    """
    role_name = role.value if isinstance(role, RandomStreamRole) else str(role)
    canonical_key = f"polyneat|{role_name}|{root_seed}|{track}|{evaluation_id}"
    digest = hashlib.sha256(canonical_key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % _SEED_MODULUS


def create_torch_generator(
    *,
    role: RandomStreamRole | str,
    root_seed: int,
    evaluation_id: str = "",
    track: str = "",
) -> torch.Generator:
    """Return a CPU ``torch.Generator`` seeded for one role.

    The generator lives on the CPU even when training runs on a GPU: draws are
    taken here and moved to the device, so the stream does not depend on which
    device the run happened to get.
    """
    generator = torch.Generator()
    generator.manual_seed(
        derive_stream_seed(
            role=role, root_seed=root_seed, evaluation_id=evaluation_id, track=track
        )
    )
    return generator


def create_numpy_generator(
    *,
    role: RandomStreamRole | str,
    root_seed: int,
    evaluation_id: str = "",
    track: str = "",
) -> np.random.Generator:
    """Return a numpy ``Generator`` seeded for one role."""
    return np.random.default_rng(
        derive_stream_seed(
            role=role, root_seed=root_seed, evaluation_id=evaluation_id, track=track
        )
    )


@dataclass(frozen=True)
class TrainingRandomStreams:
    """The three streams one training session consumes, kept separate.

    Attributes:
        parameter_initialization: Fresh weights for this session.
        batch_order: Minibatch permutation per epoch.
        augmentation: Affine and contrast draws.
    """

    parameter_initialization: torch.Generator
    batch_order: torch.Generator
    augmentation: torch.Generator

    @classmethod
    def derive(
        cls, *, root_seed: int, evaluation_id: str = "", track: str = ""
    ) -> TrainingRandomStreams:
        """Build all three streams for one evaluation of one track."""
        return cls(
            parameter_initialization=create_torch_generator(
                role=RandomStreamRole.PARAMETER_INITIALIZATION,
                root_seed=root_seed,
                evaluation_id=evaluation_id,
                track=track,
            ),
            batch_order=create_torch_generator(
                role=RandomStreamRole.BATCH_ORDER,
                root_seed=root_seed,
                evaluation_id=evaluation_id,
                track=track,
            ),
            augmentation=create_torch_generator(
                role=RandomStreamRole.AUGMENTATION,
                root_seed=root_seed,
                evaluation_id=evaluation_id,
                track=track,
            ),
        )

    def state_dict(self) -> dict:
        """Capture every stream position, for a resumable checkpoint."""
        return {
            "parameter_initialization": self.parameter_initialization.get_state(),
            "batch_order": self.batch_order.get_state(),
            "augmentation": self.augmentation.get_state(),
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore stream positions captured by :meth:`state_dict`."""
        self.parameter_initialization.set_state(state["parameter_initialization"])
        self.batch_order.set_state(state["batch_order"])
        self.augmentation.set_state(state["augmentation"])
