"""Independent, repeatable random streams for experiment stages."""

import hashlib
import random
from dataclasses import dataclass
from typing import Literal

Purpose = Literal["branch_position", "action_sampling", "sibling_rollout", "evaluation"]
_PURPOSES = frozenset(Purpose.__args__)


@dataclass(frozen=True)
class RNGStreams:
    branch_position: random.Random
    action_sampling: random.Random
    sibling_rollout: random.Random
    evaluation: random.Random


def derive_seed(run_seed: int, purpose: Purpose) -> int:
    """Derive a purpose-specific integer seed independent of call order."""
    if type(run_seed) is not int or run_seed < 0:
        raise ValueError("run_seed must be a nonnegative integer")
    if purpose not in _PURPOSES:
        raise ValueError(f"unknown RNG purpose: {purpose}")
    payload = f"lfu-tree-self-play/rng/v1:{run_seed}:{purpose}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest(), "big")


def create_rng_streams(run_seed: int) -> RNGStreams:
    """Create the four independently seeded standard-library RNGs."""
    return RNGStreams(
        branch_position=random.Random(derive_seed(run_seed, "branch_position")),
        action_sampling=random.Random(derive_seed(run_seed, "action_sampling")),
        sibling_rollout=random.Random(derive_seed(run_seed, "sibling_rollout")),
        evaluation=random.Random(derive_seed(run_seed, "evaluation")),
    )
