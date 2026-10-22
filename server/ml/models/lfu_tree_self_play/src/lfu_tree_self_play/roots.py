"""Reproducible random openings and symmetry-aware root identities."""

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from omegaconf import DictConfig

from .config import experiment_id
from .game import GameState, ReversiPyGame
from .rng import derive_seed

RootPurpose = Literal["training", "tuning", "final_evaluation"]
_PURPOSES = RootPurpose.__args__


@dataclass(frozen=True)
class RootPosition:
    """A seed, legal placement history, and its resulting opening."""

    seed: int
    placements: int
    actions: tuple[int, ...]
    state: GameState

    @property
    def key(self) -> str:
        return position_key(self.state)


def position_key(state: GameState) -> str:
    """Identify black/white disks and turn modulo the eight square symmetries."""
    planes = state.observation
    disks = planes[:2]
    if (
        not np.all((disks == 0) | (disks == 1))
        or np.any(disks[0] + disks[1] > 1)
        or state.to_play not in (-1, 1)
        or not np.all(planes[2] == state.to_play)
    ):
        raise ValueError("invalid board or turn for root identity")
    board = (disks[0] + 2 * disks[1]).astype(np.uint8)
    canonical = min(
        np.rot90(reflection, turns).tobytes()
        for reflection in (board, np.fliplr(board))
        for turns in range(4)
    )
    return hashlib.sha256(canonical + bytes([state.to_play == 1])).hexdigest()


def generate_root_position(seed: int, placements: int | None = None) -> RootPosition:
    """Uniformly draw 0–8 placements, then uniformly choose each legal square."""
    rng = random.Random(derive_seed(seed, "root_position"))
    drawn_placements = rng.randrange(9)
    if placements is None:
        placements = drawn_placements
    if type(placements) is not int or not 0 <= placements <= 8:
        raise ValueError("placements must be an integer in 0..8")
    game = ReversiPyGame()
    state = game.initial_state()
    actions = []
    for _ in range(placements):
        action = rng.choice(state.legal_actions)
        actions.append(action)
        state = game.step(state, action)
    return RootPosition(seed, placements, tuple(actions), state)


@dataclass(frozen=True)
class RootEntry:
    """An accepted opening's split and deterministic candidate coordinates."""

    purpose: RootPurpose
    run_seed: int
    index: int
    attempt: int
    root: RootPosition


def _integer(value: int, name: str, minimum: int = 0) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _candidate_seed(run_seed: int, index: int, attempt: int) -> int:
    payload = f"lfu-tree-self-play/roots/v1:{run_seed}:{index}:{attempt}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest(), "big")


def _seed_sets(config: DictConfig) -> dict[RootPurpose, tuple[int, ...]]:
    sets = {}
    seen = set()
    for purpose in _PURPOSES:
        seeds = tuple(config.seeds[purpose])
        if not seeds:
            raise ValueError("root seed sets must be nonempty")
        for seed in seeds:
            _integer(seed, "run_seed")
        if len(set(seeds)) != len(seeds) or seen.intersection(seeds):
            raise ValueError("root seeds must be distinct within and across splits")
        sets[purpose] = tuple(sorted(seeds))
        seen.update(seeds)
    return sets


@dataclass(frozen=True)
class RootRegistry:
    """A complete, immutable collection with no repeated root identities.

    Split membership does not enforce evaluation access restrictions or exclude
    positions encountered later during play.
    """

    experiment_id: str
    roots_per_seed: int
    entries: tuple[RootEntry, ...]

    def __post_init__(self) -> None:
        _integer(self.roots_per_seed, "roots_per_seed", 1)
        if (
            type(self.experiment_id) is not str
            or len(self.experiment_id) != 64
            or any(c not in "0123456789abcdef" for c in self.experiment_id)
        ):
            raise ValueError("invalid experiment ID")
        if type(self.entries) is not tuple or not self.entries:
            raise ValueError("entries must be a nonempty tuple")
        keys = set()
        seed_purposes = {}
        slots = {}
        for entry in self.entries:
            if type(entry) is not RootEntry or entry.purpose not in _PURPOSES:
                raise ValueError("invalid root entry or purpose")
            for name in ("run_seed", "index", "attempt"):
                _integer(getattr(entry, name), name)
            previous = seed_purposes.setdefault(entry.run_seed, entry.purpose)
            if previous != entry.purpose:
                raise ValueError("run seed occurs in multiple splits")
            slots.setdefault((entry.purpose, entry.run_seed), []).append(entry.index)
            if type(entry.root) is not RootPosition:
                raise ValueError("invalid root position")
            key = entry.root.key
            if key in keys:
                raise ValueError("duplicate root within or across splits")
            keys.add(key)
        if set(seed_purposes.values()) != set(_PURPOSES):
            raise ValueError("all three root splits must be nonempty")
        for indices in slots.values():
            if len(indices) != self.roots_per_seed or sorted(indices) != list(
                range(self.roots_per_seed)
            ):
                raise ValueError("incomplete or duplicate root slots")
        for entry in self.entries:
            seed = _candidate_seed(entry.run_seed, entry.index, entry.attempt)
            expected = generate_root_position(seed)
            root = entry.root
            if (
                type(root.seed) is not int
                or type(root.placements) is not int
                or type(root.actions) is not tuple
                or any(type(action) is not int for action in root.actions)
                or type(root.state) is not GameState
                or root != expected
            ):
                raise ValueError("root does not match its candidate seed and replay")
        ordered = tuple(
            sorted(
                self.entries,
                key=lambda e: (_PURPOSES.index(e.purpose), e.run_seed, e.index),
            )
        )
        object.__setattr__(self, "entries", ordered)

    def positions(self, purpose: RootPurpose) -> tuple[RootPosition, ...]:
        """Return the fixed openings belonging to one named split."""
        if purpose not in _PURPOSES:
            raise ValueError("unknown root purpose")
        return tuple(e.root for e in self.entries if e.purpose == purpose)

    def _json(self) -> str:
        document = {
            "version": 1,
            "experiment_id": self.experiment_id,
            "roots_per_seed": self.roots_per_seed,
            "entries": [
                {
                    "purpose": e.purpose,
                    "run_seed": e.run_seed,
                    "index": e.index,
                    "attempt": e.attempt,
                    "placements": e.root.placements,
                    "actions": list(e.root.actions),
                    "key": e.root.key,
                }
                for e in self.entries
            ],
        }
        return json.dumps(document, sort_keys=True, separators=(",", ":"))

    @property
    def registry_id(self) -> str:
        """Hash the versioned settings, membership, and opening provenance."""
        return hashlib.sha256(self._json().encode()).hexdigest()

    def save(self, path: str | Path) -> None:
        """Refuse overwrites; this offline registry write is not atomic."""
        with Path(path).open("x", encoding="utf-8") as file:
            file.write(self._json())

    @classmethod
    def load(cls, path: str | Path, config: DictConfig) -> "RootRegistry":
        """Replay saved candidates and verify the expected experiment and seeds."""
        document = json.loads(
            Path(path).read_text(encoding="utf-8"), object_pairs_hook=_json_object
        )
        _fields(document, {"version", "experiment_id", "roots_per_seed", "entries"})
        if type(document["version"]) is not int or document["version"] != 1:
            raise ValueError("unsupported root registry version")
        if document["experiment_id"] != experiment_id(config):
            raise ValueError("root registry experiment mismatch")
        _integer(document["roots_per_seed"], "roots_per_seed", 1)
        if type(document["entries"]) is not list:
            raise ValueError("registry entries must be a list")
        seed_sets = _seed_sets(config)
        entries = []
        for item in document["entries"]:
            _fields(
                item,
                {
                    "purpose",
                    "run_seed",
                    "index",
                    "attempt",
                    "placements",
                    "actions",
                    "key",
                },
            )
            purpose = item["purpose"]
            if type(purpose) is not str or purpose not in _PURPOSES:
                raise ValueError("unknown root purpose")
            for name in ("run_seed", "index", "attempt", "placements"):
                _integer(item[name], name)
            if item["run_seed"] not in seed_sets[purpose]:
                raise ValueError("run seed is not registered for this split")
            actions = item["actions"]
            if type(actions) is not list or any(type(a) is not int for a in actions):
                raise ValueError("actions must be an integer list")
            root = generate_root_position(
                _candidate_seed(item["run_seed"], item["index"], item["attempt"])
            )
            if (
                item["placements"] != root.placements
                or tuple(actions) != root.actions
                or item["key"] != root.key
            ):
                raise ValueError("saved root does not match candidate replay")
            entries.append(
                RootEntry(
                    purpose, item["run_seed"], item["index"], item["attempt"], root
                )
            )
        expected_seeds = {(p, seed) for p, seeds in seed_sets.items() for seed in seeds}
        if {(e.purpose, e.run_seed) for e in entries} != expected_seeds:
            raise ValueError("root registry is missing configured seeds")
        return cls(
            document["experiment_id"], document["roots_per_seed"], tuple(entries)
        )


def _json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON field")
        result[key] = value
    return result


def _fields(value, fields):
    if type(value) is not dict or set(value) != fields:
        raise ValueError("root registry fields mismatch")


def generate_root_registry(
    config: DictConfig, *, roots_per_seed: int = 1, max_attempts: int = 1000
) -> RootRegistry:
    """Fill each seed's slots with unique candidates, or fail without a partial set."""
    _integer(roots_per_seed, "roots_per_seed", 1)
    _integer(max_attempts, "max_attempts", 1)
    seed_sets = _seed_sets(config)
    entries = []
    seen = set()
    for purpose, seeds in seed_sets.items():
        for run_seed in seeds:
            for index in range(roots_per_seed):
                for attempt in range(max_attempts):
                    root = generate_root_position(
                        _candidate_seed(run_seed, index, attempt)
                    )
                    if root.key not in seen:
                        seen.add(root.key)
                        entries.append(
                            RootEntry(purpose, run_seed, index, attempt, root)
                        )
                        break
                else:
                    raise ValueError(
                        f"root attempt budget exhausted: {purpose}/{run_seed}/{index}"
                    )
    return RootRegistry(experiment_id(config), roots_per_seed, tuple(entries))
