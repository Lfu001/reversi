"""Atomic append-only units on a cooperating POSIX local filesystem."""

import fcntl
import os
import re
import uuid
from contextlib import contextmanager
from pathlib import Path

from .dataset_codec import DatasetUnit, UnitIdentity, decode_unit, encode_unit
from .game import Game
from .records import CollectionRecord
from .returns import ReturnTargets


def _sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class DatasetStore:
    def __init__(self, root: Path, *, experiment_id: str, game: Game):
        self.root = Path(root)
        self.experiment_id = experiment_id
        self.game = game
        self.root.mkdir(parents=True, exist_ok=True)
        for name in ("published", "staging", "quarantine"):
            (self.root / name).mkdir(exist_ok=True)
        _sync_directory(self.root)

    @contextmanager
    def _writer(self):
        with (self.root / "writer.lock").open("a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            yield

    def _path(self, unit_id: str) -> Path:
        if not isinstance(unit_id, str) or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", unit_id
        ):
            raise ValueError("unsafe unit ID")
        return self.root / "published" / f"{unit_id}.json"

    def _check(
        self, unit: DatasetUnit, model_generation: int | None = None
    ) -> DatasetUnit:
        if unit.identity.experiment_id != self.experiment_id:
            raise ValueError("experiment/configuration mismatch")
        if (
            model_generation is not None
            and unit.identity.model_generation != model_generation
        ):
            raise ValueError("model generation mismatch")
        return unit

    def publish(self, record: CollectionRecord, targets: ReturnTargets) -> UnitIdentity:
        destination = self._path(record.unit_id)
        data = encode_unit(record, targets, self.game)
        unit = self._check(decode_unit(data, self.game))
        with self._writer():
            if destination.exists():
                existing = self.load(record.unit_id)
                if existing.identity != unit.identity:
                    raise ValueError("unit ID replay conflict")
                return existing.identity
            staging = self.root / "staging" / uuid.uuid4().hex
            with staging.open("xb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            _sync_directory(staging.parent)
            os.rename(staging, destination)
            _sync_directory(destination.parent)
            _sync_directory(staging.parent)
        return unit.identity

    def load(self, unit_id: str, *, model_generation: int | None = None) -> DatasetUnit:
        unit = self._check(
            decode_unit(self._path(unit_id).read_bytes(), self.game), model_generation
        )
        if unit.identity.unit_id != unit_id:
            raise ValueError("filename/unit identity mismatch")
        return unit

    def enumerate_units(
        self, *, model_generation: int | None = None
    ) -> tuple[UnitIdentity, ...]:
        identities = []
        for path in sorted((self.root / "published").iterdir()):
            if not path.is_file() or path.suffix != ".json":
                raise ValueError("invalid published entry")
            identity = self.load(path.stem).identity
            if (
                model_generation is None
                or identity.model_generation == model_generation
            ):
                identities.append(identity)
        return tuple(identities)

    def quarantine_incomplete(self) -> tuple[Path, ...]:
        moved = []
        with self._writer():
            for path in sorted((self.root / "staging").iterdir()):
                destination = self.root / "quarantine" / uuid.uuid4().hex
                os.rename(path, destination)
                moved.append(destination)
            _sync_directory(self.root / "staging")
            _sync_directory(self.root / "quarantine")
        return tuple(moved)
