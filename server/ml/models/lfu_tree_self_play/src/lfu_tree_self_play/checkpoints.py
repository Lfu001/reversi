"""Immutable local checkpoint transactions; no production training integration."""

import fcntl
import hashlib
import json
import math
import os
import pickle
import random
import re
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from .config import experiment_id
from .dataset_codec import UnitIdentity, canonical_json
from .dataset_store import _sync_directory
from .model import PolicyValueModel
from .rng import RNGStreams, create_rng_streams


class ConfigurationMismatch(ValueError):
    """A committed root or snapshot belongs to a different experiment."""


_NAMES = ("branch_position", "action_sampling", "sibling_rollout", "evaluation")


def _identifier(value):
    if type(value) is not str or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", value
    ):
        raise ValueError("unsafe identifier")


def _integer(value):
    if type(value) is not int or value < 0:
        raise ValueError("expected nonnegative integer")


@dataclass(frozen=True)
class InputManifest:
    model_generation: int
    units: tuple[UnitIdentity, ...]
    start: int
    stop: int

    @property
    def selected(self):
        return self.units[self.start : self.stop]


@dataclass(frozen=True)
class EvaluationResult:
    checkpoint_id: str
    opponent: str
    start_pair: str
    seed: int
    result_json: str

    @property
    def identity(self):
        return (self.checkpoint_id, self.opponent, self.start_pair, self.seed)


@dataclass(frozen=True)
class Progress:
    model_generation: int
    stage: str
    cumulative_cost: float
    consumed_unit_ids: tuple[str, ...] = ()
    evaluations: tuple[EvaluationResult, ...] = ()


@dataclass(frozen=True)
class CheckpointManifest:
    commit_id: str
    parent_id: str | None
    sequence: int
    config_hash: str
    configuration: str
    inputs: InputManifest
    progress: Progress
    artifacts: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class RecoveredCheckpoint:
    manifest: CheckpointManifest
    path: Path


@dataclass(frozen=True)
class RestoredRuntime:
    manifest: CheckpointManifest
    model: PolicyValueModel
    optimizer: object
    scheduler: object
    rng: RNGStreams


def _rng_state(rng):
    return {
        "streams": {name: getattr(rng, name).getstate() for name in _NAMES},
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "mps": torch.mps.get_rng_state() if torch.backends.mps.is_available() else None,
    }


def _validate_rng(state):
    if set(state) != {"streams", "python", "numpy", "torch", "cuda", "mps"} or set(
        state["streams"]
    ) != set(_NAMES):
        raise ValueError("RNG state fields mismatch")
    for value in [state["python"], *state["streams"].values()]:
        random.Random().setstate(value)
    np.random.RandomState().set_state(state["numpy"])
    torch.Generator().set_state(state["torch"])
    if len(state["cuda"]) != (
        torch.cuda.device_count() if torch.cuda.is_available() else 0
    ):
        raise ValueError("CUDA RNG device mismatch")
    for index, device_state in enumerate(state["cuda"]):
        torch.Generator(device=f"cuda:{index}").set_state(device_state)
    if state["mps"] is not None:
        expected = torch.mps.get_rng_state()
        if (
            not isinstance(state["mps"], torch.Tensor)
            or state["mps"].dtype != expected.dtype
            or state["mps"].shape != expected.shape
        ):
            raise ValueError("invalid MPS RNG state")
    if (state["mps"] is not None) != torch.backends.mps.is_available():
        raise ValueError("MPS RNG device mismatch")


def _state_equal(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, dict):
        return (
            isinstance(right, dict)
            and left.keys() == right.keys()
            and all(_state_equal(value, right[key]) for key, value in left.items())
        )
    if isinstance(left, (tuple, list)):
        return (
            type(left) is type(right)
            and len(left) == len(right)
            and all(_state_equal(a, b) for a, b in zip(left, right, strict=True))
        )
    return left == right


def _decode(data):
    envelope = json.loads(data)
    if (
        set(envelope) != {"payload", "checksum"}
        or hashlib.sha256(canonical_json(envelope["payload"])).hexdigest()
        != envelope["checksum"]
    ):
        raise ValueError("manifest checksum mismatch")
    payload = envelope["payload"]
    version = payload.pop("version")
    if type(version) is not int or version != 1:
        raise ValueError("manifest version mismatch")
    inputs = payload["inputs"]
    payload["inputs"] = InputManifest(
        inputs["model_generation"],
        tuple(UnitIdentity(**u) for u in inputs["units"]),
        inputs["start"],
        inputs["stop"],
    )
    progress = payload["progress"]
    progress["consumed_unit_ids"] = tuple(progress["consumed_unit_ids"])
    progress["evaluations"] = tuple(
        EvaluationResult(**e) for e in progress["evaluations"]
    )
    payload["progress"] = Progress(**progress)
    payload["artifacts"] = tuple(tuple(a) for a in payload["artifacts"])
    return CheckpointManifest(**payload)


class CheckpointStore:
    def __init__(self, root, *, config, dataset_store):
        self.root = Path(root)
        self.config_hash = experiment_id(config)
        self.configuration = canonical_json(
            OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
        ).decode()
        self.dataset_store = dataset_store
        self.root.mkdir(parents=True, exist_ok=True)
        _sync_directory(self.root.parent)
        for name in ("staging", "committed"):
            (self.root / name).mkdir(exist_ok=True)
        _sync_directory(self.root)

    @contextmanager
    def _writer(self):
        with (self.root / "writer.lock").open("a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            yield

    def _check_root(self):
        binding = self.root / "experiment.json"
        if binding.exists():
            if json.loads(binding.read_bytes()) != {
                "version": 1,
                "config_hash": self.config_hash,
                "configuration": self.configuration,
            }:
                raise ValueError("root configuration mismatch")
        elif any((self.root / "committed").iterdir()):
            raise ValueError("missing root configuration binding")
        if self.dataset_store.experiment_id != self.config_hash:
            raise ValueError("dataset configuration mismatch")
        self.dataset_store._check_root()

    def _bind_root(self):
        self._check_root()
        binding = self.root / "experiment.json"
        if not binding.exists():
            staged = self.root / "staging" / uuid.uuid4().hex
            self._write(
                staged,
                canonical_json(
                    {
                        "version": 1,
                        "config_hash": self.config_hash,
                        "configuration": self.configuration,
                    }
                ),
            )
            _sync_directory(staged.parent)
            os.rename(staged, binding)
        _sync_directory(self.root)
        _sync_directory(self.root / "staging")

    @staticmethod
    def _write(path, data):
        with path.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())

    def pin_inputs(self, model_generation, *, start=0, stop=None):
        self._check_root()
        units = self.dataset_store.enumerate_units(model_generation=model_generation)
        inputs = InputManifest(
            model_generation, units, start, len(units) if stop is None else stop
        )
        self._validate_inputs(inputs)
        return inputs

    def _validate_inputs(self, inputs):
        for value in (inputs.model_generation, inputs.start, inputs.stop):
            _integer(value)
        if not 0 <= inputs.start <= inputs.stop <= len(inputs.units):
            raise ValueError("input range mismatch")
        ids = tuple(u.unit_id for u in inputs.units)
        if ids != tuple(sorted(set(ids))):
            raise ValueError("input identity ordering/duplicate mismatch")
        for identity in inputs.units:
            if (
                identity
                != self.dataset_store.load(
                    identity.unit_id, model_generation=inputs.model_generation
                ).identity
            ):
                raise ValueError("input checksum/identity mismatch")

    def _validate_progress(self, manifest, parent):
        progress, inputs = manifest.progress, manifest.inputs
        _identifier(manifest.commit_id)
        _integer(manifest.sequence)
        _integer(progress.model_generation)
        if (
            progress.stage not in ("collection", "learning", "evaluation")
            or type(progress.cumulative_cost) not in (float, int)
            or not math.isfinite(progress.cumulative_cost)
            or progress.cumulative_cost < 0
        ):
            raise ValueError("invalid stage/cost")
        if type(progress.consumed_unit_ids) is not tuple or len(
            set(progress.consumed_unit_ids)
        ) != len(progress.consumed_unit_ids):
            raise ValueError("invalid consumption")
        for unit_id in progress.consumed_unit_ids:
            if (
                self.dataset_store.load(unit_id).identity.model_generation
                >= progress.model_generation
            ):
                raise ValueError("consumption generation mismatch")
        evaluations = {}
        if type(progress.evaluations) is not tuple:
            raise ValueError("evaluation tuple required")
        for result in progress.evaluations:
            _identifier(result.checkpoint_id)
            _integer(result.seed)
            if (
                not result.opponent
                or not result.start_pair
                or result.identity in evaluations
            ):
                raise ValueError("evaluation identity mismatch")
            if (
                type(result.opponent) is not str
                or type(result.start_pair) is not str
                or type(result.result_json) is not str
            ):
                raise ValueError("invalid evaluation fields")
            canonical_json(json.loads(result.result_json))
            evaluations[result.identity] = result
        if parent is None:
            if (
                manifest.sequence != 0
                or manifest.parent_id is not None
                or progress.consumed_unit_ids
                or progress.evaluations
                or inputs.model_generation != progress.model_generation
            ):
                raise ValueError("genesis generation/consumption mismatch")
            return
        previous = parent.progress
        if (
            manifest.parent_id != parent.commit_id
            or manifest.sequence != parent.sequence + 1
            or progress.cumulative_cost < previous.cumulative_cost
        ):
            raise ValueError("parent/sequence/cost mismatch")
        delta = progress.model_generation - previous.model_generation
        if delta not in (0, 1):
            raise ValueError("generation mismatch")
        if delta == 0 and inputs.model_generation not in (
            previous.model_generation,
            previous.model_generation - 1,
        ):
            raise ValueError("input generation mismatch")
        if delta == 1:
            selected = tuple(u.unit_id for u in inputs.selected)
            if (
                inputs.model_generation != previous.model_generation
                or not selected
                or set(selected).intersection(previous.consumed_unit_ids)
                or progress.consumed_unit_ids != previous.consumed_unit_ids + selected
            ):
                raise ValueError("generation/input consumption mismatch")
        elif progress.consumed_unit_ids != previous.consumed_unit_ids:
            raise ValueError("consumption without generation advance")
        for result in previous.evaluations:
            if evaluations.get(result.identity) != result:
                raise ValueError("committed evaluation changed")
        evaluation_checkpoints = set()
        ancestor = parent
        while ancestor.progress.model_generation == previous.model_generation:
            evaluation_checkpoints.add(ancestor.commit_id)
            if ancestor.parent_id is None:
                break
            ancestor_path = next(
                p
                for p in (self.root / "committed").iterdir()
                if p.name[21:] == ancestor.parent_id
            )
            ancestor = _decode((ancestor_path / "manifest.json").read_bytes())
        for result in progress.evaluations:
            if (
                result.identity not in {e.identity for e in previous.evaluations}
                and result.checkpoint_id not in evaluation_checkpoints
            ):
                raise ValueError("evaluation checkpoint mismatch")

    def _validate_candidate(self, path, parent):
        manifest = _decode((path / "manifest.json").read_bytes())
        if (
            manifest.config_hash != self.config_hash
            or manifest.configuration != self.configuration
        ):
            raise ConfigurationMismatch("checkpoint configuration mismatch")
        if path.name != f"{manifest.sequence:020d}-{manifest.commit_id}":
            raise ValueError("checkpoint filename mismatch")
        self._validate_inputs(manifest.inputs)
        self._validate_progress(manifest, parent)
        actual = tuple(
            sorted(
                (
                    p.relative_to(path).as_posix(),
                    hashlib.sha256(p.read_bytes()).hexdigest(),
                )
                for p in path.rglob("*")
                if p.is_file() and p.name != "manifest.json"
            )
        )
        if actual != manifest.artifacts or not {
            "runtime.pt",
            "model/config.json",
            "model/model.safetensors",
        }.issubset({a[0] for a in actual}):
            raise ValueError("checkpoint artifact checksum mismatch")
        runtime = torch.load(
            path / "runtime.pt", map_location="cpu", weights_only=False
        )
        if set(runtime) != {"optimizer", "scheduler", "rng"} or not isinstance(
            runtime["optimizer"], dict
        ):
            raise ValueError("invalid runtime artifact")
        _validate_rng(runtime["rng"])
        if (
            parent
            and manifest.progress.model_generation == parent.progress.model_generation
        ):
            parent_path = next(
                p
                for p in (self.root / "committed").iterdir()
                if p.name[21:] == parent.commit_id
            )
            previous = torch.load(
                parent_path / "runtime.pt", map_location="cpu", weights_only=False
            )
            if (
                tuple(a for a in actual if a[0].startswith("model/"))
                != tuple(a for a in parent.artifacts if a[0].startswith("model/"))
                or not _state_equal(runtime["optimizer"], previous["optimizer"])
                or not _state_equal(runtime["scheduler"], previous["scheduler"])
            ):
                raise ValueError("state update without generation advance")
        return RecoveredCheckpoint(manifest, path)

    def recover(self):
        self._check_root()
        valid = None
        for path in sorted((self.root / "committed").iterdir()):
            try:
                candidate = self._validate_candidate(
                    path, valid.manifest if valid else None
                )
            except ConfigurationMismatch:
                raise
            except (
                ValueError,
                TypeError,
                KeyError,
                OSError,
                RuntimeError,
                EOFError,
                pickle.UnpicklingError,
            ):
                continue
            valid = candidate
        if valid is None and any((self.root / "committed").iterdir()):
            raise ValueError("no verified committed checkpoint")
        return valid

    def commit(
        self,
        commit_id,
        *,
        model,
        optimizer,
        scheduler,
        rng,
        progress,
        inputs,
        expected_parent,
    ):
        _identifier(commit_id)
        with self._writer():
            self._bind_root()
            previous = self.recover()
            paths = [
                p
                for p in (self.root / "committed").iterdir()
                if p.name[21:] == commit_id
            ]
            replay = (
                _decode((paths[0] / "manifest.json").read_bytes()) if paths else None
            )
            parent = previous.manifest if previous else None
            if replay:
                parent = None
                if replay.parent_id:
                    parent_paths = [
                        p
                        for p in (self.root / "committed").iterdir()
                        if p.name[21:] == replay.parent_id
                    ]
                    parent = _decode((parent_paths[0] / "manifest.json").read_bytes())
            elif expected_parent != (parent.commit_id if parent else None):
                raise ValueError("stale parent conflict")
            sequence = parent.sequence + 1 if parent else 0
            staging = self.root / "staging" / uuid.uuid4().hex
            staging.mkdir()
            # Saving a snapshot must not advance the caller's random generators.
            model.save_pretrained(staging / "model", safe_serialization=True)
            torch.save(
                {
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict() if scheduler else None,
                    "rng": _rng_state(rng),
                },
                staging / "runtime.pt",
            )
            artifacts = tuple(
                sorted(
                    (
                        p.relative_to(staging).as_posix(),
                        hashlib.sha256(p.read_bytes()).hexdigest(),
                    )
                    for p in staging.rglob("*")
                    if p.is_file()
                )
            )
            manifest = CheckpointManifest(
                commit_id,
                expected_parent,
                sequence,
                self.config_hash,
                self.configuration,
                inputs,
                progress,
                artifacts,
            )
            self._validate_inputs(inputs)
            self._validate_progress(manifest, parent)
            if parent and progress.model_generation == parent.progress.model_generation:
                parent_path = next(
                    p
                    for p in (self.root / "committed").iterdir()
                    if p.name[21:] == parent.commit_id
                )
                saved = torch.load(
                    parent_path / "runtime.pt", map_location="cpu", weights_only=False
                )
                parent_model = tuple(
                    a for a in parent.artifacts if a[0].startswith("model/")
                )
                model_artifacts = tuple(
                    a for a in artifacts if a[0].startswith("model/")
                )
                if (
                    model_artifacts != parent_model
                    or not _state_equal(saved["optimizer"], optimizer.state_dict())
                    or not _state_equal(
                        saved["scheduler"],
                        scheduler.state_dict() if scheduler else None,
                    )
                ):
                    raise ValueError(
                        "model/optimizer update without generation advance"
                    )
            if replay:
                if manifest != replay:
                    raise ValueError("checkpoint replay conflict")
                self._validate_candidate(paths[0], parent)
                _sync_directory(self.root / "committed")
                _sync_directory(self.root / "staging")
                return replay
            payload = {"version": 1, **asdict(manifest)}
            self._write(
                staging / "manifest.json",
                canonical_json(
                    {
                        "payload": payload,
                        "checksum": hashlib.sha256(canonical_json(payload)).hexdigest(),
                    }
                ),
            )
            for file in staging.rglob("*"):
                if file.is_file():
                    with file.open("rb") as stream:
                        os.fsync(stream.fileno())
            _sync_directory(staging / "model")
            _sync_directory(staging)
            _sync_directory(staging.parent)
            destination = self.root / "committed" / f"{sequence:020d}-{commit_id}"
            os.rename(staging, destination)
            _sync_directory(destination.parent)
            _sync_directory(staging.parent)
            return manifest

    def restore(self, recovered, *, optimizer_factory, scheduler_factory):
        self._check_root()
        current = self.recover()
        if recovered is None or current is None or recovered != current:
            raise ValueError("restore requires latest verified checkpoint")
        saved_global = _rng_state(create_rng_streams(0))
        try:
            model = PolicyValueModel.from_pretrained(
                recovered.path / "model", local_files_only=True
            )
            optimizer = optimizer_factory(model)
            scheduler = scheduler_factory(optimizer)
            runtime = torch.load(
                recovered.path / "runtime.pt", map_location="cpu", weights_only=False
            )
            optimizer.load_state_dict(runtime["optimizer"])
            if scheduler is not None:
                scheduler.load_state_dict(runtime["scheduler"])
            elif runtime["scheduler"] is not None:
                raise ValueError("scheduler mismatch")
            state = runtime["rng"]
            _validate_rng(state)
        except BaseException:
            self._restore_globals(saved_global)
            raise
        rng = create_rng_streams(0)
        for name in _NAMES:
            getattr(rng, name).setstate(state["streams"][name])
        try:
            self._restore_globals(state)
        except BaseException:
            self._restore_globals(saved_global)
            raise
        return RestoredRuntime(recovered.manifest, model, optimizer, scheduler, rng)

    @staticmethod
    def _restore_globals(state):
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"])
        if state["cuda"]:
            torch.cuda.set_rng_state_all(state["cuda"])
        if state["mps"] is not None:
            torch.mps.set_rng_state(state["mps"])
