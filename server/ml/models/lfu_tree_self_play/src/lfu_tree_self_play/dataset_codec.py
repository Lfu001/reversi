"""Versioned, explicit complete-unit wire format."""

import hashlib
import json
from dataclasses import dataclass, fields

import numpy as np

from .game import Game, GameState
from .records import CollectionRecord
from .returns import ReturnTargets, aggregate_returns


@dataclass(frozen=True)
class UnitIdentity:
    unit_id: str
    experiment_id: str
    model_generation: int
    checksum: str


@dataclass(frozen=True)
class DatasetUnit:
    record: CollectionRecord
    targets: ReturnTargets
    identity: UnitIdentity


def canonical_json(value: object) -> bytes:
    """UTF-8, sorted keys, compact separators, no nonfinite numbers."""
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def _payload(record: CollectionRecord, targets: ReturnTargets, game: Game) -> dict:
    expected = aggregate_returns(record, game)
    for field in fields(ReturnTargets):
        mapping = getattr(targets, field.name)
        if any(type(v) not in (int, float) for v in mapping.values()):
            raise ValueError("targets require finite real numbers excluding bool")
    if expected != targets:
        raise ValueError("targets do not match validated record")
    wire = record.model_dump(mode="python")
    for node, original in zip(wire["nodes"], record.nodes, strict=True):
        node["state"] = original.state.observation.tolist()
    return {
        "version": 1,
        "record": wire,
        "targets": {
            field.name: dict(getattr(targets, field.name))
            for field in fields(ReturnTargets)
        },
    }


def encode_unit(record: CollectionRecord, targets: ReturnTargets, game: Game) -> bytes:
    payload = _payload(record, targets, game)
    checksum = hashlib.sha256(canonical_json(payload)).hexdigest()
    return canonical_json({"payload": payload, "checksum": checksum})


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _decode_unit(data: bytes, game: Game) -> DatasetUnit:
    envelope = json.loads(data, object_pairs_hook=_pairs)
    if not isinstance(envelope, dict) or set(envelope) != {"payload", "checksum"}:
        raise ValueError("invalid envelope fields")
    payload = envelope["payload"]
    checksum = hashlib.sha256(canonical_json(payload)).hexdigest()
    if checksum != envelope["checksum"]:
        raise ValueError("checksum mismatch")
    if not isinstance(payload, dict) or set(payload) != {
        "version",
        "record",
        "targets",
    }:
        raise ValueError("invalid payload fields")
    if type(payload["version"]) is not int or payload["version"] != 1:
        raise ValueError("unsupported codec version")
    wire = payload["record"]
    if not isinstance(wire, dict):
        raise TypeError("invalid record")
    for key in ("nodes", "edges", "sibling_groups"):
        if not isinstance(wire[key], list):
            raise TypeError(f"{key} must be a JSON array")
    for owner, key in ((wire["branch_plan"], "positions"), (wire["rng"], "streams")):
        if not isinstance(owner[key], list):
            raise TypeError(f"{key} must be a JSON array")
    for group in wire["sibling_groups"]:
        if not isinstance(group["edge_ids"], list):
            raise TypeError("edge_ids must be a JSON array")
    target_wire = payload["targets"]
    names = {field.name for field in fields(ReturnTargets)}
    if not isinstance(target_wire, dict) or set(target_wire) != names:
        raise ValueError("invalid targets object fields")
    if any(not isinstance(mapping, dict) for mapping in target_wire.values()):
        raise ValueError("target mappings must be JSON objects")
    for node in wire["nodes"]:
        raw = node["state"]
        if any(
            type(value) not in (int, float)
            for plane in raw
            for row in plane
            for value in row
        ):
            raise ValueError("state requires real numbers excluding bool")
        planes = np.asarray(raw)
        if planes.shape != (4, 8, 8) or planes.dtype.kind not in "if":
            raise ValueError("invalid state planes")
        converted = planes.astype(np.float32)
        if not np.array_equal(planes, converted):
            raise ValueError("state cannot be represented losslessly as float32")
        node["state"] = GameState(converted)
    wire["branch_plan"]["positions"] = tuple(wire["branch_plan"]["positions"])
    wire["rng"]["streams"] = tuple(wire["rng"]["streams"])
    for group in wire["sibling_groups"]:
        group["edge_ids"] = tuple(group["edge_ids"])
    for key in ("nodes", "edges", "sibling_groups"):
        wire[key] = tuple(wire[key])
    record = CollectionRecord.model_validate(wire)
    targets = ReturnTargets(**payload["targets"])
    _payload(record, targets, game)
    return DatasetUnit(
        record,
        targets,
        UnitIdentity(
            record.unit_id, record.experiment_id, record.model_generation, checksum
        ),
    )


def decode_unit(data: bytes, game: Game) -> DatasetUnit:
    """Verify checksum, strict wire fields, game semantics, and all targets."""
    try:
        return _decode_unit(data, game)
    except (KeyError, TypeError, AttributeError, OverflowError) as error:
        raise ValueError("malformed unit") from error
