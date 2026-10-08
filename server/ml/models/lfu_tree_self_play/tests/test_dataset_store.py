"""Complete-unit persistence contracts."""

import hashlib
import json
from dataclasses import fields, replace

import pytest
from pydantic import model_serializer
from tests.test_returns import uneven_record

from lfu_tree_self_play.dataset_codec import canonical_json, decode_unit, encode_unit
from lfu_tree_self_play.dataset_store import DatasetStore
from lfu_tree_self_play.game import GameState
from lfu_tree_self_play.records import CollectionRecord, EdgeRecord
from lfu_tree_self_play.returns import ReturnTargets, aggregate_returns


class EqualitySpoofTargets:
    def __init__(self, **values):
        self.__dict__.update(values)

    def __eq__(self, other):
        return True


def _spoof_targets(targets, field_name, mapping):
    values = {field.name: getattr(targets, field.name) for field in fields(targets)}
    values[field_name] = mapping
    return EqualitySpoofTargets(**values)


def test_publish_replays_each_edge_once_with_supplied_targets(tmp_path, monkeypatch):
    import lfu_tree_self_play.returns as module

    record, game = uneven_record()
    targets = aggregate_returns(record, game)
    validate = module.validate_record
    step = game.step
    validations, transitions = [], []

    def tracked_validation(record, game):
        validations.append(record.unit_id)
        return validate(record, game)

    def tracked_step(state, action):
        transitions.append(action)
        return step(state, action)

    monkeypatch.setattr(module, "validate_record", tracked_validation)
    monkeypatch.setattr(game, "step", tracked_step)
    store = DatasetStore(tmp_path, experiment_id=record.experiment_id, game=game)
    identity = store.publish(record, targets)
    assert validations == [record.unit_id]
    assert len(transitions) == len(record.edges)
    assert (tmp_path / "published" / f"{record.unit_id}.json").is_file()
    assert identity.unit_id == record.unit_id


def test_published_bytes_and_identity_match_version_one_codec(tmp_path):
    record, game = uneven_record()
    targets = aggregate_returns(record, game)
    data = encode_unit(record, targets, game)
    # Captured from the existing version-1 codec before the publication repair.
    assert len(data) == 12284
    assert hashlib.sha256(data).hexdigest() == (
        "d4dc20f6d98ce630d15c0170a5e50bafa3c237a890296df4c5cf319e43ed53b4"
    )
    expected = decode_unit(data, game).identity
    assert expected.checksum == (
        "a077c7ca91df7e7576ebf3aee8f9c8a157d449b92957857d911fe7a76103d4da"
    )
    store = DatasetStore(tmp_path, experiment_id=record.experiment_id, game=game)
    assert store.publish(record, targets) == expected
    assert (tmp_path / "published" / f"{record.unit_id}.json").read_bytes() == data


@pytest.mark.parametrize("field_name", [field.name for field in fields(ReturnTargets)])
def test_equality_spoof_cannot_encode_mismatched_targets(field_name):
    record, game = uneven_record()
    targets = aggregate_returns(record, game)
    mapping = dict(getattr(targets, field_name))
    mapping[next(iter(mapping))] = 0.125
    supplied = _spoof_targets(targets, field_name, mapping)
    with pytest.raises(ValueError, match="targets"):
        encode_unit(record, supplied, game)


def test_equality_spoof_rejection_precedes_root_binding(tmp_path):
    record, game = uneven_record()
    targets = aggregate_returns(record, game)
    supplied = _spoof_targets(targets, "node_weights", {"root": 0.125})
    store = DatasetStore(tmp_path, experiment_id=record.experiment_id, game=game)
    with pytest.raises(ValueError, match="targets"):
        store.publish(record, supplied)
    assert not (tmp_path / "experiment.json").exists()
    assert not (tmp_path / "writer.lock").exists()
    assert not tuple((tmp_path / "published").iterdir())
    assert not tuple((tmp_path / "staging").iterdir())


class IterablePairs:
    def __iter__(self):
        return iter((("root", 1.0),))

    def values(self):
        return (1.0,)


class StringSubclass(str):
    pass


class FloatSubclass(float):
    pass


@pytest.mark.parametrize(
    "mapping",
    [
        IterablePairs(),
        [("root", 1.0)],
        None,
        {1: 1.0},
        {StringSubclass("root"): 1.0},
        {"root": float("nan")},
        {"root": float("inf")},
        {"root": float("-inf")},
        {"root": True},
        {"root": "1.0"},
        {"root": FloatSubclass(1.0)},
    ],
)
def test_encoder_rejects_noncanonical_target_maps(mapping):
    record, game = uneven_record()
    targets = aggregate_returns(record, game)
    supplied = _spoof_targets(targets, "node_weights", mapping)
    with pytest.raises((TypeError, ValueError), match="target"):
        encode_unit(record, supplied, game)


@pytest.mark.parametrize("nested", [False, True])
def test_publish_rejects_unknown_record_fields_before_binding(tmp_path, nested):
    record, game = uneven_record()
    targets = aggregate_returns(record, game)
    if nested:
        nodes = (record.nodes[0].model_copy(update={"unexpected": 1}),) + record.nodes[
            1:
        ]
        record = record.model_copy(update={"nodes": nodes})
    else:
        record = record.model_copy(update={"unexpected": 1})
    store = DatasetStore(tmp_path, experiment_id=record.experiment_id, game=game)
    with pytest.raises(ValueError):
        store.publish(record, targets)
    assert not (tmp_path / "experiment.json").exists()
    assert not tuple((tmp_path / "published").iterdir())


@pytest.mark.parametrize("nested", [False, True])
def test_publish_uses_canonical_record_instead_of_custom_serializers(tmp_path, nested):
    class BadWireRecord(CollectionRecord):
        @model_serializer(mode="wrap")
        def serialize(self, handler):
            wire = handler(self)
            wire["edges"][0]["action"] = 63
            return wire

    class BadWireEdge(EdgeRecord):
        @model_serializer(mode="wrap")
        def serialize(self, handler):
            wire = handler(self)
            wire["action"] = 63
            return wire

    record, game = uneven_record()
    targets = aggregate_returns(record, game)
    if nested:
        edge = record.edges[0]
        bad_edge = BadWireEdge(
            **{name: getattr(edge, name) for name in EdgeRecord.model_fields}
        )
        supplied = record.model_copy(update={"edges": (bad_edge,) + record.edges[1:]})
    else:
        supplied = BadWireRecord(
            **{name: getattr(record, name) for name in CollectionRecord.model_fields}
        )
    store = DatasetStore(tmp_path, experiment_id=record.experiment_id, game=game)
    identity = store.publish(supplied, targets)
    unit = store.load(record.unit_id)
    assert unit.record == record
    assert unit.identity == identity
    assert unit.record.edges[0].action == 0
    assert identity.experiment_id == record.experiment_id
    assert type(unit.record) is CollectionRecord
    assert type(unit.record.edges[0]) is EdgeRecord


@pytest.mark.parametrize("nested", [False, True])
def test_canonical_snapshot_rejects_subclass_unknown_fields(tmp_path, nested):
    class ExtraRecord(CollectionRecord):
        unexpected: int = 1

    class ExtraEdge(EdgeRecord):
        unexpected: int = 1

    record, game = uneven_record()
    targets = aggregate_returns(record, game)
    if nested:
        edge = record.edges[0]
        bad_edge = ExtraEdge(
            **{name: getattr(edge, name) for name in EdgeRecord.model_fields}
        )
        record = record.model_copy(update={"edges": (bad_edge,) + record.edges[1:]})
    else:
        record = ExtraRecord(
            **{name: getattr(record, name) for name in CollectionRecord.model_fields}
        )
    store = DatasetStore(tmp_path, experiment_id=record.experiment_id, game=game)
    with pytest.raises(ValueError):
        store.publish(record, targets)
    assert not (tmp_path / "experiment.json").exists()
    assert not tuple((tmp_path / "published").iterdir())


def test_state_subclass_cannot_spoof_terminal_result_before_binding(tmp_path):
    from tests.test_records import build_record, position

    from lfu_tree_self_play.game import ReversiPyGame

    class FalseResultState(GameState):
        @property
        def black_result(self):
            return 0

    root = position(("BBBBBBBB",) * 8, 1, ())
    record = build_record(
        root=FalseResultState(root.observation), mode="independent", positions=()
    )
    game = ReversiPyGame()
    targets = aggregate_returns(record, game)
    store = DatasetStore(tmp_path, experiment_id=record.experiment_id, game=game)
    with pytest.raises(ValueError, match="black result"):
        store.publish(record, targets)
    assert not (tmp_path / "experiment.json").exists()
    assert not (tmp_path / "writer.lock").exists()
    assert not tuple((tmp_path / "published").iterdir())


@pytest.mark.parametrize("malformation", ["boolean", "rounding", "shape"])
def test_state_snapshot_does_not_repair_invalid_observation(tmp_path, malformation):
    import numpy as np
    from tests.test_records import build_record, position

    from lfu_tree_self_play.game import ReversiPyGame

    class InvalidObservationState(GameState):
        @property
        def observation(self):
            planes = super().observation
            if malformation == "boolean":
                return planes.astype(np.bool_)
            if malformation == "rounding":
                planes = planes.astype(np.float64)
                planes[1, 0, 0] = 1e-50
                return planes
            return planes[:, :7]

    root = position(("BBBBBBBB",) * 8, 1, ())
    record = build_record(root=root, mode="independent", positions=())
    targets = aggregate_returns(record, ReversiPyGame())
    node = record.nodes[0].model_copy(
        update={"state": InvalidObservationState(root.observation)}
    )
    record = record.model_copy(update={"nodes": (node,)})
    store = DatasetStore(
        tmp_path, experiment_id=record.experiment_id, game=ReversiPyGame()
    )
    with pytest.raises(ValueError):
        store.publish(record, targets)
    assert not (tmp_path / "experiment.json").exists()
    assert not tuple((tmp_path / "published").iterdir())


@pytest.mark.parametrize("operation", ["replay", "load", "enumerate"])
@pytest.mark.parametrize("mutation", ["checksum", "targets", "transition", "extra"])
def test_existing_corrupt_units_reject_all_read_paths(tmp_path, operation, mutation):
    record, game = uneven_record()
    targets = aggregate_returns(record, game)
    store = DatasetStore(tmp_path, experiment_id=record.experiment_id, game=game)
    store.publish(record, targets)
    path = tmp_path / "published" / f"{record.unit_id}.json"
    envelope = json.loads(path.read_bytes())
    payload = envelope["payload"]
    if mutation == "checksum":
        payload["record"]["round_id"] = "tampered"
    else:
        if mutation == "targets":
            payload["targets"]["node_weights"]["root"] = 0.125
        elif mutation == "transition":
            payload["record"]["edges"][0]["action"] = 63
        else:
            payload["record"]["nodes"][0]["unexpected"] = 1
        envelope["checksum"] = hashlib.sha256(canonical_json(payload)).hexdigest()
    tampered = canonical_json(envelope)
    path.write_bytes(tampered)
    with pytest.raises(ValueError):
        if operation == "replay":
            store.publish(record, targets)
        elif operation == "load":
            store.load(record.unit_id)
        else:
            store.enumerate_units()
    assert path.read_bytes() == tampered


def test_round_trip_and_replay(tmp_path):
    record, game = uneven_record()
    targets = aggregate_returns(record, game)
    unit = decode_unit(encode_unit(record, targets, game), game)
    assert unit.record == record
    assert unit.targets == targets
    store = DatasetStore(tmp_path, experiment_id=record.experiment_id, game=game)
    identity = store.publish(record, targets)
    assert store.publish(record, targets) == identity
    assert store.load(record.unit_id).record == record
    assert store.enumerate_units() == (identity,)
    with pytest.raises(ValueError, match="generation"):
        store.load(record.unit_id, model_generation=99)


def test_integrity_and_semantics():
    record, game = uneven_record()
    targets = aggregate_returns(record, game)
    envelope = json.loads(encode_unit(record, targets, game))
    envelope["payload"]["record"]["round_id"] = "tampered"
    with pytest.raises(ValueError, match="checksum"):
        decode_unit(json.dumps(envelope).encode(), game)
    with pytest.raises(ValueError, match="targets"):
        encode_unit(record, replace(targets, node_weights={}), game)
    envelope["unexpected"] = 1
    with pytest.raises(ValueError):
        decode_unit(json.dumps(envelope).encode(), game)


def test_conflicts_and_incomplete_quarantine(tmp_path):
    record, game = uneven_record()
    targets = aggregate_returns(record, game)
    store = DatasetStore(tmp_path, experiment_id=record.experiment_id, game=game)
    identity = store.publish(record, targets)
    changed = record.model_copy(update={"round_id": "another"})
    with pytest.raises(ValueError, match="conflict"):
        store.publish(changed, targets)
    assert store.enumerate_units() == (identity,)
    (tmp_path / "staging" / "partial").write_bytes(b"{")
    assert len(store.quarantine_incomplete()) == 1
    assert store.enumerate_units() == (identity,)
    with pytest.raises(ValueError):
        store.load("../escape")
    wrong = DatasetStore(tmp_path, experiment_id="wrong", game=game)
    with pytest.raises(ValueError, match="experiment"):
        wrong.load(record.unit_id)


def test_resealed_invalid_targets_and_versions():
    import hashlib

    from lfu_tree_self_play.dataset_codec import canonical_json

    record, game = uneven_record()
    original = json.loads(encode_unit(record, aggregate_returns(record, game), game))
    for mutation in ("targets", "version", "boolean", "extra"):
        envelope = json.loads(json.dumps(original))
        payload = envelope["payload"]
        if mutation == "targets":
            payload["targets"]["node_weights"]["root"] = 0.5
        elif mutation == "version":
            payload["version"] = 2
        elif mutation == "boolean":
            payload["record"]["nodes"][0]["state"][0][0][0] = True
        else:
            payload["record"]["extra"] = 1
        envelope["checksum"] = hashlib.sha256(canonical_json(payload)).hexdigest()
        with pytest.raises(ValueError):
            decode_unit(canonical_json(envelope), game)


def _interrupted_publish(root, boundary, connection):
    import os

    import lfu_tree_self_play.dataset_store as module

    record, game = uneven_record()
    store = DatasetStore(root, experiment_id=record.experiment_id, game=game)
    rename = module.os.rename

    def pause(source, destination):
        if destination.parent.name != "published":
            return rename(source, destination)
        if boundary == "after":
            rename(source, destination)
        connection.send("paused")
        connection.recv()
        os._exit(17)

    module.os.rename = pause
    store.publish(record, aggregate_returns(record, game))


@pytest.mark.parametrize("boundary", ["before", "after"])
def test_process_interruption_and_parallel_reader(tmp_path, boundary):
    import multiprocessing

    context = multiprocessing.get_context("fork")
    parent, child = context.Pipe()
    record, game = uneven_record()
    store = DatasetStore(tmp_path, experiment_id=record.experiment_id, game=game)
    process = context.Process(
        target=_interrupted_publish, args=(tmp_path, boundary, child)
    )
    process.start()
    try:
        assert parent.poll(15)
        assert parent.recv() == "paused"
        # The writer is alive and holding flock throughout these lock-free reads.
        for _ in range(20):
            identities = store.enumerate_units()
            assert len(identities) == (boundary == "after")
            if identities:
                assert store.load(record.unit_id).record == record
        parent.send("stop")
        process.join(15)
        assert process.exitcode == 17
    finally:
        if process.is_alive():
            process.kill()
            process.join()
        parent.close()
        child.close()
    recovered = DatasetStore(tmp_path, experiment_id=record.experiment_id, game=game)
    assert len(recovered.quarantine_incomplete()) == (boundary == "before")
    assert (
        recovered.publish(record, aggregate_returns(record, game))
        == recovered.load(record.unit_id).identity
    )
    assert len(recovered.enumerate_units()) == 1


@pytest.mark.parametrize("empty", [0, 1])
def test_terminal_boundary_round_trip(empty):
    from tests.test_records import build_record, position

    from lfu_tree_self_play.game import ReversiPyGame

    root = position(
        (".WBBBBBB" if empty else "BBBBBBBB",) + ("BBBBBBBB",) * 7,
        1,
        (0,) if empty else (),
    )
    record = build_record(root=root, positions=(0,) if empty else ())
    game = ReversiPyGame()
    targets = aggregate_returns(record, game)
    unit = decode_unit(encode_unit(record, targets, game), game)
    assert unit.record == record
    assert unit.targets == targets


def test_independent_forced_pass_round_trip():
    from tests.test_records import build_record, position

    from lfu_tree_self_play.game import ReversiPyGame

    root = position(
        (
            "..BBBBWB",
            "BBBBB.WB",
            "BWBWBBWB",
            "BWBBWBWB",
            "BWWWBWWB",
            "BWBWWBWB",
            "BWWWWWBB",
            "BWWBBBBB",
        ),
        -1,
        (1, 13),
    )
    record = build_record(root=root, mode="independent", positions=(), first_action=13)
    game = ReversiPyGame()
    targets = aggregate_returns(record, game)
    unit = decode_unit(encode_unit(record, targets, game), game)
    assert unit.record == record
    assert unit.record.nodes[1].state.to_play == -1
    assert unit.targets == targets


def test_root_configuration_cannot_be_poisoned(tmp_path):
    record, game = uneven_record()
    original = DatasetStore(tmp_path, experiment_id=record.experiment_id, game=game)
    # Two handles can exist before the first publication; append must recheck root.
    foreign = DatasetStore(tmp_path, experiment_id="other-config", game=game)
    identity = original.publish(record, aggregate_returns(record, game))
    changed = record.model_copy(
        update={"experiment_id": "other-config", "unit_id": "foreign"}
    )
    with pytest.raises(ValueError, match="experiment"):
        foreign.publish(changed, aggregate_returns(changed, game))
    assert original.enumerate_units() == (identity,)


def test_replay_finishes_interrupted_directory_sync(tmp_path, monkeypatch):
    import lfu_tree_self_play.dataset_store as module

    record, game = uneven_record()
    store = DatasetStore(tmp_path, experiment_id=record.experiment_id, game=game)
    targets = aggregate_returns(record, game)
    original_sync = module._sync_directory

    def fail_published(path):
        if path == tmp_path / "published":
            raise OSError("injected post-rename fsync failure")
        original_sync(path)

    monkeypatch.setattr(module, "_sync_directory", fail_published)
    with pytest.raises(OSError, match="post-rename"):
        store.publish(record, targets)
    assert store.load(record.unit_id).record == record
    calls = []

    def record_sync(path):
        calls.append(path)
        original_sync(path)

    monkeypatch.setattr(module, "_sync_directory", record_sync)
    store.publish(record, targets)
    assert tmp_path / "published" in calls
    assert tmp_path / "staging" in calls


@pytest.mark.parametrize("container", ["pairs", "duplicate_pairs", "tuple_string"])
def test_strict_wire_containers(container):
    import hashlib

    from lfu_tree_self_play.dataset_codec import canonical_json

    record, game = uneven_record()
    envelope = json.loads(encode_unit(record, aggregate_returns(record, game), game))
    payload = envelope["payload"]
    if container == "tuple_string":
        # Existing valid edge IDs a,b can be coerced from this string.
        payload["record"]["sibling_groups"][0]["edge_ids"] = "ab"
    else:
        pairs = list(payload["targets"]["node_weights"].items())
        if container == "duplicate_pairs":
            pairs.append(pairs[0])
        payload["targets"]["node_weights"] = pairs
    envelope["checksum"] = hashlib.sha256(canonical_json(payload)).hexdigest()
    with pytest.raises(ValueError):
        decode_unit(canonical_json(envelope), game)


def _first_writer(root, experiment, connection):
    record, game = uneven_record()
    record = record.model_copy(
        update={"experiment_id": experiment, "unit_id": experiment}
    )
    store = DatasetStore(root, experiment_id=experiment, game=game)
    connection.send("ready")
    connection.recv()
    try:
        store.publish(record, aggregate_returns(record, game))
        connection.send("published")
    except ValueError as error:
        connection.send(str(error))


def test_two_first_writers_bind_one_experiment(tmp_path):
    import multiprocessing

    context = multiprocessing.get_context("fork")
    pipes = [context.Pipe(), context.Pipe()]
    processes = [
        context.Process(target=_first_writer, args=(tmp_path, experiment, child))
        for experiment, (_, child) in zip(("config-a", "config-b"), pipes, strict=True)
    ]
    for process in processes:
        process.start()
    try:
        for parent, _ in pipes:
            assert parent.poll(15)
            assert parent.recv() == "ready"
        for parent, _ in pipes:
            parent.send("go")
        outcomes = []
        for parent, _ in pipes:
            assert parent.poll(15)
            outcomes.append(parent.recv())
        assert outcomes.count("published") == 1
        assert sum("experiment" in outcome for outcome in outcomes) == 1
        winner = ("config-a", "config-b")[outcomes.index("published")]
        _, game = uneven_record()
        identities = DatasetStore(
            tmp_path, experiment_id=winner, game=game
        ).enumerate_units()
        assert len(identities) == 1
        assert identities[0].experiment_id == winner
    finally:
        for process in processes:
            process.join(15)
            if process.is_alive():
                process.kill()
                process.join()
        for parent, child in pipes:
            parent.close()
            child.close()


def test_new_root_parent_directory_is_synced(tmp_path, monkeypatch):
    import lfu_tree_self_play.dataset_store as module

    calls = []
    sync = module._sync_directory

    def tracked(path):
        calls.append(path)
        sync(path)

    monkeypatch.setattr(module, "_sync_directory", tracked)
    _, game = uneven_record()
    root = tmp_path / "new-parent" / "dataset"
    DatasetStore(root, experiment_id="experiment", game=game)
    assert tmp_path in calls
    assert root.parent in calls
    assert root in calls
