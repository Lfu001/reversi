"""Complete-unit persistence contracts."""

import json
from dataclasses import replace

import pytest
from tests.test_returns import uneven_record

from lfu_tree_self_play.dataset_codec import decode_unit, encode_unit
from lfu_tree_self_play.dataset_store import DatasetStore
from lfu_tree_self_play.returns import aggregate_returns


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
