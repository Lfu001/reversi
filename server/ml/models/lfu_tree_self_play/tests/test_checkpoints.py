"""Checkpoint and committed-consumption transactions."""

import os
import random
from dataclasses import replace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from tests.test_returns import uneven_record

from lfu_tree_self_play.checkpoints import CheckpointStore, EvaluationResult, Progress
from lfu_tree_self_play.config import experiment_id
from lfu_tree_self_play.dataset_store import DatasetStore
from lfu_tree_self_play.model import PolicyValueConfig, PolicyValueModel
from lfu_tree_self_play.returns import aggregate_returns
from lfu_tree_self_play.rng import create_rng_streams


def setup_store(tmp_path):
    config = OmegaConf.create(
        {
            "experiment": {"name": "checkpoints"},
            "seeds": {"training": [1], "tuning": [2], "final_evaluation": [3]},
        }
    )
    record, game = uneven_record()
    record = record.model_copy(update={"experiment_id": experiment_id(config)})
    dataset = DatasetStore(
        tmp_path / "dataset", experiment_id=experiment_id(config), game=game
    )
    identity = dataset.publish(record, aggregate_returns(record, game))
    store = CheckpointStore(
        tmp_path / "checkpoints", config=config, dataset_store=dataset
    )
    model = PolicyValueModel(PolicyValueConfig(width=4, num_blocks=1))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    model(torch.randn(2, 4, 8, 8)).value.sum().backward()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    optimizer.step()
    scheduler.step()
    return store, model, optimizer, scheduler, create_rng_streams(42), identity


def commit(
    store,
    model,
    optimizer,
    scheduler,
    rng,
    identity,
    *,
    name="genesis",
    parent=None,
    generation=None,
    consumed=(),
    evaluations=(),
):
    generation = identity.model_generation if generation is None else generation
    return store.commit(
        name,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        rng=rng,
        progress=Progress(generation, "evaluation", 1.0, consumed, evaluations),
        inputs=store.pin_inputs(identity.model_generation),
        expected_parent=parent,
    )


def test_roundtrip(tmp_path):
    store, model, optimizer, scheduler, rng, identity = setup_store(tmp_path)
    manifest = commit(store, model, optimizer, scheduler, rng, identity)
    draws = (
        random.random(),
        np.random.rand(),
        torch.rand(3),
        *(
            getattr(rng, n).random()
            for n in (
                "branch_position",
                "action_sampling",
                "sibling_rollout",
                "evaluation",
            )
        ),
    )
    restored = store.restore(
        store.recover(),
        optimizer_factory=lambda m: torch.optim.Adam(m.parameters()),
        scheduler_factory=lambda o: torch.optim.lr_scheduler.StepLR(o, step_size=1),
    )
    assert restored.manifest == manifest
    assert all(
        torch.equal(v, restored.model.state_dict()[k])
        for k, v in model.state_dict().items()
    )
    assert (
        restored.optimizer.state_dict()["param_groups"]
        == optimizer.state_dict()["param_groups"]
    )
    for key, value in optimizer.state_dict()["state"].items():
        assert all(
            torch.equal(v, restored.optimizer.state_dict()["state"][key][k])
            for k, v in value.items()
        )
    assert restored.scheduler.state_dict() == scheduler.state_dict()
    actual = (
        random.random(),
        np.random.rand(),
        torch.rand(3),
        *(
            getattr(restored.rng, n).random()
            for n in (
                "branch_position",
                "action_sampling",
                "sibling_rollout",
                "evaluation",
            )
        ),
    )
    assert (
        draws[:2] == actual[:2]
        and torch.equal(draws[2], actual[2])
        and draws[3:] == actual[3:]
    )


def test_consumption_replay_and_fallback(tmp_path):
    store, model, optimizer, scheduler, rng, identity = setup_store(tmp_path)
    first = commit(store, model, optimizer, scheduler, rng, identity)
    second = commit(
        store,
        model,
        optimizer,
        scheduler,
        rng,
        identity,
        name="update",
        parent=first.commit_id,
        generation=identity.model_generation + 1,
        consumed=(identity.unit_id,),
    )
    assert (
        commit(
            store,
            model,
            optimizer,
            scheduler,
            rng,
            identity,
            name="update",
            parent=first.commit_id,
            generation=identity.model_generation + 1,
            consumed=(identity.unit_id,),
        )
        == second
    )
    with pytest.raises(ValueError, match="conflict"):
        commit(
            store,
            model,
            optimizer,
            scheduler,
            rng,
            identity,
            name="update",
            parent=first.commit_id,
        )
    (store.recover().path / "runtime.pt").write_bytes(b"broken")
    assert store.recover().manifest == first


def test_mismatches(tmp_path):
    store, model, optimizer, scheduler, rng, identity = setup_store(tmp_path)
    first = commit(store, model, optimizer, scheduler, rng, identity)
    with pytest.raises(ValueError, match="configuration"):
        CheckpointStore(
            store.root,
            config=OmegaConf.create({"wrong": 1}),
            dataset_store=store.dataset_store,
        ).recover()
    inputs = store.pin_inputs(identity.model_generation)
    for bad in (
        replace(inputs, stop=2),
        replace(inputs, model_generation=99),
        replace(inputs, units=(replace(identity, checksum="0" * 64),)),
    ):
        with pytest.raises(ValueError):
            store.commit(
                "bad",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                rng=rng,
                progress=Progress(
                    identity.model_generation + 1, "learning", 2.0, (identity.unit_id,)
                ),
                inputs=bad,
                expected_parent=first.commit_id,
            )
    with pytest.raises(ValueError, match="consumption"):
        commit(
            store,
            model,
            optimizer,
            scheduler,
            rng,
            identity,
            name="bad",
            parent=first.commit_id,
            generation=identity.model_generation + 1,
        )


def test_interruption_and_evaluation(tmp_path, monkeypatch):
    store, model, optimizer, scheduler, rng, identity = setup_store(tmp_path)
    first = commit(store, model, optimizer, scheduler, rng, identity)
    rename = os.rename

    def stop(source, destination):
        if destination.parent.name == "committed":
            raise RuntimeError("stop")
        rename(source, destination)

    monkeypatch.setattr(os, "rename", stop)
    with pytest.raises(RuntimeError):
        commit(
            store,
            model,
            optimizer,
            scheduler,
            rng,
            identity,
            name="update",
            parent=first.commit_id,
            generation=identity.model_generation + 1,
            consumed=(identity.unit_id,),
        )
    assert store.recover().manifest == first
    monkeypatch.setattr(os, "rename", rename)
    result = EvaluationResult(first.commit_id, "opponent", "pair", 5, '{"score":1.0}')
    evaluated = commit(
        store,
        model,
        optimizer,
        scheduler,
        rng,
        identity,
        name="evaluation",
        parent=first.commit_id,
        evaluations=(result,),
    )
    assert (
        store.recover().manifest.progress.evaluations == evaluated.progress.evaluations
    )


def test_pinned_range_survives_later_lexical_insert_and_stale_writer(tmp_path):
    store, model, optimizer, scheduler, rng, identity = setup_store(tmp_path)
    first = commit(store, model, optimizer, scheduler, rng, identity)
    pinned = store.pin_inputs(identity.model_generation)
    original = store.dataset_store.load(identity.unit_id)
    added = original.record.model_copy(update={"unit_id": "000-later"})
    store.dataset_store.publish(
        added, aggregate_returns(added, store.dataset_store.game)
    )
    assert store.pin_inputs(identity.model_generation).units[0].unit_id == "000-later"
    updated = store.commit(
        "update",
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        rng=rng,
        progress=Progress(
            identity.model_generation + 1, "learning", 2.0, (identity.unit_id,)
        ),
        inputs=pinned,
        expected_parent=first.commit_id,
    )
    assert updated.inputs.selected == (identity,)
    with pytest.raises(ValueError, match="stale parent"):
        commit(
            store,
            model,
            optimizer,
            scheduler,
            rng,
            identity,
            name="stale",
            parent=first.commit_id,
        )


def test_post_rename_retry_finishes_durability(tmp_path, monkeypatch):
    import lfu_tree_self_play.checkpoints as module

    store, model, optimizer, scheduler, rng, identity = setup_store(tmp_path)
    first = commit(store, model, optimizer, scheduler, rng, identity)
    sync = module._sync_directory

    def stop(path):
        if path.name == "committed":
            raise RuntimeError("directory sync stop")
        sync(path)

    monkeypatch.setattr(module, "_sync_directory", stop)
    with pytest.raises(RuntimeError):
        commit(
            store,
            model,
            optimizer,
            scheduler,
            rng,
            identity,
            name="update",
            parent=first.commit_id,
            generation=identity.model_generation + 1,
            consumed=(identity.unit_id,),
        )
    assert store.recover().manifest.progress.consumed_unit_ids == (identity.unit_id,)
    synced = []
    monkeypatch.setattr(
        module, "_sync_directory", lambda path: (synced.append(path.name), sync(path))
    )
    retried = commit(
        store,
        model,
        optimizer,
        scheduler,
        rng,
        identity,
        name="update",
        parent=first.commit_id,
        generation=identity.model_generation + 1,
        consumed=(identity.unit_id,),
    )
    assert retried == store.recover().manifest
    assert "committed" in synced and "staging" in synced


def test_resealed_semantic_corruption_and_failed_restore_rollback(tmp_path):
    import hashlib
    import json

    from lfu_tree_self_play.dataset_codec import canonical_json

    store, model, optimizer, scheduler, rng, identity = setup_store(tmp_path)
    first = commit(store, model, optimizer, scheduler, rng, identity)
    second = commit(
        store,
        model,
        optimizer,
        scheduler,
        rng,
        identity,
        name="update",
        parent=first.commit_id,
        generation=identity.model_generation + 1,
        consumed=(identity.unit_id,),
    )
    path = store.recover().path / "manifest.json"
    envelope = json.loads(path.read_bytes())
    envelope["payload"]["progress"]["model_generation"] += 2
    envelope["checksum"] = hashlib.sha256(
        canonical_json(envelope["payload"])
    ).hexdigest()
    path.write_bytes(canonical_json(envelope))
    assert store.recover().manifest == first != second
    state = torch.get_rng_state().clone()
    python_state = random.getstate()

    def bad_optimizer(model):
        torch.rand(5)
        random.random()
        raise RuntimeError("factory failure")

    with pytest.raises(RuntimeError, match="factory failure"):
        store.restore(
            store.recover(),
            optimizer_factory=bad_optimizer,
            scheduler_factory=lambda o: None,
        )
    assert (
        torch.equal(state, torch.get_rng_state()) and python_state == random.getstate()
    )


def test_evaluation_replay_conflict_and_generation_rejection(tmp_path):
    store, model, optimizer, scheduler, rng, identity = setup_store(tmp_path)
    first = commit(store, model, optimizer, scheduler, rng, identity)
    result = EvaluationResult(first.commit_id, "opponent", "pair", 5, '{"score":1.0}')
    evaluated = commit(
        store,
        model,
        optimizer,
        scheduler,
        rng,
        identity,
        name="evaluated",
        parent=first.commit_id,
        evaluations=(result,),
    )
    for bad in ((result, result), (replace(result, result_json='{"score":0.0}'),), ()):
        with pytest.raises(ValueError):
            commit(
                store,
                model,
                optimizer,
                scheduler,
                rng,
                identity,
                name="bad",
                parent=evaluated.commit_id,
                evaluations=bad,
            )
    with pytest.raises(ValueError, match="generation"):
        commit(
            store,
            model,
            optimizer,
            scheduler,
            rng,
            identity,
            name="jump",
            parent=evaluated.commit_id,
            generation=identity.model_generation + 2,
        )


def test_same_generation_cannot_hide_model_update(tmp_path):
    store, model, optimizer, scheduler, rng, identity = setup_store(tmp_path)
    first = commit(store, model, optimizer, scheduler, rng, identity)
    with torch.no_grad():
        next(model.parameters()).add_(1)
    with pytest.raises(ValueError, match="generation advance"):
        commit(
            store,
            model,
            optimizer,
            scheduler,
            rng,
            identity,
            name="hidden",
            parent=first.commit_id,
        )
    assert store.recover().manifest == first


def test_all_corrupt_commits_fail_closed(tmp_path):
    store, model, optimizer, scheduler, rng, identity = setup_store(tmp_path)
    commit(store, model, optimizer, scheduler, rng, identity)
    (store.recover().path / "runtime.pt").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="no verified"):
        store.recover()


def test_first_writer_configuration_race(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    store, model, optimizer, scheduler, rng, identity = setup_store(tmp_path)
    config = OmegaConf.create({"experiment": {"name": "other"}})
    other_dataset = DatasetStore(
        tmp_path / "other-data",
        experiment_id=experiment_id(config),
        game=store.dataset_store.game,
    )
    original = store.dataset_store.load(identity.unit_id).record
    other_record = original.model_copy(update={"experiment_id": experiment_id(config)})
    other_identity = other_dataset.publish(
        other_record, aggregate_returns(other_record, other_dataset.game)
    )
    other = CheckpointStore(store.root, config=config, dataset_store=other_dataset)
    barrier = Barrier(2)

    def publish(candidate, unit):
        barrier.wait()
        try:
            commit(candidate, model, optimizer, scheduler, rng, unit)
            return "committed"
        except ValueError as error:
            assert "configuration" in str(error)
            return "rejected"

    with ThreadPoolExecutor(max_workers=2) as workers:
        results = [
            workers.submit(publish, candidate, unit)
            for candidate, unit in ((store, identity), (other, other_identity))
        ]
        assert sorted(result.result() for result in results) == [
            "committed",
            "rejected",
        ]


def test_commit_ids_with_shared_suffix_do_not_alias(tmp_path):
    store, model, optimizer, scheduler, rng, identity = setup_store(tmp_path)
    first = commit(store, model, optimizer, scheduler, rng, identity, name="prefix-id")
    second = commit(
        store,
        model,
        optimizer,
        scheduler,
        rng,
        identity,
        name="id",
        parent=first.commit_id,
    )
    assert second.commit_id == "id" and second.parent_id == "prefix-id"
    assert store.recover().manifest == second


def test_incremental_evaluation_retains_fixed_checkpoint_identity(tmp_path):
    store, model, optimizer, scheduler, rng, identity = setup_store(tmp_path)
    first = commit(store, model, optimizer, scheduler, rng, identity)
    pair_a = EvaluationResult(first.commit_id, "opponent", "pair-a", 5, '{"score":1}')
    partial = commit(
        store,
        model,
        optimizer,
        scheduler,
        rng,
        identity,
        name="partial-evaluation",
        parent=first.commit_id,
        evaluations=(pair_a,),
    )
    pair_b = EvaluationResult(first.commit_id, "opponent", "pair-b", 5, '{"score":0}')
    complete = commit(
        store,
        model,
        optimizer,
        scheduler,
        rng,
        identity,
        name="complete-evaluation",
        parent=partial.commit_id,
        evaluations=(pair_a, pair_b),
    )
    assert store.recover().manifest == complete


@pytest.mark.parametrize("mode", ["train", "eval", "mixed"])
def test_model_mode_roundtrip_preserves_subsequent_behavior(tmp_path, mode):
    store, model, optimizer, scheduler, rng, identity = setup_store(tmp_path)
    if mode == "eval":
        model.eval()
    elif mode == "mixed":
        model.train()
        model.stem[1].eval()
    expected_modes = {name: module.training for name, module in model.named_modules()}
    commit(store, model, optimizer, scheduler, rng, identity)
    inputs = torch.arange(2 * 4 * 8 * 8, dtype=torch.float32).reshape(2, 4, 8, 8) / 100
    with torch.no_grad():
        expected = model(inputs)
    restored = store.restore(
        store.recover(),
        optimizer_factory=lambda m: torch.optim.Adam(m.parameters()),
        scheduler_factory=lambda o: torch.optim.lr_scheduler.StepLR(o, step_size=1),
    )
    with torch.no_grad():
        actual = restored.model(inputs)
    assert torch.equal(actual.policy_logits, expected.policy_logits)
    assert torch.equal(actual.value, expected.value)
    assert {
        name: module.training for name, module in restored.model.named_modules()
    } == expected_modes
    assert all(
        torch.equal(value, restored.model.state_dict()[name])
        for name, value in model.state_dict().items()
    )


def test_model_mode_change_requires_generation_and_invalid_flags_recover_previous(
    tmp_path,
):
    import hashlib
    import json

    from lfu_tree_self_play.dataset_codec import canonical_json

    store, model, optimizer, scheduler, rng, identity = setup_store(tmp_path)
    first = commit(store, model, optimizer, scheduler, rng, identity)
    model.eval()
    with pytest.raises(ValueError, match="generation advance"):
        commit(
            store,
            model,
            optimizer,
            scheduler,
            rng,
            identity,
            name="mode-only",
            parent=first.commit_id,
        )
    commit(
        store,
        model,
        optimizer,
        scheduler,
        rng,
        identity,
        name="update",
        parent=first.commit_id,
        generation=identity.model_generation + 1,
        consumed=(identity.unit_id,),
    )
    candidate = store.recover().path
    original_runtime = torch.load(candidate / "runtime.pt", weights_only=False)
    original_manifest = json.loads((candidate / "manifest.json").read_bytes())
    for flags in ({**original_runtime["model_modes"], "": 1}, {"": False}):
        runtime = {**original_runtime, "model_modes": flags}
        torch.save(runtime, candidate / "runtime.pt")
        envelope = json.loads(json.dumps(original_manifest))
        artifacts = envelope["payload"]["artifacts"]
        for artifact in artifacts:
            if artifact[0] == "runtime.pt":
                artifact[1] = hashlib.sha256(
                    (candidate / "runtime.pt").read_bytes()
                ).hexdigest()
        envelope["checksum"] = hashlib.sha256(
            canonical_json(envelope["payload"])
        ).hexdigest()
        (candidate / "manifest.json").write_bytes(canonical_json(envelope))
        assert store.recover().manifest == first
