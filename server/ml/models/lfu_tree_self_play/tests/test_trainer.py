"""A real collection/update and crash recovery through the trainer boundary."""

import json
import math
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from shutil import copytree

import pytest
import torch
from accelerate import Accelerator
from omegaconf import OmegaConf
from tests.test_losses import clipping_batch
from tests.test_recovery import equal

from lfu_tree_self_play.checkpoints import CheckpointStore, Progress
from lfu_tree_self_play.collector import IndependentCollector
from lfu_tree_self_play.config import experiment_id
from lfu_tree_self_play.dataset_store import DatasetStore
from lfu_tree_self_play.game import ReversiPyGame
from lfu_tree_self_play.model import PolicyValueConfig, PolicyValueModel
from lfu_tree_self_play.rng import create_rng_streams
from lfu_tree_self_play.runtime_codec import load_runtime
from lfu_tree_self_play.trainer import TrainingOptions, _diagnostics, train_one_epoch
from lfu_tree_self_play.tree_collector import TreeCollector
from lfu_tree_self_play.verification import endgame_position


def optimizer_factory(model):
    return torch.optim.Adam(model.parameters(), lr=0.01)


def scheduler_factory(optimizer):
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)


def setup(root, *, independent=False):
    torch.set_num_threads(1)
    torch.manual_seed(123)
    config = OmegaConf.create({"experiment": {"name": "trainer"}})
    dataset = DatasetStore(
        root / "dataset", experiment_id=experiment_id(config), game=ReversiPyGame()
    )
    store = CheckpointStore(root / "checkpoints", config=config, dataset_store=dataset)
    if store.recover() is None:
        model = PolicyValueModel(PolicyValueConfig(width=4, num_blocks=1)).eval()
        common = {
            "experiment_id": experiment_id(config),
            "round_id": "round-0",
            "model_generation": 0,
            "run_seed": 42,
        }
        if independent:
            collector = IndependentCollector(model, dataset.game, **common)
            collector.collect_paths(
                endgame_position(-1), paths_per_root=2, unit_ids=("a", "b")
            ).publish_all(dataset)
        else:
            collector = TreeCollector(model, game=dataset.game, levels=1, **common)
            for uid in ("a", "b"):
                tree = collector.collect_tree(endgame_position(-1), unit_id=uid)
                dataset.publish(tree.record, tree.targets)
        optimizer = optimizer_factory(model)
        store.commit(
            "collected",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler_factory(optimizer),
            rng=create_rng_streams(42),
            inputs=store.pin_inputs(0),
            progress=Progress(0, "collection", 1.0),
            expected_parent=None,
        )
    return store


def train(store, **kwargs):
    return train_one_epoch(
        store,
        optimizer_factory=optimizer_factory,
        scheduler_factory=scheduler_factory,
        accelerator=Accelerator(cpu=True),
        **kwargs,
    )


def test_one_epoch_metrics_and_idempotence(tmp_path, monkeypatch):
    store = setup(tmp_path)
    initial = store.restore(
        store.recover(),
        optimizer_factory=optimizer_factory,
        scheduler_factory=scheduler_factory,
    )
    result = train(store)
    assert result.manifest.progress.model_generation == 1
    assert result.manifest.progress.consumed_unit_ids == ("a", "b")
    metrics = result.report["metrics"]
    assert metrics["gradient_norm"] > 0
    assert metrics["num_roots"] == metrics["num_units"] == 2
    assert metrics["before"]["kl"] == 0
    assert metrics["before"]["clip_fraction"] == 0
    assert metrics["after"]["kl"] > 0
    assert all(math.isfinite(value) for value in metrics["after"].values())
    assert result.manifest.progress.cumulative_cost == 1.0 + metrics["learning_seconds"]
    restored = store.restore(
        store.recover(),
        optimizer_factory=optimizer_factory,
        scheduler_factory=scheduler_factory,
    )
    assert restored.scheduler.last_epoch == 1
    assert {int(state["step"]) for state in restored.optimizer.state.values()} == {1}
    assert any(
        not torch.equal(value, restored.model.state_dict()[key])
        for key, value in initial.model.state_dict().items()
    )
    for key, value in initial.model.state_dict().items():
        if "running_" in key or "num_batches_tracked" in key:
            assert torch.equal(value, restored.model.state_dict()[key])
    assert (
        json.loads((store.recover().path / "training.json").read_bytes())
        == result.report
    )

    def unexpected(*args, **kwargs):
        raise AssertionError("second update")

    monkeypatch.setattr(torch.optim.Adam, "step", unexpected)
    assert train(store) == result
    with pytest.raises(ValueError, match="plan mismatch"):
        train(store, options=TrainingOptions(value_coefficient=0.5))


def test_diagnostics_match_hand_calculation():
    batch, output = clipping_batch()
    mask = batch.observations[:, 3].flatten(1).bool()
    old = torch.full((4, 64), -torch.inf, dtype=torch.float64)
    old[mask] = math.log(0.5)
    metrics = _diagnostics(lambda _: output, [batch], Accelerator(cpu=True), [old], 0.2)
    assert metrics["kl"] == pytest.approx(0.5 * math.log(4 / 3))
    assert metrics["clip_fraction"] == 1
    assert metrics["policy_entropy"] == pytest.approx(
        -0.75 * math.log(0.75) - 0.25 * math.log(0.25)
    )
    assert metrics["value_error"] == pytest.approx((1 + 0.5 + 0.25 * 0.25) / 1.875)


@pytest.mark.parametrize(
    "groups", [(("a", "a"),), (("a",),), (("a", "b"),), ((), ("a", "b"))]
)
def test_invalid_root_groups_do_not_commit(tmp_path, groups):
    store = setup(tmp_path)
    with pytest.raises(ValueError):
        train(store, root_groups=groups)
    assert store.recover().manifest.commit_id == "collected"


def test_independent_paths_share_one_root(tmp_path):
    store = setup(tmp_path, independent=True)
    result = train(store, root_groups=(("a", "b"),))
    assert result.report["metrics"]["num_roots"] == 1
    assert result.report["metrics"]["num_units"] == 2
    assert train(store) == result


def test_leave_one_out_runs_from_same_data(tmp_path):
    result = train(setup(tmp_path), options=TrainingOptions(advantage="leave_one_out"))
    assert result.report["plan"]["options"]["advantage"] == "leave_one_out"


def test_corrupt_report_falls_back_to_unconsumed_plan(tmp_path):
    store = setup(tmp_path)
    result = train(store)
    (store.recover().path / "training.json").write_text("{}")
    assert store.recover().manifest.progress.consumed_unit_ids == ()
    # Corrupt committed destinations require operator repair, not overwrite.
    assert store.recover().manifest.commit_id == result.manifest.parent_id


def test_pending_plan_keeps_inputs_and_rejects_changed_options(tmp_path, monkeypatch):
    store = setup(tmp_path)

    def stop(*args, **kwargs):
        raise RuntimeError("interrupted")

    with monkeypatch.context() as patch:
        patch.setattr(Accelerator, "backward", stop)
        with pytest.raises(RuntimeError, match="interrupted"):
            train(store)
    pending = store.recover().manifest
    assert pending.progress.model_generation == 0
    assert pending.progress.consumed_unit_ids == ()
    with pytest.raises(ValueError, match="plan mismatch"):
        train(store, options=TrainingOptions(advantage="leave_one_out"))
    original = store.dataset_store.load("a")
    store.dataset_store.publish(
        original.record.model_copy(update={"unit_id": "0-later"}), original.targets
    )
    result = train(store)
    assert result.manifest.inputs == pending.inputs
    assert result.manifest.progress.consumed_unit_ids == ("a", "b")


def test_next_generation_advances_optimizer_once_and_rejects_old_manifest(tmp_path):
    store = setup(tmp_path)
    train(store)
    runtime = store.restore(
        store.recover(),
        optimizer_factory=optimizer_factory,
        scheduler_factory=scheduler_factory,
    )
    old_inputs = runtime.manifest.inputs
    store.commit(
        "evaluated",
        model=runtime.model,
        optimizer=runtime.optimizer,
        scheduler=runtime.scheduler,
        rng=runtime.rng,
        progress=replace(runtime.manifest.progress, stage="evaluation"),
        inputs=old_inputs,
        expected_parent=runtime.manifest.commit_id,
    )
    with pytest.raises(ValueError, match="already consumed"):
        train(store)
    collector = TreeCollector(
        runtime.model,
        game=store.dataset_store.game,
        experiment_id=store.config_hash,
        round_id="round-1",
        model_generation=1,
        run_seed=43,
        levels=1,
    )
    tree = collector.collect_tree(endgame_position(-1), unit_id="c")
    store.dataset_store.publish(tree.record, tree.targets)
    store.commit(
        "collected-1",
        model=runtime.model,
        optimizer=runtime.optimizer,
        scheduler=runtime.scheduler,
        rng=runtime.rng,
        progress=replace(runtime.manifest.progress, stage="collection"),
        inputs=store.pin_inputs(1),
        expected_parent="evaluated",
    )
    result = train(store)
    assert result.manifest.progress.model_generation == 2
    assert result.manifest.progress.consumed_unit_ids == ("a", "b", "c")
    updated = store.restore(
        store.recover(),
        optimizer_factory=optimizer_factory,
        scheduler_factory=scheduler_factory,
    )
    assert {int(state["step"]) for state in updated.optimizer.state.values()} == {2}
    assert updated.scheduler.last_epoch == 2


def test_empty_manifest_rejected_without_update(tmp_path):
    store = setup(tmp_path)
    runtime = store.restore(
        store.recover(),
        optimizer_factory=optimizer_factory,
        scheduler_factory=scheduler_factory,
    )
    store.commit(
        "empty",
        model=runtime.model,
        optimizer=runtime.optimizer,
        scheduler=runtime.scheduler,
        rng=runtime.rng,
        progress=runtime.manifest.progress,
        inputs=replace(runtime.manifest.inputs, stop=0),
        expected_parent="collected",
    )
    with pytest.raises(ValueError, match="empty"):
        train(store)
    assert store.recover().manifest.commit_id == "empty"


def crash_worker(root, boundary):
    store = setup(root)
    backward = Accelerator.backward
    step = torch.optim.Adam.step
    rename = os.rename

    def stop(name):
        if boundary == name:
            os._exit(73)

    def interrupted_backward(self, *args, **kwargs):
        backward(self, *args, **kwargs)
        stop("backward")

    def interrupted_step(self, *args, **kwargs):
        result = step(self, *args, **kwargs)
        stop("step")
        return result

    def interrupted_rename(source, destination):
        destination = Path(destination)
        is_update = destination.name.endswith("-updated")
        if is_update:
            stop("before-rename")
        rename(source, destination)
        if is_update:
            stop("after-rename")

    Accelerator.backward = interrupted_backward
    torch.optim.Adam.step = interrupted_step
    os.rename = interrupted_rename
    train(store)


def execute(root, boundary=""):
    return subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), str(root), boundary],
        env={
            **os.environ,
            "PYTHONPATH": str(Path.cwd() / "src") + os.pathsep + str(Path.cwd()),
            "OMP_NUM_THREADS": "1",
        },
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )


@pytest.mark.parametrize(
    "boundary", ["backward", "step", "before-rename", "after-rename"]
)
def test_process_death_resumes_to_same_update(tmp_path, boundary):
    reference = tmp_path / "reference"
    interrupted = tmp_path / "interrupted"
    setup(reference)
    copytree(reference, interrupted)
    baseline = execute(reference)
    assert baseline.returncode == 0, baseline.stderr
    stopped = execute(interrupted, boundary)
    assert stopped.returncode == 73, stopped.stderr
    resumed = execute(interrupted)
    assert resumed.returncode == 0, resumed.stderr
    a, b = setup(reference).recover(), setup(interrupted).recover()
    assert a.manifest.commit_id == b.manifest.commit_id
    assert a.manifest.inputs == b.manifest.inputs
    assert (
        a.manifest.progress.consumed_unit_ids == b.manifest.progress.consumed_unit_ids
    )
    equal(load_runtime(a.path), load_runtime(b.path))
    from safetensors.torch import load_file

    equal(
        load_file(a.path / "model/model.safetensors"),
        load_file(b.path / "model/model.safetensors"),
    )
    reports = [
        json.loads((item.path / "training.json").read_bytes()) for item in (a, b)
    ]
    for report in reports:
        report["metrics"].pop("learning_seconds")
    assert reports[0] == reports[1]
    assert len(list((interrupted / "checkpoints/committed").iterdir())) == 3


if __name__ == "__main__":
    crash_worker(Path(sys.argv[1]), sys.argv[2])
