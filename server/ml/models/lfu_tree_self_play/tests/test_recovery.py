"""Real-store G1 fixtures and abrupt subprocess recovery (no production trainer)."""

import json
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from tests.test_records import build_record, position
from tests.test_returns import uneven_record

from lfu_tree_self_play.checkpoints import CheckpointStore, EvaluationResult, Progress
from lfu_tree_self_play.config import experiment_id
from lfu_tree_self_play.dataset_codec import encode_unit
from lfu_tree_self_play.dataset_store import DatasetStore
from lfu_tree_self_play.game import ReversiPyGame
from lfu_tree_self_play.model import PolicyValueConfig, PolicyValueModel
from lfu_tree_self_play.returns import aggregate_returns
from lfu_tree_self_play.rng import create_rng_streams

BOUNDARIES = (
    "collection-partial",
    "collection-complete",
    "publication-before",
    "publication-after",
    "learning",
    "checkpoint-staging",
    "checkpoint-before",
    "checkpoint-after",
    "evaluation-computed",
    "evaluation-before",
    "evaluation-after",
)


def stores(root):
    config = OmegaConf.create({"experiment": {"name": "recovery"}})
    record, game = uneven_record()
    record = record.model_copy(update={"experiment_id": experiment_id(config)})
    dataset = DatasetStore(
        root / "dataset", experiment_id=experiment_id(config), game=game
    )
    return (
        record,
        dataset,
        CheckpointStore(root / "checkpoints", config=config, dataset_store=dataset),
    )


def restore(checkpoints):
    return checkpoints.restore(
        checkpoints.recover(),
        optimizer_factory=lambda m: torch.optim.Adam(m.parameters()),
        scheduler_factory=lambda o: torch.optim.lr_scheduler.StepLR(o, step_size=1),
    )


def run_fixture(root, boundary):
    torch.set_num_threads(1)
    random.seed(11)
    np.random.seed(12)
    torch.manual_seed(13)
    record, dataset, checkpoints = stores(root)

    def stop(name):
        if boundary == name:
            os._exit(73)

    if checkpoints.recover() is None:
        model = PolicyValueModel(PolicyValueConfig(width=4, num_blocks=1))
        model.train()
        model.stem[1].eval()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        checkpoints.commit(
            "genesis",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            rng=create_rng_streams(42),
            progress=Progress(0, "collection", 0.0),
            inputs=checkpoints.pin_inputs(0),
            expected_parent=None,
        )
    runtime = restore(checkpoints)
    model, optimizer, scheduler, rng = (
        runtime.model,
        runtime.optimizer,
        runtime.scheduler,
        runtime.rng,
    )
    progress = runtime.manifest.progress
    original_rename = os.rename

    def rename(source, destination):
        destination = Path(destination)
        if destination.parent.name == "published":
            stop("publication-before")
            original_rename(source, destination)
            stop("publication-after")
        elif destination.parent.name == "committed":
            kind = "checkpoint" if progress.model_generation == 0 else "evaluation"
            stop(f"{kind}-before")
            original_rename(source, destination)
            stop(f"{kind}-after")
        else:
            original_rename(source, destination)

    os.rename = rename
    if progress.model_generation == 0:
        for index in range(2):
            unit = record.model_copy(update={"unit_id": f"unit-{index}"})
            if any(i.unit_id == unit.unit_id for i in dataset.enumerate_units()):
                continue
            scratch = root / "collection.partial"
            scratch.write_text('{"incomplete":')
            stop("collection-partial")
            targets = aggregate_returns(unit, dataset.game)
            scratch.write_bytes(encode_unit(unit, targets, dataset.game))
            stop("collection-complete")
            dataset.publish(unit, targets)
            scratch.unlink()
        inputs = checkpoints.pin_inputs(0)
        optimizer.zero_grad()
        batch = torch.randn(2, 4, 8, 8)
        scalar = random.random() + float(np.random.rand())
        scalar += sum(
            getattr(rng, n).random()
            for n in ("branch_position", "action_sampling", "sibling_rollout")
        )
        model(batch).value.sum().mul(scalar).backward()
        optimizer.step()
        scheduler.step()
        stop("learning")
        save = torch.save

        def save_runtime(value, path, *args, **kwargs):
            save(value, path, *args, **kwargs)
            if Path(path).name == "runtime.pt":
                stop("checkpoint-staging")

        torch.save = save_runtime
        checkpoints.commit(
            "learned",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            rng=rng,
            progress=Progress(
                1, "learning", 2.0, tuple(i.unit_id for i in inputs.selected)
            ),
            inputs=inputs,
            expected_parent="genesis",
        )
        torch.save = save
        runtime = restore(checkpoints)
        model, optimizer, scheduler, rng = (
            runtime.model,
            runtime.optimizer,
            runtime.scheduler,
            runtime.rng,
        )
        progress = runtime.manifest.progress
    for index in range(2):
        latest = checkpoints.recover().manifest
        if len(latest.progress.evaluations) > index:
            continue
        score = (
            rng.evaluation.random()
            + random.random()
            + float(np.random.rand())
            + torch.rand(()).item()
        )
        result = EvaluationResult(
            "learned",
            "fixed-opponent",
            f"pair-{index}",
            100 + index,
            json.dumps({"score": score, "games": [1, -1], "pair": index}),
        )
        stop("evaluation-computed")
        checkpoints.commit(
            f"evaluated-{index}",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            rng=rng,
            progress=Progress(
                1,
                "evaluation",
                3.0 + index,
                latest.progress.consumed_unit_ids,
                (*latest.progress.evaluations, result),
            ),
            inputs=latest.inputs,
            expected_parent=latest.commit_id,
        )
    os.rename = original_rename
    latest = checkpoints.recover()
    runtime = restore(checkpoints)
    snapshot = {
        "progress": latest.manifest.progress,
        "inputs": latest.manifest.inputs,
        "units": tuple(
            json.loads((dataset.root / "published" / f"{i.unit_id}.json").read_bytes())
            for i in dataset.enumerate_units()
        ),
        "model": runtime.model.state_dict(),
        "modes": {n: m.training for n, m in runtime.model.named_modules()},
        "optimizer": runtime.optimizer.state_dict(),
        "scheduler": runtime.scheduler.state_dict(),
        "runtime": torch.load(latest.path / "runtime.pt", weights_only=False),
    }
    with torch.no_grad():
        snapshot["behavior"] = runtime.model(torch.ones(2, 4, 8, 8)).value
    torch.save(snapshot, root / "snapshot.pt")


def execute(root, boundary=""):
    env = {
        **os.environ,
        "PYTHONPATH": str(Path.cwd() / "src") + os.pathsep + str(Path.cwd()),
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    return subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), str(root), boundary],
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )


def equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, np.ndarray):
        assert np.array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            equal(a, b)
    else:
        assert left == right


@pytest.fixture(scope="module")
def baseline(tmp_path_factory):
    root = tmp_path_factory.mktemp("uninterrupted")
    result = execute(root)
    assert result.returncode == 0, result.stderr
    return torch.load(root / "snapshot.pt", weights_only=False)


@pytest.mark.parametrize("boundary", BOUNDARIES)
def test_abrupt_boundary_resume_matches_uninterrupted(tmp_path, baseline, boundary):
    result = execute(tmp_path, boundary)
    assert result.returncode == 73, result.stderr
    _, dataset, checkpoints = stores(tmp_path)
    stopped = checkpoints.recover().manifest
    visible = dataset.enumerate_units()
    if boundary.startswith("collection"):
        assert (tmp_path / "collection.partial").exists() and not visible
    if boundary == "publication-before":
        assert not visible and tuple((dataset.root / "staging").iterdir())
    if boundary == "publication-after":
        assert len(visible) == 1
    if boundary in ("learning", "checkpoint-staging", "checkpoint-before"):
        assert stopped.progress.model_generation == 0
        assert stopped.progress.consumed_unit_ids == ()
    if boundary in ("checkpoint-staging", "checkpoint-before", "evaluation-before"):
        assert tuple((checkpoints.root / "staging").iterdir())
    if boundary == "checkpoint-after":
        assert stopped.commit_id == "learned"
    if boundary in ("evaluation-computed", "evaluation-before"):
        assert stopped.progress.evaluations == ()
    if boundary == "evaluation-after":
        assert len(stopped.progress.evaluations) == 1
    dataset.quarantine_incomplete()
    for _ in range(2):
        result = execute(tmp_path)
        assert result.returncode == 0, result.stderr
        equal(baseline, torch.load(tmp_path / "snapshot.pt", weights_only=False))
    manifests = [
        json.loads(p.read_bytes())["payload"]
        for p in sorted((checkpoints.root / "committed").glob("*/manifest.json"))
    ]
    assert [m["commit_id"] for m in manifests] == [
        "genesis",
        "learned",
        "evaluated-0",
        "evaluated-1",
    ]
    assert checkpoints.recover().manifest.progress.consumed_unit_ids == (
        "unit-0",
        "unit-1",
    )
    assert {i.model_generation for i in dataset.enumerate_units()} == {0}
    assert checkpoints.recover().manifest.inputs.model_generation == 0
    assert checkpoints.recover().manifest.progress.model_generation == 1
    results = checkpoints.recover().manifest.progress.evaluations
    assert (
        len({(r.checkpoint_id, r.opponent, r.start_pair, r.seed) for r in results}) == 2
    )


@pytest.mark.parametrize("kind", ["h0", "h1", "forced-pass", "uneven"])
def test_g1_semantic_roundtrip_and_branch_conservation(tmp_path, kind):
    if kind == "uneven":
        record, game = uneven_record()
    else:
        game = ReversiPyGame()
        if kind == "forced-pass":
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
            record = build_record(
                root=root, mode="independent", positions=(), first_action=13
            )
        else:
            empty = kind == "h1"
            root = position((".WBBBBBB",) + ("BBBBBBBB",) * 7, 1, (0,))
            record = build_record(root=root, positions=(0,) if empty else ())
    targets = aggregate_returns(record, game)
    store = DatasetStore(tmp_path, experiment_id=record.experiment_id, game=game)
    store.publish(record, targets)
    loaded = store.load(record.unit_id)
    assert loaded.record == record and loaded.targets == targets
    if kind in ("h0", "h1"):
        assert len(record.sibling_groups) == (kind == "h1")
        assert len(record.edges) == (2 if kind == "h1" else 1)
        assert targets.node_black_returns[record.root_node_id] == 1.0
    for group in record.sibling_groups:
        assert sum(targets.edge_weights[e] for e in group.edge_ids) == pytest.approx(
            targets.node_weights[group.parent_node_id]
        )
    if kind == "uneven":
        assert targets.node_black_returns["root"] == 0.25
        assert targets.edge_weights["c"] == targets.edge_weights["e"] == 0.25
        assert loaded.record.edges[2].action == loaded.record.edges[4].action
        assert (
            loaded.record.edges[2].child_node_id != loaded.record.edges[4].child_node_id
        )
    if kind == "forced-pass":
        assert loaded.record.nodes[1].state.to_play == -1
        assert targets.edge_parent_targets[record.edges[1].edge_id] == 1.0


def test_repeated_interruptions_preserve_completed_evaluation_pair(tmp_path, baseline):
    for boundary in (
        "collection-partial",
        "publication-after",
        "learning",
        "checkpoint-after",
        "evaluation-after",
        "evaluation-computed",
    ):
        result = execute(tmp_path, boundary)
        assert result.returncode == 73, result.stderr
    _, _, checkpoints = stores(tmp_path)
    partial = checkpoints.recover().manifest.progress
    assert len(partial.evaluations) == 1
    assert partial.evaluations[0].start_pair == "pair-0"
    assert partial.consumed_unit_ids == ("unit-0", "unit-1")
    result = execute(tmp_path)
    assert result.returncode == 0, result.stderr
    equal(baseline, torch.load(tmp_path / "snapshot.pt", weights_only=False))
    assert len(tuple((checkpoints.root / "committed").iterdir())) == 4


if __name__ == "__main__":
    run_fixture(Path(sys.argv[1]), sys.argv[2])
