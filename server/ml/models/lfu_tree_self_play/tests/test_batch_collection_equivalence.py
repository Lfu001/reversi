"""Batch scheduling changes cost, never the sampled games or training weights."""

from collections.abc import Iterator

import pytest
import torch
from tests.test_records import position

from lfu_tree_self_play.batch_inference import FrozenBatchEvaluator
from lfu_tree_self_play.batch_scheduler import BatchScheduler, SchedulerLimits
from lfu_tree_self_play.collection_session import CollectionLane
from lfu_tree_self_play.collector import IndependentCollector
from lfu_tree_self_play.dataset_store import DatasetStore
from lfu_tree_self_play.game import ReversiPyGame
from lfu_tree_self_play.model import PolicyValueConfig, PolicyValueModel
from lfu_tree_self_play.tree_collector import TreeCollector
from lfu_tree_self_play.verification import endgame_position


def make_inputs(evaluator, kind, root, levels, strategy):
    lanes, reference = [], []
    for lane_index, seed in enumerate([19, 88, 101]):
        kwargs = {
            "experiment_id": "batch-test",
            "round_id": "round",
            "model_generation": 2,
            "run_seed": seed,
        }
        ids = [f"lane{lane_index}-a", f"lane{lane_index}-b"]
        if kind == "independent":
            collector = IndependentCollector(evaluator, ReversiPyGame(), **kwargs)
            reference.extend(
                collector.collect_paths(root, paths_per_root=2, unit_ids=ids).units
            )
            collector = IndependentCollector(evaluator, ReversiPyGame(), **kwargs)
            sessions = collector.iter_sessions(root, paths_per_root=2, unit_ids=ids)
        else:
            collector = TreeCollector(
                evaluator, levels=levels, strategy=strategy, **kwargs
            )
            reference.extend(collector.collect_tree(root, unit_id=uid) for uid in ids)
            collector = TreeCollector(
                evaluator, levels=levels, strategy=strategy, **kwargs
            )

            def iter_trees(collector=collector, ids=ids) -> Iterator:
                for uid in ids:
                    yield collector.create_session(root, unit_id=uid)

            sessions = iter_trees()
        lanes.append(CollectionLane(collector, sessions))
    return lanes, reference


def assert_units_equal(actual, expected):
    left, right = actual.record, expected.record
    for name in type(left).model_fields:
        if name not in ("nodes", "edges", "cost"):
            assert getattr(left, name) == getattr(right, name)
    assert len(left.nodes) == len(right.nodes)
    assert len(left.edges) == len(right.edges)
    for a, b in zip(left.nodes, right.nodes, strict=True):
        assert a.model_copy(update={"old_value": None}) == b.model_copy(
            update={"old_value": None}
        )
        if a.old_value is None:
            assert b.old_value is None
        else:
            assert abs(a.old_value - b.old_value) <= 1e-6
    for a, b in zip(left.edges, right.edges, strict=True):
        assert a.model_copy(update={"old_log_probability": 0.0}) == b.model_copy(
            update={"old_log_probability": 0.0}
        )
        assert abs(a.old_log_probability - b.old_log_probability) <= 1e-6
    assert actual.targets == expected.targets
    if left.mode == "independent":
        assert actual.path_weight == expected.path_weight
        assert actual.weighted_targets == expected.weighted_targets
    else:
        assert actual.branch_execution == expected.branch_execution


@pytest.mark.parametrize("batch,active", [(1, 1), (2, 3), (8, 2), (32, 3)])
@pytest.mark.parametrize(
    "kind,levels,strategy",
    [
        ("independent", 0, "uniform"),
        ("tree", 0, "uniform"),
        ("tree", 1, "uniform"),
        ("tree", 2, "uniform"),
        ("tree", 2, "stratified"),
        ("tree", 2, "fixed"),
    ],
)
def test_fixed_seed_records_match_serial_reference(
    batch, active, kind, levels, strategy, tmp_path
):
    torch.manual_seed(11)
    evaluator = FrozenBatchEvaluator(
        PolicyValueModel(PolicyValueConfig(width=4, num_blocks=1)), model_generation=2
    )
    root = (
        ReversiPyGame().initial_state() if strategy == "fixed" else endgame_position(1)
    )
    lanes, expected = make_inputs(evaluator, kind, root, levels, strategy)
    actual = []
    store = DatasetStore(tmp_path, experiment_id="batch-test", game=ReversiPyGame())

    def publish(unit):
        store.publish(unit.record, unit.targets)
        actual.append(unit)

    metrics = BatchScheduler(
        evaluator, limits=SchedulerLimits(active, batch, 4 * 1024**3)
    ).run(lanes, on_complete=publish)
    assert metrics.completed_units == 6
    assert metrics.stop_reason == "complete"
    assert sum(size * count for size, count in metrics.batch_sizes.items()) == sum(
        unit.record.cost.evaluated_positions for unit in actual
    )
    if active > 1 and batch > 1:
        assert any(size > 1 for size in metrics.batch_sizes)
    by_id = {unit.record.unit_id: unit for unit in actual}
    for unit in expected:
        received = by_id[unit.record.unit_id]
        assert_units_equal(received, unit)
        assert store.load(unit.record.unit_id).record == received.record


@pytest.mark.parametrize("kind", ["independent", "tree"])
@pytest.mark.parametrize(
    "root",
    [
        position(("BBBBBBBB",) * 8, 1, ()),
        position((".WBBBBBB",) + ("BBBBBBBB",) * 7, 1, (0,)),
        position((".WBBBBBB",) + ("BBBBBBBB",) * 6 + ("BBBBBBB.",), 1, (0,)),
        position((".WBBBBBB",) + ("BBBBBBBB",) * 6 + ("BBBBBBW.",), 1, (0,)),
    ],
)
def test_terminal_pass_early_terminal_and_duplicate_siblings(kind, root):
    torch.manual_seed(11)
    evaluator = FrozenBatchEvaluator(
        PolicyValueModel(PolicyValueConfig(width=4, num_blocks=1)), model_generation=2
    )
    lanes, expected = make_inputs(evaluator, kind, root, 2, "uniform")
    actual = []
    metrics = BatchScheduler(evaluator, limits=SchedulerLimits(3, 8, 4 * 1024**3)).run(
        lanes, on_complete=actual.append
    )
    assert metrics.completed_units == len(expected)
    by_id = {unit.record.unit_id: unit for unit in actual}
    for unit in expected:
        assert_units_equal(by_id[unit.record.unit_id], unit)
    if root.is_terminal:
        assert metrics.inference_batches == 0


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires MPS")
@pytest.mark.parametrize("kind", ["independent", "tree"])
def test_mps_batch_collection_matches_same_device_reference(kind):
    torch.manual_seed(88)
    evaluator = FrozenBatchEvaluator(
        PolicyValueModel(PolicyValueConfig(width=8, num_blocks=1)).to("mps"),
        model_generation=2,
    )
    lanes, expected = make_inputs(evaluator, kind, endgame_position(1), 2, "uniform")
    actual = []
    metrics = BatchScheduler(
        evaluator,
        limits=SchedulerLimits(3, 8, 4 * 1024**3, 1024**3),
    ).run(lanes, on_complete=actual.append)
    by_id = {unit.record.unit_id: unit for unit in actual}
    for unit in expected:
        assert_units_equal(by_id[unit.record.unit_id], unit)
    assert metrics.completed_units == 6
    assert metrics.device_allocated_sampled_peak_bytes > 0
    assert metrics.device_peak_bytes >= metrics.device_allocated_sampled_peak_bytes
    assert any(size > 1 for size in metrics.batch_sizes)


@pytest.mark.parametrize("kind", ["independent", "tree"])
def test_collectors_share_snapshot_and_reject_generation_mismatch(kind):
    torch.manual_seed(1)
    evaluator = FrozenBatchEvaluator(
        PolicyValueModel(PolicyValueConfig(width=4, num_blocks=1)), model_generation=2
    )
    kwargs = {
        "experiment_id": "exp",
        "round_id": "round",
        "model_generation": 2,
        "run_seed": 88,
    }
    if kind == "independent":
        collector = IndependentCollector(evaluator, ReversiPyGame(), **kwargs)
        with pytest.raises(ValueError, match="generation"):
            IndependentCollector(
                evaluator, ReversiPyGame(), **(kwargs | {"model_generation": 3})
            )
    else:
        collector = TreeCollector(evaluator, **kwargs)
        with pytest.raises(ValueError, match="generation"):
            TreeCollector(evaluator, **(kwargs | {"model_generation": 3}))
    assert collector.evaluator is evaluator
