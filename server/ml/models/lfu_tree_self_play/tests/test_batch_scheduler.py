"""Fair dispatch and fail-closed resource limits with real collection sessions."""

from types import SimpleNamespace

import pytest
import torch

from lfu_tree_self_play.batch_inference import EvaluationRequest, FrozenBatchEvaluator
from lfu_tree_self_play.batch_scheduler import (
    BatchScheduler,
    MemoryMeasurementError,
    MemorySample,
    SchedulerLimits,
    SystemMemoryProbe,
)
from lfu_tree_self_play.collection_session import CollectionLane, CollectionSession
from lfu_tree_self_play.model import PolicyValueConfig, PolicyValueModel
from lfu_tree_self_play.verification import endgame_position


class Probe:
    def __init__(self, rss=100, device=None, fail_at=None):
        self.rss, self.device, self.fail_at = rss, device, fail_at
        self.calls = 0

    def sample(self):
        self.calls += 1
        if self.calls == self.fail_at:
            raise MemoryMeasurementError("unavailable")
        return MemorySample(self.rss, self.device)


def setup_lanes(count=4, steps=2):
    torch.manual_seed(1)
    evaluator = FrozenBatchEvaluator(
        PolicyValueModel(PolicyValueConfig(width=4, num_blocks=1)), model_generation=0
    )
    order, aborted = [], []
    lanes = []
    for index in range(count):

        def generate(index=index):
            try:
                for step in range(steps):
                    result = yield EvaluationRequest(
                        index * 10 + step, endgame_position(1)
                    )
                    order.append(result.token)
                return index
            except GeneratorExit:
                aborted.append(index)
                raise

        session = CollectionSession(generate(), evaluator, on_release=lambda: None)
        lanes.append(
            CollectionLane(SimpleNamespace(evaluator=evaluator), iter([session]))
        )
    return evaluator, lanes, order, aborted


def test_round_robin_fairness_and_partial_batch():
    evaluator, lanes, order, _ = setup_lanes(count=3)
    completed = []
    clock = iter(range(1000))
    metrics = BatchScheduler(
        evaluator,
        limits=SchedulerLimits(3, 2, 100),
        memory_probe=Probe(),
        clock=lambda: next(clock),
    ).run(lanes, on_complete=completed.append)
    assert order == [0, 10, 20, 1, 11, 21]
    assert sorted(completed) == [0, 1, 2]
    assert metrics.batch_sizes == {2: 3}
    assert metrics.inference_batches == 3
    assert metrics.evaluated_positions == 6
    assert len(metrics.request_wait_seconds) == 6
    assert all(seconds >= 0 for seconds in metrics.request_wait_seconds)
    assert metrics.rss_sampled_peak_bytes == 100


def test_physical_batches_not_multiplied_by_unit_count():
    evaluator, lanes, _, _ = setup_lanes(count=3, steps=1)
    metrics = BatchScheduler(
        evaluator, limits=SchedulerLimits(3, 8, 100), memory_probe=Probe()
    ).run(lanes, on_complete=lambda unit: None)
    assert metrics.batch_sizes == {3: 1}
    assert metrics.inference_batches == 1
    assert metrics.evaluated_positions == 3


@pytest.mark.parametrize("probe", [Probe(rss=101), Probe(fail_at=1)])
def test_memory_failure_blocks_admission_and_dispatch(probe):
    evaluator, lanes, order, _ = setup_lanes()
    scheduler = BatchScheduler(
        evaluator, limits=SchedulerLimits(2, 2, 100), memory_probe=probe
    )
    with pytest.raises((MemoryError, MemoryMeasurementError)):
        scheduler.run(lanes, on_complete=lambda unit: pytest.fail("no complete unit"))
    assert order == []
    assert scheduler.metrics.inference_batches == 0


def test_failure_aborts_unfinished_sessions():
    evaluator, lanes, _, aborted = setup_lanes()
    # First dispatch's post-forward memory check fails before any result resumes.
    probe = Probe(fail_at=5)
    scheduler = BatchScheduler(
        evaluator, limits=SchedulerLimits(2, 2, 100), memory_probe=probe
    )
    with pytest.raises(MemoryMeasurementError):
        scheduler.run(lanes, on_complete=lambda unit: pytest.fail("unfinished"))
    assert sorted(aborted) == [0, 1]
    assert scheduler.metrics.discarded_units == 2


def test_transition_error_keeps_original_exception_and_aborts_other_sessions():
    evaluator, lanes, _, aborted = setup_lanes(count=2)

    def broken():
        yield EvaluationRequest(99, endgame_position(1))
        raise ValueError("transition failed")

    lanes[0].sessions = iter(
        [CollectionSession(broken(), evaluator, on_release=lambda: None)]
    )
    scheduler = BatchScheduler(
        evaluator, limits=SchedulerLimits(2, 2, 100), memory_probe=Probe()
    )
    with pytest.raises(ValueError, match="transition failed"):
        scheduler.run(lanes, on_complete=lambda unit: pytest.fail("unfinished"))
    assert aborted == [1]


def test_callback_failure_does_not_publish_unfinished_units():
    evaluator, lanes, _, aborted = setup_lanes(count=2)

    def publish(unit):
        raise OSError("storage unavailable")

    scheduler = BatchScheduler(
        evaluator, limits=SchedulerLimits(2, 1, 100), memory_probe=Probe()
    )
    with pytest.raises(OSError, match="storage unavailable"):
        scheduler.run(lanes, on_complete=publish)
    assert aborted == [1]
    assert scheduler.metrics.completed_units == 0


def test_device_measurement_required_when_limit_configured():
    evaluator, lanes, _, _ = setup_lanes(count=1)
    scheduler = BatchScheduler(
        evaluator, limits=SchedulerLimits(1, 1, 100, 50), memory_probe=Probe()
    )
    with pytest.raises(MemoryMeasurementError, match="device"):
        scheduler.run(lanes, on_complete=lambda unit: None)


def test_linux_current_rss_converts_pages_to_bytes(monkeypatch):
    import lfu_tree_self_play.batch_scheduler as module

    monkeypatch.setattr(module.sys, "platform", "linux")
    monkeypatch.setattr(module.Path, "read_text", lambda path: "200 17 3 4")
    monkeypatch.setattr(module.os, "sysconf", lambda name: 4096)
    assert SystemMemoryProbe(torch.device("cpu")).sample().rss_bytes == 69632


def test_duplicate_owner_or_snapshot_rejected_before_rng_consumption():
    evaluator, lanes, order, _ = setup_lanes(count=2)
    lanes[1].owner = lanes[0].owner
    scheduler = BatchScheduler(
        evaluator, limits=SchedulerLimits(2, 2, 100), memory_probe=Probe()
    )
    with pytest.raises(ValueError, match="owner"):
        scheduler.run(lanes, on_complete=lambda unit: None)
    assert order == []
    lanes[1].owner = SimpleNamespace(evaluator=object())
    with pytest.raises(ValueError, match="evaluator"):
        scheduler.run(lanes, on_complete=lambda unit: None)


@pytest.mark.parametrize("limits", [(0, 2, 100), (2, True, 100), (2, 2, -1)])
def test_invalid_limits(limits):
    with pytest.raises(ValueError):
        SchedulerLimits(*limits)


def test_system_memory_probe_current_rss_and_device_allocation():
    sample = SystemMemoryProbe(torch.device("cpu")).sample()
    assert sample.rss_bytes > 0
    assert sample.device_allocated_bytes is None
