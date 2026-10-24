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


def test_ready_inference_is_not_starved_by_terminal_only_lane():
    evaluator, _, _, _ = setup_lanes(count=0)
    events = []

    def waiting():
        yield EvaluationRequest(0, endgame_position(1))
        events.append("ready")
        return "ready"

    def terminal(index):
        yield from ()
        events.append(f"terminal-{index}")
        return index

    def terminal_sessions():
        for index in range(100):
            yield CollectionSession(terminal(index), evaluator, on_release=lambda: None)

    lanes = [
        CollectionLane(
            SimpleNamespace(evaluator=evaluator),
            iter([CollectionSession(waiting(), evaluator, on_release=lambda: None)]),
        ),
        CollectionLane(SimpleNamespace(evaluator=evaluator), terminal_sessions()),
    ]
    metrics = BatchScheduler(
        evaluator, limits=SchedulerLimits(2, 2, 100), memory_probe=Probe()
    ).run(lanes, on_complete=lambda unit: None)
    assert events.index("ready") < events.index("terminal-1")
    assert metrics.completed_units == 101


def test_completed_payload_is_released_before_next_single_active_unit():
    import gc
    import weakref

    evaluator, _, _, _ = setup_lanes(count=0)
    references = []

    class Payload:
        pass

    def generate(index):
        payload = Payload()
        references.append(weakref.ref(payload))
        yield EvaluationRequest(index, endgame_position(1))
        return payload

    def sessions():
        for index in range(3):
            yield CollectionSession(generate(index), evaluator, on_release=lambda: None)

    class PayloadProbe:
        def sample(self):
            gc.collect()
            return MemorySample(
                100 + 1000 * sum(ref() is not None for ref in references), None
            )

    lane = CollectionLane(SimpleNamespace(evaluator=evaluator), sessions())
    metrics = BatchScheduler(
        evaluator, limits=SchedulerLimits(1, 1, 1100), memory_probe=PayloadProbe()
    ).run([lane], on_complete=lambda unit: None)
    assert metrics.completed_units == 3
    assert all(reference() is None for reference in references)


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


def test_current_rss_uses_portable_process_memory_info(monkeypatch):
    import psutil

    monkeypatch.setattr(
        psutil,
        "Process",
        lambda: SimpleNamespace(memory_info=lambda: SimpleNamespace(rss=69632)),
    )
    assert SystemMemoryProbe(torch.device("cpu")).sample().rss_bytes == 69632


@pytest.mark.parametrize("device", ["cuda:1", "mps"])
def test_device_memory_uses_unified_accelerator_api(monkeypatch, device):
    monkeypatch.setattr(
        torch.accelerator,
        "memory_allocated",
        lambda actual: 123456 if actual == torch.device(device) else -1,
    )
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device: 0)
    monkeypatch.setattr(torch.mps, "current_allocated_memory", lambda: 0)
    assert (
        SystemMemoryProbe(torch.device(device)).sample().device_allocated_bytes
        == 123456
    )


def test_process_memory_failure_is_reported(monkeypatch):
    import psutil

    def unavailable():
        raise psutil.AccessDenied(pid=1)

    monkeypatch.setattr(psutil, "Process", unavailable)
    with pytest.raises(MemoryMeasurementError):
        SystemMemoryProbe(torch.device("cpu")).sample()


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
