"""Bounded, round-robin inference across independent collection RNG owners."""

import time
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Protocol

import psutil
import torch

from .batch_inference import EvaluationRequest, FrozenBatchEvaluator
from .collection_session import CollectionLane, CollectionSession, SessionComplete


class MemoryMeasurementError(RuntimeError):
    """A configured memory limit cannot be checked reliably."""


def _positive_int(value: object, name: str) -> None:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class SchedulerLimits:
    max_active_units: int
    max_batch_size: int
    rss_limit_bytes: int
    device_limit_bytes: int | None = None

    def __post_init__(self):
        for name in ("max_active_units", "max_batch_size", "rss_limit_bytes"):
            _positive_int(getattr(self, name), name)
        if self.device_limit_bytes is not None:
            _positive_int(self.device_limit_bytes, "device_limit_bytes")


@dataclass(frozen=True)
class MemorySample:
    rss_bytes: int
    device_allocated_bytes: int | None


class MemoryProbe(Protocol):
    def sample(self) -> MemorySample: ...


class SystemMemoryProbe:
    """Current process RSS and live tensor allocation, not an OS hard cap."""

    def __init__(self, device: torch.device) -> None:
        self.device = torch.device(device)

    def sample(self) -> MemorySample:
        try:
            rss = psutil.Process().memory_info().rss
            allocated = (
                None
                if self.device.type == "cpu"
                else int(torch.accelerator.memory_allocated(self.device))
            )
            if rss <= 0 or (allocated is not None and allocated < 0):
                raise MemoryMeasurementError("invalid memory measurement")
            return MemorySample(rss, allocated)
        except (
            psutil.Error,
            OSError,
            ValueError,
            AttributeError,
            RuntimeError,
        ) as error:
            raise MemoryMeasurementError(str(error)) from error


@dataclass
class SchedulerMetrics:
    """Physical batch metrics; persisted collection record schema stays at v1."""

    batch_sizes: dict[int, int] = field(default_factory=dict)
    request_wait_seconds: list[float] = field(default_factory=list)
    inference_batches: int = 0
    evaluated_positions: int = 0
    inference_seconds: float = 0.0
    rss_sampled_peak_bytes: int = 0
    device_allocated_sampled_peak_bytes: int | None = None
    device_peak_bytes: int | None = None
    completed_units: int = 0
    discarded_units: int = 0
    stop_reason: str = "running"


@dataclass
class _Active:
    lane: CollectionLane
    session: CollectionSession
    request: EvaluationRequest | None = None
    ready_at: float = 0.0


class BatchScheduler:
    """Dispatch one ready position per lane under the same limits for A/B/C."""

    def __init__(
        self,
        evaluator: FrozenBatchEvaluator,
        *,
        limits: SchedulerLimits,
        memory_probe: MemoryProbe | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        if evaluator.device.type != "cpu" and limits.device_limit_bytes is None:
            raise ValueError(
                "device_limit_bytes is required for accelerator collection"
            )
        self.evaluator = evaluator
        self.limits = limits
        self._probe = (
            memory_probe
            if memory_probe is not None
            else SystemMemoryProbe(evaluator.device)
        )
        self._clock = clock
        self.metrics = SchedulerMetrics()

    def _check_memory(self) -> None:
        sample = self._probe.sample()
        if type(sample.rss_bytes) is not int or sample.rss_bytes <= 0:
            raise MemoryMeasurementError("invalid RSS measurement")
        allocated = sample.device_allocated_bytes
        if allocated is not None and (type(allocated) is not int or allocated < 0):
            raise MemoryMeasurementError("invalid device measurement")
        self.metrics.rss_sampled_peak_bytes = max(
            self.metrics.rss_sampled_peak_bytes, sample.rss_bytes
        )
        if allocated is not None:
            self.metrics.device_allocated_sampled_peak_bytes = max(
                self.metrics.device_allocated_sampled_peak_bytes or 0, allocated
            )
        if sample.rss_bytes > self.limits.rss_limit_bytes:
            raise MemoryError("process RSS limit exceeded")
        if self.limits.device_limit_bytes is not None:
            if allocated is None:
                raise MemoryMeasurementError(
                    "configured device limit cannot be measured"
                )
            if allocated > self.limits.device_limit_bytes:
                raise MemoryError("device allocation limit exceeded")

    def run(
        self, lanes: Sequence[CollectionLane], *, on_complete: Callable[[object], None]
    ) -> SchedulerMetrics:
        self.metrics = SchedulerMetrics()
        owners = [id(lane.owner) for lane in lanes]
        if len(set(owners)) != len(owners):
            raise ValueError("duplicate RNG owner across lanes")
        if any(
            getattr(lane.owner, "evaluator", None) is not self.evaluator
            for lane in lanes
        ):
            raise ValueError(
                "all lanes must share the scheduler evaluator and generation"
            )
        pending = deque(lanes)
        active: deque[_Active] = deque()

        def advance(item: _Active) -> bool:
            result = item.session.poll()
            if isinstance(result, SessionComplete):
                self._check_memory()
                on_complete(result.value)
                self.metrics.completed_units += 1
                self._check_memory()
                pending.append(item.lane)
                return False
            item.request = result
            item.ready_at = self._clock()
            return True

        def admit(lane: CollectionLane) -> None:
            try:
                session = next(lane.sessions)
            except StopIteration:
                return
            item = _Active(lane, session)
            active.append(item)
            if session.evaluator is not self.evaluator:
                raise ValueError("session evaluator does not match scheduler")
            if not advance(item):
                active.remove(item)

        def dispatch() -> None:
            # Keep batch temporaries scoped here so completed payloads are
            # released before the next admission pass allocates another unit.
            self._check_memory()
            selected = list(active)[: self.limits.max_batch_size]
            dispatched = self._clock()
            requests = [item.request for item in selected]
            outputs = self.evaluator.evaluate(requests)
            self.metrics.inference_batches += 1
            self.metrics.evaluated_positions += len(selected)
            self.metrics.batch_sizes[len(selected)] = (
                self.metrics.batch_sizes.get(len(selected), 0) + 1
            )
            self.metrics.inference_seconds += sum(
                output.inference_seconds for output in outputs
            )
            self.metrics.request_wait_seconds.extend(
                max(0.0, dispatched - item.ready_at) for item in selected
            )
            self._check_memory()
            for item, output in zip(selected, outputs, strict=True):
                item.session.resume(output)
                # Remove only after success, so failed resumes remain abortable.
                if advance(item):
                    active.remove(item)
                    active.append(item)
                else:
                    active.remove(item)

        try:
            while pending or active:
                # A terminal unit consumes its admission turn too; replenishing
                # that lane must not starve an already-ready inference request.
                admissions = min(
                    len(pending), self.limits.max_active_units - len(active)
                )
                for _ in range(admissions):
                    self._check_memory()
                    admit(pending.popleft())
                if active:
                    dispatch()
            if self.evaluator.device.type != "cpu":
                torch.accelerator.synchronize(self.evaluator.device)
                self.metrics.device_peak_bytes = int(
                    torch.accelerator.max_memory_allocated(self.evaluator.device)
                )
            self.metrics.stop_reason = "complete"
            return self.metrics
        except BaseException as error:
            self.metrics.stop_reason = type(error).__name__
            for item in active:
                if not item.session.completed:
                    self.metrics.discarded_units += 1
                try:
                    item.session.abort()
                except BaseException as cleanup_error:  # noqa: BLE001 - preserve cancellation and clean every lane
                    error.add_note(f"session cleanup failed: {cleanup_error}")
            raise
