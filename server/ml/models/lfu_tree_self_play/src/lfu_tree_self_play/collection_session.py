"""Cooperative collection sessions; no model, sampling, or scheduling policy."""

from collections.abc import Callable, Generator, Iterator
from dataclasses import dataclass
from typing import Protocol

from .batch_inference import EvaluationOutput, EvaluationRequest


class Evaluator(Protocol):
    def evaluate(
        self, requests: list[EvaluationRequest]
    ) -> tuple[EvaluationOutput, ...]: ...


@dataclass(frozen=True)
class SessionComplete[T]:
    value: T


class CollectionSession[T]:
    """A single-owner generator stopped at an inference boundary."""

    def __init__(
        self,
        generator: Generator[EvaluationRequest, EvaluationOutput, T],
        evaluator: Evaluator,
        *,
        on_release: Callable[[], None],
    ) -> None:
        self.evaluator = evaluator
        self._generator = generator
        self._on_release = on_release
        self._pending: EvaluationRequest | SessionComplete[T] | None = None
        self._aborted = False

    def _advance(self, output: EvaluationOutput | None = None) -> None:
        try:
            self._pending = (
                next(self._generator)
                if output is None
                else self._generator.send(output)
            )
        except StopIteration as completed:
            self._pending = SessionComplete(completed.value)
            self._on_release()
        except BaseException:
            self.abort()
            raise

    def poll(self) -> EvaluationRequest | SessionComplete[T]:
        if self._aborted:
            raise RuntimeError("session was aborted")
        if self._pending is None:
            self._advance()
        assert self._pending is not None
        return self._pending

    def resume(self, output: EvaluationOutput) -> None:
        if self._aborted or not isinstance(self._pending, EvaluationRequest):
            raise RuntimeError("session is not waiting for evaluation")
        if output.token != self._pending.token:
            raise ValueError("evaluation token does not match pending request")
        self._advance(output)

    def abort(self) -> None:
        if not self._aborted and not isinstance(self._pending, SessionComplete):
            self._aborted = True
            try:
                self._generator.close()
            finally:
                self._pending = None
                self._on_release()


@dataclass
class CollectionLane[T]:
    """Ordered sessions from one RNG owner, with at most one active session."""

    owner: object
    sessions: Iterator[CollectionSession[T]]


def drain_session[T](session: CollectionSession[T]) -> T:
    """Sequential reference execution through the same inference boundaries."""
    try:
        while True:
            pending = session.poll()
            if isinstance(pending, SessionComplete):
                return pending.value
            session.resume(session.evaluator.evaluate([pending])[0])
    finally:
        session.abort()
