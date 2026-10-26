"""Replay a fixed collection tree with shared or repeated path work."""

import math
import time
from collections.abc import Generator
from dataclasses import dataclass, fields
from typing import Literal

import psutil
import torch

from .batch_inference import EvaluationOutput, EvaluationRequest, FrozenBatchEvaluator
from .collection_session import CollectionSession
from .game import Game, GameState, ReversiPyGame
from .model import masked_policy_logits
from .record_validation import validate_record
from .records import CollectionRecord, CostRecord, EdgeRecord, NodeRecord
from .returns import aggregate_returns
from .schedule import summarize_branch_execution
from .tree_collector import CollectedTree

ReplayMode = Literal["shared", "recompute"]
_TOLERANCE = 1e-6


def _same_fields(expected, actual, *, excluding: frozenset[str] = frozenset()) -> None:
    for name in type(expected).model_fields:
        if name not in excluding and getattr(expected, name) != getattr(actual, name):
            raise ValueError(f"Replay mismatch in {type(expected).__name__}.{name}")


def _numeric_error(expected: float | None, actual: float | None, label: str) -> float:
    if expected is None or actual is None:
        if expected is not actual:
            raise ValueError(f"Replay mismatch in {label}")
        return 0.0
    if not math.isfinite(expected) or not math.isfinite(actual):
        raise ValueError(f"Nonfinite replay value in {label}")
    error = abs(expected - actual)
    if error > _TOLERANCE:
        raise ValueError(f"Replay mismatch in {label}: error {error}")
    return error


@dataclass(frozen=True)
class ReplayResult:
    record: CollectionRecord


class TreeReplay:
    """Own one replay session at a time using a fixed evaluator snapshot."""

    def __init__(
        self, evaluator: FrozenBatchEvaluator, *, game: Game | None = None
    ) -> None:
        self._evaluator = evaluator
        self._game = game if game is not None else ReversiPyGame()
        self._active = False

    @property
    def evaluator(self) -> FrozenBatchEvaluator:
        return self._evaluator

    def create_session(
        self, reference: CollectedTree, *, mode: ReplayMode
    ) -> CollectionSession[ReplayResult]:
        """Validate and prepare work before the caller starts its replay timer."""
        if self._active:
            raise RuntimeError("A replay session is already active")
        if mode not in ("shared", "recompute"):
            raise ValueError(f"Unknown replay mode: {mode}")
        record = reference.record
        if record.mode != "tree":
            raise ValueError("Replay requires a tree collection record")
        if record.model_generation != self.evaluator.model_generation:
            raise ValueError("Replay model generation does not match the reference")
        validate_record(record, self._game)
        compare_trees(reference, finalize_replay(ReplayResult(record), self._game))
        nodes = {node.node_id: node for node in record.nodes}
        edges = {edge.edge_id: edge for edge in record.edges}
        outgoing: dict[str, list[EdgeRecord]] = {
            node.node_id: [] for node in record.nodes
        }
        for edge in record.edges:
            outgoing[edge.parent_node_id].append(edge)
        paths: list[tuple[EdgeRecord, ...]] = []
        if mode == "recompute":
            for node in record.nodes:
                if node.state.is_terminal:
                    path = []
                    while node.incoming_edge_id is not None:
                        edge = edges[node.incoming_edge_id]
                        path.append(edge)
                        node = nodes[edge.parent_node_id]
                    paths.append(tuple(reversed(path)))
        self._active = True
        return CollectionSession(
            self._generate(record, mode, nodes, outgoing, paths),
            self.evaluator,
            on_release=self._release,
        )

    def _release(self) -> None:
        self._active = False

    def _generate(
        self,
        reference: CollectionRecord,
        mode: ReplayMode,
        nodes: dict[str, NodeRecord],
        outgoing: dict[str, list[EdgeRecord]],
        paths: list[tuple[EdgeRecord, ...]],
    ) -> Generator[EvaluationRequest, EvaluationOutput, ReplayResult]:
        started = time.perf_counter()
        evaluated = 0
        transitions = 0
        inference_seconds = 0.0
        transition_seconds = 0.0
        actual_nodes: dict[str, NodeRecord] = {}
        actual_edges: dict[str, EdgeRecord] = {}

        def save_node(node: NodeRecord, state: GameState, value: float | None) -> None:
            actual = node.model_copy(
                update={
                    "state": state,
                    "old_value": value,
                    "black_result": state.black_result if state.is_terminal else None,
                }
            )
            _same_fields(node, actual, excluding=frozenset({"old_value"}))
            _numeric_error(node.old_value, value, f"node {node.node_id} old_value")
            if node.node_id in actual_nodes:
                previous = actual_nodes[node.node_id]
                _same_fields(previous, actual, excluding=frozenset({"old_value"}))
                _numeric_error(
                    previous.old_value, value, f"repeated node {node.node_id}"
                )
            else:
                actual_nodes[node.node_id] = actual

        def evaluate(node: NodeRecord, state: GameState):
            nonlocal evaluated, inference_seconds
            token = evaluated
            output = yield EvaluationRequest(token=token, state=state)
            evaluated += 1
            inference_seconds += output.inference_seconds
            save_node(node, state, output.old_value)
            legal = torch.zeros((1, 64), dtype=torch.bool)
            legal[0, list(state.legal_actions)] = True
            return masked_policy_logits(
                output.policy_logits.reshape(1, 64), legal
            ).log_softmax(-1)[0]

        def step(edge: EdgeRecord, state: GameState, log_probabilities) -> GameState:
            nonlocal transitions, transition_seconds
            log_probability = float(log_probabilities[edge.action].item())
            _numeric_error(
                edge.old_log_probability,
                log_probability,
                f"edge {edge.edge_id} log_probability",
            )
            if edge.edge_id in actual_edges:
                _numeric_error(
                    actual_edges[edge.edge_id].old_log_probability,
                    log_probability,
                    f"repeated edge {edge.edge_id}",
                )
            else:
                actual_edges[edge.edge_id] = edge.model_copy(
                    update={"old_log_probability": log_probability}
                )
            before = time.perf_counter()
            child = self._game.step(state, edge.action)
            transition_seconds += time.perf_counter() - before
            transitions += 1
            if child != nodes[edge.child_node_id].state:
                raise ValueError(f"Replay state mismatch at edge {edge.edge_id}")
            return child

        def shared(node: NodeRecord, state: GameState):
            if state.is_terminal:
                save_node(node, state, None)
                return
            probabilities = yield from evaluate(node, state)
            for edge in outgoing[node.node_id]:
                child = step(edge, state, probabilities)
                yield from shared(nodes[edge.child_node_id], child)

        root = nodes[reference.root_node_id]
        if mode == "shared":
            yield from shared(root, root.state)
        else:
            for path in paths:
                state = root.state
                for edge in path:
                    probabilities = yield from evaluate(
                        nodes[edge.parent_node_id], state
                    )
                    state = step(edge, state, probabilities)
                leaf = nodes[path[-1].child_node_id] if path else root
                save_node(leaf, state, None)
        elapsed = time.perf_counter() - started
        cost = CostRecord(
            environment_transitions=transitions,
            evaluated_positions=evaluated,
            inference_batches=evaluated,
            collection_seconds=elapsed,
            inference_seconds=inference_seconds,
            transition_seconds=transition_seconds,
            management_seconds=max(
                0.0, elapsed - inference_seconds - transition_seconds
            ),
            peak_memory_bytes=psutil.Process().memory_info().rss,
        )
        return ReplayResult(
            reference.model_copy(
                update={
                    "nodes": tuple(
                        actual_nodes[node.node_id] for node in reference.nodes
                    ),
                    "edges": tuple(
                        actual_edges[edge.edge_id] for edge in reference.edges
                    ),
                    "cost": cost,
                }
            )
        )


def finalize_replay(result: ReplayResult, game: Game) -> CollectedTree:
    """Compute validated targets and branch accounting outside replay timing."""
    validate_record(result.record, game)
    return CollectedTree(
        record=result.record,
        targets=aggregate_returns(result.record, game),
        branch_execution=summarize_branch_execution(result.record, game),
    )


def compare_trees(reference: CollectedTree, actual: CollectedTree) -> dict:
    """Require exact discrete data and weights, allowing 1e-6 model error."""
    _same_fields(
        reference.record, actual.record, excluding=frozenset({"nodes", "edges", "cost"})
    )
    if len(reference.record.nodes) != len(actual.record.nodes) or len(
        reference.record.edges
    ) != len(actual.record.edges):
        raise ValueError("Replay node or edge count mismatch")
    max_value_error = 0.0
    max_log_probability_error = 0.0
    for expected, observed in zip(
        reference.record.nodes, actual.record.nodes, strict=True
    ):
        _same_fields(expected, observed, excluding=frozenset({"old_value"}))
        max_value_error = max(
            max_value_error,
            _numeric_error(
                expected.old_value, observed.old_value, f"node {expected.node_id}"
            ),
        )
    for expected, observed in zip(
        reference.record.edges, actual.record.edges, strict=True
    ):
        _same_fields(expected, observed, excluding=frozenset({"old_log_probability"}))
        max_log_probability_error = max(
            max_log_probability_error,
            _numeric_error(
                expected.old_log_probability,
                observed.old_log_probability,
                f"edge {expected.edge_id}",
            ),
        )
    for field in fields(reference.targets):
        if getattr(reference.targets, field.name) != getattr(
            actual.targets, field.name
        ):
            raise ValueError(f"Replay target mismatch in {field.name}")
    if reference.branch_execution != actual.branch_execution:
        raise ValueError("Replay branch execution mismatch")
    return {
        "passed": True,
        "max_value_error": max_value_error,
        "max_log_probability_error": max_log_probability_error,
    }
