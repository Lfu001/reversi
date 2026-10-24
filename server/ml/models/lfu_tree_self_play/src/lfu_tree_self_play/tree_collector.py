"""Tree self-play collector with pre-sampled width-four branching."""

import random
import re
import resource
import sys
import time
from collections import Counter
from collections.abc import Generator
from dataclasses import dataclass

import numpy as np
import torch

from .batch_inference import EvaluationOutput, EvaluationRequest, FrozenBatchEvaluator
from .collection_session import CollectionSession, drain_session
from .dataset_codec import UnitIdentity
from .dataset_store import DatasetStore
from .game import Game, GameState, ReversiPyGame
from .model import PolicyValueModel
from .record_validation import validate_record
from .records import (
    CollectionRecord,
    CostRecord,
    EdgeRecord,
    NodeRecord,
    RNGRecord,
    RNGStreamRecord,
    SamplingRecord,
    SiblingGroupRecord,
)
from .returns import ReturnTargets, aggregate_returns
from .rng import RNGStreams, create_rng_streams, derive_seed
from .roots import RootEntry, RootPosition
from .sampler import sample_policy
from .schedule import (
    BranchExecution,
    BranchPath,
    PositionStrategy,
    sample_branch_plan,
    summarize_branch_execution,
)

RootInput = GameState | RootPosition | RootEntry
_UNIT_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")


def _require_nonempty_str(value: object, name: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _require_unit_id(value: object) -> str:
    if type(value) is not str or not _UNIT_ID_PATTERN.fullmatch(value):
        raise ValueError("unit_id must be a safe identifier")
    return value


def _require_nonnegative_int(value: object, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _extract_root_state(root: object) -> GameState:
    if type(root) is RootEntry:
        state = root.root.state
    elif type(root) is RootPosition:
        state = root.state
    elif type(root) is GameState:
        state = root
    else:
        raise ValueError("root must be a GameState, RootPosition, or RootEntry")

    planes = state.observation
    if planes.shape != (4, 8, 8) or not bool(np.isfinite(planes).all()):
        raise ValueError("root observation must have finite (4, 8, 8) values")
    if not bool(np.isin(planes[[0, 1, 3]], (0, 1)).all()):
        raise ValueError("root board and legal-move planes must be binary")
    if not bool((planes[0] + planes[1] <= 1).all()):
        raise ValueError("root black and white disks cannot overlap")
    if not bool((planes[3] * (planes[0] + planes[1]) == 0).all()):
        raise ValueError("root legal moves cannot target occupied squares")
    turn = planes[2, 0, 0]
    if turn not in (-1, 1) or not bool((planes[2] == turn).all()):
        raise ValueError("root turn plane must be uniformly +1 or -1")
    return state


def _validate_streams(streams: object) -> RNGStreams:
    if not isinstance(streams, RNGStreams) or any(
        not isinstance(getattr(streams, name), random.Random)
        for name in (
            "branch_position",
            "action_sampling",
            "sibling_rollout",
            "evaluation",
        )
    ):
        raise ValueError("streams must be an RNGStreams instance")
    return streams


def _peak_memory_bytes() -> int:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_bytes = int(rss) if sys.platform == "darwin" else int(rss) * 1024
    cuda_bytes = (
        int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
    )
    return max(0, rss_bytes, cuda_bytes)


@dataclass(frozen=True)
class CollectedTree:
    """A validated self-play tree with aggregated returns and branch counts."""

    record: CollectionRecord
    targets: ReturnTargets
    branch_execution: BranchExecution

    @property
    def planned_branch_count(self) -> int:
        """Total sibling groups budgeted if every path reaches all planned depths."""
        return self.branch_execution.planned_count

    @property
    def executed_branch_count(self) -> int:
        """Total sibling groups actually executed before paths reached terminal."""
        return self.branch_execution.executed_count

    @property
    def terminal_leaf_count(self) -> int:
        """Total terminal leaves in the completed tree (at most 16 for 2 levels)."""
        return sum(node.state.is_terminal for node in self.record.nodes)


class TreeCollector:
    """Collect self-play trees from a frozen round-start policy/value model."""

    def __init__(
        self,
        model: PolicyValueModel | FrozenBatchEvaluator,
        *,
        experiment_id: str,
        round_id: str,
        model_generation: int,
        run_seed: int,
        streams: RNGStreams | None = None,
        levels: int = 2,
        strategy: PositionStrategy = "uniform",
        game: Game | None = None,
    ) -> None:
        if type(model) not in (PolicyValueModel, FrozenBatchEvaluator):
            raise ValueError("model must be a PolicyValueModel")
        self.experiment_id = _require_nonempty_str(experiment_id, "experiment_id")
        self.round_id = _require_nonempty_str(round_id, "round_id")
        self.model_generation = _require_nonnegative_int(
            model_generation, "model_generation"
        )
        self.run_seed = _require_nonnegative_int(run_seed, "run_seed")
        if type(levels) is not int or levels not in (0, 1, 2):
            raise ValueError("levels must be an integer in {0, 1, 2}")
        if strategy not in ("uniform", "stratified", "fixed"):
            raise ValueError(f"unknown position strategy: {strategy}")
        self.levels = levels
        self.strategy: PositionStrategy = strategy
        self._primary_seeds = (
            ("branch", "branch_position", derive_seed(run_seed, "branch_position")),
            ("action", "action_sampling", derive_seed(run_seed, "action_sampling")),
            ("sibling", "sibling_rollout", derive_seed(run_seed, "sibling_rollout")),
        )
        self.streams = (
            create_rng_streams(run_seed)
            if streams is None
            else _validate_streams(streams)
        )
        if game is None:
            game = ReversiPyGame()
        elif not hasattr(game, "initial_state") or not hasattr(game, "step"):
            raise ValueError("game must implement initial_state and step")
        self.game = game
        if type(model) is FrozenBatchEvaluator:
            if model.model_generation != self.model_generation:
                raise ValueError("model_generation does not match evaluator")
            self._evaluator = model
        else:
            self._evaluator = FrozenBatchEvaluator(
                model, model_generation=self.model_generation
            )
        self._session_active = False

    @property
    def evaluator(self) -> FrozenBatchEvaluator:
        return self._evaluator

    def _release_session(self) -> None:
        self._session_active = False

    def create_session(
        self, root: RootInput, *, unit_id: str
    ) -> CollectionSession[CollectedTree]:
        if self._session_active:
            raise RuntimeError("collector already has an active session")
        checked_id = _require_unit_id(unit_id)
        state = _extract_root_state(root)
        self._session_active = True

        def transaction():
            names = (
                "branch_position",
                "action_sampling",
                "sibling_rollout",
                "evaluation",
            )
            saved = {name: getattr(self.streams, name).getstate() for name in names}
            try:
                return (yield from self._generate_tree(state, unit_id=checked_id))
            except BaseException:
                for name, snapshot in saved.items():
                    getattr(self.streams, name).setstate(snapshot)
                raise

        return CollectionSession(
            transaction(), self.evaluator, on_release=self._release_session
        )

    def _next_continuation_seed(self, used_seeds: set[int]) -> int:
        while True:
            candidate = self.streams.sibling_rollout.getrandbits(64)
            if candidate not in used_seeds:
                used_seeds.add(candidate)
                return candidate

    def collect_tree(self, root: RootInput, *, unit_id: str) -> CollectedTree:
        """Build one complete self-play tree, advancing all branches to terminal states."""
        return drain_session(self.create_session(root, unit_id=unit_id))

    def _generate_tree(
        self, root: GameState, *, unit_id: str
    ) -> Generator[EvaluationRequest, EvaluationOutput, CollectedTree]:
        checked_unit_id = _require_unit_id(unit_id)
        root_state = _extract_root_state(root)

        collection_start = time.perf_counter()
        inference_seconds = 0.0
        transition_seconds = 0.0
        evaluated_positions = 0
        inference_batches = 0
        environment_transitions = 0

        planes = root_state.observation
        empty_squares = int(np.count_nonzero(planes[0] + planes[1] == 0))
        plan = sample_branch_plan(
            empty_squares,
            self.streams.branch_position,
            levels=self.levels,
            strategy=self.strategy,
        )

        stream_records: list[RNGStreamRecord] = [
            RNGStreamRecord(stream_id=stream_id, purpose=purpose, seed=seed)
            for stream_id, purpose, seed in self._primary_seeds
        ]
        used_continuation_seeds: set[int] = {self._primary_seeds[2][2]}
        draw_counts: Counter[str] = Counter()
        nodes: list[NodeRecord] = []
        edges: list[EdgeRecord | None] = []
        sibling_groups: list[SiblingGroupRecord | None] = []

        def visit(
            state: GameState,
            incoming_edge_id: str | None,
            placement_depth: int,
            branch_path: BranchPath,
            stream_id: str,
            stream_rng: random.Random,
        ) -> Generator[EvaluationRequest, EvaluationOutput, str]:
            nonlocal inference_seconds, transition_seconds
            nonlocal evaluated_positions, inference_batches, environment_transitions

            node_id = f"n{len(nodes)}"
            if state.is_terminal:
                nodes.append(
                    NodeRecord(
                        node_id=node_id,
                        incoming_edge_id=incoming_edge_id,
                        state=state,
                        placement_depth=placement_depth,
                        model_generation=self.model_generation,
                        old_value=None,
                        black_result=state.black_result,
                    )
                )
                return node_id

            should_branch, next_path = branch_path.visit(state, placement_depth)
            sample_count = plan.width if should_branch else 1

            evaluation = yield EvaluationRequest(len(nodes), state)
            old_value = evaluation.old_value
            draws = sample_policy(
                evaluation.policy_logits,
                torch.from_numpy(state.observation[3]).flatten().bool(),
                stream_rng,
                count=sample_count,
            )
            inference_seconds += evaluation.inference_seconds
            evaluated_positions += 1
            inference_batches += 1

            nodes.append(
                NodeRecord(
                    node_id=node_id,
                    incoming_edge_id=incoming_edge_id,
                    state=state,
                    placement_depth=placement_depth,
                    model_generation=self.model_generation,
                    old_value=old_value,
                    black_result=None,
                )
            )

            if not should_branch:
                draw = draws[0]
                edge_id = f"e{len(edges)}"
                edge_slot = len(edges)
                edges.append(None)
                draw_index = draw_counts[stream_id]
                draw_counts[stream_id] += 1

                step_start = time.perf_counter()
                child_state = self.game.step(state, draw.action)
                transition_seconds += time.perf_counter() - step_start
                environment_transitions += 1

                child_id = yield from visit(
                    child_state,
                    edge_id,
                    placement_depth + 1,
                    next_path,
                    stream_id,
                    stream_rng,
                )
                edges[edge_slot] = EdgeRecord(
                    edge_id=edge_id,
                    parent_node_id=node_id,
                    child_node_id=child_id,
                    action=draw.action,
                    old_log_probability=draw.old_log_probability,
                    model_generation=self.model_generation,
                    sampling=SamplingRecord(
                        stream_id=stream_id,
                        draw_index=draw_index,
                        continuation_seed=None,
                    ),
                    sibling_group_id=None,
                    sample_index=None,
                )
                return node_id

            group_id = f"g{len(sibling_groups)}"
            group_slot = len(sibling_groups)
            sibling_groups.append(None)

            # Mint draw indices and continuation seeds for all siblings upfront
            # so deeper branches inside earlier siblings cannot shift sibling seeds.
            sibling_specs = []
            for sample_index, draw in enumerate(draws):
                draw_index = draw_counts[stream_id]
                draw_counts[stream_id] += 1
                continuation_seed = self._next_continuation_seed(
                    used_continuation_seeds
                )
                sibling_specs.append(
                    (sample_index, draw, draw_index, continuation_seed)
                )

            member_edge_ids: list[str] = []
            for sample_index, draw, draw_index, continuation_seed in sibling_specs:
                edge_id = f"e{len(edges)}"
                edge_slot = len(edges)
                edges.append(None)
                child_stream_id = f"continuation-{edge_id}"
                stream_records.append(
                    RNGStreamRecord(
                        stream_id=child_stream_id,
                        purpose="sibling_rollout",
                        seed=continuation_seed,
                    )
                )
                child_rng = random.Random(continuation_seed)

                step_start = time.perf_counter()
                child_state = self.game.step(state, draw.action)
                transition_seconds += time.perf_counter() - step_start
                environment_transitions += 1

                child_id = yield from visit(
                    child_state,
                    edge_id,
                    placement_depth + 1,
                    next_path,
                    child_stream_id,
                    child_rng,
                )
                edges[edge_slot] = EdgeRecord(
                    edge_id=edge_id,
                    parent_node_id=node_id,
                    child_node_id=child_id,
                    action=draw.action,
                    old_log_probability=draw.old_log_probability,
                    model_generation=self.model_generation,
                    sampling=SamplingRecord(
                        stream_id=stream_id,
                        draw_index=draw_index,
                        continuation_seed=continuation_seed,
                    ),
                    sibling_group_id=group_id,
                    sample_index=sample_index,
                )
                member_edge_ids.append(edge_id)

            sibling_groups[group_slot] = SiblingGroupRecord(
                group_id=group_id,
                parent_node_id=node_id,
                edge_ids=tuple(member_edge_ids),
                distinct_action_count=len({draw.action for draw in draws}),
            )
            return node_id

        root_node_id = yield from visit(
            root_state,
            None,
            0,
            BranchPath(plan),
            "action",
            self.streams.action_sampling,
        )

        collection_seconds = time.perf_counter() - collection_start
        management_seconds = max(
            0.0, collection_seconds - inference_seconds - transition_seconds
        )
        record = CollectionRecord(
            unit_id=checked_unit_id,
            experiment_id=self.experiment_id,
            round_id=self.round_id,
            model_generation=self.model_generation,
            mode="tree",
            root_node_id=root_node_id,
            branch_plan=plan,
            rng=RNGRecord(
                algorithm="python.random.MT19937-v1",
                run_seed=self.run_seed,
                streams=tuple(stream_records),
            ),
            nodes=tuple(nodes),
            edges=tuple(edge for edge in edges if edge is not None),
            sibling_groups=tuple(
                group for group in sibling_groups if group is not None
            ),
            cost=CostRecord(
                environment_transitions=environment_transitions,
                evaluated_positions=evaluated_positions,
                inference_batches=inference_batches,
                collection_seconds=collection_seconds,
                inference_seconds=inference_seconds,
                transition_seconds=transition_seconds,
                management_seconds=management_seconds,
                peak_memory_bytes=_peak_memory_bytes(),
            ),
        )
        targets = aggregate_returns(record, self.game)
        branch_execution = summarize_branch_execution(record, self.game)
        validate_record(record, self.game)
        return CollectedTree(
            record=record,
            targets=targets,
            branch_execution=branch_execution,
        )

    def collect_and_publish(
        self, root: RootInput, store: DatasetStore, *, unit_id: str
    ) -> tuple[CollectedTree, UnitIdentity]:
        """Collect one complete self-play tree and publish it atomically to a DatasetStore."""
        if type(store) is not DatasetStore:
            raise ValueError("store must be a DatasetStore")
        if store.experiment_id != self.experiment_id:
            raise ValueError("experiment/configuration mismatch")
        tree = self.collect_tree(root, unit_id=unit_id)
        identity = store.publish(tree.record, tree.targets)
        return tree, identity


def collect_tree(
    model: PolicyValueModel,
    root: RootInput,
    *,
    unit_id: str,
    experiment_id: str,
    round_id: str,
    model_generation: int,
    run_seed: int,
    streams: RNGStreams | None = None,
    levels: int = 2,
    strategy: PositionStrategy = "uniform",
    game: Game | None = None,
) -> CollectedTree:
    """Collect a single validated self-play tree from a frozen policy/value snapshot."""
    collector = TreeCollector(
        model,
        experiment_id=experiment_id,
        round_id=round_id,
        model_generation=model_generation,
        run_seed=run_seed,
        streams=streams,
        levels=levels,
        strategy=strategy,
        game=game,
    )
    return collector.collect_tree(root, unit_id=unit_id)
