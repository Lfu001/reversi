"""Independent self-play trajectory collection under a frozen old policy."""

import math
import re
import resource
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch

from .dataset_codec import UnitIdentity
from .dataset_store import DatasetStore
from .game import Game, GameState
from .model import PolicyValueModel
from .records import (
    BranchPlan,
    CollectionRecord,
    CostRecord,
    EdgeRecord,
    NodeRecord,
    RNGRecord,
    RNGStreamRecord,
    SamplingRecord,
)
from .returns import ReturnTargets, aggregate_returns
from .rng import RNGStreams, create_rng_streams, derive_seed
from .roots import RootEntry, RootPosition
from .sampler import FrozenPolicySampler

_UNIT_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
_DEFAULT_BRANCH_WIDTH = 4


def _require_nonempty_id(value: object, name: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _require_unit_id(value: object) -> str:
    if type(value) is not str or not _UNIT_ID_PATTERN.fullmatch(value):
        raise ValueError("unsafe unit ID")
    return value


def _require_nonnegative_int(value: object, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _extract_root_state(root: GameState | RootPosition | RootEntry) -> GameState:
    if type(root) is GameState:
        state = root
    elif type(root) is RootPosition:
        state = root.state
    elif type(root) is RootEntry:
        state = root.root.state
    else:
        raise ValueError("root must be a GameState, RootPosition, or RootEntry")
    _validate_root_state(state)
    return state


def _validate_root_state(state: GameState) -> None:
    planes = state.observation
    if planes.shape != (4, 8, 8) or not bool(np.isfinite(planes).all()):
        raise ValueError("invalid root observation shape or finite values")
    if not bool(np.isin(planes[[0, 1, 3]], (0, 1)).all()):
        raise ValueError("invalid root binary planes")
    if not bool((planes[0] + planes[1] <= 1).all()):
        raise ValueError("invalid root disk overlap")
    if not bool((planes[3] * (planes[0] + planes[1]) == 0).all()):
        raise ValueError("invalid root legal move on occupied square")
    turn = planes[2, 0, 0]
    if turn not in (-1, 1) or not bool((planes[2] == turn).all()):
        raise ValueError("invalid root turn plane")


def _peak_memory_bytes() -> int:
    rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform != "darwin":
        rss *= 1024
    cuda_bytes = (
        int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
    )
    return max(rss, cuda_bytes, 0)


def scale_return_targets(targets: ReturnTargets, path_weight: float) -> ReturnTargets:
    """Scale a completed unit's node and edge weights by its root path weight."""
    if type(targets) is not ReturnTargets:
        raise ValueError("targets must be a ReturnTargets instance")
    if (
        type(path_weight) not in (int, float)
        or not math.isfinite(path_weight)
        or not 0.0 < float(path_weight) <= 1.0
    ):
        raise ValueError("path_weight must be a finite number in (0, 1]")
    weight = float(path_weight)
    return ReturnTargets(
        node_black_returns=targets.node_black_returns,
        edge_black_returns=targets.edge_black_returns,
        node_weights={
            node_id: value * weight for node_id, value in targets.node_weights.items()
        },
        edge_weights={
            edge_id: value * weight for edge_id, value in targets.edge_weights.items()
        },
        node_player_targets=targets.node_player_targets,
        edge_parent_targets=targets.edge_parent_targets,
    )


def _is_nonstring_sequence(value: object) -> bool:
    return type(value) not in (str, bytes) and isinstance(value, Sequence)


def weight_independent_records(
    records: Sequence[CollectionRecord], game: Game
) -> tuple[ReturnTargets, ...]:
    """Validate K independent paths from the same root and weight each by 1/K."""
    if not _is_nonstring_sequence(records):
        raise ValueError("records must be a sequence of CollectionRecord")
    if not records:
        raise ValueError("at least one independent record is required")

    unscaled: list[ReturnTargets] = []
    seen_units: set[str] = set()
    seen_draws: set[tuple[int, str, int]] = set()
    first = records[0]
    first_root: GameState | None = None

    for record in records:
        targets = aggregate_returns(record, game)
        if record.mode != "independent":
            raise ValueError(f"expected independent mode: {record.unit_id}")
        if record.unit_id in seen_units:
            raise ValueError(f"duplicate unit_id in root group: {record.unit_id}")
        seen_units.add(record.unit_id)
        if (
            record.experiment_id != first.experiment_id
            or record.round_id != first.round_id
            or record.model_generation != first.model_generation
        ):
            raise ValueError("independent root group metadata mismatch")
        nodes = {node.node_id: node for node in record.nodes}
        root_state = nodes[record.root_node_id].state
        if first_root is None:
            first_root = root_state
        elif root_state != first_root:
            raise ValueError("independent root group must share the same root state")
        for edge in record.edges:
            draw_event = (
                record.rng.run_seed,
                edge.sampling.stream_id,
                edge.sampling.draw_index,
            )
            if draw_event in seen_draws:
                raise ValueError(f"duplicate sampling draw across paths: {draw_event}")
            seen_draws.add(draw_event)
        unscaled.append(targets)

    path_weight = 1.0 / len(unscaled)
    return tuple(scale_return_targets(targets, path_weight) for targets in unscaled)


@dataclass(frozen=True)
class IndependentUnit:
    """One completed independent game with unit-level and root-weighted targets."""

    record: CollectionRecord
    targets: ReturnTargets
    path_weight: float
    weighted_targets: ReturnTargets


@dataclass(frozen=True)
class IndependentRootCollection:
    """K independent trajectories collected from a single root position."""

    root_state: GameState
    units: tuple[IndependentUnit, ...]

    def __post_init__(self) -> None:
        if type(self.units) is not tuple or not self.units:
            raise ValueError("units must be a non-empty tuple of IndependentUnit")

    @property
    def path_weight(self) -> float:
        return 1.0 / len(self.units)

    @property
    def records(self) -> tuple[CollectionRecord, ...]:
        return tuple(unit.record for unit in self.units)

    @property
    def weighted_targets(self) -> tuple[ReturnTargets, ...]:
        return tuple(unit.weighted_targets for unit in self.units)

    def publish_all(self, store: DatasetStore) -> tuple[UnitIdentity, ...]:
        """Publish every completed independent unit atomically to DatasetStore."""
        return tuple(store.publish(unit.record, unit.targets) for unit in self.units)


class IndependentCollector:
    """Generate complete independent self-play games under a fixed pi_old model."""

    def __init__(
        self,
        model: PolicyValueModel | FrozenPolicySampler,
        game: Game,
        *,
        experiment_id: str,
        round_id: str,
        model_generation: int,
        run_seed: int,
        rng_streams: RNGStreams | None = None,
        initial_draw_index: int = 0,
    ) -> None:
        self._experiment_id = _require_nonempty_id(experiment_id, "experiment_id")
        self._round_id = _require_nonempty_id(round_id, "round_id")
        self._model_generation = _require_nonnegative_int(
            model_generation, "model_generation"
        )
        self._run_seed = _require_nonnegative_int(run_seed, "run_seed")
        self._draw_index = _require_nonnegative_int(
            initial_draw_index, "initial_draw_index"
        )
        self._primary_streams = (
            RNGStreamRecord(
                stream_id="branch",
                purpose="branch_position",
                seed=derive_seed(self._run_seed, "branch_position"),
            ),
            RNGStreamRecord(
                stream_id="action",
                purpose="action_sampling",
                seed=derive_seed(self._run_seed, "action_sampling"),
            ),
            RNGStreamRecord(
                stream_id="sibling",
                purpose="sibling_rollout",
                seed=derive_seed(self._run_seed, "sibling_rollout"),
            ),
        )
        if type(model) is FrozenPolicySampler:
            self._sampler = model
        elif type(model) is PolicyValueModel:
            self._sampler = FrozenPolicySampler(model)
        else:
            raise ValueError("model must be a PolicyValueModel or FrozenPolicySampler")
        if rng_streams is not None and type(rng_streams) is not RNGStreams:
            raise ValueError("rng_streams must be an RNGStreams instance")
        self._streams = (
            rng_streams
            if rng_streams is not None
            else create_rng_streams(self._run_seed)
        )
        self._game = game

    @property
    def draw_index(self) -> int:
        """Return the next action_sampling draw index."""
        return self._draw_index

    def collect_game(
        self,
        root: GameState | RootPosition | RootEntry,
        *,
        unit_id: str,
    ) -> IndependentUnit:
        """Play one independent game to termination and return its validated unit."""
        checked_unit_id = _require_unit_id(unit_id)
        root_state = _extract_root_state(root)
        saved_rng_state = self._streams.action_sampling.getstate()
        saved_draw_index = self._draw_index

        try:
            start_collection = time.perf_counter()
            inference_seconds = 0.0
            transition_seconds = 0.0
            evaluated_positions = 0
            inference_batches = 0
            environment_transitions = 0

            nodes: list[NodeRecord] = []
            edges: list[EdgeRecord] = []
            current = root_state
            incoming_edge_id: str | None = None
            depth = 0

            while not current.is_terminal:
                node_id = f"n{depth}"
                edge_id = f"e{depth}"
                child_id = f"n{depth + 1}"

                infer_start = time.perf_counter()
                evaluation = self._sampler.evaluate_and_sample(
                    current, self._streams.action_sampling, count=1
                )
                inference_seconds += time.perf_counter() - infer_start
                evaluated_positions += 1
                inference_batches += 1

                sample = evaluation.samples[0]
                draw_index = self._draw_index
                self._draw_index += 1

                nodes.append(
                    NodeRecord(
                        node_id=node_id,
                        incoming_edge_id=incoming_edge_id,
                        state=current,
                        placement_depth=depth,
                        model_generation=self._model_generation,
                        old_value=evaluation.old_value,
                        black_result=None,
                    )
                )

                step_start = time.perf_counter()
                next_state = self._game.step(current, sample.action)
                transition_seconds += time.perf_counter() - step_start
                environment_transitions += 1

                edges.append(
                    EdgeRecord(
                        edge_id=edge_id,
                        parent_node_id=node_id,
                        child_node_id=child_id,
                        action=sample.action,
                        old_log_probability=sample.old_log_probability,
                        model_generation=self._model_generation,
                        sampling=SamplingRecord(
                            stream_id="action",
                            draw_index=draw_index,
                            continuation_seed=None,
                        ),
                        sibling_group_id=None,
                        sample_index=None,
                    )
                )

                current = next_state
                incoming_edge_id = edge_id
                depth += 1

            nodes.append(
                NodeRecord(
                    node_id=f"n{depth}",
                    incoming_edge_id=incoming_edge_id,
                    state=current,
                    placement_depth=depth,
                    model_generation=self._model_generation,
                    old_value=None,
                    black_result=current.black_result,
                )
            )

            collection_seconds = time.perf_counter() - start_collection
            management_seconds = max(
                0.0, collection_seconds - inference_seconds - transition_seconds
            )
            record = CollectionRecord(
                unit_id=checked_unit_id,
                experiment_id=self._experiment_id,
                round_id=self._round_id,
                model_generation=self._model_generation,
                mode="independent",
                root_node_id="n0",
                branch_plan=BranchPlan(width=_DEFAULT_BRANCH_WIDTH, positions=()),
                rng=RNGRecord(
                    algorithm="python.random.MT19937-v1",
                    run_seed=self._run_seed,
                    streams=self._primary_streams,
                ),
                nodes=tuple(nodes),
                edges=tuple(edges),
                sibling_groups=(),
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
            targets = aggregate_returns(record, self._game)
            return IndependentUnit(
                record=record,
                targets=targets,
                path_weight=1.0,
                weighted_targets=targets,
            )
        except BaseException:
            self._streams.action_sampling.setstate(saved_rng_state)
            self._draw_index = saved_draw_index
            raise

    def collect_paths(
        self,
        root: GameState | RootPosition | RootEntry,
        *,
        paths_per_root: int = 1,
        unit_id_prefix: str | None = None,
        unit_ids: Sequence[str] | None = None,
    ) -> IndependentRootCollection:
        """Collect K independent trajectories from the same root and weight each by 1/K."""
        if type(paths_per_root) is not int or paths_per_root < 1:
            raise ValueError("paths_per_root must be a positive integer")
        if (unit_id_prefix is None) == (unit_ids is None):
            raise ValueError("specify exactly one of unit_id_prefix or unit_ids")
        if unit_id_prefix is not None:
            _require_unit_id(unit_id_prefix)
            resolved_ids = tuple(
                _require_unit_id(f"{unit_id_prefix}-{index}")
                for index in range(paths_per_root)
            )
        else:
            assert unit_ids is not None
            if not _is_nonstring_sequence(unit_ids) or len(unit_ids) != paths_per_root:
                raise ValueError("unit_ids must match paths_per_root")
            resolved_ids = tuple(_require_unit_id(uid) for uid in unit_ids)
            if len(set(resolved_ids)) != len(resolved_ids):
                raise ValueError("unit_ids must be distinct")

        root_state = _extract_root_state(root)
        saved_rng_state = self._streams.action_sampling.getstate()
        saved_draw_index = self._draw_index

        try:
            raw_units = [
                self.collect_game(root_state, unit_id=uid) for uid in resolved_ids
            ]
            weighted = weight_independent_records(
                [unit.record for unit in raw_units], self._game
            )
            path_weight = 1.0 / paths_per_root
            units = tuple(
                IndependentUnit(
                    record=unit.record,
                    targets=unit.targets,
                    path_weight=path_weight,
                    weighted_targets=weighted_targets,
                )
                for unit, weighted_targets in zip(raw_units, weighted, strict=True)
            )
            return IndependentRootCollection(root_state=root_state, units=units)
        except BaseException:
            self._streams.action_sampling.setstate(saved_rng_state)
            self._draw_index = saved_draw_index
            raise


def collect_independent_game(
    model: PolicyValueModel | FrozenPolicySampler,
    game: Game,
    root: GameState | RootPosition | RootEntry,
    *,
    unit_id: str,
    experiment_id: str,
    round_id: str,
    model_generation: int,
    run_seed: int,
    rng_streams: RNGStreams | None = None,
    initial_draw_index: int = 0,
) -> IndependentUnit:
    """Collect a single completed independent game from root."""
    collector = IndependentCollector(
        model,
        game,
        experiment_id=experiment_id,
        round_id=round_id,
        model_generation=model_generation,
        run_seed=run_seed,
        rng_streams=rng_streams,
        initial_draw_index=initial_draw_index,
    )
    return collector.collect_game(root, unit_id=unit_id)


def collect_independent_paths(
    model: PolicyValueModel | FrozenPolicySampler,
    game: Game,
    root: GameState | RootPosition | RootEntry,
    *,
    paths_per_root: int = 1,
    unit_id_prefix: str | None = None,
    unit_ids: Sequence[str] | None = None,
    experiment_id: str,
    round_id: str,
    model_generation: int,
    run_seed: int,
    rng_streams: RNGStreams | None = None,
    initial_draw_index: int = 0,
) -> IndependentRootCollection:
    """Collect K independent games from root and weight each path by 1/K."""
    collector = IndependentCollector(
        model,
        game,
        experiment_id=experiment_id,
        round_id=round_id,
        model_generation=model_generation,
        run_seed=run_seed,
        rng_streams=rng_streams,
        initial_draw_index=initial_draw_index,
    )
    return collector.collect_paths(
        root,
        paths_per_root=paths_per_root,
        unit_id_prefix=unit_id_prefix,
        unit_ids=unit_ids,
    )
