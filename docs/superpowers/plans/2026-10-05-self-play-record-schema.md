# Self-play Record Schema Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** T06 の独立対局と分岐木を同じ型・検証器で扱い、不正な親子関係とモデル世代混在を拒否する。

**Architecture:** Pydantic の strict / frozen model が個々の値と構築時の型を検証する。共通検証器が入口でモデルを再検証し、ID、グラフ、ゲーム遷移、終局、分岐、乱数参照の整合性を検証する。A は一本道、B/C は同じ木の意味モデルを使う。

**Tech Stack:** CPython 3.14、uv、Pydantic v2、NumPy、既存 reversi-py、pytest、Ruff 0.16.8。

**Spec:** [T06 design](../specs/2026-10-04-self-play-record-schema-design.md)

## Global Constraints

- Python: `>=3.14,<3.15`。Pydantic: `>=2.12,<3` を直接依存に追加する。
- 共通基底は `strict=True`、`frozen=True`、`extra="forbid"`、`allow_inf_nan=False`、`validate_default=True`、`revalidate_instances="always"`。
- `GameState` は `InstanceOf[GameState]` として保持する。既存のゲーム API と model / config / RNG の実装は変更しない。
- nodes / edges / sibling groups / RNG streams は tuple とする。整数としての bool、数値文字列、NaN / Inf、未知 field を拒否する。
- 個々の model 構築時は Pydantic `ValidationError`。共通検証器の入口以降は `RecordValidationError(ValueError)`。
- 保存形式、atomic store、経路重み・リターン計算、collector、trainer、checkpoint、停止・再開は今回の範囲外。G1 通過前に保存データを trainer へ接続しない。
- 同じ action と同じ子局面を持つ兄弟標本を統合しない。強制パスは独立した辺や着手数にしない。

## Review Focus

- 未検証の `model_copy` 更新でネストされた型違反を渡しても入口で拒否する。Task 2 の再検証試験を持つ。
- `True` と `1` が等しい Python の性質で schema version / result の制約を迂回できない。Task 1 で strict integer に対して試験する。
- 同じ action の兄弟が等しい `GameState` を持っても、異なる標本 ID を維持する。Task 2 の重複標本試験を持つ。
- 強制パスで親子の手番が同じでも、局面の実遷移と着手数で受理する。Task 2 で既知局面を使う。
- 根から到達しない閉じた循環や、完成と称した途中葉を拒否する。Task 2 のグラフ・完了試験を持つ。

---

## File Map

以下の `P` は `server/ml/models/lfu_tree_self_play` の省略表記。実際の編集はこのディレクトリ内で行う。

| ファイル | 責任 |
| --- | --- |
| `P/pyproject.toml`, `P/uv.lock` | 直接依存の追加と再現可能な環境 |
| `P/src/lfu_tree_self_play/records.py` | record 型と局所的な値制約 |
| `P/src/lfu_tree_self_play/record_validation.py` | record 全体の意味検証 |
| `P/tests/test_record_types.py` | 暗黙変換拒否・範囲・frozen・tuple の試験 |
| `P/tests/test_records.py` | 正常単位の fixture と不正な単位の拒否試験 |
| `P/docs/records.md` | 公開 API、値の視点、分岐・乱数・コスト、制約、試験結果 |
| `P/README.md`, `P/docs/milestones/01-data-storage.md` | T06 実装と検証の参照 |

### Task 1: Strict record types と直接依存

**Files:** Modify `P/pyproject.toml`, `P/uv.lock`; Create `P/src/lfu_tree_self_play/records.py`, `P/tests/test_record_types.py`.

**Interfaces:** Consumes `GameState` と `derive_seed(run_seed: int, purpose: Purpose) -> int`。Produces 下記 record 型。局所的な型・範囲違反は `ValidationError`。

- [ ] **Step 1: Baseline の確認**

`P` で実行する。現行 lockfile を変更せず baseline を確認し、エラーがある場合は原因を記録する。

```sh
uv sync --locked
uv run --locked pytest -q
```

- [ ] **Step 2: 型制約の failing tests**

`test_record_types.py` に次の最小例から始める。Pydantic の利用前は record import の欠落で失敗する。

```python
import pytest
from pydantic import ValidationError
from lfu_tree_self_play.records import BranchPlan, CostRecord, NodeRecord
from lfu_tree_self_play.game import ReversiPyGame


def costs(**updates):
    values = dict(environment_transitions=0, evaluated_positions=0,
                  inference_batches=0, collection_seconds=0.0,
                  inference_seconds=0.0, transition_seconds=0.0,
                  management_seconds=0.0, peak_memory_bytes=0)
    values.update(updates)
    return values


@pytest.mark.parametrize("bad", [True, "1", 1.5, -1])
def test_transition_count_requires_nonnegative_integer(bad):
    data = costs()
    data["environment_transitions"] = bad
    with pytest.raises(ValidationError):
        CostRecord(**data)


@pytest.mark.parametrize("bad", [True, "1", float("nan"), float("inf"), -0.1])
def test_collection_seconds_rejects_invalid_value(bad):
    data = costs()
    data["collection_seconds"] = bad
    with pytest.raises(ValidationError):
        CostRecord(**data)


def test_branch_positions_require_tuple_and_integer_entries():
    with pytest.raises(ValidationError):
        BranchPlan(width=4, positions=[0])
    with pytest.raises(ValidationError):
        BranchPlan(width=4, positions=(True,))


def test_frozen_record_rejects_assignment_and_unknown_field():
    plan = BranchPlan(width=4, positions=())
    with pytest.raises(ValidationError):
        plan.width = 2
    with pytest.raises(ValidationError):
        BranchPlan(width=4, positions=(), position=(0,))


def test_state_keeps_existing_game_contract():
    state = ReversiPyGame().initial_state()
    node = NodeRecord(node_id="n0", incoming_edge_id=None, state=state,
                      placement_depth=0, model_generation=0,
                      old_value=0.0, black_result=None)
    assert node.state == state
    observation = node.state.observation
    observation[0] = 0
    assert node.state == state
```

- [ ] **Step 3: Pydantic の追加と red の確認**

```sh
uv add 'pydantic>=2.12,<3'
uv run --locked pytest tests/test_record_types.py -q
```

依存変更は Pydantic と必要な依存の範囲に限定し、Torch 等を一括更新しない。期待する red は record 型が未実装であること。ネットワーク・ビルドの失敗を red として扱わない。

- [ ] **Step 4: Record 型の実装**

次のフィールドと制約を実装する。整数 Literal の代わりに strict int と範囲を使い、bool との等価比較を避ける。

```python
from typing import Annotated, Literal
from pydantic import (BaseModel, BeforeValidator, ConfigDict,
                      Field, InstanceOf, StrictInt, StrictStr)
from .game import GameState


def _numeric(value: object) -> object:
    if type(value) not in (int, float):
        raise ValueError("expected a Python int or float, excluding bool")
    return value


RecordId = Annotated[StrictStr, Field(min_length=1, pattern=r"\S")]
Count = Annotated[StrictInt, Field(ge=0)]
Real = Annotated[float, BeforeValidator(_numeric)]
Seconds = Annotated[Real, Field(ge=0)]
Value = Annotated[Real, Field(ge=-1, le=1)]
Result = Annotated[StrictInt, Field(ge=-1, le=1)]


class RecordModel(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid",
                             allow_inf_nan=False, validate_default=True,
                             revalidate_instances="always")


class BranchPlan(RecordModel):
    width: Annotated[StrictInt, Field(ge=2)]
    positions: tuple[Count, ...]


class RNGStreamRecord(RecordModel):
    stream_id: RecordId
    purpose: Literal["branch_position", "action_sampling", "sibling_rollout"]
    seed: Count


class RNGRecord(RecordModel):
    algorithm: Literal["python.random.MT19937-v1"]
    run_seed: Count
    streams: tuple[RNGStreamRecord, ...]


class SamplingRecord(RecordModel):
    stream_id: RecordId
    draw_index: Count
    continuation_seed: Count | None


class NodeRecord(RecordModel):
    node_id: RecordId
    incoming_edge_id: RecordId | None
    state: InstanceOf[GameState]
    placement_depth: Count
    model_generation: Count
    old_value: Value | None
    black_result: Result | None


class EdgeRecord(RecordModel):
    edge_id: RecordId
    parent_node_id: RecordId
    child_node_id: RecordId
    action: Annotated[StrictInt, Field(ge=0, lt=64)]
    old_log_probability: Annotated[Real, Field(le=0)]
    model_generation: Count
    sampling: SamplingRecord
    sibling_group_id: RecordId | None
    sample_index: Count | None


class SiblingGroupRecord(RecordModel):
    group_id: RecordId
    parent_node_id: RecordId
    edge_ids: tuple[RecordId, ...]
    distinct_action_count: Count


class CostRecord(RecordModel):
    environment_transitions: Count
    evaluated_positions: Count
    inference_batches: Count
    collection_seconds: Seconds
    inference_seconds: Seconds
    transition_seconds: Seconds
    management_seconds: Seconds
    peak_memory_bytes: Count


class CollectionRecord(RecordModel):
    schema_version: Annotated[StrictInt, Field(ge=1, le=1)] = 1
    unit_id: RecordId
    experiment_id: RecordId
    round_id: RecordId
    model_generation: Count
    mode: Literal["independent", "tree"]
    root_node_id: RecordId
    branch_plan: BranchPlan
    rng: RNGRecord
    nodes: tuple[NodeRecord, ...]
    edges: tuple[EdgeRecord, ...]
    sibling_groups: tuple[SiblingGroupRecord, ...]
    cost: CostRecord
```

- [ ] **Step 5: 値の範囲と nested model の試験を増やして green を確認**

schema version / black result に bool、世代・seed・sample index に負数、value に範囲外、log probability に正数、ID に空白のみを渡す試験を追加する。すべて `ValidationError` を期待する。`model_copy(update=...)` が再検証前に未検証値を持てることは Task 2 で扱う。

```sh
uv run --locked pytest tests/test_record_types.py -q
uv run --locked pytest -q
git diff --check
```

- [ ] **Step 6: 対象ファイルだけをコミット**

Conventional Commit: `feat(ml): add strict self-play record types`。

### Task 2: 完了単位を検証する共通検証器

**Files:** Create `P/src/lfu_tree_self_play/record_validation.py`, `P/tests/test_records.py`.

**Interfaces:** Consumes Task 1 の全 record 型と `Game.step(state, action) -> GameState`。Produces `validate_record(record: CollectionRecord, game: Game) -> None`, `RecordValidationError(ValueError)`。以下の私有関数は同じモジュール内に置く。

- [ ] **Step 1: 正常 fixture と failing acceptance tests**

`test_records.py` に import、次の fixture builder と acceptance tests を追加する。局面はゲーム backend で進め、全枝を実際に終局させる。fixtures は collector の製品実装にしない。

```python
import math
from collections import Counter
import numpy as np
import pytest
from lfu_tree_self_play.game import GameState, ReversiPyGame
from lfu_tree_self_play.rng import derive_seed
from lfu_tree_self_play.verification import endgame_position
from lfu_tree_self_play.records import (
    BranchPlan, CollectionRecord, CostRecord, EdgeRecord, NodeRecord,
    RNGRecord, RNGStreamRecord, SamplingRecord, SiblingGroupRecord,
)
from lfu_tree_self_play.record_validation import (
    RecordValidationError, validate_record,
)


def build_record(root=None, mode="tree", positions=(0, 1), width=2,
                 first_action=None, root_actions=None):
    game = ReversiPyGame()
    root = endgame_position(1) if root is None else root
    nodes, edges, groups = [], [], []
    streams = [RNGStreamRecord(stream_id=stream_id, purpose=purpose,
                              seed=derive_seed(7, purpose))
               for stream_id, purpose in (
                   ("branch", "branch_position"),
                   ("action", "action_sampling"),
                   ("sibling", "sibling_rollout"))]
    draws = Counter()

    def visit(state, incoming, depth, stream_id):
        node_id = f"n{len(nodes)}"
        nodes.append(NodeRecord(node_id=node_id, incoming_edge_id=incoming,
                                state=state, placement_depth=depth,
                                model_generation=0,
                                old_value=None if state.is_terminal else 0.0,
                                black_result=state.black_result))
        if state.is_terminal:
            return node_id
        branch = mode == "tree" and depth in positions
        group_id = f"g{len(groups)}" if branch else None
        if branch:
            groups.append(None)
        group_slot = len(groups) - 1
        member_ids = []
        for index in range(width if branch else 1):
            edge_id = f"e{len(edges)}"
            # Always repeat the first legal action at a branch: the validator
            # must keep these independent samples even when boards are equal.
            action = (root_actions[index] if depth == 0 and branch and root_actions
                      else first_action if depth == 0 and first_action is not None
                      else state.legal_actions[0])
            continuation_seed = len(edges) + 1000 if branch else None
            child_stream = f"continuation-{edge_id}" if branch else stream_id
            if branch:
                streams.append(RNGStreamRecord(stream_id=child_stream,
                    purpose="sibling_rollout", seed=continuation_seed))
            draw_index = draws[stream_id]
            draws[stream_id] += 1
            edges.append(None)
            edge_slot = len(edges) - 1
            child_id = visit(game.step(state, action), edge_id, depth + 1,
                             child_stream)
            edges[edge_slot] = EdgeRecord(edge_id=edge_id,
                parent_node_id=node_id, child_node_id=child_id, action=action,
                old_log_probability=-math.log(len(state.legal_actions)),
                model_generation=0,
                sampling=SamplingRecord(stream_id=stream_id,
                    draw_index=draw_index, continuation_seed=continuation_seed),
                sibling_group_id=group_id,
                sample_index=index if branch else None)
            member_ids.append(edge_id)
        if branch:
            groups[group_slot] = SiblingGroupRecord(group_id=group_id,
                parent_node_id=node_id, edge_ids=tuple(member_ids),
                distinct_action_count=1)
        return node_id

    root_id = visit(root, None, 0, "action")
    return CollectionRecord(unit_id="unit", experiment_id="experiment",
        round_id="round", model_generation=0, mode=mode, root_node_id=root_id,
        branch_plan=BranchPlan(width=width, positions=positions),
        rng=RNGRecord(algorithm="python.random.MT19937-v1", run_seed=7,
                      streams=tuple(streams)),
        nodes=tuple(nodes), edges=tuple(edges), sibling_groups=tuple(groups),
        cost=CostRecord(environment_transitions=len(edges),
            evaluated_positions=len(nodes)-sum(n.state.is_terminal for n in nodes),
            inference_batches=0, collection_seconds=0.0,
            inference_seconds=0.0, transition_seconds=0.0,
            management_seconds=0.0, peak_memory_bytes=0))


@pytest.mark.parametrize(("mode", "positions"), [
    ("independent", ()), ("tree", ()), ("tree", (0,)), ("tree", (0, 1)),
])
def test_one_validator_accepts_independent_and_tree_records(mode, positions):
    record = build_record(mode=mode, positions=positions)
    assert validate_record(record, ReversiPyGame()) is None


def test_duplicate_actions_remain_distinct_samples():
    record = build_record(positions=(0,))
    by_id = {edge.edge_id: edge for edge in record.edges}
    members = [by_id[key] for key in record.sibling_groups[0].edge_ids]
    assert len({edge.action for edge in members}) == 1
    assert len({edge.child_node_id for edge in members}) == 2
    assert validate_record(record, ReversiPyGame()) is None
```

```sh
uv run --locked pytest tests/test_records.py -q
```

期待する red は未実装の検証器 import。fixture の構築失敗は先に直す。

- [ ] **Step 2: 再検証入口とグラフ検証を実装**

次の入口と索引 helper を使用し、検証の責任を小さな私有関数に分ける。公開 API は 1 個に保つ。

```python
from collections import defaultdict
import numpy as np
from pydantic import ValidationError
from .game import Game
from .records import CollectionRecord
from .rng import derive_seed


class RecordValidationError(ValueError):
    """A complete collection unit violates its schema or game contracts."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RecordValidationError(message)


def _index(values: tuple, field: str) -> dict:
    result = {}
    for value in values:
        key = getattr(value, field)
        _require(key not in result, f"duplicate {field}: {key}")
        result[key] = value
    return result


def validate_record(record: CollectionRecord, game: Game) -> None:
    try:
        checked = CollectionRecord.model_validate(record)
    except ValidationError as error:
        raise RecordValidationError(str(error)) from error
    nodes = _index(checked.nodes, "node_id")
    edges = _index(checked.edges, "edge_id")
    groups = _index(checked.sibling_groups, "group_id")
    streams = _index(checked.rng.streams, "stream_id")
    outgoing = _validate_graph(checked, nodes, edges)
    _validate_positions(checked, game, nodes, outgoing)
    _validate_branching(checked, nodes, edges, groups, outgoing)
    _validate_rng(checked, streams)
```

`_validate_graph(record, nodes, edges) -> dict[str, list[EdgeRecord]]` の実装は以下の順序とする。

```python
def _validate_graph(record, nodes, edges):
    root_id = record.root_node_id
    _require(root_id in nodes, f"missing root: {root_id}")
    incoming, outgoing = defaultdict(list), defaultdict(list)
    for edge in edges.values():
        _require(edge.parent_node_id in nodes, f"missing parent: {edge.edge_id}")
        _require(edge.child_node_id in nodes, f"missing child: {edge.edge_id}")
        _require(edge.parent_node_id != edge.child_node_id,
                 f"self edge: {edge.edge_id}")
        incoming[edge.child_node_id].append(edge)
        outgoing[edge.parent_node_id].append(edge)
    for node in nodes.values():
        actual = incoming[node.node_id]
        if node.node_id == root_id:
            _require(node.incoming_edge_id is None and not actual,
                     f"root has incoming edge: {node.node_id}")
            _require(node.placement_depth == 0, f"root depth: {node.node_id}")
        else:
            _require(len(actual) == 1, f"parent count: {node.node_id}")
            _require(node.incoming_edge_id == actual[0].edge_id,
                     f"incoming edge mismatch: {node.node_id}")
    seen, pending = set(), [root_id]
    while pending:
        node_id = pending.pop()
        _require(node_id not in seen, f"cycle or repeated child: {node_id}")
        seen.add(node_id)
        pending.extend(edge.child_node_id for edge in outgoing[node_id])
    _require(seen == set(nodes), "unreachable nodes or disconnected cycle")
    return outgoing
```

- [ ] **Step 3: 局面・モデル世代・完了単位を検証**

`_validate_positions(record, game, nodes, outgoing) -> None` に次を実装する。

```python
def _validate_positions(record, game, nodes, outgoing):
    for node in nodes.values():
        key = node.node_id
        _require(node.model_generation == record.model_generation,
                 f"model generation: {key}")
        planes = node.state.observation
        _require(planes.shape == (4, 8, 8) and bool(np.isfinite(planes).all()),
                 f"observation shape or finite values: {key}")
        _require(bool(np.isin(planes[[0, 1, 3]], (0, 1)).all()),
                 f"binary planes: {key}")
        _require(bool((planes[0] + planes[1] <= 1).all()), f"disk overlap: {key}")
        _require(bool((planes[3] * (planes[0] + planes[1]) == 0).all()),
                 f"occupied legal action: {key}")
        turn = planes[2, 0, 0]
        _require(turn in (-1, 1) and bool((planes[2] == turn).all()),
                 f"turn plane: {key}")
        if node.state.is_terminal:
            _require(not outgoing[key], f"terminal has children: {key}")
            _require(node.old_value is None, f"terminal has old value: {key}")
            _require(node.black_result == node.state.black_result,
                     f"black result: {key}")
        else:
            _require(bool(outgoing[key]), f"incomplete leaf: {key}")
            _require(node.old_value is not None and node.black_result is None,
                     f"nonterminal value/result: {key}")
    for edge in record.edges:
        parent, child = nodes[edge.parent_node_id], nodes[edge.child_node_id]
        _require(edge.model_generation == record.model_generation,
                 f"model generation: {edge.edge_id}")
        _require(child.placement_depth == parent.placement_depth + 1,
                 f"placement depth: {edge.edge_id}")
        _require(edge.action in parent.state.legal_actions,
                 f"illegal action: {edge.edge_id}")
        try:
            next_state = game.step(parent.state, edge.action)
        except ValueError as error:
            raise RecordValidationError(f"game transition: {edge.edge_id}") from error
        _require(next_state == child.state, f"transition mismatch: {edge.edge_id}")
```

- [ ] **Step 4: 分岐予定と兄弟群を検証**

`_validate_branching(record, nodes, edges, groups, outgoing) -> None` に次の規則を実装する。

```python
def _validate_branching(record, nodes, edges, groups, outgoing):
    plan = record.branch_plan
    root_planes = nodes[record.root_node_id].state.observation
    empty = int(np.count_nonzero(root_planes[0] + root_planes[1] == 0))
    _require(plan.positions == tuple(sorted(set(plan.positions))), "branch positions order")
    _require(all(position < empty for position in plan.positions), "branch positions range")
    if record.mode == "independent":
        _require(not plan.positions and not groups, "independent branching")
    group_by_parent = {}
    grouped_edges = set()
    for group in groups.values():
        key = group.group_id
        _require(group.parent_node_id in nodes, f"group parent: {key}")
        _require(group.parent_node_id not in group_by_parent, f"duplicate parent group: {key}")
        group_by_parent[group.parent_node_id] = group
        _require(len(group.edge_ids) == plan.width, f"group width: {key}")
        _require(len(set(group.edge_ids)) == len(group.edge_ids), f"duplicate group member: {key}")
        _require(all(edge_id in edges for edge_id in group.edge_ids), f"missing member: {key}")
        members = [edges[edge_id] for edge_id in group.edge_ids]
        _require(set(group.edge_ids) == {edge.edge_id for edge in outgoing[group.parent_node_id]},
                 f"group coverage: {key}")
        for index, edge in enumerate(members):
            _require(edge.parent_node_id == group.parent_node_id and
                     edge.sibling_group_id == key and edge.sample_index == index,
                     f"group member mismatch: {edge.edge_id}")
            _require(edge.edge_id not in grouped_edges, f"multiple groups: {edge.edge_id}")
            grouped_edges.add(edge.edge_id)
        _require(group.distinct_action_count == len({edge.action for edge in members}),
                 f"distinct actions: {key}")
    for node in nodes.values():
        branching = record.mode == "tree" and node.placement_depth in plan.positions
        if node.state.is_terminal:
            _require(node.node_id not in group_by_parent, f"terminal group: {node.node_id}")
        elif branching:
            _require(node.node_id in group_by_parent, f"missing branch: {node.node_id}")
        else:
            _require(node.node_id not in group_by_parent and len(outgoing[node.node_id]) == 1,
                     f"unexpected branch: {node.node_id}")
    for edge in record.edges:
        if edge.edge_id not in grouped_edges:
            _require(edge.sibling_group_id is None and edge.sample_index is None,
                     f"ungrouped edge metadata: {edge.edge_id}")
```

- [ ] **Step 5: 乱数の参照と標本識別を検証**

`_validate_rng(record, streams) -> None` に主要 seed の照合、抽出イベントの重複拒否、追加 continuation stream の照合を実装する。

```python
def _validate_rng(record, streams):
    primary = {"branch": "branch_position", "action": "action_sampling",
               "sibling": "sibling_rollout"}
    for stream_id, purpose in primary.items():
        _require(stream_id in streams, f"missing RNG stream: {stream_id}")
        stream = streams[stream_id]
        _require(stream.purpose == purpose and
                 stream.seed == derive_seed(record.rng.run_seed, purpose),
                 f"primary RNG stream: {stream_id}")
    continuations = {}
    for stream in streams.values():
        if stream.stream_id not in primary:
            _require(stream.purpose == "sibling_rollout", f"continuation purpose: {stream.stream_id}")
            continuations.setdefault(stream.seed, []).append(stream.stream_id)
    seen = set()
    group_seeds = defaultdict(set)
    for edge in record.edges:
        sampling = edge.sampling
        _require(sampling.stream_id in streams, f"sampling stream: {edge.edge_id}")
        _require(streams[sampling.stream_id].purpose in ("action_sampling", "sibling_rollout"),
                 f"sampling purpose: {edge.edge_id}")
        event = (sampling.stream_id, sampling.draw_index)
        _require(event not in seen, f"duplicate draw: {edge.edge_id}")
        seen.add(event)
        if edge.sibling_group_id is None:
            _require(sampling.continuation_seed is None, f"normal edge continuation: {edge.edge_id}")
        else:
            seed = sampling.continuation_seed
            _require(seed is not None and seed in continuations,
                     f"missing continuation stream: {edge.edge_id}")
            seeds = group_seeds[edge.sibling_group_id]
            _require(seed not in seeds, f"duplicate continuation seed: {edge.edge_id}")
            seeds.add(seed)
```

抽出系列の独立性をこの参照検証だけで証明したと記載しない。draw index は系列内の抽出イベント番号で、PRNG 内部の呼出回数ではない。

- [ ] **Step 6: 不正例を先に追加し、各拒否理由を red / green で確認**

型は正しいが意味の違う更新には `model_copy` を用いる。次の pattern でノードと辺の全試験を記述し、非破壊であることも確認する。

```python
def replace_node(record, index, **updates):
    nodes = list(record.nodes)
    nodes[index] = nodes[index].model_copy(update=updates)
    return record.model_copy(update={"nodes": tuple(nodes)})


def replace_edge(record, index, **updates):
    edges = list(record.edges)
    edges[index] = edges[index].model_copy(update=updates)
    return record.model_copy(update={"edges": tuple(edges)})


def test_node_generation_mix_is_rejected():
    original = build_record()
    invalid = replace_node(original, 1, model_generation=1)
    with pytest.raises(RecordValidationError, match="model generation"):
        validate_record(invalid, ReversiPyGame())
    assert original.nodes[1].model_generation == 0


def test_nested_unvalidated_update_is_rejected():
    record = replace_edge(build_record(), 0, action=True)
    with pytest.raises(RecordValidationError, match="action"):
        validate_record(record, ReversiPyGame())


@pytest.mark.parametrize(("updates", "message"), [
    ({"child_node_id": "absent"}, "missing child"),
    ({"parent_node_id": "absent"}, "missing parent"),
    ({"model_generation": 1}, "model generation"),
    ({"action": 63}, "illegal action"),
])
def test_bad_edge_metadata(updates, message):
    record = replace_edge(build_record(), 0, **updates)
    with pytest.raises(RecordValidationError, match=message):
        validate_record(record, ReversiPyGame())
```

辺 action=63 は fixture の親合法手に含まれないことを test 内で assert する。含まれる fixture に変わった場合は `next(a for a in range(64) if a not in parent.legal_actions)` を選ぶ。

さらに以下の具体的な mutation を試験化し、対応する意味違反の message を期待する。

| fixture の変更 | 対応する検証 |
| --- | --- |
| nodes / edges / groups / streams の先頭を tuple の末尾にも追加 | `duplicate` |
| 根を欠落 ID に置換、根 incoming edge を設定 | `missing root` / `root has incoming` |
| 同じ child に別の親から辺を追加 | `parent count` |
| 根以外の node の incoming ID を他の辺 ID に変更 | `incoming edge mismatch` |
| 根から孤立した 2-node cycle を追加し incoming を双方に設定 | `unreachable` |
| 最後の非終局 node の出辺と子を除去 | `incomplete leaf` |
| child の合法手 plane を別の空きマスへ移す | `transition mismatch` |
| 終局 result を他の整数に変更、または None に変更 | `black result` |
| 終局 old value を 0.0 に変更 | `terminal has old value` |
| node depth を +1、白 node の turn plane を +1 に変更 | `placement depth` / `transition mismatch` |
| plan positions を逆順・重複・H 以上に変更 | `branch positions` |
| 幅を変えず group member を 1 本削除 | `group width` |
| group member / edge sample index / group parent を変更 | `group member` / `group coverage` |
| distinct_action_count を 1 から 2 に変更 | `distinct actions` |
| plan の予定位置を移動し、既存群は維持 | `unexpected branch` / `missing branch` |
| 主要 stream の seed を変更、sampling に branch stream を指定 | `primary RNG` / `sampling purpose` |
| sibling の draw index を同じ stream の他 edge と重複させる | `duplicate draw` |
| 兄弟の continuation seed を同じ値へ変更 | `duplicate continuation seed` |
| 構築後の cost に数値文字列または未知 field を追加 | 入り口で `RecordValidationError` |

値が Pydantic の局所制約に反する mutation は入口の型エラーを期待し、意味エラーの試験とは分ける。自己辺・終局に子を追加する試験では、最初に検出されるグラフ違反を許容するが、途中葉等の試験はそれぞれの intended check まで到達する fixture を作る。

- [ ] **Step 7: H=0 / H=1 / 強制パス / 早期終局を試験**

既知の盤面を明示する helper と以下のテストを追加する。子孫数が異なる tree も受理するが、平均・経路重みは計算しない。

```python
def position(rows, turn, legal):
    planes = np.zeros((4, 8, 8), dtype=np.float32)
    for row, cells in enumerate(rows):
        for col, cell in enumerate(cells):
            if cell in ("B", "W"):
                planes[0 if cell == "B" else 1, row, col] = 1
    planes[2] = turn
    for action in legal:
        planes[3, action // 8, action % 8] = 1
    return GameState(planes)


def test_terminal_root_and_one_empty_root():
    terminal = position(("BBBBBBBB",) * 8, 1, ())
    record = build_record(root=terminal, positions=())
    assert len(record.nodes) == 1 and not record.edges
    validate_record(record, ReversiPyGame())
    one_empty = position((".WBBBBBB",) + ("BBBBBBBB",) * 7, 1, (0,))
    record = build_record(root=one_empty, positions=(0,))
    assert len(record.sibling_groups) == 1
    validate_record(record, ReversiPyGame())


def test_early_terminal_does_not_require_unreached_branch():
    root = position((".WBBBBBB",) + ("BBBBBBBB",) * 6 + ("BBBBBBB.",), 1, (0,))
    record = build_record(root=root, positions=(1,))
    assert not record.sibling_groups
    validate_record(record, ReversiPyGame())


def test_forced_pass_preserves_same_player_after_placement():
    root = position(("..BBBBWB", "BBBBB.WB", "BWBWBBWB", "BWBBWBWB",
                     "BWWWBWWB", "BWBWWBWB", "BWWWWWBB", "BWWBBBBB"),
                    -1, (1, 13))
    after = ReversiPyGame().step(root, 13)
    assert after.to_play == root.to_play
    record = build_record(root=root, mode="independent", positions=(), first_action=13)
    by_id = {node.node_id: node for node in record.nodes}
    child = by_id[record.edges[0].child_node_id]
    assert child.placement_depth == 1 and child.state.to_play == root.to_play
    assert validate_record(record, ReversiPyGame()) is None


def test_unequal_descendant_counts_are_valid():
    game = ReversiPyGame()
    pending, seen = [endgame_position(1)], set()
    mixed = None
    while pending and mixed is None:
        state = pending.pop()
        if state in seen or state.is_terminal:
            continue
        seen.add(state)
        children = [(action, game.step(state, action)) for action in state.legal_actions]
        terminal = [action for action, child in children if child.is_terminal]
        continuing = [action for action, child in children if not child.is_terminal]
        if terminal and continuing:
            mixed = (state, terminal[0], continuing[0])
        pending.extend(child for action, child in children if not child.is_terminal)
    assert mixed is not None, "known ending must contain an early-terminal fork"
    state, ending_action, continuing_action = mixed
    record = build_record(root=state, positions=(0, 1),
                          root_actions=(ending_action, continuing_action))
    by_id = {node.node_id: node for node in record.nodes}
    by_edge = {edge.edge_id: edge for edge in record.edges}
    group = next(group for group in record.sibling_groups
                 if group.parent_node_id == record.root_node_id)
    left, right = [by_id[by_edge[key].child_node_id] for key in group.edge_ids]
    assert left.state.is_terminal and not right.state.is_terminal
    assert any(group.parent_node_id == right.node_id for group in record.sibling_groups)
    assert validate_record(record, game) is None
```

- [ ] **Step 8: 全 acceptance tests を green にしてコミット**

```sh
uv run --locked pytest tests/test_record_types.py tests/test_records.py -q
uv run --locked pytest -q
git diff --check
```

Conventional Commit: `feat(ml): validate complete self-play collection records`。

### Task 3: 利用契約と T06 の検証結果

**Files:** Create `P/docs/records.md`; Modify `P/README.md`, `P/docs/milestones/01-data-storage.md`。

**Interfaces:** Consumes Task 1 の型と Task 2 の API。Produces 利用方法と T06 合格条件への証拠。公開 CLI、collector、trainer は追加しない。

- [ ] **Step 1: 利用例と境界を記載**

`docs/records.md` は型一覧、旧 value の着手者視点、黒視点結果、パスの扱い、兄弟標本の識別、予定位置の意味、乱数情報、コストの単位、検証の限界を記載する。API 利用例は次の形にする。

```python
from lfu_tree_self_play.game import ReversiPyGame
from lfu_tree_self_play.records import CollectionRecord
from lfu_tree_self_play.record_validation import RecordValidationError, validate_record

def accept_complete_unit(record: CollectionRecord) -> None:
    validate_record(record, ReversiPyGame())
```

Pydantic 構築時の `ValidationError` と単位検証の `RecordValidationError` を区別する。`model_copy` の未検証更新は入口で再検証される。`InstanceOf[GameState]` には JSON 永続化を提供しない。保存形式と round-trip は T08 で扱う。

- [ ] **Step 2: README と T06 正本へリンク追加**

README の詳細一覧に `[収集 record と共通検証器](docs/records.md)` を追加する。T06 見出し下に成果の API、独立対局と木の共通 acceptance tests、親子・世代混在の rejection tests へのリンクを追加する。既存の成果・合格条件を置き換えず、G1 完了とは記載しない。

- [ ] **Step 3: 必要な全体検証を実行**

`P` で以下を実行し、実行日・結果・テスト件数を `docs/records.md` へ記録する。

```sh
uv sync --locked
uv run --locked pytest -q
uvx --from ruff==0.16.8 ruff check --target-version py312 src tests
uvx --from ruff==0.16.8 ruff format --check --target-version py312 src tests
git diff --check
```

baseline と比較し、無関係な既存失敗があれば別記する。変更していない重い G0 Monte Carlo 実験は再実行しない。

- [ ] **Step 4: 文書をコミットして実装全体をレビュー**

Conventional Commit: `docs(ml): document collection record validation`。レビューで確認する点は正本の全成果の表現、全 acceptance tests、保存・学習への範囲漏れ、Pydantic strict / frozen / 再検証の抜け、fixture に依存した過剰な仮定。レビュー後の修正には該当する test を再実行する。

## 実行方法の提案

3 task が同じ record API と fixture に強く依存するため、このセッションでの Native 実行を推奨する。実装後に独立した reviewer が全体を確認する。計画のレビューと実行方法の選択を受けてから製品コード・依存の変更を始める。

## 実行状況（2026-10-05）

上の Native 実行は当初の提案です。ユーザーは subagent による実装・独立レビュー・修正ループと、その後の PR 作成を承認し、既存 GameState / config / model / RNG の構造を維持する方針で実行しています。Task 1 の型と Task 2 の検証器は個別レビューを通過し、Task 3 は利用契約と検証証拠を追加します。

計画内のコード例は実装ガイド、承認済み仕様は合格条件の正本です。実行時には、共有観測を自己比較する例を独立した配列コピーと identity 確認へ修正し、意図した field の拒否を試すよう ID tests を修正しました。子孫数が異なる木の探索例はその終盤局面で目的の分岐を発見できなかったため、Random(955) の実遷移で到達する明示的盤面（action 12 は終局、10 は継続）へ置き換えています。根の規則に従った合法手 mask は仕様どおり game adapter の信頼境界とし、既存 Game API を拡張せず表現チェックと全保存遷移の replay を行います。

Task 3 の全体 lint で既存 5 test files の I001 が見つかり、実装前 baseline と同じ内容であることを確認しました。追加の限定 Task 4 で import 順序だけを修正し、全体 pytest（208 passed）、Ruff lint / format を再確認しています。製品 API・テストの assertions・依存には変更を加えていません。
