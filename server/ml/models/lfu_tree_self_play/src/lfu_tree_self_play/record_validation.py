"""Validate a complete independent game or self-play tree."""

from collections import defaultdict

import numpy as np
from pydantic import ValidationError

from .game import Game
from .records import (
    CollectionRecord,
    EdgeRecord,
    NodeRecord,
    RNGStreamRecord,
    SiblingGroupRecord,
)
from .rng import derive_seed


class RecordValidationError(ValueError):
    """A complete collection unit violates its schema or game contracts."""


def validate_record(record: CollectionRecord, game: Game) -> None:
    """Reject invalid complete records without modifying the input."""
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


def _validate_graph(
    record: CollectionRecord,
    nodes: dict[str, NodeRecord],
    edges: dict[str, EdgeRecord],
) -> dict[str, list[EdgeRecord]]:
    root_id = record.root_node_id
    _require(root_id in nodes, f"missing root: {root_id}")
    incoming, outgoing = defaultdict(list), defaultdict(list)
    for edge in edges.values():
        _require(edge.parent_node_id in nodes, f"missing parent: {edge.edge_id}")
        _require(edge.child_node_id in nodes, f"missing child: {edge.edge_id}")
        _require(
            edge.parent_node_id != edge.child_node_id, f"self edge: {edge.edge_id}"
        )
        incoming[edge.child_node_id].append(edge)
        outgoing[edge.parent_node_id].append(edge)
    for node in nodes.values():
        actual = incoming[node.node_id]
        if node.node_id == root_id:
            _require(
                node.incoming_edge_id is None and not actual,
                f"root has incoming edge: {node.node_id}",
            )
            _require(node.placement_depth == 0, f"root depth: {node.node_id}")
        else:
            _require(len(actual) == 1, f"parent count: {node.node_id}")
            _require(
                node.incoming_edge_id == actual[0].edge_id,
                f"incoming edge mismatch: {node.node_id}",
            )
    seen, pending = set(), [root_id]
    while pending:
        node_id = pending.pop()
        _require(node_id not in seen, f"cycle or repeated child: {node_id}")
        seen.add(node_id)
        pending.extend(edge.child_node_id for edge in outgoing[node_id])
    _require(seen == set(nodes), "unreachable nodes or disconnected cycle")
    return outgoing


def _validate_positions(
    record: CollectionRecord,
    game: Game,
    nodes: dict[str, NodeRecord],
    outgoing: dict[str, list[EdgeRecord]],
) -> None:
    for node in nodes.values():
        key = node.node_id
        _require(
            node.model_generation == record.model_generation, f"model generation: {key}"
        )
        planes = node.state.observation
        _require(
            planes.shape == (4, 8, 8) and bool(np.isfinite(planes).all()),
            f"observation shape or finite values: {key}",
        )
        _require(
            bool(np.isin(planes[[0, 1, 3]], (0, 1)).all()), f"binary planes: {key}"
        )
        _require(bool((planes[0] + planes[1] <= 1).all()), f"disk overlap: {key}")
        _require(
            bool((planes[3] * (planes[0] + planes[1]) == 0).all()),
            f"occupied legal action: {key}",
        )
        turn = planes[2, 0, 0]
        _require(
            turn in (-1, 1) and bool((planes[2] == turn).all()), f"turn plane: {key}"
        )
        if node.state.is_terminal:
            _require(not outgoing[key], f"terminal has children: {key}")
            _require(node.old_value is None, f"terminal has old value: {key}")
            _require(
                node.black_result == node.state.black_result, f"black result: {key}"
            )
        else:
            _require(bool(outgoing[key]), f"incomplete leaf: {key}")
            _require(
                node.old_value is not None and node.black_result is None,
                f"nonterminal value/result: {key}",
            )
    for edge in record.edges:
        parent, child = nodes[edge.parent_node_id], nodes[edge.child_node_id]
        _require(
            edge.model_generation == record.model_generation,
            f"model generation: {edge.edge_id}",
        )
        _require(
            child.placement_depth == parent.placement_depth + 1,
            f"placement depth: {edge.edge_id}",
        )
        _require(
            edge.action in parent.state.legal_actions, f"illegal action: {edge.edge_id}"
        )
        try:
            next_state = game.step(parent.state, edge.action)
        except ValueError as error:
            raise RecordValidationError(f"game transition: {edge.edge_id}") from error
        _require(next_state == child.state, f"transition mismatch: {edge.edge_id}")


def _validate_branching(
    record: CollectionRecord,
    nodes: dict[str, NodeRecord],
    edges: dict[str, EdgeRecord],
    groups: dict[str, SiblingGroupRecord],
    outgoing: dict[str, list[EdgeRecord]],
) -> None:
    plan = record.branch_plan
    root_planes = nodes[record.root_node_id].state.observation
    empty = int(np.count_nonzero(root_planes[0] + root_planes[1] == 0))
    _require(
        plan.positions == tuple(sorted(set(plan.positions))), "branch positions order"
    )
    _require(
        all(position < empty for position in plan.positions), "branch positions range"
    )
    if record.mode == "independent":
        _require(not plan.positions and not groups, "independent branching")
    group_by_parent = {}
    grouped_edges = set()
    for group in groups.values():
        key = group.group_id
        _require(group.parent_node_id in nodes, f"group parent: {key}")
        _require(
            group.parent_node_id not in group_by_parent,
            f"duplicate parent group: {key}",
        )
        group_by_parent[group.parent_node_id] = group
        _require(len(group.edge_ids) == plan.width, f"group width: {key}")
        _require(
            len(set(group.edge_ids)) == len(group.edge_ids),
            f"duplicate group member: {key}",
        )
        _require(
            all(edge_id in edges for edge_id in group.edge_ids),
            f"missing member: {key}",
        )
        members = [edges[edge_id] for edge_id in group.edge_ids]
        _require(
            set(group.edge_ids)
            == {edge.edge_id for edge in outgoing[group.parent_node_id]},
            f"group coverage: {key}",
        )
        for index, edge in enumerate(members):
            _require(
                edge.parent_node_id == group.parent_node_id
                and edge.sibling_group_id == key
                and edge.sample_index == index,
                f"group member mismatch: {edge.edge_id}",
            )
            _require(
                edge.edge_id not in grouped_edges, f"multiple groups: {edge.edge_id}"
            )
            grouped_edges.add(edge.edge_id)
        _require(
            group.distinct_action_count == len({edge.action for edge in members}),
            f"distinct actions: {key}",
        )
    for node in nodes.values():
        branching = record.mode == "tree" and node.placement_depth in plan.positions
        if node.state.is_terminal:
            _require(
                node.node_id not in group_by_parent, f"terminal group: {node.node_id}"
            )
        elif branching:
            _require(node.node_id in group_by_parent, f"missing branch: {node.node_id}")
        else:
            _require(
                node.node_id not in group_by_parent
                and len(outgoing[node.node_id]) == 1,
                f"unexpected branch: {node.node_id}",
            )
    for edge in record.edges:
        if edge.edge_id not in grouped_edges:
            _require(
                edge.sibling_group_id is None and edge.sample_index is None,
                f"ungrouped edge metadata: {edge.edge_id}",
            )


def _validate_rng(
    record: CollectionRecord, streams: dict[str, RNGStreamRecord]
) -> None:
    primary = {
        "branch": "branch_position",
        "action": "action_sampling",
        "sibling": "sibling_rollout",
    }
    for stream_id, purpose in primary.items():
        _require(stream_id in streams, f"missing RNG stream: {stream_id}")
        stream = streams[stream_id]
        _require(
            stream.purpose == purpose
            and stream.seed == derive_seed(record.rng.run_seed, purpose),
            f"primary RNG stream: {stream_id}",
        )
    continuation_seeds = set()
    for stream in streams.values():
        if stream.stream_id not in primary:
            _require(
                stream.purpose == "sibling_rollout",
                f"continuation purpose: {stream.stream_id}",
            )
            continuation_seeds.add(stream.seed)
    seen = set()
    group_seeds = defaultdict(set)
    for edge in record.edges:
        sampling = edge.sampling
        _require(sampling.stream_id in streams, f"sampling stream: {edge.edge_id}")
        _require(
            streams[sampling.stream_id].purpose
            in ("action_sampling", "sibling_rollout"),
            f"sampling purpose: {edge.edge_id}",
        )
        event = (sampling.stream_id, sampling.draw_index)
        _require(event not in seen, f"duplicate draw: {edge.edge_id}")
        seen.add(event)
        if edge.sibling_group_id is None:
            _require(
                sampling.continuation_seed is None,
                f"normal edge continuation: {edge.edge_id}",
            )
        else:
            seed = sampling.continuation_seed
            _require(
                seed is not None and seed in continuation_seeds,
                f"missing continuation stream: {edge.edge_id}",
            )
            seeds = group_seeds[edge.sibling_group_id]
            _require(seed not in seeds, f"duplicate continuation seed: {edge.edge_id}")
            seeds.add(seed)
