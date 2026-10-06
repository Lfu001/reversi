"""Hand-calculated return tests using an enumerated game and Reversi."""

from dataclasses import FrozenInstanceError

import pytest
from tests.test_records import build_record, position

from lfu_tree_self_play.game import ReversiPyGame
from lfu_tree_self_play.record_validation import RecordValidationError
from lfu_tree_self_play.returns import aggregate_returns


def uneven_record():
    """A real valid topology replayed by a tiny enumerated game adapter."""
    template = build_record(positions=(0, 1))
    # Prune the first sample to an early +1 terminal. Keep the second
    # sample's duplicate-action children with outcomes -1 and 0.
    root = position(("........",) * 8, -1, (0, 1))
    right = position(("B.......",) + ("........",) * 7, -1, (2,))
    shared = position(("BB......",) + ("........",) * 7, 1, (3, 4))
    win = position(("B.......",) + ("........",) * 7, 1, ())
    loss = position(("W.......",) + ("........",) * 7, -1, ())
    draw = position(("BW......",) + ("........",) * 7, 1, ())
    states = (root, win, right, shared, loss, shared, draw)
    ids = ("root", "win", "right", "sample1", "loss", "sample2", "draw")
    incoming = (None, "a", "b", "c", "d", "e", "f")
    depths = (0, 1, 1, 2, 3, 2, 3)
    nodes = tuple(
        template.nodes[0].model_copy(
            update={
                "node_id": nid,
                "incoming_edge_id": inc,
                "state": state,
                "placement_depth": depth,
                "old_value": None if state.is_terminal else 0.0,
                "black_result": state.black_result,
            }
        )
        for nid, inc, state, depth in zip(ids, incoming, states, depths, strict=True)
    )
    descriptions = (
        ("a", "root", "win", 0, "g0", 0),
        ("b", "root", "right", 1, "g0", 1),
        ("c", "right", "sample1", 2, "g1", 0),
        ("d", "sample1", "loss", 3, None, None),
        ("e", "right", "sample2", 2, "g1", 1),
        ("f", "sample2", "draw", 4, None, None),
    )
    streams = list(template.rng.streams[:3])
    edges = []
    for index, (eid, parent, child, action, group, sample) in enumerate(descriptions):
        seed = index + 100 if group else None
        if group:
            streams.append(
                streams[2].model_copy(update={"stream_id": eid, "seed": seed})
            )
        edges.append(
            template.edges[0].model_copy(
                update={
                    "edge_id": eid,
                    "parent_node_id": parent,
                    "child_node_id": child,
                    "action": action,
                    "sibling_group_id": group,
                    "sample_index": sample,
                    "sampling": template.edges[0].sampling.model_copy(
                        update={
                            "stream_id": "action",
                            "draw_index": index,
                            "continuation_seed": seed,
                        }
                    ),
                }
            )
        )
    groups = tuple(
        template.sibling_groups[0].model_copy(
            update={
                "group_id": gid,
                "parent_node_id": pid,
                "edge_ids": eids,
                "distinct_action_count": distinct,
            }
        )
        for gid, pid, eids, distinct in (
            ("g0", "root", ("a", "b"), 2),
            ("g1", "right", ("c", "e"), 1),
        )
    )
    record = template.model_copy(
        update={
            "root_node_id": "root",
            "nodes": nodes,
            "edges": tuple(edges),
            "sibling_groups": groups,
            "rng": template.rng.model_copy(update={"streams": tuple(streams)}),
        }
    )

    class EnumeratedGame:
        def step(self, state, action):
            return {
                (root, 0): win,
                (root, 1): right,
                (right, 2): shared,
                (shared, 3): loss,
                (shared, 4): draw,
            }[state, action]

    return record, EnumeratedGame()


def test_recursive_average_keeps_duplicate_samples_and_early_terminal_mass():
    record, game = uneven_record()
    result = aggregate_returns(record, game)
    assert dict(result.node_black_returns) == {
        "root": 0.25,
        "win": 1.0,
        "right": -0.5,
        "sample1": -1.0,
        "loss": -1.0,
        "sample2": 0.0,
        "draw": 0.0,
    }
    assert dict(result.edge_black_returns) == {
        "a": 1.0,
        "b": -0.5,
        "c": -1.0,
        "d": -1.0,
        "e": 0.0,
        "f": 0.0,
    }
    assert dict(result.node_weights) == {
        "root": 1.0,
        "win": 0.5,
        "right": 0.5,
        "sample1": 0.25,
        "loss": 0.25,
        "sample2": 0.25,
        "draw": 0.25,
    }
    assert dict(result.edge_weights) == {
        "a": 0.5,
        "b": 0.5,
        "c": 0.25,
        "d": 0.25,
        "e": 0.25,
        "f": 0.25,
    }
    assert dict(result.edge_parent_targets) == {
        "a": -1.0,
        "b": 0.5,
        "c": 1.0,
        "d": -1.0,
        "e": 0.0,
        "f": 0.0,
    }
    assert result.node_player_targets["root"] == -0.25
    assert sum(result.node_weights[key] for key in ("win", "loss", "draw")) == 1.0


def test_immutable_output_and_order_independence():
    record, game = uneven_record()
    result = aggregate_returns(record, game)
    before = record.model_dump()
    shuffled = record.model_copy(
        update={"nodes": record.nodes[::-1], "edges": record.edges[::-1]}
    )
    assert aggregate_returns(shuffled, game) == result
    assert record.model_dump() == before
    with pytest.raises(TypeError):
        result.node_weights["root"] = 2.0
    with pytest.raises(FrozenInstanceError):
        result.node_weights = {}


@pytest.mark.parametrize("width", [2, 3, 4])
def test_branch_width_conserves_weight(width):
    record = build_record(positions=(0,), width=width)
    result = aggregate_returns(record, ReversiPyGame())
    group = record.sibling_groups[0]
    assert [result.edge_weights[eid] for eid in group.edge_ids] == [1 / width] * width
    assert sum(result.edge_weights[eid] for eid in group.edge_ids) == pytest.approx(1.0)


@pytest.mark.parametrize("empty", [0, 1])
def test_h_zero_and_one(empty):
    root = position(
        (".WBBBBBB" if empty else "BBBBBBBB",) + ("BBBBBBBB",) * 7,
        1,
        (0,) if empty else (),
    )
    record = build_record(root=root, positions=(0,) if empty else ())
    result = aggregate_returns(record, ReversiPyGame())
    assert result.node_black_returns[record.root_node_id] == 1.0
    assert result.node_weights[record.root_node_id] == 1.0
    assert len(result.edge_weights) == 2 * empty


def test_forced_pass_uses_actual_parent_turn():
    root = position(
        (
            "..BBBBWB",
            "BBBBB.WB",
            "BWBWBBWB",
            "BWBBWBWB",
            "BWWWBWWB",
            "BWBWWBWB",
            "BWWWWWBB",
            "BWWBBBBB",
        ),
        -1,
        (1, 13),
    )
    record = build_record(root=root, mode="independent", positions=(), first_action=13)
    result = aggregate_returns(record, ReversiPyGame())
    assert record.nodes[1].state.to_play == -1
    assert result.edge_black_returns[record.edges[0].edge_id] == -1.0
    assert result.edge_parent_targets[record.edges[0].edge_id] == 1.0
    assert result.edge_parent_targets[record.edges[1].edge_id] == 1.0
    assert set(result.edge_weights.values()) == {1.0}


def test_invalid_record_rejected():
    record, game = uneven_record()
    with pytest.raises(RecordValidationError, match="missing root"):
        aggregate_returns(record.model_copy(update={"root_node_id": "missing"}), game)
