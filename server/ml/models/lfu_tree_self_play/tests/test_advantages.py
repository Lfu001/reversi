"""Hand-calculated B/C targets on complete records, including forced passes."""

from dataclasses import FrozenInstanceError

import pytest
from tests.test_records import build_record, position, replace_edge, replace_node
from tests.test_returns import uneven_record

from lfu_tree_self_play.advantages import (
    AdvantageTargets,
    compute_advantages,
    leave_one_out_advantages,
)
from lfu_tree_self_play.game import ReversiPyGame
from lfu_tree_self_play.record_validation import RecordValidationError


@pytest.mark.parametrize(
    ("returns", "expected"),
    [
        ((1.0, -1.0), (2.0, -2.0)),
        ((1.0, 0.0, -1.0), (1.5, 0.0, -1.5)),
        ((1.0, 1.0, -1.0, 0.0), (1.0, 1.0, -5 / 3, -1 / 3)),
        ((0.5, 0.25, 0.25), (0.25, -0.125, -0.125)),
    ],
)
def test_leave_one_out_matches_hand_calculation_without_scaling(returns, expected):
    assert leave_one_out_advantages(returns) == pytest.approx(expected)


@pytest.mark.parametrize("value", [-1.0, 0.0, 1.0, 0.1, 1 / 3])
@pytest.mark.parametrize("width", [2, 3, 4, 7])
def test_equal_conditional_returns_are_exactly_zero(value, width):
    assert leave_one_out_advantages((value,) * width) == (0.0,) * width


@pytest.mark.parametrize(
    "returns", [(), (1.0,), (1.0, float("nan")), (float("inf"), 0.0)]
)
def test_invalid_sibling_returns_are_rejected(returns):
    with pytest.raises(ValueError):
        leave_one_out_advantages(returns)


def test_nested_groups_use_conditional_returns_and_each_parent_old_value():
    record, game = uneven_record()
    old_values = {"root": 0.25, "right": -0.25, "sample1": 0.5, "sample2": -0.5}
    record = record.model_copy(
        update={
            "nodes": tuple(
                node.model_copy(update={"old_value": old_values[node.node_id]})
                if node.node_id in old_values
                else node
                for node in record.nodes
            )
        }
    )

    result = compute_advantages(record, game)

    # The root is white: its targets are -1 and +0.5, even though
    # the first subtree has one leaf and the second has two leaves.
    assert dict(result.standard_edge_advantages) == {
        "a": -1.25,
        "b": 0.25,
        "c": 1.25,
        "d": -1.5,
        "e": 0.25,
        "f": 0.5,
    }
    assert dict(result.leave_one_out_edge_advantages) == {
        "a": -1.5,
        "b": 1.5,
        # c and e are the same action with independent continuation outcomes.
        "c": 1.0,
        "d": -1.5,
        "e": -1.0,
        "f": 0.5,
    }
    assert result.zero_advantage_group_fraction == 0.0


@pytest.mark.parametrize("color", [1, -1])
@pytest.mark.parametrize("width", [2, 3, 4])
def test_root_win_loss_draw_samples_in_each_players_perspective(color, width):
    record = build_record(positions=(0,), width=width)
    root = position(("........",) * 8, color, (0, 1, 2))
    terminals = (
        position(("B.......",) + ("........",) * 7, -color, ()),
        position(("W.......",) + ("........",) * 7, -color, ()),
        position(("BW......",) + ("........",) * 7, -color, ()),
    )
    root_node = record.nodes[0].model_copy(update={"state": root, "old_value": 0.25})
    edges = tuple(
        record.edges[0].model_copy(
            update={
                "edge_id": f"edge-{i}",
                "child_node_id": f"child-{i}",
                "action": i % 3,
                "sample_index": i,
                "sampling": record.edges[0].sampling.model_copy(
                    update={"draw_index": i, "continuation_seed": 1000 + i}
                ),
            }
        )
        for i in range(width)
    )
    children = tuple(
        root_node.model_copy(
            update={
                "node_id": edge.child_node_id,
                "incoming_edge_id": edge.edge_id,
                "state": terminals[edge.action],
                "placement_depth": 1,
                "old_value": None,
                "black_result": terminals[edge.action].black_result,
            }
        )
        for edge in edges
    )
    group = record.sibling_groups[0].model_copy(
        update={
            "edge_ids": tuple(edge.edge_id for edge in edges),
            "distinct_action_count": min(width, 3),
        }
    )
    continuation = record.rng.streams[3]
    streams = record.rng.streams[:3] + tuple(
        continuation.model_copy(update={"stream_id": f"stream-{i}", "seed": 1000 + i})
        for i in range(width)
    )
    record = record.model_copy(
        update={
            "nodes": (root_node,) + children,
            "edges": edges,
            "sibling_groups": (group,),
            "rng": record.rng.model_copy(update={"streams": streams}),
        }
    )

    class TerminalGame:
        def step(self, state, action):
            assert state == root
            return terminals[action]

    result = compute_advantages(record, TerminalGame())
    black_returns = (1, -1, 0, 1)[:width]
    expected = {
        2: (2, -2),
        3: (1.5, -1.5, 0),
        4: (1, -5 / 3, -1 / 3, 1),
    }[width]
    assert tuple(result.standard_edge_advantages.values()) == tuple(
        color * value - 0.25 for value in black_returns
    )
    assert tuple(result.leave_one_out_edge_advantages.values()) == pytest.approx(
        tuple(color * value for value in expected)
    )


@pytest.mark.parametrize("mode", ["independent", "tree"])
def test_unbranched_records_use_standard_advantages(mode):
    record = build_record(mode=mode, positions=())
    record = record.model_copy(
        update={
            "nodes": tuple(
                node.model_copy(update={"old_value": 0.375})
                if not node.state.is_terminal
                else node
                for node in record.nodes
            )
        }
    )
    result = compute_advantages(record, ReversiPyGame())
    by_id = {node.node_id: node for node in record.nodes}
    black_result = record.nodes[-1].black_result
    expected = {
        edge.edge_id: black_result * by_id[edge.parent_node_id].state.to_play - 0.375
        for edge in record.edges
    }
    assert result.standard_edge_advantages == expected
    assert result.leave_one_out_edge_advantages == expected
    assert result.zero_advantage_group_fraction is None


def test_forced_pass_keeps_actual_white_parent_perspective():
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
    record = replace_node(record, 0, old_value=-0.25)
    record = replace_node(record, 1, old_value=0.25)
    assert record.nodes[0].state.to_play == record.nodes[1].state.to_play == -1
    result = compute_advantages(record, ReversiPyGame())
    for values in (
        result.standard_edge_advantages,
        result.leave_one_out_edge_advantages,
    ):
        assert values[record.edges[0].edge_id] == 1.25
        assert values[record.edges[1].edge_id] == 0.75


def test_equal_result_groups_at_identical_states_remain_separate():
    record = build_record(positions=(0, 1), width=3)
    result = compute_advantages(record, ReversiPyGame())
    grouped = {eid for group in record.sibling_groups for eid in group.edge_ids}
    assert all(result.leave_one_out_edge_advantages[eid] == 0.0 for eid in grouped)
    assert result.zero_advantage_group_fraction == 1.0
    assert len(result.leave_one_out_edge_advantages) == len(record.edges)
    assert any(result.standard_edge_advantages[eid] != 0 for eid in grouped)


def test_same_state_groups_and_different_turns_are_not_pooled():
    record, game = uneven_record()
    draw = next(node for node in record.nodes if node.node_id == "draw")
    edges = list(record.edges)
    groups = list(record.sibling_groups)
    nodes = list(record.nodes)
    streams = list(record.rng.streams)
    # sample1 and sample2 have identical black-to-play states. Give each its
    # own two-sample group: [-1, 0] versus [0, 0]. Their ancestors play white.
    for index, parent in enumerate(("sample1", "sample2")):
        group_id = f"extra-group-{index}"
        slot = next(i for i, edge in enumerate(edges) if edge.parent_node_id == parent)
        first = edges[slot]
        second = first.model_copy(
            update={
                "edge_id": f"extra-edge-{index}",
                "child_node_id": f"extra-draw-{index}",
                "action": 4,
            }
        )
        members = []
        for sample_index, edge in enumerate((first, second)):
            seed = 10000 + index * 2 + sample_index
            streams.append(
                streams[-1].model_copy(update={"stream_id": edge.edge_id, "seed": seed})
            )
            members.append(
                edge.model_copy(
                    update={
                        "sibling_group_id": group_id,
                        "sample_index": sample_index,
                        "sampling": edge.sampling.model_copy(
                            update={
                                "stream_id": edge.edge_id,
                                "draw_index": 0,
                                "continuation_seed": seed,
                            }
                        ),
                    }
                )
            )
        edges[slot] = members[0]
        edges.append(members[1])
        nodes.append(
            draw.model_copy(
                update={
                    "node_id": second.child_node_id,
                    "incoming_edge_id": second.edge_id,
                }
            )
        )
        groups.append(
            groups[0].model_copy(
                update={
                    "group_id": group_id,
                    "parent_node_id": parent,
                    "edge_ids": tuple(edge.edge_id for edge in members),
                    "distinct_action_count": 2 - index,
                }
            )
        )
    record = record.model_copy(
        update={
            "nodes": tuple(nodes),
            "edges": tuple(edges),
            "sibling_groups": tuple(groups),
            "branch_plan": record.branch_plan.model_copy(
                update={"positions": (0, 1, 2)}
            ),
            "rng": record.rng.model_copy(update={"streams": tuple(streams)}),
        }
    )

    result = compute_advantages(record, game)

    assert dict(result.leave_one_out_edge_advantages) == {
        "a": -1.25,
        "b": 1.25,
        "c": 0.5,
        "e": -0.5,
        "d": -1.0,
        "extra-edge-0": 1.0,
        "f": 0.0,
        "extra-edge-1": 0.0,
    }
    assert result.zero_advantage_group_fraction == 0.25


def test_zero_group_fraction_counts_groups_without_path_weighting():
    record, game = uneven_record()
    draw = next(node.state for node in record.nodes if node.node_id == "draw")
    record = record.model_copy(
        update={
            "nodes": tuple(
                node.model_copy(update={"state": draw, "black_result": 0})
                if node.node_id == "loss"
                else node
                for node in record.nodes
            )
        }
    )

    class DrawContinuationGame:
        def step(self, state, action):
            return draw if action == 3 else game.step(state, action)

    result = compute_advantages(record, DrawContinuationGame())
    assert result.zero_advantage_group_fraction == 0.5
    assert result.leave_one_out_edge_advantages["a"] == -1.0
    assert result.leave_one_out_edge_advantages["b"] == 1.0
    assert result.leave_one_out_edge_advantages["c"] == 0.0
    assert result.leave_one_out_edge_advantages["e"] == 0.0


def test_terminal_root_has_no_policy_targets_or_groups():
    root = position(("BBBBBBBB",) * 8, 1, ())
    record = build_record(root=root, positions=())
    result = compute_advantages(record, ReversiPyGame())
    assert result.standard_edge_advantages == {}
    assert result.leave_one_out_edge_advantages == {}
    assert result.zero_advantage_group_fraction is None


def test_targets_are_immutable_detached_and_order_independent():
    record, game = uneven_record()
    before = record.model_dump()
    result = compute_advantages(record, game)
    shuffled = record.model_copy(
        update={
            "nodes": record.nodes[::-1],
            "edges": record.edges[::-1],
            "sibling_groups": record.sibling_groups[::-1],
        }
    )
    assert compute_advantages(shuffled, game) == result
    assert record.model_dump() == before
    for values in (
        result.standard_edge_advantages,
        result.leave_one_out_edge_advantages,
    ):
        assert all(type(value) is float for value in values.values())
        with pytest.raises(TypeError):
            values["a"] = 999.0
    with pytest.raises(FrozenInstanceError):
        result.zero_advantage_group_fraction = 1.0
    source = {"a": 0.5}
    detached = AdvantageTargets(source, source, None)
    source["a"] = 999.0
    assert detached.standard_edge_advantages["a"] == 0.5
    assert detached.leave_one_out_edge_advantages["a"] == 0.5


@pytest.mark.parametrize("old_value", [None, float("nan"), 2.0])
def test_missing_or_invalid_old_value_is_rejected(old_value):
    record = replace_node(build_record(), 0, old_value=old_value)
    with pytest.raises(RecordValidationError):
        compute_advantages(record, ReversiPyGame())


def test_cross_parent_sibling_metadata_is_rejected():
    record, game = uneven_record()
    record = replace_edge(record, 2, sibling_group_id="g0")
    with pytest.raises(RecordValidationError, match="group member mismatch"):
        compute_advantages(record, game)
