"""Complete records built with real game transitions, then narrowly corrupted."""

import math
from collections import Counter

import numpy as np
import pytest

from lfu_tree_self_play.game import GameState, ReversiPyGame
from lfu_tree_self_play.record_validation import RecordValidationError, validate_record
from lfu_tree_self_play.records import (
    BranchPlan,
    CollectionRecord,
    CostRecord,
    EdgeRecord,
    NodeRecord,
    RNGRecord,
    RNGStreamRecord,
    SamplingRecord,
    SiblingGroupRecord,
)
from lfu_tree_self_play.rng import derive_seed
from lfu_tree_self_play.verification import endgame_position


def build_record(
    root=None,
    mode="tree",
    positions=(0, 1),
    width=2,
    first_action=None,
    root_actions=None,
):
    game = ReversiPyGame()
    root = endgame_position(1) if root is None else root
    nodes, edges, groups = [], [], []
    streams = [
        RNGStreamRecord(
            stream_id=stream_id, purpose=purpose, seed=derive_seed(7, purpose)
        )
        for stream_id, purpose in (
            ("branch", "branch_position"),
            ("action", "action_sampling"),
            ("sibling", "sibling_rollout"),
        )
    ]
    draws = Counter()

    def visit(state, incoming, depth, stream_id):
        node_id = f"n{len(nodes)}"
        nodes.append(
            NodeRecord(
                node_id=node_id,
                incoming_edge_id=incoming,
                state=state,
                placement_depth=depth,
                model_generation=0,
                old_value=None if state.is_terminal else 0.0,
                black_result=state.black_result,
            )
        )
        if state.is_terminal:
            return node_id
        branch = mode == "tree" and depth in positions
        group_id = f"g{len(groups)}" if branch else None
        group_slot = len(groups)
        if branch:
            groups.append(None)
        member_ids, actions = [], []
        for index in range(width if branch else 1):
            edge_id = f"e{len(edges)}"
            # Duplicate actions remain independent samples, with separate children.
            action = (
                root_actions[index]
                if depth == 0 and branch and root_actions
                else first_action
                if depth == 0 and first_action is not None
                else state.legal_actions[0]
            )
            continuation_seed = len(edges) + 1000 if branch else None
            child_stream = f"continuation-{edge_id}" if branch else stream_id
            if branch:
                streams.append(
                    RNGStreamRecord(
                        stream_id=child_stream,
                        purpose="sibling_rollout",
                        seed=continuation_seed,
                    )
                )
            draw_index = draws[stream_id]
            draws[stream_id] += 1
            edges.append(None)
            edge_slot = len(edges) - 1
            child_id = visit(game.step(state, action), edge_id, depth + 1, child_stream)
            edges[edge_slot] = EdgeRecord(
                edge_id=edge_id,
                parent_node_id=node_id,
                child_node_id=child_id,
                action=action,
                old_log_probability=-math.log(len(state.legal_actions)),
                model_generation=0,
                sampling=SamplingRecord(
                    stream_id=stream_id,
                    draw_index=draw_index,
                    continuation_seed=continuation_seed,
                ),
                sibling_group_id=group_id,
                sample_index=index if branch else None,
            )
            member_ids.append(edge_id)
            actions.append(action)
        if branch:
            groups[group_slot] = SiblingGroupRecord(
                group_id=group_id,
                parent_node_id=node_id,
                edge_ids=tuple(member_ids),
                distinct_action_count=len(set(actions)),
            )
        return node_id

    root_id = visit(root, None, 0, "action")
    return CollectionRecord(
        unit_id="unit",
        experiment_id="experiment",
        round_id="round",
        model_generation=0,
        mode=mode,
        root_node_id=root_id,
        branch_plan=BranchPlan(width=width, positions=positions),
        rng=RNGRecord(
            algorithm="python.random.MT19937-v1", run_seed=7, streams=tuple(streams)
        ),
        nodes=tuple(nodes),
        edges=tuple(edges),
        sibling_groups=tuple(groups),
        cost=CostRecord(
            environment_transitions=len(edges),
            evaluated_positions=sum(not node.state.is_terminal for node in nodes),
            inference_batches=0,
            collection_seconds=0.0,
            inference_seconds=0.0,
            transition_seconds=0.0,
            management_seconds=0.0,
            peak_memory_bytes=0,
        ),
    )


def replace_node(record, index, **updates):
    nodes = list(record.nodes)
    nodes[index] = nodes[index].model_copy(update=updates)
    return record.model_copy(update={"nodes": tuple(nodes)})


def replace_edge(record, index, **updates):
    edges = list(record.edges)
    edges[index] = edges[index].model_copy(update=updates)
    return record.model_copy(update={"edges": tuple(edges)})


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


@pytest.mark.parametrize(
    ("mode", "positions"),
    [
        ("independent", ()),
        ("tree", ()),
        ("tree", (0,)),
        ("tree", (0, 1)),
    ],
)
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


@pytest.mark.parametrize("width", [3, 4])
def test_branch_width_is_not_hardcoded(width):
    record = build_record(positions=(0,), width=width)
    assert len(record.sibling_groups[0].edge_ids) == width
    assert validate_record(record, ReversiPyGame()) is None


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
    after = ReversiPyGame().step(root, 13)
    assert after.to_play == root.to_play
    record = build_record(root=root, mode="independent", positions=(), first_action=13)
    by_id = {node.node_id: node for node in record.nodes}
    child = by_id[record.edges[0].child_node_id]
    assert child.placement_depth == 1 and child.state.to_play == root.to_play
    assert validate_record(record, ReversiPyGame()) is None


def test_unequal_descendant_counts_are_valid():
    game = ReversiPyGame()
    # Reachable under Random(955) from the initial state by placements
    # 44, 43, 34, 45, 20, 26, 54, 42, 41, 19, 25, 37, 30, 17, 8, 21.
    # Action 12 eliminates all white disks; action 10 leaves a continuing game.
    state = position(
        (
            "........",
            "B.......",
            ".B.WWW..",
            ".BBBW.B.",
            "..BBWB..",
            ".BBBBB..",
            "......B.",
            "........",
        ),
        1,
        (10, 11, 12, 13, 14, 29),
    )
    assert game.step(state, 12).is_terminal
    assert not game.step(state, 10).is_terminal
    record = build_record(root=state, positions=(0, 1), root_actions=(12, 10))
    by_id = {node.node_id: node for node in record.nodes}
    by_edge = {edge.edge_id: edge for edge in record.edges}
    group = record.sibling_groups[0]
    left, right = [by_id[by_edge[key].child_node_id] for key in group.edge_ids]
    assert left.state.is_terminal and not right.state.is_terminal
    assert any(group.parent_node_id == right.node_id for group in record.sibling_groups)
    assert validate_record(record, game) is None


def test_validation_preserves_input_and_does_not_constrain_measured_costs():
    record = build_record()
    record = record.model_copy(
        update={"cost": record.cost.model_copy(update={"environment_transitions": 0})}
    )
    before = record.model_dump()
    observations = [node.state.observation for node in record.nodes]
    validate_record(record, ReversiPyGame())
    assert record.model_dump() == before
    for node, expected in zip(record.nodes, observations, strict=True):
        np.testing.assert_array_equal(node.state.observation, expected)


@pytest.mark.parametrize(
    ("field", "bad"),
    [
        ("action", True),
        ("action", "1"),
        ("old_log_probability", float("nan")),
        ("old_log_probability", float("inf")),
        ("sample_index", -1),
    ],
)
def test_nested_unvalidated_edge_updates_are_rejected(field, bad):
    record = replace_edge(build_record(), 0, **{field: bad})
    with pytest.raises(RecordValidationError, match=field):
        validate_record(record, ReversiPyGame())


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"environment_transitions": "1"}, "environment_transitions"),
        ({"collection_seconds": -1}, "collection_seconds"),
        ({"unknown": 1}, "unknown"),
    ],
)
def test_nested_unvalidated_cost_updates_are_rejected(updates, message):
    record = build_record()
    invalid = record.model_copy(update={"cost": record.cost.model_copy(update=updates)})
    with pytest.raises(RecordValidationError, match=message):
        validate_record(invalid, ReversiPyGame())


@pytest.mark.parametrize("field", ["nodes", "edges", "sibling_groups", "streams"])
def test_duplicate_ids_are_rejected(field):
    record = build_record()
    if field == "streams":
        values = record.rng.streams
        invalid = record.model_copy(
            update={"rng": record.rng.model_copy(update={field: values + values[:1]})}
        )
    else:
        values = getattr(record, field)
        invalid = record.model_copy(update={field: values + values[:1]})
    with pytest.raises(RecordValidationError, match="duplicate"):
        validate_record(invalid, ReversiPyGame())


def test_missing_root_is_rejected():
    record = build_record().model_copy(update={"root_node_id": "absent"})
    with pytest.raises(RecordValidationError, match="missing root"):
        validate_record(record, ReversiPyGame())


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"incoming_edge_id": "e0"}, "root has incoming"),
        ({"placement_depth": 1}, "root depth"),
    ],
)
def test_invalid_root_metadata_is_rejected(updates, message):
    record = replace_node(build_record(), 0, **updates)
    with pytest.raises(RecordValidationError, match=message):
        validate_record(record, ReversiPyGame())


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"child_node_id": "absent"}, "missing child"),
        ({"parent_node_id": "absent"}, "missing parent"),
        ({"child_node_id": "n0"}, "self edge"),
    ],
)
def test_bad_graph_edge_references_are_rejected(updates, message):
    record = replace_edge(build_record(), 0, **updates)
    with pytest.raises(RecordValidationError, match=message):
        validate_record(record, ReversiPyGame())


def test_multiple_parents_are_rejected():
    record = build_record()
    extra = record.edges[0].model_copy(
        update={"edge_id": "extra", "parent_node_id": record.nodes[2].node_id}
    )
    invalid = record.model_copy(update={"edges": record.edges + (extra,)})
    with pytest.raises(RecordValidationError, match="parent count"):
        validate_record(invalid, ReversiPyGame())


def test_wrong_incoming_edge_is_rejected():
    record = replace_node(build_record(), 1, incoming_edge_id="e1")
    with pytest.raises(RecordValidationError, match="incoming edge mismatch"):
        validate_record(record, ReversiPyGame())


def test_disconnected_cycle_is_rejected():
    record = build_record()
    left = record.nodes[1].model_copy(
        update={"node_id": "left", "incoming_edge_id": "right-left"}
    )
    right = record.nodes[1].model_copy(
        update={"node_id": "right", "incoming_edge_id": "left-right"}
    )
    forward = record.edges[0].model_copy(
        update={
            "edge_id": "left-right",
            "parent_node_id": "left",
            "child_node_id": "right",
        }
    )
    backward = forward.model_copy(
        update={
            "edge_id": "right-left",
            "parent_node_id": "right",
            "child_node_id": "left",
        }
    )
    invalid = record.model_copy(
        update={
            "nodes": record.nodes + (left, right),
            "edges": record.edges + (forward, backward),
        }
    )
    with pytest.raises(RecordValidationError, match="unreachable"):
        validate_record(invalid, ReversiPyGame())


def test_isolated_node_is_rejected():
    record = build_record()
    orphan = record.nodes[1].model_copy(
        update={"node_id": "orphan", "incoming_edge_id": None}
    )
    invalid = record.model_copy(update={"nodes": record.nodes + (orphan,)})
    with pytest.raises(RecordValidationError, match="parent count"):
        validate_record(invalid, ReversiPyGame())


def test_node_generation_mix_is_rejected_without_changing_original():
    original = build_record()
    invalid = replace_node(original, 1, model_generation=1)
    with pytest.raises(RecordValidationError, match="model generation"):
        validate_record(invalid, ReversiPyGame())
    assert original.nodes[1].model_generation == 0


def test_edge_generation_mix_is_rejected():
    record = replace_edge(build_record(), 0, model_generation=1)
    with pytest.raises(RecordValidationError, match="model generation"):
        validate_record(record, ReversiPyGame())


def test_illegal_action_is_rejected():
    record = build_record()
    action = next(
        action
        for action in range(64)
        if action not in record.nodes[0].state.legal_actions
    )
    assert action not in record.nodes[0].state.legal_actions
    invalid = replace_edge(record, 0, action=action)
    with pytest.raises(RecordValidationError, match="illegal action"):
        validate_record(invalid, ReversiPyGame())


def test_incomplete_leaf_is_rejected_after_removing_its_child():
    record = build_record(mode="independent", positions=())
    last = record.edges[-1]
    invalid = record.model_copy(
        update={
            "edges": record.edges[:-1],
            "nodes": tuple(
                node for node in record.nodes if node.node_id != last.child_node_id
            ),
        }
    )
    with pytest.raises(RecordValidationError, match="incomplete leaf"):
        validate_record(invalid, ReversiPyGame())


def test_changed_child_legal_mask_is_rejected_as_transition_mismatch():
    record = build_record()
    planes = record.nodes[1].state.observation
    legal = record.nodes[1].state.legal_actions[0]
    empty_nonlegal = next(
        action
        for action in range(64)
        if planes[0, action // 8, action % 8] + planes[1, action // 8, action % 8] == 0
        and action not in record.nodes[1].state.legal_actions
    )
    planes[3, legal // 8, legal % 8] = 0
    planes[3, empty_nonlegal // 8, empty_nonlegal % 8] = 1
    invalid = replace_node(record, 1, state=GameState(planes))
    with pytest.raises(RecordValidationError, match="transition mismatch"):
        validate_record(invalid, ReversiPyGame())


@pytest.mark.parametrize("result", [None, 0])
def test_terminal_result_must_match_board(result):
    record = build_record()
    index = next(
        index
        for index, node in enumerate(record.nodes)
        if node.state.is_terminal and node.black_result != result
    )
    invalid = replace_node(record, index, black_result=result)
    with pytest.raises(RecordValidationError, match="black result"):
        validate_record(invalid, ReversiPyGame())


def test_terminal_old_value_is_rejected():
    record = build_record()
    index = next(
        index for index, node in enumerate(record.nodes) if node.state.is_terminal
    )
    invalid = replace_node(record, index, old_value=0.0)
    with pytest.raises(RecordValidationError, match="terminal has old value"):
        validate_record(invalid, ReversiPyGame())


def test_terminal_children_are_rejected_in_an_otherwise_connected_graph():
    record = build_record()
    terminal = next(node for node in record.nodes if node.state.is_terminal)
    child = terminal.model_copy(
        update={"node_id": "extra", "incoming_edge_id": "extra-edge"}
    )
    edge = record.edges[0].model_copy(
        update={
            "edge_id": "extra-edge",
            "parent_node_id": terminal.node_id,
            "child_node_id": "extra",
        }
    )
    invalid = record.model_copy(
        update={"nodes": record.nodes + (child,), "edges": record.edges + (edge,)}
    )
    with pytest.raises(RecordValidationError, match="terminal has children"):
        validate_record(invalid, ReversiPyGame())


@pytest.mark.parametrize("updates", [{"old_value": None}, {"black_result": 1}])
def test_nonterminal_value_and_result_contract(updates):
    record = replace_node(build_record(), 0, **updates)
    with pytest.raises(RecordValidationError, match="nonterminal value/result"):
        validate_record(record, ReversiPyGame())


def test_placement_depth_counts_edges():
    record = build_record()
    invalid = replace_node(
        record, 1, placement_depth=record.nodes[1].placement_depth + 1
    )
    with pytest.raises(RecordValidationError, match="placement depth"):
        validate_record(invalid, ReversiPyGame())


def test_changed_child_turn_is_rejected_as_transition_mismatch():
    record = build_record()
    index = next(
        index
        for index, node in enumerate(record.nodes)
        if index > 0 and node.state.to_play == -1
    )
    planes = record.nodes[index].state.observation
    planes[2] = 1
    invalid = replace_node(record, index, state=GameState(planes))
    with pytest.raises(RecordValidationError, match="transition mismatch"):
        validate_record(invalid, ReversiPyGame())


@pytest.mark.parametrize(
    ("kind", "message"),
    [
        ("finite", "observation shape or finite"),
        ("shape", "observation shape or finite"),
        ("binary", "binary planes"),
        ("overlap", "disk overlap"),
        ("occupied", "occupied legal action"),
        ("turn-zero", "turn plane"),
        ("turn-varies", "turn plane"),
    ],
)
def test_observation_contract(kind, message):
    record = build_record()
    planes = record.nodes[0].state.observation
    if kind == "finite":
        planes[0, 0, 0] = float("nan")
    elif kind == "binary":
        planes[0, 0, 0] = 0.5
    elif kind == "overlap":
        planes[0, 0, 0] = planes[1, 0, 0] = 1
    elif kind == "occupied":
        square = np.argwhere(planes[0] + planes[1] == 1)[0]
        planes[3, square[0], square[1]] = 1
    elif kind == "turn-zero":
        planes[2] = 0
    elif kind == "turn-varies":
        planes[2, 0, 1] = -planes[2, 0, 0]
    if kind == "shape":
        # Deliberately bypass GameState construction to exercise the validator's
        # defensive shape check on an already-existing instance.
        state = object.__new__(GameState)
        object.__setattr__(state, "_planes", planes[:, :7])
    else:
        state = GameState(planes)
    invalid = replace_node(record, 0, state=state)
    with pytest.raises(RecordValidationError, match=message):
        validate_record(invalid, ReversiPyGame())


def test_game_value_error_is_wrapped_with_edge_id():
    class RejectingGame:
        def step(self, state, action):
            raise ValueError("backend rejected transition")

    with pytest.raises(RecordValidationError, match="game transition: e0") as caught:
        validate_record(build_record(), RejectingGame())
    assert isinstance(caught.value.__cause__, ValueError)


def replace_group(record, index, **updates):
    groups = list(record.sibling_groups)
    groups[index] = groups[index].model_copy(update=updates)
    return record.model_copy(update={"sibling_groups": tuple(groups)})


@pytest.mark.parametrize("positions", [(1, 0), (0, 0), (64,)])
def test_invalid_branch_positions_are_rejected(positions):
    record = build_record()
    invalid = record.model_copy(
        update={
            "branch_plan": record.branch_plan.model_copy(
                update={"positions": positions}
            )
        }
    )
    with pytest.raises(RecordValidationError, match="branch positions"):
        validate_record(invalid, ReversiPyGame())


def test_branch_position_at_root_empty_count_is_rejected():
    root = position((".WBBBBBB",) + ("BBBBBBBB",) * 7, 1, (0,))
    record = build_record(root=root, positions=(0,))
    invalid = record.model_copy(
        update={"branch_plan": BranchPlan(width=2, positions=(1,))}
    )
    with pytest.raises(RecordValidationError, match="branch positions range"):
        validate_record(invalid, ReversiPyGame())


def test_independent_record_cannot_have_branches():
    record = build_record().model_copy(update={"mode": "independent"})
    with pytest.raises(RecordValidationError, match="independent branching"):
        validate_record(record, ReversiPyGame())


@pytest.mark.parametrize(
    ("kind", "message"),
    [
        ("width", "group width"),
        ("duplicate", "duplicate group member"),
        ("absent", "missing member"),
        ("coverage", "group coverage"),
        ("parent-absent", "group parent"),
        ("parent-changed", "group coverage"),
        ("distinct", "distinct actions"),
    ],
)
def test_bad_sibling_group_metadata(kind, message):
    record = build_record(positions=(0,))
    members = record.sibling_groups[0].edge_ids
    if kind == "width":
        updates = {"edge_ids": members[:1]}
    elif kind == "duplicate":
        updates = {"edge_ids": (members[0], members[0])}
    elif kind == "absent":
        updates = {"edge_ids": (members[0], "absent")}
    elif kind == "coverage":
        updates = {"edge_ids": (members[0], record.edges[1].edge_id)}
    elif kind == "parent-absent":
        updates = {"parent_node_id": "absent"}
    elif kind == "parent-changed":
        updates = {"parent_node_id": record.nodes[1].node_id}
    else:
        assert (
            len({edge.action for edge in record.edges if edge.edge_id in members}) == 1
        )
        updates = {"distinct_action_count": 2}
    invalid = replace_group(record, 0, **updates)
    with pytest.raises(RecordValidationError, match=message):
        validate_record(invalid, ReversiPyGame())


def test_multiple_groups_for_one_parent_are_rejected():
    record = build_record(positions=(0,))
    extra = record.sibling_groups[0].model_copy(update={"group_id": "extra"})
    invalid = record.model_copy(
        update={"sibling_groups": record.sibling_groups + (extra,)}
    )
    with pytest.raises(RecordValidationError, match="duplicate parent group"):
        validate_record(invalid, ReversiPyGame())


@pytest.mark.parametrize(
    "updates", [{"sample_index": 0}, {"sibling_group_id": "absent"}]
)
def test_sibling_member_metadata_must_match_group(updates):
    record = build_record(positions=(0,))
    second = record.sibling_groups[0].edge_ids[1]
    index = next(
        index for index, edge in enumerate(record.edges) if edge.edge_id == second
    )
    invalid = replace_edge(record, index, **updates)
    with pytest.raises(RecordValidationError, match="group member mismatch"):
        validate_record(invalid, ReversiPyGame())


def test_sibling_member_order_matches_sample_indices():
    record = build_record(positions=(0,))
    invalid = replace_group(record, 0, edge_ids=record.sibling_groups[0].edge_ids[::-1])
    with pytest.raises(RecordValidationError, match="group member mismatch"):
        validate_record(invalid, ReversiPyGame())


def test_missing_planned_branch_is_rejected():
    record = build_record(positions=(0,)).model_copy(update={"sibling_groups": ()})
    with pytest.raises(RecordValidationError, match="missing branch"):
        validate_record(record, ReversiPyGame())


def test_moved_branch_plan_is_rejected():
    record = build_record(positions=(0,))
    invalid = record.model_copy(
        update={"branch_plan": BranchPlan(width=2, positions=(1,))}
    )
    with pytest.raises(RecordValidationError, match="unexpected branch"):
        validate_record(invalid, ReversiPyGame())


@pytest.mark.parametrize(
    "updates",
    [
        {"sibling_group_id": "absent"},
        {"sample_index": 0},
    ],
)
def test_normal_edges_cannot_carry_group_metadata(updates):
    record = replace_edge(build_record(positions=()), 0, **updates)
    with pytest.raises(RecordValidationError, match="ungrouped edge metadata"):
        validate_record(record, ReversiPyGame())


def test_identical_samples_cannot_be_collapsed_to_one():
    record = build_record(positions=(0,))
    by_edge = {edge.edge_id: edge for edge in record.edges}
    first, second = [by_edge[key] for key in record.sibling_groups[0].edge_ids]
    assert first.action == second.action
    by_node = {node.node_id: node for node in record.nodes}
    assert by_node[first.child_node_id].state == by_node[second.child_node_id].state
    removed_nodes, pending = set(), [second.child_node_id]
    while pending:
        node_id = pending.pop()
        removed_nodes.add(node_id)
        pending.extend(
            edge.child_node_id
            for edge in record.edges
            if edge.parent_node_id == node_id
        )
    invalid = record.model_copy(
        update={
            "nodes": tuple(
                node for node in record.nodes if node.node_id not in removed_nodes
            ),
            "edges": tuple(
                edge
                for edge in record.edges
                if edge != second and edge.parent_node_id not in removed_nodes
            ),
            "sibling_groups": (
                record.sibling_groups[0].model_copy(
                    update={"edge_ids": (first.edge_id,)}
                ),
            ),
        }
    )
    with pytest.raises(RecordValidationError, match="group width"):
        validate_record(invalid, ReversiPyGame())


def replace_stream(record, index, **updates):
    streams = list(record.rng.streams)
    streams[index] = streams[index].model_copy(update=updates)
    return record.model_copy(
        update={"rng": record.rng.model_copy(update={"streams": tuple(streams)})}
    )


def replace_sampling(record, index, **updates):
    return replace_edge(
        record, index, sampling=record.edges[index].sampling.model_copy(update=updates)
    )


@pytest.mark.parametrize("stream_id", ["branch", "action", "sibling"])
def test_missing_primary_rng_stream_is_rejected(stream_id):
    record = build_record()
    streams = tuple(
        stream for stream in record.rng.streams if stream.stream_id != stream_id
    )
    invalid = record.model_copy(
        update={"rng": record.rng.model_copy(update={"streams": streams})}
    )
    with pytest.raises(RecordValidationError, match="missing RNG stream"):
        validate_record(invalid, ReversiPyGame())


@pytest.mark.parametrize("index", [0, 1, 2])
@pytest.mark.parametrize("field", ["seed", "purpose"])
def test_primary_rng_seed_and_purpose_match_run_seed(index, field):
    record = build_record()
    stream = record.rng.streams[index]
    updates = (
        {"seed": stream.seed + 1}
        if field == "seed"
        else {
            "purpose": "action_sampling"
            if stream.purpose != "action_sampling"
            else "sibling_rollout",
        }
    )
    invalid = replace_stream(record, index, **updates)
    with pytest.raises(RecordValidationError, match="primary RNG stream"):
        validate_record(invalid, ReversiPyGame())


def test_continuation_stream_requires_sibling_purpose():
    record = replace_stream(build_record(), 3, purpose="action_sampling")
    with pytest.raises(RecordValidationError, match="continuation purpose"):
        validate_record(record, ReversiPyGame())


@pytest.mark.parametrize(
    ("stream_id", "message"),
    [
        ("absent", "sampling stream"),
        ("branch", "sampling purpose"),
    ],
)
def test_sampling_stream_reference_and_purpose(stream_id, message):
    record = replace_sampling(build_record(), 0, stream_id=stream_id)
    with pytest.raises(RecordValidationError, match=message):
        validate_record(record, ReversiPyGame())


def test_siblings_cannot_reuse_an_action_draw_event():
    record = build_record(positions=(0,))
    members = record.sibling_groups[0].edge_ids
    indices = [
        next(index for index, edge in enumerate(record.edges) if edge.edge_id == key)
        for key in members
    ]
    first, second = indices
    assert (
        record.edges[first].sampling.stream_id
        == record.edges[second].sampling.stream_id
    )
    invalid = replace_sampling(
        record, second, draw_index=record.edges[first].sampling.draw_index
    )
    with pytest.raises(RecordValidationError, match="duplicate draw"):
        validate_record(invalid, ReversiPyGame())


@pytest.mark.parametrize("seed", [None, 999999])
def test_branch_sample_requires_matching_continuation_stream(seed):
    record = build_record()
    assert seed is None or seed not in {stream.seed for stream in record.rng.streams}
    invalid = replace_sampling(record, 0, continuation_seed=seed)
    with pytest.raises(RecordValidationError, match="missing continuation stream"):
        validate_record(invalid, ReversiPyGame())


def test_primary_sibling_stream_does_not_substitute_for_additional_continuation():
    record = build_record()
    invalid = replace_sampling(record, 0, continuation_seed=record.rng.streams[2].seed)
    with pytest.raises(RecordValidationError, match="missing continuation stream"):
        validate_record(invalid, ReversiPyGame())


def test_siblings_require_distinct_continuation_seeds():
    record = build_record(positions=(0,))
    second = record.sibling_groups[0].edge_ids[1]
    index = next(
        index for index, edge in enumerate(record.edges) if edge.edge_id == second
    )
    invalid = replace_sampling(
        record, index, continuation_seed=record.edges[0].sampling.continuation_seed
    )
    with pytest.raises(RecordValidationError, match="duplicate continuation seed"):
        validate_record(invalid, ReversiPyGame())


def test_normal_edge_cannot_have_continuation_seed():
    record = replace_sampling(
        build_record(mode="independent", positions=()), 0, continuation_seed=1000
    )
    with pytest.raises(RecordValidationError, match="normal edge continuation"):
        validate_record(record, ReversiPyGame())
