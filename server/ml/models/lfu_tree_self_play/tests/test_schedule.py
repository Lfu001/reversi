"""Branch schedules obey the research contract before any game is played."""

import random
from collections import Counter

import pytest
from tests.test_records import build_record, position

from lfu_tree_self_play.game import ReversiPyGame
from lfu_tree_self_play.records import BranchPlan
from lfu_tree_self_play.rng import create_rng_streams


@pytest.mark.parametrize("strategy", ["uniform", "stratified", "fixed"])
@pytest.mark.parametrize("levels", [0, 1, 2])
def test_zero_empty_squares_never_schedule_a_branch(strategy, levels):
    from lfu_tree_self_play.schedule import sample_branch_plan

    rng = random.Random(7)
    before = rng.getstate()
    plan = sample_branch_plan(0, rng, strategy=strategy, levels=levels)
    assert plan.positions == ()
    assert plan.width == 4
    assert rng.getstate() == before


@pytest.mark.parametrize("strategy", ["uniform", "stratified"])
@pytest.mark.parametrize("levels, expected", [(0, ()), (1, (0,)), (2, (0,))])
def test_one_empty_square_caps_the_number_of_random_branches(
    strategy, levels, expected
):
    from lfu_tree_self_play.schedule import sample_branch_plan

    assert (
        sample_branch_plan(
            1, random.Random(7), strategy=strategy, levels=levels
        ).positions
        == expected
    )


@pytest.mark.parametrize("strategy", ["uniform", "stratified", "fixed"])
def test_no_branch_condition_does_not_consume_position_rng(strategy):
    from lfu_tree_self_play.schedule import sample_branch_plan

    rng = random.Random(7)
    before = rng.getstate()
    assert sample_branch_plan(60, rng, strategy=strategy, levels=0).positions == ()
    assert rng.getstate() == before


@pytest.mark.parametrize(
    "empty_squares, levels, expected",
    [
        (1, 2, ()),
        (20, 2, ()),
        (21, 2, (20,)),
        (40, 2, (20,)),
        (41, 2, (20, 40)),
        (60, 1, (20,)),
        (60, 2, (20, 40)),
    ],
)
def test_fixed_positions_outside_candidate_range_are_not_replaced(
    empty_squares, levels, expected
):
    from lfu_tree_self_play.schedule import sample_branch_plan

    rng = random.Random(7)
    before = rng.getstate()
    plan = sample_branch_plan(empty_squares, rng, strategy="fixed", levels=levels)
    assert plan.positions == expected
    assert rng.getstate() == before


@pytest.mark.parametrize("strategy", ["uniform", "stratified", "fixed"])
@pytest.mark.parametrize("levels", [1, 2])
def test_plans_are_sorted_unique_bounded_and_reproducible(strategy, levels):
    from lfu_tree_self_play.schedule import sample_branch_plan

    first, second = create_rng_streams(42), create_rng_streams(42)
    for empty_squares in range(65):
        # Action draws must not influence the next tree's position selection.
        for _ in range(5):
            first.action_sampling.random()
        plan = sample_branch_plan(
            empty_squares, first.branch_position, strategy=strategy, levels=levels
        )
        replay = sample_branch_plan(
            empty_squares, second.branch_position, strategy=strategy, levels=levels
        )
        assert plan == replay
        assert plan.positions == tuple(sorted(set(plan.positions)))
        assert all(0 <= d < empty_squares for d in plan.positions)
        assert len(plan.positions) <= levels
        if strategy != "fixed":
            assert len(plan.positions) == min(empty_squares, levels)
    assert first.branch_position.getstate() == second.branch_position.getstate()


@pytest.mark.parametrize("levels", [1, 2])
def test_uniform_selection_covers_all_positions_and_pairs(levels):
    from lfu_tree_self_play.schedule import sample_branch_plan

    rng = random.Random(11)
    counts = Counter(
        sample_branch_plan(4, rng, levels=levels).positions for _ in range(6000)
    )
    expected = (
        {(0,), (1,), (2,), (3,)}
        if levels == 1
        else {(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)}
    )
    assert set(counts) == expected
    target = 6000 / len(expected)
    assert all(abs(count - target) < 150 for count in counts.values())


@pytest.mark.parametrize("levels", [1, 2])
def test_stratification_selects_intervals_equally_then_positions_uniformly(levels):
    from lfu_tree_self_play.schedule import sample_branch_plan

    # H=8 gives [0,3), [3,6), [6,8): interval probabilities, not
    # individual-position probabilities, are equal despite unequal sizes.
    rng = random.Random(19)
    selections, positions = Counter(), Counter()
    for _ in range(9000):
        plan = sample_branch_plan(8, rng, strategy="stratified", levels=levels)
        intervals = tuple(0 if d < 3 else 1 if d < 6 else 2 for d in plan.positions)
        assert len(set(intervals)) == levels
        selections[intervals] += 1
        positions.update(plan.positions)
    expected = {(0,), (1,), (2,)} if levels == 1 else {(0, 1), (0, 2), (1, 2)}
    assert set(selections) == expected
    assert all(abs(count - 3000) < 200 for count in selections.values())
    for interval, target in [
        ((0, 1, 2), 1000 * levels),
        ((3, 4, 5), 1000 * levels),
        ((6, 7), 1500 * levels),
    ]:
        assert all(abs(positions[d] - target) < 200 for d in interval)


def test_branch_position_draws_leave_action_rng_unchanged():
    from lfu_tree_self_play.schedule import sample_branch_plan

    first, second = create_rng_streams(7), create_rng_streams(7)
    for _ in range(100):
        sample_branch_plan(60, first.branch_position)
    assert [first.action_sampling.random() for _ in range(20)] == [
        second.action_sampling.random() for _ in range(20)
    ]


@pytest.mark.parametrize("empty_squares", [-1, 65, True, 1.5, "60"])
def test_invalid_candidate_count_is_rejected_without_consuming_rng(empty_squares):
    from lfu_tree_self_play.schedule import sample_branch_plan

    rng = random.Random(7)
    before = rng.getstate()
    with pytest.raises(ValueError, match="empty_squares"):
        sample_branch_plan(empty_squares, rng)
    assert rng.getstate() == before


@pytest.mark.parametrize("levels", [-1, 3, True, 1.5, "2"])
def test_invalid_branch_level_is_rejected(levels):
    from lfu_tree_self_play.schedule import sample_branch_plan

    with pytest.raises(ValueError, match="levels"):
        sample_branch_plan(60, random.Random(7), levels=levels)


def test_unknown_selection_strategy_is_rejected_even_without_candidates():
    from lfu_tree_self_play.schedule import sample_branch_plan

    with pytest.raises(ValueError, match="strategy"):
        sample_branch_plan(0, random.Random(7), strategy="adaptive")


def test_each_path_consumes_a_scheduled_position_only_once():
    from lfu_tree_self_play.schedule import BranchPath

    state = ReversiPyGame().initial_state()
    original = BranchPath(BranchPlan(width=4, positions=(0, 1)))
    branch, advanced = original.visit(state, 0)
    assert branch is True
    assert original.executed_positions == ()
    assert advanced.executed_positions == (0,)
    assert advanced.planned_count == 2
    assert advanced.executed_count == 1
    assert advanced.unexecuted_positions == (1,)
    repeated, unchanged = advanced.visit(state, 0)
    assert repeated is False
    assert unchanged == advanced


def test_sibling_paths_inherit_the_plan_but_execute_independently():
    from lfu_tree_self_play.schedule import BranchPath

    game = ReversiPyGame()
    root = game.initial_state()
    _, parent = BranchPath(BranchPlan(width=4, positions=(0, 1))).visit(root, 0)
    first, first_path = parent.visit(game.step(root, 19), 1)
    second, second_path = parent.visit(game.step(root, 26), 1)
    assert first is second is True
    assert first_path.executed_positions == second_path.executed_positions == (0, 1)
    assert parent.executed_positions == (0,)


def test_early_terminal_keeps_unreached_position_without_replacement():
    from lfu_tree_self_play.schedule import BranchPath, sample_branch_plan

    root = position((".WBBBBBB",) + ("BBBBBBBB",) * 6 + ("BBBBBBB.",), 1, (0,))
    plan = sample_branch_plan(2, random.Random(7))
    _, progress = BranchPath(plan).visit(root, 0)
    terminal = ReversiPyGame().step(root, 0)
    assert terminal.is_terminal
    branch, finished = progress.visit(terminal, 1)
    assert branch is False
    assert finished.plan.positions == (0, 1)
    assert finished.executed_positions == (0,)
    assert finished.unexecuted_positions == (1,)


def test_full_terminal_root_has_no_executed_branch():
    from lfu_tree_self_play.schedule import BranchPath, sample_branch_plan

    root = position(("BBBBBBBB",) * 8, 1, ())
    path = BranchPath(sample_branch_plan(0, random.Random(7)))
    branch, finished = path.visit(root, 0)
    assert branch is False
    assert finished.planned_count == finished.executed_count == 0


def test_forced_pass_does_not_add_a_position_or_repeat_a_branch():
    from lfu_tree_self_play.schedule import BranchPath

    before = position(
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
    path = BranchPath(BranchPlan(width=4, positions=(1, 2)))
    assert path.visit(before, 0)[0] is False
    after_pass = ReversiPyGame().step(before, 13)
    assert after_pass.to_play == before.to_play == -1
    branch, progress = path.visit(after_pass, 1)
    assert branch is True
    assert progress.executed_positions == (1,)
    assert progress.visit(after_pass, 1)[0] is False
    next_state = ReversiPyGame().step(after_pass, 0)
    assert not next_state.is_terminal
    branch, progress = progress.visit(next_state, 2)
    assert branch is True
    assert progress.executed_positions == (1, 2)


@pytest.mark.parametrize("depth", [-1, True, 1.5, "1"])
def test_path_rejects_invalid_placement_depth(depth):
    from lfu_tree_self_play.schedule import BranchPath

    with pytest.raises(ValueError, match="placement_depth"):
        BranchPath(BranchPlan(width=4, positions=())).visit(
            ReversiPyGame().initial_state(), depth
        )


@pytest.mark.parametrize("positions", [(1, 0), (0, 0)])
def test_path_rejects_unordered_or_duplicate_plan_positions(positions):
    from lfu_tree_self_play.schedule import BranchPath

    with pytest.raises(ValueError, match="plan.positions"):
        BranchPath(BranchPlan(width=4, positions=positions))


@pytest.mark.parametrize("executed", [(1,), (0, 0), (True,), [0]])
def test_path_rejects_invalid_execution_history(executed):
    from lfu_tree_self_play.schedule import BranchPath

    with pytest.raises(ValueError, match="executed_positions"):
        BranchPath(BranchPlan(width=4, positions=(0,)), executed)


def test_summary_counts_actual_groups_at_each_planned_position():
    from lfu_tree_self_play.schedule import summarize_branch_execution

    record = build_record(width=4)
    summary = summarize_branch_execution(record, ReversiPyGame())
    assert summary.planned_positions == (0, 1)
    assert summary.planned_by_position == (1, 4)
    assert summary.planned_count == 5
    assert summary.executions_by_position == (1, 4)
    assert summary.executed_count == 5


def test_summary_keeps_zero_count_for_unreached_branch():
    from lfu_tree_self_play.schedule import (
        sample_branch_plan,
        summarize_branch_execution,
    )

    root = position((".WBBBBBB",) + ("BBBBBBBB",) * 6 + ("BBBBBBB.",), 1, (0,))
    plan = sample_branch_plan(2, random.Random(7))
    record = build_record(root=root, positions=plan.positions, width=plan.width)
    summary = summarize_branch_execution(record, ReversiPyGame())
    assert summary.planned_positions == (0, 1)
    assert summary.planned_by_position == (1, 4)
    assert summary.planned_count == 5
    assert summary.executions_by_position == (1, 0)
    assert summary.executed_count == 1
    assert record.branch_plan == plan


def test_independent_record_summary_has_no_scheduled_or_executed_branch():
    from lfu_tree_self_play.schedule import summarize_branch_execution

    record = build_record(mode="independent", positions=())
    summary = summarize_branch_execution(record, ReversiPyGame())
    assert summary.planned_positions == summary.executions_by_position == ()
    assert summary.planned_count == summary.executed_count == 0


def test_summary_rejects_missing_required_branch_instead_of_reporting_zero():
    from lfu_tree_self_play.record_validation import RecordValidationError
    from lfu_tree_self_play.schedule import summarize_branch_execution

    record = build_record().model_copy(update={"sibling_groups": ()})
    with pytest.raises(RecordValidationError, match="missing branch"):
        summarize_branch_execution(record, ReversiPyGame())


def test_saved_early_terminal_record_preserves_planned_and_executed_counts(tmp_path):
    from lfu_tree_self_play.dataset_store import DatasetStore
    from lfu_tree_self_play.returns import aggregate_returns
    from lfu_tree_self_play.schedule import (
        sample_branch_plan,
        summarize_branch_execution,
    )

    root = position((".WBBBBBB",) + ("BBBBBBBB",) * 6 + ("BBBBBBB.",), 1, (0,))
    plan = sample_branch_plan(2, create_rng_streams(7).branch_position)
    record = build_record(root=root, positions=plan.positions, width=plan.width)
    game = ReversiPyGame()
    store = DatasetStore(tmp_path, experiment_id=record.experiment_id, game=game)
    store.publish(record, aggregate_returns(record, game))
    loaded = store.load(record.unit_id).record
    assert loaded.branch_plan == plan
    summary = summarize_branch_execution(loaded, game)
    assert summary.planned_count == 5
    assert summary.executed_count == 1
    assert summary.executions_by_position == (1, 0)


@pytest.mark.parametrize("strategy", ["uniform", "stratified", "fixed"])
def test_schedule_and_path_history_match_complete_real_tree(strategy):
    from lfu_tree_self_play.schedule import BranchPath, sample_branch_plan

    game = ReversiPyGame()
    root = game.initial_state()
    plan = sample_branch_plan(
        60, create_rng_streams(7).branch_position, strategy=strategy
    )
    record = build_record(root=root, positions=plan.positions, width=plan.width)
    group_parents = {group.parent_node_id for group in record.sibling_groups}
    outgoing = {}
    for edge in record.edges:
        outgoing.setdefault(edge.parent_node_id, []).append(edge.child_node_id)
    histories = {record.root_node_id: BranchPath(plan)}
    for node in record.nodes:
        branch, progress = histories[node.node_id].visit(
            node.state, node.placement_depth
        )
        assert branch == (node.node_id in group_parents)
        for child in outgoing.get(node.node_id, ()):
            histories[child] = progress
    assert all(
        node.state.is_terminal for node in record.nodes if node.node_id not in outgoing
    )
