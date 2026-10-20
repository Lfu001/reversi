"""Acceptance tests for pre-scheduled width-four tree self-play collection."""

import math
import random

import numpy as np
import pytest
import torch
from tests.test_records import position

from lfu_tree_self_play.dataset_store import DatasetStore
from lfu_tree_self_play.game import GameState, ReversiPyGame
from lfu_tree_self_play.model import PolicyValueConfig, PolicyValueModel
from lfu_tree_self_play.record_validation import validate_record
from lfu_tree_self_play.returns import aggregate_returns
from lfu_tree_self_play.rng import create_rng_streams
from lfu_tree_self_play.roots import RootEntry, generate_root_position
from lfu_tree_self_play.schedule import summarize_branch_execution
from lfu_tree_self_play.tree_collector import TreeCollector, collect_tree
from lfu_tree_self_play.verification import endgame_position


def small_model(seed: int = 15) -> PolicyValueModel:
    torch.manual_seed(seed)
    return PolicyValueModel(PolicyValueConfig(width=8, num_blocks=1)).eval()


@pytest.mark.parametrize("strategy", ["uniform", "stratified", "fixed"])
@pytest.mark.parametrize("levels", [0, 1, 2])
def test_tree_collector_builds_valid_complete_trees_across_schedules(
    strategy, levels
) -> None:
    game = ReversiPyGame()
    model = small_model()
    root = endgame_position(1) if strategy != "fixed" else game.initial_state()
    tree = collect_tree(
        model,
        root,
        unit_id=f"tree-{strategy}-{levels}",
        experiment_id="exp-87",
        round_id="round-0",
        model_generation=2,
        run_seed=87,
        levels=levels,
        strategy=strategy,
        game=game,
    )

    validate_record(tree.record, game)
    assert tree.record.mode == "tree"
    assert tree.record.branch_plan.width == 4
    assert len(tree.record.branch_plan.positions) == levels
    assert 1 <= tree.terminal_leaf_count <= 4**levels
    assert tree.targets == aggregate_returns(tree.record, game)
    assert tree.branch_execution == summarize_branch_execution(tree.record, game)
    assert tree.planned_branch_count == sum(4**i for i in range(levels))
    assert 0 <= tree.executed_branch_count <= tree.planned_branch_count
    assert tree.record.cost.environment_transitions == len(tree.record.edges)
    assert tree.record.cost.evaluated_positions == sum(
        not node.state.is_terminal for node in tree.record.nodes
    )
    assert tree.record.cost.inference_batches == tree.record.cost.evaluated_positions
    assert math.isclose(
        sum(
            tree.targets.node_weights[node.node_id]
            for node in tree.record.nodes
            if node.state.is_terminal
        ),
        1.0,
        rel_tol=0,
        abs_tol=1e-12,
    )


def test_full_two_level_tree_reaches_sixteen_terminal_leaves_and_five_groups() -> None:
    game = ReversiPyGame()
    tree = collect_tree(
        small_model(),
        endgame_position(-1),
        unit_id="sixteen-leaves",
        experiment_id="exp-87",
        round_id="round-0",
        model_generation=0,
        run_seed=19,
        levels=2,
        strategy="uniform",
        game=game,
    )

    assert len(tree.record.branch_plan.positions) == 2
    assert tree.planned_branch_count == 5
    assert tree.executed_branch_count == 5
    assert tree.branch_execution.planned_by_position == (1, 4)
    assert tree.branch_execution.executions_by_position == (1, 4)
    assert len(tree.record.sibling_groups) == 5
    assert tree.terminal_leaf_count == 16


def test_root_position_and_root_entry_produce_identical_trees() -> None:
    model = small_model()
    root_pos = generate_root_position(12, placements=4)
    root_entry = RootEntry(
        purpose="training", run_seed=12, index=0, attempt=0, root=root_pos
    )

    from_state = collect_tree(
        model,
        root_pos.state,
        unit_id="u0",
        experiment_id="exp",
        round_id="r0",
        model_generation=0,
        run_seed=42,
    )
    from_root_pos = collect_tree(
        model,
        root_pos,
        unit_id="u0",
        experiment_id="exp",
        round_id="r0",
        model_generation=0,
        run_seed=42,
    )
    from_entry = collect_tree(
        model,
        root_entry,
        unit_id="u0",
        experiment_id="exp",
        round_id="r0",
        model_generation=0,
        run_seed=42,
    )

    assert (
        from_state.record.nodes == from_root_pos.record.nodes == from_entry.record.nodes
    )
    assert (
        from_state.record.edges == from_root_pos.record.edges == from_entry.record.edges
    )
    assert (
        from_state.record.sibling_groups
        == from_root_pos.record.sibling_groups
        == from_entry.record.sibling_groups
    )
    assert from_state.targets == from_root_pos.targets == from_entry.targets


def test_position_and_action_rng_streams_are_isolated() -> None:
    model = small_model()
    root = endgame_position(1)
    first_streams = create_rng_streams(99)
    second_streams = create_rng_streams(99)

    # Advancing evaluation RNG must never perturb tree collection.
    for _ in range(50):
        first_streams.evaluation.random()

    first_collector = TreeCollector(
        model,
        experiment_id="exp",
        round_id="r0",
        model_generation=0,
        run_seed=99,
        streams=first_streams,
    )
    second_collector = TreeCollector(
        model,
        experiment_id="exp",
        round_id="r0",
        model_generation=0,
        run_seed=99,
        streams=second_streams,
    )

    first_tree = first_collector.collect_tree(root, unit_id="u1")
    second_tree = second_collector.collect_tree(root, unit_id="u1")
    assert first_tree.record.branch_plan == second_tree.record.branch_plan
    assert first_tree.record.nodes == second_tree.record.nodes
    assert first_tree.record.edges == second_tree.record.edges
    assert first_tree.record.sibling_groups == second_tree.record.sibling_groups

    # Advancing action_sampling and sibling_rollout on first_streams must not
    # affect the next tree's branch_position draw.
    for _ in range(25):
        first_streams.action_sampling.random()
        first_streams.sibling_rollout.getrandbits(64)

    next_first = first_collector.collect_tree(root, unit_id="u2")
    next_second = second_collector.collect_tree(root, unit_id="u2")
    assert next_first.record.branch_plan == next_second.record.branch_plan
    assert (
        first_streams.branch_position.getstate()
        == second_streams.branch_position.getstate()
    )


def test_duplicate_sibling_actions_are_preserved_with_independent_rollouts() -> None:
    # Build a position at depth 0 with only 1 legal move (square 5), followed by
    # multi-action endgame play so all 4 siblings MUST draw the same action at
    # the root branch yet continue on distinct continuation seeds.
    base = endgame_position(1)
    planes = base.observation
    planes[3] = 0
    planes[3, 5 // 8, 5 % 8] = 1
    single_legal_root = GameState(planes)

    # Fix branch positions to (0,) via H=6 and a seed, or levels=1 where position 0 is drawn.
    for seed in range(1, 50):
        streams = create_rng_streams(seed)
        tree = collect_tree(
            small_model(),
            single_legal_root,
            unit_id="dup-siblings",
            experiment_id="exp",
            round_id="r0",
            model_generation=0,
            run_seed=seed,
            streams=streams,
            levels=1,
        )
        if tree.record.branch_plan.positions == (0,):
            break
    else:
        pytest.fail("expected a seed that schedules position 0")

    assert len(tree.record.sibling_groups) == 1
    group = tree.record.sibling_groups[0]
    assert len(group.edge_ids) == 4
    assert group.distinct_action_count == 1

    edge_map = {edge.edge_id: edge for edge in tree.record.edges}
    root_edges = [edge_map[edge_id] for edge_id in group.edge_ids]
    assert [edge.action for edge in root_edges] == [5, 5, 5, 5]
    assert [edge.sample_index for edge in root_edges] == [0, 1, 2, 3]
    assert len({edge.child_node_id for edge in root_edges}) == 4
    continuation_seeds = [edge.sampling.continuation_seed for edge in root_edges]
    assert all(seed is not None for seed in continuation_seeds)
    assert len(set(continuation_seeds)) == 4
    assert tree.terminal_leaf_count == 4


def test_terminal_root_h0_does_not_branch_infer_or_consume_rng() -> None:
    terminal_root = position(("BBBBBBBB",) * 8, 1, ())
    assert terminal_root.is_terminal

    streams = create_rng_streams(7)
    before = (
        streams.branch_position.getstate(),
        streams.action_sampling.getstate(),
        streams.sibling_rollout.getstate(),
        streams.evaluation.getstate(),
    )

    tree = collect_tree(
        small_model(),
        terminal_root,
        unit_id="terminal-root",
        experiment_id="exp",
        round_id="r0",
        model_generation=0,
        run_seed=7,
        streams=streams,
        levels=2,
    )

    after = (
        streams.branch_position.getstate(),
        streams.action_sampling.getstate(),
        streams.sibling_rollout.getstate(),
        streams.evaluation.getstate(),
    )
    assert before == after
    assert tree.record.branch_plan.positions == ()
    assert tree.planned_branch_count == 0
    assert tree.executed_branch_count == 0
    assert tree.terminal_leaf_count == 1
    assert len(tree.record.nodes) == 1
    assert tree.record.edges == ()
    assert tree.record.sibling_groups == ()
    assert tree.record.cost.evaluated_positions == 0
    assert tree.record.cost.inference_batches == 0
    assert tree.record.cost.environment_transitions == 0
    assert tree.record.nodes[0].old_value is None
    assert tree.record.nodes[0].black_result == 1


def test_one_empty_square_h1_caps_random_branches_and_skips_out_of_range_fixed() -> (
    None
):
    one_empty_root = position((".WBBBBBB",) + ("BBBBBBBB",) * 7, 1, (0,))

    random_tree = collect_tree(
        small_model(),
        one_empty_root,
        unit_id="h1-uniform",
        experiment_id="exp",
        round_id="r0",
        model_generation=0,
        run_seed=7,
        levels=2,
        strategy="uniform",
    )
    assert random_tree.record.branch_plan.positions == (0,)
    assert random_tree.planned_branch_count == 1
    assert random_tree.executed_branch_count == 1
    assert random_tree.terminal_leaf_count == 4
    assert random_tree.record.sibling_groups[0].distinct_action_count == 1

    fixed_tree = collect_tree(
        small_model(),
        one_empty_root,
        unit_id="h1-fixed",
        experiment_id="exp",
        round_id="r0",
        model_generation=0,
        run_seed=7,
        levels=2,
        strategy="fixed",
    )
    assert fixed_tree.record.branch_plan.positions == ()
    assert fixed_tree.planned_branch_count == 0
    assert fixed_tree.executed_branch_count == 0
    assert fixed_tree.terminal_leaf_count == 1


def test_early_terminal_records_unreached_planned_branches_without_resampling(
    tmp_path,
) -> None:
    # Two empty squares (0 and 63), but placing at 0 flips the only white disk
    # and leaves square 63 with no legal moves for either player -> terminal at depth 1.
    early_root = position((".WBBBBBB",) + ("BBBBBBBB",) * 6 + ("BBBBBBB.",), 1, (0,))
    game = ReversiPyGame()
    store = DatasetStore(tmp_path, experiment_id="exp-early", game=game)
    collector = TreeCollector(
        small_model(),
        experiment_id="exp-early",
        round_id="r0",
        model_generation=1,
        run_seed=7,
        levels=2,
        strategy="uniform",
        game=game,
    )

    tree, identity = collector.collect_and_publish(
        early_root, store, unit_id="early-unit"
    )
    assert identity.unit_id == "early-unit"
    assert tree.record.branch_plan.positions == (0, 1)
    assert tree.branch_execution.planned_by_position == (1, 4)
    assert tree.planned_branch_count == 5
    assert tree.branch_execution.executions_by_position == (1, 0)
    assert tree.executed_branch_count == 1
    assert tree.terminal_leaf_count == 4

    loaded = store.load("early-unit")
    assert loaded.record == tree.record
    assert loaded.targets == tree.targets
    loaded_summary = summarize_branch_execution(loaded.record, game)
    assert loaded_summary == tree.branch_execution


def test_forced_pass_maintains_placement_depth_and_turn_perspective() -> None:
    # Position where black plays at 0, white has no legal reply (forced pass),
    # and black plays again at depth 1 before reaching terminal at depth 2.
    pass_root = position(
        (
            ".WBBBBBB",
            "BBBBBBBB",
            "BBBBBBBB",
            "BBBBBBBB",
            "BBBBBBBB",
            "BBBBBBBB",
            "BBBBBBBB",
            "BBBBBBW.",
        ),
        1,
        (0,),
    )
    game = ReversiPyGame()
    after_first = game.step(pass_root, 0)
    assert not after_first.is_terminal
    assert after_first.to_play == 1

    tree = collect_tree(
        small_model(),
        pass_root,
        unit_id="forced-pass",
        experiment_id="exp",
        round_id="r0",
        model_generation=0,
        run_seed=11,
        levels=2,
        strategy="uniform",
        game=game,
    )
    validate_record(tree.record, game)
    assert tree.record.branch_plan.positions == (0, 1)
    assert tree.executed_branch_count == 5
    assert tree.terminal_leaf_count == 16
    for node in tree.record.nodes:
        if node.placement_depth in (0, 1):
            assert node.state.to_play == 1


def test_collector_freezes_round_start_model_and_preserves_caller_state() -> None:
    torch.manual_seed(23)
    model = PolicyValueModel(PolicyValueConfig(width=8, num_blocks=1)).train()
    root = endgame_position(1)
    collector = TreeCollector(
        model,
        experiment_id="exp",
        round_id="r0",
        model_generation=0,
        run_seed=23,
    )

    model_snapshot = {k: v.clone() for k, v in model.state_dict().items()}
    py_rng = random.getstate()
    np_rng = np.random.get_state()
    torch_rng = torch.get_rng_state().clone()

    first = collector.collect_tree(root, unit_id="u1")

    assert model.training
    assert all(p.requires_grad for p in model.parameters())
    assert all(p.grad is None for p in model.parameters())
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, model_snapshot[key], rtol=0, atol=0)
    assert random.getstate() == py_rng
    assert np.array_equal(np.random.get_state()[1], np_rng[1])
    assert torch.equal(torch.get_rng_state(), torch_rng)

    # Mutate original model weights and buffers heavily; collector snapshot must not change.
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(50.0)
        for buffer in model.buffers():
            buffer.fill_(25.0)

    replay = TreeCollector(
        small_model(23),
        experiment_id="exp",
        round_id="r0",
        model_generation=0,
        run_seed=23,
    ).collect_tree(root, unit_id="u1")

    assert first.record.nodes == replay.record.nodes
    assert first.record.edges == replay.record.edges
    assert first.targets == replay.targets


def test_invalid_inputs_are_rejected_without_advancing_rng(tmp_path) -> None:
    model = small_model()
    root = endgame_position(1)
    streams = create_rng_streams(31)
    collector = TreeCollector(
        model,
        experiment_id="exp-31",
        round_id="r0",
        model_generation=0,
        run_seed=31,
        streams=streams,
    )
    before = (
        streams.branch_position.getstate(),
        streams.action_sampling.getstate(),
        streams.sibling_rollout.getstate(),
        streams.evaluation.getstate(),
    )

    with pytest.raises(ValueError, match="unit_id"):
        collector.collect_tree(root, unit_id="bad/unit")
    with pytest.raises(ValueError, match="root"):
        collector.collect_tree("not-a-root", unit_id="u1")  # type: ignore[arg-type]

    bad_planes = root.observation
    bad_planes[2] = 0
    with pytest.raises(ValueError, match="turn"):
        collector.collect_tree(GameState(bad_planes), unit_id="u1")

    wrong_store = DatasetStore(
        tmp_path, experiment_id="other-experiment", game=ReversiPyGame()
    )
    with pytest.raises(ValueError, match="mismatch"):
        collector.collect_and_publish(root, wrong_store, unit_id="u1")

    after = (
        streams.branch_position.getstate(),
        streams.action_sampling.getstate(),
        streams.sibling_rollout.getstate(),
        streams.evaluation.getstate(),
    )
    assert before == after

    with pytest.raises(ValueError, match="model"):
        TreeCollector(
            object(),  # type: ignore[arg-type]
            experiment_id="exp",
            round_id="r0",
            model_generation=0,
            run_seed=1,
        )
    with pytest.raises(ValueError, match="levels"):
        TreeCollector(
            model,
            experiment_id="exp",
            round_id="r0",
            model_generation=0,
            run_seed=1,
            levels=3,
        )
    with pytest.raises(ValueError, match="strategy"):
        TreeCollector(
            model,
            experiment_id="exp",
            round_id="r0",
            model_generation=0,
            run_seed=1,
            strategy="adaptive",  # type: ignore[arg-type]
        )
