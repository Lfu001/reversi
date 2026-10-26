"""Replay contracts exercise real evaluation and real Reversi transitions."""

import random
from dataclasses import replace

import numpy as np
import pytest
import torch

from lfu_tree_self_play.batch_inference import FrozenBatchEvaluator
from lfu_tree_self_play.batch_scheduler import BatchScheduler, SchedulerLimits
from lfu_tree_self_play.collection_session import CollectionLane, drain_session
from lfu_tree_self_play.game import GameState, ReversiPyGame
from lfu_tree_self_play.model import PolicyValueConfig, PolicyValueModel
from lfu_tree_self_play.tree_collector import TreeCollector
from lfu_tree_self_play.tree_replay import TreeReplay, compare_trees, finalize_replay


class CountingGame:
    def __init__(self):
        self.game = ReversiPyGame()
        self.steps = 0

    def initial_state(self):
        return self.game.initial_state()

    def step(self, state, action):
        self.steps += 1
        return self.game.step(state, action)


@pytest.fixture(scope="module")
def evaluator():
    torch.manual_seed(89)
    return FrozenBatchEvaluator(
        PolicyValueModel(PolicyValueConfig(width=4, num_blocks=1)),
        model_generation=3,
    )


def endgame():
    game = ReversiPyGame()
    state = game.initial_state()
    while not state.is_terminal and np.count_nonzero(state.observation[:2]) < 60:
        state = game.step(state, state.legal_actions[0])
    return state


def reference_tree(evaluator, *, levels=2, strategy="uniform", root=None):
    return TreeCollector(
        evaluator,
        experiment_id="replay",
        round_id="round",
        model_generation=3,
        run_seed=89,
        levels=levels,
        strategy=strategy,
    ).collect_tree(endgame() if root is None else root, unit_id="tree")


def path_costs(record):
    """Count root paths independently through each terminal's incoming links."""
    nodes = {node.node_id: node for node in record.nodes}
    edges = {edge.edge_id: edge for edge in record.edges}
    steps = 0
    for leaf in record.nodes:
        if leaf.state.is_terminal:
            node = leaf
            while node.incoming_edge_id is not None:
                edge = edges[node.incoming_edge_id]
                steps += 1
                node = nodes[edge.parent_node_id]
    return steps


@pytest.mark.parametrize("levels", [0, 1, 2])
@pytest.mark.parametrize("mode", ["shared", "recompute"])
@pytest.mark.parametrize("strategy", ["uniform", "stratified", "fixed"])
def test_replay_preserves_tree_and_performs_exact_physical_work(
    evaluator, levels, mode, strategy
):
    # Skipping an occurrence or validating inside generation breaks these counters.
    reference = reference_tree(evaluator, levels=levels, strategy=strategy)
    game = CountingGame()
    owner = TreeReplay(evaluator, game=game)
    session = owner.create_session(reference, mode=mode)
    game.steps = 0  # Upfront validation is deliberately outside replay work.
    result = drain_session(session)
    want_steps = (
        len(reference.record.edges)
        if mode == "shared"
        else path_costs(reference.record)
    )
    want_evals = (
        sum(not node.state.is_terminal for node in reference.record.nodes)
        if mode == "shared"
        else want_steps
    )
    assert game.steps == want_steps
    assert result.record.cost.environment_transitions == want_steps
    assert result.record.cost.evaluated_positions == want_evals
    assert result.record.cost.inference_batches == want_evals
    assert result.record.cost.peak_memory_bytes > 0
    actual = finalize_replay(result, ReversiPyGame())
    assert compare_trees(reference, actual)["passed"] is True
    assert result.record.rng == reference.record.rng
    assert result.record.sibling_groups == reference.record.sibling_groups


def test_duplicate_sibling_samples_keep_separate_nodes_and_work(evaluator):
    # Deduplicating identical sibling actions breaks both identity and physical counts.
    reference = reference_tree(evaluator)
    assert any(
        group.distinct_action_count < len(group.edge_ids)
        for group in reference.record.sibling_groups
    )
    shared = drain_session(
        TreeReplay(evaluator).create_session(reference, mode="shared")
    )
    recompute = drain_session(
        TreeReplay(evaluator).create_session(reference, mode="recompute")
    )
    assert shared.record.cost.environment_transitions == len(reference.record.edges)
    assert recompute.record.cost.environment_transitions == path_costs(reference.record)
    assert (
        recompute.record.cost.environment_transitions
        > shared.record.cost.environment_transitions
    )
    assert compare_trees(reference, finalize_replay(recompute, ReversiPyGame()))[
        "passed"
    ]


@pytest.mark.parametrize("mode", ["shared", "recompute"])
def test_forced_pass_keeps_actual_player_turn_and_targets(evaluator, mode):
    # Treating every placement as a turn flip corrupts these real forced passes.
    game = ReversiPyGame()
    root = game.initial_state()
    for _ in range(46):
        root = game.step(root, root.legal_actions[0])
    reference = reference_tree(evaluator, root=root, levels=0)
    nodes = {node.node_id: node for node in reference.record.nodes}
    first = reference.record.edges[0]
    assert (
        nodes[first.parent_node_id].state.to_play
        == nodes[first.child_node_id].state.to_play
    )
    assert not nodes[first.child_node_id].state.is_terminal
    result = drain_session(TreeReplay(evaluator).create_session(reference, mode=mode))
    assert compare_trees(reference, finalize_replay(result, game))["passed"]


@pytest.mark.parametrize("mode", ["shared", "recompute"])
def test_early_terminal_root_performs_no_evaluation_or_step(evaluator, mode):
    # A terminal tree with one unfilled square must not fabricate placement work.
    planes = np.zeros((4, 8, 8), dtype=np.float32)
    planes[0] = 1
    planes[0, 0, 0] = 0
    planes[2] = 1
    root = GameState(planes)
    assert root.is_terminal
    reference = reference_tree(evaluator, root=root)
    assert reference.terminal_leaf_count == 1
    assert reference.executed_branch_count == 0
    game = CountingGame()
    session = TreeReplay(evaluator, game=game).create_session(reference, mode=mode)
    game.steps = 0
    result = drain_session(session)
    assert game.steps == 0
    assert result.record.cost.evaluated_positions == 0
    assert compare_trees(reference, finalize_replay(result, ReversiPyGame()))["passed"]


def test_shared_and_recompute_sessions_run_together_in_batch_scheduler(evaluator):
    # Batching must resume each owner's token stream without interchanging outputs.
    reference = reference_tree(evaluator)
    owners = [TreeReplay(evaluator), TreeReplay(evaluator)]
    sessions = [
        owner.create_session(reference, mode=mode)
        for owner, mode in zip(owners, ["shared", "recompute"], strict=True)
    ]
    completed = []
    scheduler = BatchScheduler(
        evaluator,
        limits=SchedulerLimits(
            max_active_units=2, max_batch_size=2, rss_limit_bytes=8_000_000_000
        ),
    )
    metrics = scheduler.run(
        [
            CollectionLane(owner, iter([session]))
            for owner, session in zip(owners, sessions, strict=True)
        ],
        on_complete=completed.append,
    )
    assert metrics.completed_units == 2
    assert metrics.batch_sizes.get(2, 0) > 0
    for result in completed:
        assert compare_trees(reference, finalize_replay(result, ReversiPyGame()))[
            "passed"
        ]


def test_each_recomputed_occurrence_is_checked_against_first_value(evaluator):
    # Opposite sub-tolerance deviations are each near reference, yet disagree with
    # one another by more than tolerance and must not be silently deduplicated.
    reference = reference_tree(evaluator)
    root = reference.record.nodes[0].state

    class DriftingEvaluator:
        model_generation = evaluator.model_generation
        root_calls = 0

        def evaluate(self, requests):
            outputs = evaluator.evaluate(requests)
            changed = []
            for request, output in zip(requests, outputs, strict=True):
                if request.state == root:
                    self.root_calls += 1
                    output = replace(
                        output,
                        old_value=output.old_value
                        + (0.75e-6 if self.root_calls == 1 else -0.75e-6),
                    )
                changed.append(output)
            return tuple(changed)

    owner = TreeReplay(DriftingEvaluator())
    with pytest.raises(ValueError, match="repeated node"):
        drain_session(owner.create_session(reference, mode="recompute"))
    # Failure releases ownership even when the caller never aborts explicitly.
    owner.create_session(reference, mode="shared").abort()


def test_replay_does_not_consume_random_state(evaluator):
    reference = reference_tree(evaluator)
    before = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())
    drain_session(TreeReplay(evaluator).create_session(reference, mode="recompute"))
    assert random.getstate() == before[0]
    after_numpy = np.random.get_state()
    assert after_numpy[0] == before[1][0]
    np.testing.assert_array_equal(after_numpy[1], before[1][1])
    assert after_numpy[2:] == before[1][2:]
    assert torch.equal(torch.get_rng_state(), before[2])


def test_replay_leaves_source_model_parameters_gradients_and_modes_unchanged():
    # Replay must not update the supplied training model through its evaluator.
    model = PolicyValueModel(PolicyValueConfig(width=4, num_blocks=1))
    model.train()
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    frozen = FrozenBatchEvaluator(model, model_generation=3)
    reference = reference_tree(frozen)
    weights = {name: value.clone() for name, value in model.state_dict().items()}
    gradients = [parameter.grad.clone() for parameter in model.parameters()]
    modes = [module.training for module in model.modules()]
    for mode in ("shared", "recompute"):
        drain_session(TreeReplay(frozen).create_session(reference, mode=mode))
    for name, value in model.state_dict().items():
        assert torch.equal(value, weights[name])
    for parameter, gradient in zip(model.parameters(), gradients, strict=True):
        assert torch.equal(parameter.grad, gradient)
    assert [module.training for module in model.modules()] == modes


def test_owner_releases_after_abort_completion_and_failure(evaluator):
    reference = reference_tree(evaluator)
    owner = TreeReplay(evaluator)
    first = owner.create_session(reference, mode="shared")
    with pytest.raises(RuntimeError):
        owner.create_session(reference, mode="shared")
    first.abort()
    drain_session(owner.create_session(reference, mode="shared"))
    corrupt = replace(
        reference,
        record=reference.record.model_copy(
            update={
                "nodes": tuple(
                    node.model_copy(update={"old_value": 0.99})
                    if not node.state.is_terminal
                    else node
                    for node in reference.record.nodes
                ),
            }
        ),
    )
    with pytest.raises(ValueError):
        drain_session(owner.create_session(corrupt, mode="shared"))
    drain_session(owner.create_session(reference, mode="shared"))


def test_generation_and_mode_fail_before_any_session_work(evaluator):
    reference = reference_tree(evaluator)
    stale = replace(
        reference, record=reference.record.model_copy(update={"model_generation": 4})
    )
    owner = TreeReplay(evaluator)
    with pytest.raises(ValueError):
        owner.create_session(stale, mode="shared")
    with pytest.raises(ValueError):
        owner.create_session(reference, mode="invalid")
    drain_session(owner.create_session(reference, mode="shared"))


def test_compare_rejects_discrete_weight_and_numeric_mismatches(evaluator):
    reference = reference_tree(evaluator)
    changed_record = reference.record.model_copy(update={"unit_id": "different"})
    with pytest.raises(ValueError):
        compare_trees(reference, replace(reference, record=changed_record))
    weights = dict(reference.targets.node_weights)
    weights[reference.record.root_node_id] = 0.5
    with pytest.raises(ValueError):
        compare_trees(
            reference,
            replace(
                reference, targets=replace(reference.targets, node_weights=weights)
            ),
        )
    nodes = list(reference.record.nodes)
    nodes[0] = nodes[0].model_copy(update={"old_value": nodes[0].old_value + 1e-4})
    with pytest.raises(ValueError):
        compare_trees(
            reference,
            replace(
                reference,
                record=reference.record.model_copy(update={"nodes": tuple(nodes)}),
            ),
        )
    edges = list(reference.record.edges)
    edges[0] = edges[0].model_copy(
        update={"old_log_probability": edges[0].old_log_probability - 1e-4}
    )
    with pytest.raises(ValueError):
        compare_trees(
            reference,
            replace(
                reference,
                record=reference.record.model_copy(update={"edges": tuple(edges)}),
            ),
        )


def test_compare_reports_permitted_numerical_errors(evaluator):
    # Exact float comparison would reject acceptable batching roundoff.
    reference = reference_tree(evaluator)
    nodes = list(reference.record.nodes)
    nodes[0] = nodes[0].model_copy(update={"old_value": nodes[0].old_value + 0.5e-6})
    edges = list(reference.record.edges)
    edges[0] = edges[0].model_copy(
        update={"old_log_probability": edges[0].old_log_probability - 0.5e-6}
    )
    actual = replace(
        reference,
        record=reference.record.model_copy(
            update={"nodes": tuple(nodes), "edges": tuple(edges)}
        ),
    )
    comparison = compare_trees(reference, actual)
    assert comparison["passed"] is True
    assert comparison["max_value_error"] == pytest.approx(0.5e-6)
    assert comparison["max_log_probability_error"] == pytest.approx(0.5e-6)


@pytest.mark.parametrize(
    "target_field",
    [
        "node_black_returns",
        "edge_black_returns",
        "node_weights",
        "edge_weights",
        "node_player_targets",
        "edge_parent_targets",
    ],
)
def test_compare_rejects_any_changed_target_mapping(evaluator, target_field):
    # A comparison that skips any derived target map can certify wrong training data.
    reference = reference_tree(evaluator)
    changed = dict(getattr(reference.targets, target_field))
    key = next(iter(changed))
    changed[key] += 0.01
    targets = replace(reference.targets, **{target_field: changed})
    with pytest.raises(ValueError):
        compare_trees(reference, replace(reference, targets=targets))
