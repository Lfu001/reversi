"""Resuming collection must preserve the sequential draw order and cleanup."""

import random

import pytest
import torch

from lfu_tree_self_play.batch_inference import FrozenBatchEvaluator
from lfu_tree_self_play.collection_session import SessionComplete, drain_session
from lfu_tree_self_play.collector import IndependentCollector
from lfu_tree_self_play.game import ReversiPyGame
from lfu_tree_self_play.model import PolicyValueConfig, PolicyValueModel
from lfu_tree_self_play.tree_collector import TreeCollector
from lfu_tree_self_play.verification import endgame_position


def make_collector(kind, seed=88):
    torch.manual_seed(7)
    evaluator = FrozenBatchEvaluator(
        PolicyValueModel(PolicyValueConfig(width=8, num_blocks=1)), model_generation=2
    )
    kwargs = {
        "experiment_id": "exp",
        "round_id": "round",
        "model_generation": 2,
        "run_seed": seed,
    }
    if kind == "independent":
        return IndependentCollector(evaluator, ReversiPyGame(), **kwargs)
    return TreeCollector(evaluator, **kwargs)


@pytest.mark.parametrize("kind", ["independent", "tree"])
def test_serial_drain_matches_legacy_contract(kind):
    first, second = make_collector(kind), make_collector(kind)
    root = endgame_position(1)
    if kind == "independent":
        reference = first.collect_game(root, unit_id="unit")
        session = second.create_session(root, unit_id="unit")
    else:
        reference = first.collect_tree(root, unit_id="unit")
        session = second.create_session(root, unit_id="unit")
    actual = drain_session(session)
    assert actual.record.nodes == reference.record.nodes
    assert actual.record.edges == reference.record.edges
    assert actual.record.rng == reference.record.rng
    assert actual.record.sibling_groups == reference.record.sibling_groups
    assert actual.targets == reference.targets


def test_independent_lane_preserves_global_draw_order_and_one_over_k_weights():
    first, second = make_collector("independent"), make_collector("independent")
    root = endgame_position(1)
    expected = first.collect_paths(root, paths_per_root=3, unit_ids=["a", "b", "c"])
    actual = [
        drain_session(session)
        for session in second.iter_sessions(
            root, paths_per_root=3, unit_ids=["a", "b", "c"]
        )
    ]
    assert second.draw_index == first.draw_index
    for left, right in zip(actual, expected.units, strict=True):
        assert left.record.edges == right.record.edges
        assert left.weighted_targets == right.weighted_targets
        assert left.path_weight == 1 / 3


@pytest.mark.parametrize("kind", ["independent", "tree"])
def test_abort_restores_session_entry_rng(kind):
    collector = make_collector(kind)
    root = endgame_position(1)
    session = collector.create_session(root, unit_id="unit")
    request = session.poll()
    session.resume(collector.evaluator.evaluate([request])[0])
    with pytest.raises(RuntimeError, match="active"):
        collector.create_session(root, unit_id="other")
    session.abort()
    with pytest.raises(RuntimeError):
        session.resume(collector.evaluator.evaluate([request])[0])
    actual = drain_session(collector.create_session(root, unit_id="unit"))
    replay = drain_session(make_collector(kind).create_session(root, unit_id="unit"))
    assert actual.record.edges == replay.record.edges
    assert actual.record.rng == replay.record.rng


def test_resume_rejects_wrong_token_without_sampling():
    from dataclasses import replace

    collector = make_collector("tree")
    session = collector.create_session(endgame_position(1), unit_id="unit")
    request = session.poll()
    result = collector.evaluator.evaluate([request])[0]
    with pytest.raises(ValueError, match="token"):
        session.resume(replace(result, token=request.token + 1))
    assert session.poll() is request
    session.abort()


def test_terminal_session_consumes_no_rng_or_inference():
    game = ReversiPyGame()
    state = endgame_position(1)
    rng = random.Random(1)
    while not state.is_terminal:
        state = game.step(state, rng.choice(state.legal_actions))
    collector = make_collector("tree")
    before = collector.streams.action_sampling.getstate()
    session = collector.create_session(state, unit_id="terminal")
    result = session.poll()
    assert isinstance(result, SessionComplete)
    assert result.value.record.cost.inference_batches == 0
    assert collector.streams.action_sampling.getstate() == before
