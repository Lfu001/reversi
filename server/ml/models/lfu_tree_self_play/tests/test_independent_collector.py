"""Acceptance tests for independent self-play trajectory collection."""

import math
import random
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch
from tests.test_records import build_record, position

from lfu_tree_self_play.collector import (
    IndependentCollector,
    collect_independent_game,
    collect_independent_paths,
    scale_return_targets,
    weight_independent_records,
)
from lfu_tree_self_play.config import experiment_id, load_config
from lfu_tree_self_play.dataset_store import DatasetStore
from lfu_tree_self_play.game import GameState, ReversiPyGame
from lfu_tree_self_play.model import (
    PolicyValueConfig,
    PolicyValueModel,
    masked_policy_logits,
)
from lfu_tree_self_play.record_validation import validate_record
from lfu_tree_self_play.rng import create_rng_streams
from lfu_tree_self_play.roots import generate_root_position, generate_root_registry
from lfu_tree_self_play.schedule import summarize_branch_execution
from lfu_tree_self_play.verification import endgame_position


def tiny_model(seed: int = 11) -> PolicyValueModel:
    torch.manual_seed(seed)
    return PolicyValueModel(PolicyValueConfig(width=8, num_blocks=1))


def test_independent_game_reaches_terminal_and_matches_frozen_policy_outputs():
    game = ReversiPyGame()
    model = tiny_model(19).train()
    reference = deepcopy(model).eval()
    root = generate_root_position(seed=42, placements=4)

    unit = collect_independent_game(
        model,
        game,
        root,
        unit_id="game-0",
        experiment_id="experiment",
        round_id="round-0",
        model_generation=3,
        run_seed=101,
    )
    record = unit.record
    validate_record(record, game)
    summary = summarize_branch_execution(record, game)

    assert record.mode == "independent"
    assert record.branch_plan.positions == ()
    assert record.sibling_groups == ()
    assert summary.planned_count == 0
    assert summary.executed_count == 0
    assert record.nodes[-1].state.is_terminal
    assert record.nodes[-1].old_value is None
    assert record.nodes[-1].black_result in (-1, 0, 1)
    assert len(record.nodes) == len(record.edges) + 1
    assert unit.path_weight == 1.0
    assert unit.targets == unit.weighted_targets
    assert set(unit.targets.node_weights.values()) == {1.0}
    assert set(unit.targets.edge_weights.values()) == {1.0}

    for depth, (node, edge) in enumerate(
        zip(record.nodes[:-1], record.edges, strict=True)
    ):
        assert node.placement_depth == depth
        assert node.model_generation == 3
        assert edge.model_generation == 3
        assert edge.sibling_group_id is None
        assert edge.sample_index is None
        assert edge.sampling.stream_id == "action"
        assert edge.sampling.draw_index == depth
        assert edge.sampling.continuation_seed is None

        observations = torch.from_numpy(node.state.observation).unsqueeze(0)
        with torch.no_grad():
            output = reference(observations)
            expected_value = float(output.value[0].double().item())
            mask = observations[:, 3].flatten(1).bool()
            logs = masked_policy_logits(
                output.policy_logits.double(), mask
            ).log_softmax(-1)[0]
        assert node.old_value == pytest.approx(expected_value)
        assert edge.old_log_probability == pytest.approx(float(logs[edge.action]))
        assert math.exp(
            float(logs[edge.action]) - edge.old_log_probability
        ) == pytest.approx(1.0)

    cost = record.cost
    assert cost.environment_transitions == len(record.edges)
    assert cost.evaluated_positions == len(record.edges)
    assert cost.inference_batches == len(record.edges)
    assert cost.collection_seconds >= 0.0
    assert cost.inference_seconds >= 0.0
    assert cost.transition_seconds >= 0.0
    assert cost.management_seconds >= 0.0
    assert cost.peak_memory_bytes > 0


def test_h_zero_terminal_root_produces_single_node_without_consuming_rng():
    game = ReversiPyGame()
    model = tiny_model(5)
    terminal_root = position(("BBBBBBBB",) * 8, 1, ())
    streams = create_rng_streams(77)
    before_action = streams.action_sampling.getstate()

    collector = IndependentCollector(
        model,
        game,
        experiment_id="experiment",
        round_id="round-0",
        model_generation=0,
        run_seed=77,
        rng_streams=streams,
        initial_draw_index=5,
    )
    unit = collector.collect_game(terminal_root, unit_id="terminal-root")

    assert len(unit.record.nodes) == 1
    assert len(unit.record.edges) == 0
    assert unit.record.nodes[0].state.is_terminal
    assert unit.record.nodes[0].old_value is None
    assert unit.record.nodes[0].black_result == 1
    assert unit.record.cost.environment_transitions == 0
    assert unit.record.cost.evaluated_positions == 0
    assert unit.record.cost.inference_batches == 0
    assert collector.draw_index == 5
    assert streams.action_sampling.getstate() == before_action


def test_forced_pass_preserves_actual_mover_perspective_targets():
    game = ReversiPyGame()
    model = tiny_model(9)
    # Only action 13 forces a pass where white (-1) moves twice in a row.
    forced_pass_root = position(
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
        (13,),
    )
    unit = collect_independent_game(
        model,
        game,
        forced_pass_root,
        unit_id="forced-pass",
        experiment_id="experiment",
        round_id="round-0",
        model_generation=1,
        run_seed=13,
    )
    record = unit.record
    nodes_by_id = {node.node_id: node for node in record.nodes}
    assert len(record.edges) >= 2
    assert record.nodes[0].state.to_play == -1
    assert record.nodes[1].state.to_play == -1
    assert record.nodes[-1].state.is_terminal
    assert record.nodes[-1].black_result == -1
    assert unit.targets.edge_parent_targets[record.edges[0].edge_id] == 1.0
    assert unit.targets.edge_parent_targets[record.edges[1].edge_id] == 1.0
    for edge in record.edges:
        parent_turn = nodes_by_id[edge.parent_node_id].state.to_play
        assert unit.targets.edge_black_returns[edge.edge_id] == -1.0
        assert unit.targets.edge_parent_targets[edge.edge_id] == -1.0 * parent_turn


def test_multiple_paths_from_same_root_weight_each_path_by_inverse_k_and_round_trip(
    tmp_path,
):
    config_path = Path(__file__).resolve().parents[1] / "configs" / "example.toml"
    config = load_config(config_path)
    exp_id = experiment_id(config)
    registry = generate_root_registry(config, roots_per_seed=1)
    root_entry = registry.entries[0]
    game = ReversiPyGame()
    model = tiny_model(23)

    collection = collect_independent_paths(
        model,
        game,
        root_entry,
        paths_per_root=4,
        unit_id_prefix="root0",
        experiment_id=exp_id,
        round_id="round-0",
        model_generation=2,
        run_seed=config.seeds.training[0],
    )

    assert len(collection.units) == 4
    assert collection.path_weight == pytest.approx(0.25)
    assert collection.root_state == root_entry.root.state

    root_weight_sum = 0.0
    first_edge_weight_sum = 0.0
    for index, unit in enumerate(collection.units):
        assert unit.record.unit_id == f"root0-{index}"
        assert unit.record.nodes[-1].state.is_terminal
        assert unit.path_weight == pytest.approx(0.25)
        assert set(unit.targets.node_weights.values()) == {1.0}
        assert set(unit.targets.edge_weights.values()) == {1.0}
        assert set(unit.weighted_targets.node_weights.values()) == {0.25}
        assert set(unit.weighted_targets.edge_weights.values()) == {0.25}
        assert (
            unit.weighted_targets.node_black_returns == unit.targets.node_black_returns
        )
        assert (
            unit.weighted_targets.edge_black_returns == unit.targets.edge_black_returns
        )
        assert (
            unit.weighted_targets.node_player_targets
            == unit.targets.node_player_targets
        )
        assert (
            unit.weighted_targets.edge_parent_targets
            == unit.targets.edge_parent_targets
        )
        root_weight_sum += unit.weighted_targets.node_weights[unit.record.root_node_id]
        first_edge_weight_sum += unit.weighted_targets.edge_weights[
            unit.record.edges[0].edge_id
        ]

    assert root_weight_sum == pytest.approx(1.0)
    assert first_edge_weight_sum == pytest.approx(1.0)

    store = DatasetStore(tmp_path / "dataset", experiment_id=exp_id, game=game)
    identities = collection.publish_all(store)
    assert len(identities) == 4

    loaded_records = [
        store.load(identity.unit_id, model_generation=2).record
        for identity in identities
    ]
    reweighted = weight_independent_records(loaded_records, game)
    assert reweighted == collection.weighted_targets


def test_collector_is_isolated_from_training_model_mutations_and_other_rng_streams():
    game = ReversiPyGame()
    model = tiny_model(31).train()
    root = endgame_position(1)
    streams_a = create_rng_streams(99)
    streams_b = create_rng_streams(99)

    branch_before = streams_a.branch_position.getstate()
    sibling_before = streams_a.sibling_rollout.getstate()
    eval_before = streams_a.evaluation.getstate()
    py_before = random.getstate()
    np_before = np.random.get_state()[1].copy()
    torch_before = torch.random.get_rng_state().clone()

    collector_a = IndependentCollector(
        model,
        game,
        experiment_id="experiment",
        round_id="round-0",
        model_generation=0,
        run_seed=99,
        rng_streams=streams_a,
    )
    first_batch = collector_a.collect_paths(
        root, paths_per_root=3, unit_id_prefix="batch"
    )

    assert streams_a.branch_position.getstate() == branch_before
    assert streams_a.sibling_rollout.getstate() == sibling_before
    assert streams_a.evaluation.getstate() == eval_before
    assert random.getstate() == py_before
    assert np.array_equal(np.random.get_state()[1], np_before)
    assert torch.equal(torch.random.get_rng_state(), torch_before)
    assert model.training
    assert all(parameter.grad is None for parameter in model.parameters())

    # Mutate original model weights and buffers heavily; collector_b initialized
    # before mutation must produce identical trajectories to collector_a.
    collector_b = IndependentCollector(
        model,
        game,
        experiment_id="experiment",
        round_id="round-0",
        model_generation=0,
        run_seed=99,
        rng_streams=streams_b,
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(50.0)
        for buffer in model.buffers():
            buffer.fill_(25.0)
    second_batch = collector_b.collect_paths(
        root, paths_per_root=3, unit_id_prefix="batch"
    )

    for unit_a, unit_b in zip(first_batch.units, second_batch.units, strict=True):
        assert unit_a.record.nodes == unit_b.record.nodes
        assert unit_a.record.edges == unit_b.record.edges
        assert unit_a.weighted_targets == unit_b.weighted_targets


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"unit_id": ""}, "unsafe unit ID"),
        ({"unit_id": "../bad"}, "unsafe unit ID"),
        ({"experiment_id": "   "}, "experiment_id"),
        ({"round_id": ""}, "round_id"),
        ({"model_generation": -1}, "model_generation"),
        ({"model_generation": True}, "model_generation"),
        ({"run_seed": -1}, "run_seed"),
        ({"run_seed": False}, "run_seed"),
        ({"initial_draw_index": -1}, "initial_draw_index"),
    ],
)
def test_invalid_collector_or_game_arguments_do_not_consume_rng(kwargs, match):
    game = ReversiPyGame()
    model = tiny_model(3)
    root = endgame_position(1)
    streams = create_rng_streams(7)
    before = streams.action_sampling.getstate()

    params = {
        "unit_id": "valid-id",
        "experiment_id": "exp-a",
        "round_id": "round-0",
        "model_generation": 0,
        "run_seed": 7,
        "rng_streams": streams,
        "initial_draw_index": 0,
        **kwargs,
    }
    with pytest.raises(ValueError, match=match):
        collect_independent_game(model, game, root, **params)
    assert streams.action_sampling.getstate() == before


@pytest.mark.parametrize(
    ("paths_per_root", "unit_id_prefix", "unit_ids"),
    [
        (0, "p", None),
        (-1, "p", None),
        (True, "p", None),
        (2, None, None),
        (2, "p", ("a", "b")),
        (2, None, ("only-one",)),
        (2, None, ("dup", "dup")),
        (2, None, "ab"),
    ],
)
def test_invalid_collect_paths_arguments_do_not_consume_rng(
    paths_per_root, unit_id_prefix, unit_ids
):
    game = ReversiPyGame()
    model = tiny_model(3)
    root = endgame_position(1)
    streams = create_rng_streams(7)
    before = streams.action_sampling.getstate()
    collector = IndependentCollector(
        model,
        game,
        experiment_id="experiment",
        round_id="round-0",
        model_generation=0,
        run_seed=7,
        rng_streams=streams,
        initial_draw_index=4,
    )
    with pytest.raises(ValueError):
        collector.collect_paths(
            root,
            paths_per_root=paths_per_root,
            unit_id_prefix=unit_id_prefix,
            unit_ids=unit_ids,
        )
    assert streams.action_sampling.getstate() == before
    assert collector.draw_index == 4


def test_failed_game_transition_restores_rng_state_and_draw_index():
    class FailingGame:
        def initial_state(self) -> GameState:
            return endgame_position(1)

        def step(self, state: GameState, action: int) -> GameState:
            raise RuntimeError("simulated transition failure")

    model = tiny_model(3)
    streams = create_rng_streams(7)
    before = streams.action_sampling.getstate()
    collector = IndependentCollector(
        model,
        FailingGame(),
        experiment_id="experiment",
        round_id="round-0",
        model_generation=0,
        run_seed=7,
        rng_streams=streams,
        initial_draw_index=10,
    )
    with pytest.raises(RuntimeError, match="simulated transition failure"):
        collector.collect_game(endgame_position(1), unit_id="fail-0")
    assert streams.action_sampling.getstate() == before
    assert collector.draw_index == 10


def test_weight_independent_records_rejects_mismatched_or_non_independent_groups():
    game = ReversiPyGame()
    model = tiny_model(11)
    root_a = endgame_position(1)
    root_b = endgame_position(-1)

    collection = collect_independent_paths(
        model,
        game,
        root_a,
        paths_per_root=2,
        unit_ids=("u-0", "u-1"),
        experiment_id="experiment",
        round_id="round-0",
        model_generation=0,
        run_seed=7,
    )
    other_root_unit = collect_independent_game(
        model,
        game,
        root_b,
        unit_id="u-2",
        experiment_id="experiment",
        round_id="round-0",
        model_generation=0,
        run_seed=8,
    )
    same_draw_unit = collect_independent_game(
        model,
        game,
        root_a,
        unit_id="u-dup-draw",
        experiment_id="experiment",
        round_id="round-0",
        model_generation=0,
        run_seed=7,
    )

    with pytest.raises(ValueError, match="at least one"):
        weight_independent_records((), game)
    with pytest.raises(ValueError, match="expected independent mode"):
        weight_independent_records((build_record(mode="tree"),), game)
    with pytest.raises(ValueError, match="duplicate unit_id"):
        weight_independent_records((collection.records[0], collection.records[0]), game)
    with pytest.raises(ValueError, match="same root state"):
        weight_independent_records(
            (collection.records[0], other_root_unit.record), game
        )
    with pytest.raises(ValueError, match="duplicate sampling draw"):
        weight_independent_records((collection.records[0], same_draw_unit.record), game)
    with pytest.raises(ValueError, match="metadata mismatch"):
        mismatched = same_draw_unit.record.model_copy(
            update={"round_id": "other-round"}
        )
        weight_independent_records((collection.records[0], mismatched), game)


@pytest.mark.parametrize("bad_weight", [0.0, -0.5, 1.5, math.nan, math.inf, True])
def test_scale_return_targets_rejects_invalid_weights(bad_weight):
    game = ReversiPyGame()
    unit = collect_independent_game(
        tiny_model(3),
        game,
        endgame_position(1),
        unit_id="u0",
        experiment_id="experiment",
        round_id="round-0",
        model_generation=0,
        run_seed=7,
    )
    with pytest.raises(ValueError, match="path_weight"):
        scale_return_targets(unit.targets, bad_weight)
