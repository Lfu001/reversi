"""Root openings are legal, reproducible, and separated by board identity."""

import random

import numpy as np
import pytest

from lfu_tree_self_play.game import GameState, ReversiPyGame


@pytest.mark.parametrize("placements", range(9))
def test_generate_root_replays_legal_placements(placements):
    from lfu_tree_self_play.roots import generate_root_position

    root = generate_root_position(42, placements)
    state = ReversiPyGame().initial_state()
    for action in root.actions:
        assert action in state.legal_actions
        state = ReversiPyGame().step(state, action)

    assert len(root.actions) == placements
    assert root.state == state
    assert root.state.observation[:2].sum() == 4 + placements
    assert root.placements == placements
    assert root.seed == 42


def test_generation_reproduces_without_consuming_other_rngs():
    from lfu_tree_self_play.rng import create_rng_streams
    from lfu_tree_self_play.roots import generate_root_position

    random.seed(42)
    global_state = random.getstate()
    streams = create_rng_streams(42)
    first = generate_root_position(42)
    for _ in range(20):
        streams.action_sampling.random()
    second = generate_root_position(42)

    assert first == second
    assert random.getstate() == global_state
    lengths = {generate_root_position(seed).placements for seed in range(100)}
    assert lengths == set(range(9))


@pytest.mark.parametrize("seed", [-1, True, 1.5])
def test_generation_rejects_invalid_seed(seed):
    from lfu_tree_self_play.roots import generate_root_position

    with pytest.raises(ValueError):
        generate_root_position(seed)


@pytest.mark.parametrize("placements", [-1, 9, True, 2.5])
def test_generation_rejects_invalid_placement_count(placements):
    from lfu_tree_self_play.roots import generate_root_position

    with pytest.raises(ValueError):
        generate_root_position(42, placements)


def test_position_key_equates_all_eight_symmetries():
    from lfu_tree_self_play.roots import position_key

    game = ReversiPyGame()
    state = game.step(game.step(game.initial_state(), 19), 18)
    board = state.observation
    key = position_key(state)

    for reflected in (board, np.flip(board, axis=2)):
        for turns in range(4):
            transformed = GameState(np.rot90(reflected, turns, axes=(1, 2)))
            assert position_key(transformed) == key


def test_position_key_distinguishes_turn_and_disk_colors():
    from lfu_tree_self_play.roots import position_key

    state = ReversiPyGame().step(ReversiPyGame().initial_state(), 19)
    opposite_turn = state.observation
    opposite_turn[2] *= -1
    swapped_colors = state.observation
    swapped_colors[[0, 1]] = swapped_colors[[1, 0]]

    assert position_key(state) != position_key(GameState(opposite_turn))
    assert position_key(state) != position_key(GameState(swapped_colors))


@pytest.mark.parametrize("corruption", ["overlap", "nonbinary", "turn"])
def test_position_key_rejects_invalid_identity_planes(corruption):
    from lfu_tree_self_play.roots import position_key

    planes = ReversiPyGame().initial_state().observation
    if corruption == "overlap":
        planes[0, 3, 3] = 1
    elif corruption == "nonbinary":
        planes[0, 0, 0] = 0.5
    else:
        planes[2, 0, 0] = 0
    with pytest.raises(ValueError):
        position_key(GameState(planes))
