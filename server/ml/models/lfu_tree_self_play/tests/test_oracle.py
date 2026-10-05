"""Exact fixed-policy results on reachable, enumerable Reversi endings."""

import math
import random

import numpy as np
import pytest

from lfu_tree_self_play.game import GameState, ReversiPyGame
from lfu_tree_self_play.oracle import expected_black_return, sample_black_return

POSITIONS = (
    (
        (
            "BWWWW.W.",
            "BWWW.WW.",
            "BWWWWWWB",
            "BBBBWWW.",
            "BBWWBWBB",
            "WWWWBWBW",
            "WBBBWBWW",
            "WB.BBBBW",
        ),
        1,
        (5, 7, 12, 15, 31),
        -0.8743055555555556,
    ),
    (
        (
            "BWWWW.WB",
            "BWWW.WB.",
            "BWWWWBWB",
            "BBBBBWW.",
            "BBWBBWBB",
            "WWBWBWBW",
            "WBBBWBWW",
            "WB.BBBBW",
        ),
        -1,
        (12, 15, 31, 58),
        -0.8298611111111112,
    ),
)


def position(rows: tuple[str, ...], turn: int, actions: tuple[int, ...]) -> GameState:
    planes = np.zeros((4, 8, 8), dtype=np.float32)
    for row, cells in enumerate(rows):
        for col, cell in enumerate(cells):
            if cell == "B":
                planes[0, row, col] = 1
            elif cell == "W":
                planes[1, row, col] = 1
    planes[2] = turn
    for action in actions:
        planes[3, action // 8, action % 8] = 1
    return GameState(planes)


def uniform(state: GameState) -> dict[int, float]:
    return {action: 1 / len(state.legal_actions) for action in state.legal_actions}


@pytest.mark.parametrize("rows,turn,actions,expected", POSITIONS)
def test_enumeration_returns_known_black_perspective_value(
    rows: tuple[str, ...], turn: int, actions: tuple[int, ...], expected: float
) -> None:
    state = position(rows, turn, actions)
    assert math.isclose(
        expected_black_return(ReversiPyGame(), state, uniform), expected, abs_tol=1e-12
    )


def test_sampling_uses_the_same_fixed_policy_as_enumeration() -> None:
    rows, turn, actions, expected = POSITIONS[1]
    state = position(rows, turn, actions)
    rng = random.Random(17)
    samples = [
        sample_black_return(ReversiPyGame(), state, uniform, rng) for _ in range(2000)
    ]
    assert set(samples) <= {-1, 0, 1}
    assert abs(sum(samples) / len(samples) - expected) < 0.06


def test_oracle_rejects_probability_on_illegal_move() -> None:
    rows, turn, actions, _ = POSITIONS[0]
    state = position(rows, turn, actions)
    with pytest.raises(ValueError, match="legal actions"):
        expected_black_return(ReversiPyGame(), state, lambda _: {0: 1.0})
