"""Conditional policy gradients with the opponent held fixed."""

import random

import numpy as np
import pytest
from lfu_tree_self_play.game import ReversiPyGame
from lfu_tree_self_play.verification import (
    endgame_position,
    exact_conditional_value_and_gradient,
    monte_carlo_value_and_gradient,
)


@pytest.mark.parametrize("color", [1, -1])
def test_exact_gradient_matches_finite_difference_for_each_color(color: int) -> None:
    game = ReversiPyGame()
    state = endgame_position(color)
    theta = np.zeros(64, dtype=np.float64)
    value, gradient = exact_conditional_value_and_gradient(game, state, color, theta)
    assert abs(gradient).max() > 1e-4
    action = state.legal_actions[0]
    epsilon = 1e-5
    higher = theta.copy()
    lower = theta.copy()
    higher[action] += epsilon
    lower[action] -= epsilon
    upper_value, _ = exact_conditional_value_and_gradient(game, state, color, higher)
    lower_value, _ = exact_conditional_value_and_gradient(game, state, color, lower)
    assert gradient[action] == pytest.approx(
        (upper_value - lower_value) / (2 * epsilon), abs=1e-8
    )
    assert -1 <= value <= 1


@pytest.mark.parametrize("color", [1, -1])
def test_independent_sibling_monte_carlo_matches_exact_gradient(color: int) -> None:
    game = ReversiPyGame()
    state = endgame_position(color)
    theta = np.zeros(64, dtype=np.float64)
    exact_value, exact_gradient = exact_conditional_value_and_gradient(
        game, state, color, theta
    )
    estimate = monte_carlo_value_and_gradient(
        game, state, color, theta, random.Random(314159 + color), pairs=3000
    )
    assert abs(estimate.value - exact_value) <= 5 * estimate.value_se + 0.01
    assert np.all(
        np.abs(estimate.gradient - exact_gradient) <= 5 * estimate.gradient_se + 0.01
    )


def test_conditional_gradient_rejects_invalid_color() -> None:
    with pytest.raises(ValueError, match="color"):
        exact_conditional_value_and_gradient(
            ReversiPyGame(), endgame_position(1), 0, np.zeros(64)
        )


def test_terminal_position_has_zero_conditional_gradient() -> None:
    game = ReversiPyGame()
    state = endgame_position(1)
    while not state.is_terminal:
        state = game.step(state, state.legal_actions[0])

    value, gradient = exact_conditional_value_and_gradient(game, state, 1, np.zeros(64))

    assert value == state.black_result
    assert np.array_equal(gradient, np.zeros(64))
