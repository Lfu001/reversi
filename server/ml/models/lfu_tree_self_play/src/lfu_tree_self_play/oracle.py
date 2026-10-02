"""Exact and sampled returns for a fixed policy on small Reversi endings."""

import math
import random
from collections.abc import Callable, Mapping

from lfu_tree_self_play.game import Game, GameState

Policy = Callable[[GameState], Mapping[int, float]]


def _probabilities(state: GameState, policy: Policy) -> tuple[tuple[int, float], ...]:
    probabilities = policy(state)
    if set(probabilities) != set(state.legal_actions):
        raise ValueError(
            "policy must assign probabilities to exactly the legal actions"
        )
    items = tuple(
        (action, float(probabilities[action])) for action in state.legal_actions
    )
    if any(
        not math.isfinite(probability) or probability < 0 for _, probability in items
    ):
        raise ValueError("policy probabilities must be finite and nonnegative")
    if not math.isclose(
        math.fsum(probability for _, probability in items), 1.0, abs_tol=1e-10
    ):
        raise ValueError("policy probabilities must sum to one")
    return items


def expected_black_return(game: Game, state: GameState, policy: Policy) -> float:
    """Enumerate terminal outcomes under a fixed state-dependent policy."""
    cache: dict[GameState, float] = {}

    def visit(current: GameState) -> float:
        if current.is_terminal:
            assert current.black_result is not None
            return float(current.black_result)
        if current not in cache:
            cache[current] = math.fsum(
                probability * visit(game.step(current, action))
                for action, probability in _probabilities(current, policy)
            )
        return cache[current]

    return visit(state)


def sample_black_return(
    game: Game, state: GameState, policy: Policy, rng: random.Random
) -> int:
    """Sample one complete game from the same fixed policy."""
    current = state
    while not current.is_terminal:
        actions, weights = zip(*_probabilities(current, policy), strict=True)
        current = game.step(current, rng.choices(actions, weights=weights, k=1)[0])
    assert current.black_result is not None
    return current.black_result
