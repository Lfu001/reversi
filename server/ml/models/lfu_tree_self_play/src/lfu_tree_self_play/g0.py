"""Small, reproducible checks for the G0 mathematical contracts."""

import math
import random
from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray

from lfu_tree_self_play.game import Game, GameState
from lfu_tree_self_play.model import masked_policy_logits
from lfu_tree_self_play.oracle import expected_black_return
from lfu_tree_self_play.rng import derive_seed

_ENDGAMES = {
    1: (
        "BWWWW.W.",
        "BWWW.WW.",
        "BWWWWWWB",
        "BBBBWWW.",
        "BBWWBWBB",
        "WWWWBWBW",
        "WBBBWBWW",
        "WB.BBBBW",
    ),
    -1: (
        "BWWWW.WB",
        "BWWW.WB.",
        "BWWWWBWB",
        "BBBBBWW.",
        "BBWBBWBB",
        "WWBWBWBW",
        "WBBBWBWW",
        "WB.BBBBW",
    ),
}
_LEGAL = {1: (5, 7, 12, 15, 31), -1: (12, 15, 31, 58)}


def endgame_position(color: int) -> GameState:
    """Return a reachable six-empty ending with variable outcomes."""
    if color not in _ENDGAMES:
        raise ValueError("color must be +1 (black) or -1 (white)")
    planes = np.zeros((4, 8, 8), dtype=np.float32)
    for row, cells in enumerate(_ENDGAMES[color]):
        for col, cell in enumerate(cells):
            if cell == "B":
                planes[0, row, col] = 1
            elif cell == "W":
                planes[1, row, col] = 1
    planes[2] = color
    for action in _LEGAL[color]:
        planes[3, action // 8, action % 8] = 1
    return GameState(planes)


def _check_inputs(color: int, theta: NDArray[np.float64]) -> None:
    if color not in (-1, 1):
        raise ValueError("color must be +1 or -1")
    if theta.shape != (64,) or not np.isfinite(theta).all():
        raise ValueError("theta must be a finite 64-element vector")


def exact_conditional_value_and_gradient(
    game: Game, state: GameState, color: int, theta: NDArray[np.float64]
) -> tuple[float, NDArray[np.float64]]:
    """Differentiate exact return for one color while the opponent stays uniform."""
    _check_inputs(color, theta)
    parameters = torch.tensor(theta, dtype=torch.float64, requires_grad=True)
    zero_logits = torch.zeros_like(parameters)
    cache: dict[GameState, torch.Tensor] = {}

    def visit(current: GameState) -> torch.Tensor:
        if current.is_terminal:
            assert current.black_result is not None
            return parameters.new_tensor(float(color * current.black_result))
        if current not in cache:
            mask = torch.zeros(64, dtype=torch.bool)
            mask[list(current.legal_actions)] = True
            logits = parameters if current.to_play == color else zero_logits
            probabilities = masked_policy_logits(
                logits.unsqueeze(0), mask.unsqueeze(0)
            ).softmax(dim=-1)[0]
            cache[current] = torch.stack(
                [
                    probabilities[action] * visit(game.step(current, action))
                    for action in current.legal_actions
                ]
            ).sum()
        return cache[current]

    value = visit(state)
    if not value.requires_grad:
        return float(value), np.zeros_like(theta, dtype=np.float64)
    gradient = torch.autograd.grad(value, parameters)[0]
    return float(value.detach()), gradient.detach().numpy().copy()


@dataclass(frozen=True)
class MonteCarloEstimate:
    value: float
    value_se: float
    gradient: NDArray[np.float64]
    gradient_se: NDArray[np.float64]


def monte_carlo_value_and_gradient(
    game: Game,
    state: GameState,
    color: int,
    theta: NDArray[np.float64],
    rng: random.Random,
    *,
    pairs: int,
) -> MonteCarloEstimate:
    """Use independent sibling rollouts and a leave-one-out score estimator."""
    _check_inputs(color, theta)
    if pairs < 2:
        raise ValueError("pairs must be at least two")

    def sample() -> tuple[float, NDArray[np.float64]]:
        current = state
        score = np.zeros(64, dtype=np.float64)
        while not current.is_terminal:
            actions = current.legal_actions
            if current.to_play == color:
                logits = theta[list(actions)]
                weights = np.exp(logits - logits.max())
                probabilities = weights / weights.sum()
            else:
                probabilities = np.full(len(actions), 1 / len(actions))
            selected = rng.choices(actions, weights=probabilities, k=1)[0]
            if current.to_play == color:
                score[list(actions)] -= probabilities
                score[selected] += 1
            current = game.step(current, selected)
        assert current.black_result is not None
        return float(color * current.black_result), score

    values = np.empty(pairs, dtype=np.float64)
    gradients = np.empty((pairs, 64), dtype=np.float64)
    for index in range(pairs):
        first_return, first_score = sample()
        second_return, second_score = sample()
        values[index] = (first_return + second_return) / 2
        gradients[index] = (
            (first_return - second_return) * (first_score - second_score) / 2
        )
    return MonteCarloEstimate(
        value=float(values.mean()),
        value_se=float(values.std(ddof=1) / math.sqrt(pairs)),
        gradient=gradients.mean(axis=0),
        gradient_se=gradients.std(axis=0, ddof=1) / math.sqrt(pairs),
    )


def leave_one_out_advantages(returns: tuple[float, ...]) -> tuple[float, ...]:
    """Subtract each sibling's mean return without changing the reward scale."""
    if len(returns) < 2:
        raise ValueError("leave-one-out requires at least two siblings")
    if not all(math.isfinite(value) for value in returns):
        raise ValueError("sibling returns must be finite")
    total = math.fsum(returns)
    return tuple(value - (total - value) / (len(returns) - 1) for value in returns)


def _legal_probabilities(
    state: GameState, theta: NDArray[np.float64]
) -> NDArray[np.float64]:
    logits = theta[list(state.legal_actions)]
    weights = np.exp(logits - logits.max())
    return weights / weights.sum()


def importance_ratio(
    state: GameState,
    action: int,
    old_theta: NDArray[np.float64],
    new_theta: NDArray[np.float64],
) -> float:
    """Compare the selected action's probabilities under identical legal masks."""
    _check_inputs(state.to_play, old_theta)
    _check_inputs(state.to_play, new_theta)
    if action not in state.legal_actions:
        raise ValueError("action must be legal")
    index = state.legal_actions.index(action)
    return float(
        _legal_probabilities(state, new_theta)[index]
        / _legal_probabilities(state, old_theta)[index]
    )


@dataclass(frozen=True)
class SharedRecomputedComparison:
    generated_equal: bool
    shared_loss: float
    recomputed_loss: float
    shared_policy_loss: float
    recomputed_policy_loss: float
    shared_value_loss: float
    recomputed_value_loss: float
    prefix_weight: float
    branch_weights: tuple[float, float]
    shared_edge_count: int
    recomputed_edge_count: int
    prefix_return: float
    branch_returns: tuple[int, int]
    branch_advantages: tuple[float, float]
    clipped_edge_count: int


def compare_shared_recomputed(
    game: Game,
    root: GameState,
    sibling_actions: tuple[int, int],
    rng: random.Random,
) -> SharedRecomputedComparison:
    """Replay two seeded branches with and without their common first edge."""
    if root.is_terminal:
        raise ValueError("root must have a legal move")
    if len(sibling_actions) != 2:
        raise ValueError("exactly two sibling actions are required")
    prefix_action = root.legal_actions[0]
    branch = game.step(root, prefix_action)
    if branch.is_terminal or any(
        action not in branch.legal_actions for action in sibling_actions
    ):
        raise ValueError("both sibling actions must be legal after the prefix")

    paths: list[tuple[tuple[int, ...], tuple[GameState, ...]]] = []
    for sibling_action in sibling_actions:
        branch_rng = random.Random(rng.getrandbits(64))
        actions = [prefix_action, sibling_action]
        states = [root, branch]
        current = game.step(branch, sibling_action)
        while not current.is_terminal:
            states.append(current)
            action = branch_rng.choice(current.legal_actions)
            actions.append(action)
            current = game.step(current, action)
        states.append(current)
        paths.append((tuple(actions), tuple(states)))

    replayed = []
    for actions, _ in paths:
        current = root
        states = [current]
        for action in actions:
            current = game.step(current, action)
            states.append(current)
        replayed.append(tuple(states))

    branch_returns = tuple(states[-1].black_result for _, states in paths)
    replay_returns = tuple(states[-1].black_result for states in replayed)
    assert all(result is not None for result in branch_returns + replay_returns)
    weight = 1 / len(paths)
    prefix_return = math.fsum(branch_returns) * weight
    replay_prefix_return = math.fsum(replay_returns) * weight
    branch_advantages = tuple(
        advantage * branch.to_play
        for advantage in leave_one_out_advantages(branch_returns)
    )
    replay_advantages = tuple(
        advantage * branch.to_play
        for advantage in leave_one_out_advantages(replay_returns)
    )

    # One logical prefix edge contains two sample contributions. Each branch
    # below it keeps its own identity, even when both actions happen to match.
    shared_edges = [
        (
            root,
            prefix_action,
            1.0,
            tuple(
                (
                    sample_id,
                    0,
                    weight,
                    prefix_return * root.to_play,
                    prefix_return * root.to_play,
                )
                for sample_id in range(len(paths))
            ),
        )
    ]
    for sample_id, (actions, states) in enumerate(paths):
        for index, (state, action) in enumerate(
            zip(states[1:-1], actions[1:], strict=True), start=1
        ):
            policy_advantage = (
                branch_advantages[sample_id]
                if index == 1
                else branch_returns[sample_id] * state.to_play
            )
            value_target = branch_returns[sample_id] * state.to_play
            shared_edges.append(
                (
                    state,
                    action,
                    weight,
                    ((sample_id, index, weight, policy_advantage, value_target),),
                )
            )

    recomputed_edges = []
    for sample_id, ((actions, _), states) in enumerate(
        zip(paths, replayed, strict=True)
    ):
        for index, (state, action) in enumerate(zip(states[:-1], actions, strict=True)):
            policy_advantage = (
                replay_prefix_return * root.to_play
                if index == 0
                else replay_advantages[sample_id]
                if index == 1
                else replay_returns[sample_id] * state.to_play
            )
            value_target = (
                replay_prefix_return * root.to_play
                if index == 0
                else replay_returns[sample_id] * state.to_play
            )
            recomputed_edges.append(
                (
                    sample_id,
                    index,
                    state,
                    action,
                    weight,
                    policy_advantage,
                    value_target,
                )
            )

    expanded_shared = sorted(
        (
            sample_id,
            index,
            state,
            action,
            sample_weight,
            policy_advantage,
            value_target,
        )
        for state, action, _, contributions in shared_edges
        for sample_id, index, sample_weight, policy_advantage, value_target in contributions
    )
    expanded_recomputed = sorted(recomputed_edges)
    generated_equal = expanded_shared == expanded_recomputed

    old_theta = np.zeros(64, dtype=np.float64)
    new_theta = (np.arange(64, dtype=np.float64) % 7) * 0.3

    def surrogate(ratio: float, advantage: float) -> float:
        clipped_ratio = min(max(ratio, 0.8), 1.2)
        return min(ratio * advantage, clipped_ratio * advantage)

    shared_policy_loss = 0.0
    shared_value_loss = 0.0
    clipped_edge_count = 0
    for state, action, total_weight, contributions in shared_edges:
        assert math.isclose(
            total_weight,
            math.fsum(sample_weight for _, _, sample_weight, _, _ in contributions),
        )
        ratio = importance_ratio(state, action, old_theta, new_theta)
        for _, _, sample_weight, policy_advantage, value_target in contributions:
            if surrogate(ratio, policy_advantage) != ratio * policy_advantage:
                clipped_edge_count += 1
            shared_policy_loss -= (
                sample_weight * surrogate(ratio, policy_advantage) / 60
            )
            shared_value_loss += sample_weight * value_target**2 / 60

    recomputed_policy_loss = -math.fsum(
        sample_weight
        * surrogate(
            importance_ratio(state, action, old_theta, new_theta), policy_advantage
        )
        / 60
        for _, _, state, action, sample_weight, policy_advantage, _ in recomputed_edges
    )
    recomputed_value_loss = math.fsum(
        sample_weight * value_target**2 / 60
        for _, _, _, _, sample_weight, _, value_target in recomputed_edges
    )

    return SharedRecomputedComparison(
        generated_equal=generated_equal,
        shared_loss=shared_policy_loss + shared_value_loss,
        recomputed_loss=recomputed_policy_loss + recomputed_value_loss,
        shared_policy_loss=shared_policy_loss,
        recomputed_policy_loss=recomputed_policy_loss,
        shared_value_loss=shared_value_loss,
        recomputed_value_loss=recomputed_value_loss,
        prefix_weight=shared_edges[0][2],
        branch_weights=(weight, weight),
        shared_edge_count=len(shared_edges),
        recomputed_edge_count=len(recomputed_edges),
        prefix_return=prefix_return,
        branch_returns=branch_returns,
        branch_advantages=branch_advantages,
        clipped_edge_count=clipped_edge_count,
    )


def run_g0_verification(*, seed: int = 314159, pairs: int = 3000) -> dict:
    """Run and summarize the reproducible G0 numerical checks."""
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if pairs < 2:
        raise ValueError("pairs must be at least two")
    from lfu_tree_self_play.game import ReversiPyGame

    game = ReversiPyGame()
    theta = np.zeros(64, dtype=np.float64)
    conditions: dict[str, dict] = {}
    ratio_errors = []
    for color, label in ((1, "black"), (-1, "white")):
        state = endgame_position(color)
        oracle_black = expected_black_return(
            game,
            state,
            lambda current: {
                action: 1 / len(current.legal_actions)
                for action in current.legal_actions
            },
        )
        exact_value, exact_gradient = exact_conditional_value_and_gradient(
            game, state, color, theta
        )
        estimate = monte_carlo_value_and_gradient(
            game,
            state,
            color,
            theta,
            random.Random(derive_seed(seed, "evaluation") + color),
            pairs=pairs,
        )
        value_error = abs(estimate.value - exact_value)
        value_tolerance = 5 * estimate.value_se + 0.01
        gradient_errors = np.abs(estimate.gradient - exact_gradient)
        gradient_tolerances = 5 * estimate.gradient_se + 0.01
        gradient_passed = bool(np.all(gradient_errors <= gradient_tolerances))
        oracle_agrees = abs(exact_value - color * oracle_black) <= 1e-12
        conditions[label] = {
            "exact_value": exact_value,
            "oracle_black_value": oracle_black,
            "monte_carlo_value": estimate.value,
            "value_standard_error": estimate.value_se,
            "value_error": value_error,
            "value_tolerance": value_tolerance,
            "exact_gradient": exact_gradient.tolist(),
            "monte_carlo_gradient": estimate.gradient.tolist(),
            "gradient_standard_error": estimate.gradient_se.tolist(),
            "gradient_passed": gradient_passed,
            "oracle_agrees_with_exact": oracle_agrees,
            "passed": value_error <= value_tolerance
            and gradient_passed
            and oracle_agrees,
        }
        varied_theta = np.linspace(-0.2, 0.2, 64)
        ratio_errors.extend(
            abs(importance_ratio(state, action, varied_theta, varied_theta) - 1)
            for action in state.legal_actions
        )

    max_ratio_error = max(ratio_errors)
    conditions["importance_ratio"] = {
        "max_error_at_theta_equal_old": max_ratio_error,
        "passed": max_ratio_error <= 1e-12,
    }
    advantages = leave_one_out_advantages((1.0, 1.0, 1.0))
    conditions["leave_one_out"] = {
        "same_result_advantages": advantages,
        "passed": all(advantage == 0 for advantage in advantages),
    }
    root = endgame_position(1)
    branch = game.step(root, root.legal_actions[0])
    first, second = branch.legal_actions[:2]

    def mixed_case(siblings: tuple[int, int]) -> tuple[int, SharedRecomputedComparison]:
        start = derive_seed(seed, "sibling_rollout") % 1_000_000
        for offset in range(100):
            case_seed = start + offset
            comparison = compare_shared_recomputed(
                game, root, siblings, random.Random(case_seed)
            )
            if (
                len(set(comparison.branch_returns)) > 1
                and comparison.clipped_edge_count > 0
            ):
                return case_seed, comparison
        raise ValueError("no mixed-return clipped sibling case found")

    comparisons = [
        mixed_case(siblings) for siblings in ((first, second), (first, first))
    ]
    conditions["shared_recomputed"] = {
        "distinct_and_duplicate_siblings": [
            {
                "case_seed": case_seed,
                "generated_equal": item.generated_equal,
                "shared_loss": item.shared_loss,
                "recomputed_loss": item.recomputed_loss,
                "shared_policy_loss": item.shared_policy_loss,
                "recomputed_policy_loss": item.recomputed_policy_loss,
                "shared_value_loss": item.shared_value_loss,
                "recomputed_value_loss": item.recomputed_value_loss,
                "prefix_weight": item.prefix_weight,
                "branch_weights": item.branch_weights,
                "shared_edge_count": item.shared_edge_count,
                "recomputed_edge_count": item.recomputed_edge_count,
                "prefix_return": item.prefix_return,
                "branch_returns": item.branch_returns,
                "branch_advantages": item.branch_advantages,
                "clipped_edge_count": item.clipped_edge_count,
            }
            for case_seed, item in comparisons
        ],
        "passed": all(
            item.generated_equal
            and abs(item.shared_loss - item.recomputed_loss) <= 1e-12
            and abs(item.shared_policy_loss - item.recomputed_policy_loss) <= 1e-12
            and abs(item.shared_value_loss - item.recomputed_value_loss) <= 1e-12
            and item.prefix_weight == 1
            and item.branch_weights == (0.5, 0.5)
            and item.shared_edge_count + 1 == item.recomputed_edge_count
            and len(set(item.branch_returns)) > 1
            and item.clipped_edge_count > 0
            for _, item in comparisons
        ),
    }
    return {
        "gate": "G0 mathematical verification",
        "seed": seed,
        "pairs_per_color": pairs,
        "conditions": conditions,
        "passed": all(condition["passed"] for condition in conditions.values()),
    }
