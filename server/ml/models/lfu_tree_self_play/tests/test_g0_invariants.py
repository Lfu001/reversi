"""Algebra and path-weight invariants for the G0 gate."""

import random

import numpy as np
import pytest
from lfu_tree_self_play.g0 import (
    compare_shared_recomputed,
    endgame_position,
    importance_ratio,
    leave_one_out_advantages,
)
from lfu_tree_self_play.game import ReversiPyGame


def test_identical_sibling_returns_have_zero_relative_advantage() -> None:
    assert leave_one_out_advantages((1.0, 1.0, 1.0)) == pytest.approx((0.0, 0.0, 0.0))
    assert leave_one_out_advantages((1.0, -1.0)) == pytest.approx((2.0, -2.0))


def test_importance_ratio_is_one_when_policies_match() -> None:
    state = endgame_position(1)
    old = np.linspace(-0.2, 0.2, 64)
    for action in state.legal_actions:
        assert importance_ratio(state, action, old, old.copy()) == pytest.approx(1)


@pytest.mark.parametrize("same_action", [False, True])
def test_shared_and_recomputed_paths_have_same_data_and_loss(
    same_action: bool,
) -> None:
    game = ReversiPyGame()
    root = endgame_position(1)
    branch = game.step(root, root.legal_actions[0])
    first, second = branch.legal_actions[:2]
    result = compare_shared_recomputed(
        game,
        root,
        (first, first if same_action else second),
        random.Random(0 if same_action else 1),
    )
    assert result.generated_equal
    assert result.shared_loss == pytest.approx(result.recomputed_loss, abs=1e-12)
    assert result.shared_policy_loss == pytest.approx(
        result.recomputed_policy_loss, abs=1e-12
    )
    assert result.shared_value_loss == pytest.approx(
        result.recomputed_value_loss, abs=1e-12
    )
    assert result.shared_value_loss > 0
    assert result.prefix_weight == pytest.approx(1.0)
    assert result.branch_weights == pytest.approx((0.5, 0.5))
    assert result.shared_edge_count + 1 == result.recomputed_edge_count
    assert result.prefix_return == pytest.approx(sum(result.branch_returns) / 2)
    assert result.branch_advantages == pytest.approx(
        (
            (result.branch_returns[0] - result.branch_returns[1]) * branch.to_play,
            (result.branch_returns[1] - result.branch_returns[0]) * branch.to_play,
        )
    )
    assert result.clipped_edge_count > 0
    assert result.branch_returns == ((-1, 1) if same_action else (1, -1))


def test_leave_one_out_requires_two_siblings() -> None:
    with pytest.raises(ValueError, match="two"):
        leave_one_out_advantages((1.0,))
