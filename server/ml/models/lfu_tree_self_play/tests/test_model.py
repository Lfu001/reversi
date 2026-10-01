"""Contract tests for the shared policy/value model."""

import numpy as np
import pytest
import torch
from lfu_tree_self_play.game import ReversiPyGame


@pytest.mark.parametrize("batch_size", [1, 3])
def test_model_returns_policy_and_value_for_batched_positions(batch_size: int) -> None:
    from lfu_tree_self_play.model import PolicyValueModel

    game = ReversiPyGame()
    first = game.initial_state()
    positions = [first, game.step(first, 19), game.step(first, 26)]
    observations = torch.from_numpy(
        np.stack([position.observation for position in positions[:batch_size]])
    )

    result = PolicyValueModel()(observations)

    assert result.policy_logits.shape == (batch_size, 64)
    assert result.value.shape == (batch_size,)
    assert torch.isfinite(result.policy_logits).all()
    assert torch.isfinite(result.value).all()
    assert ((result.value >= -1) & (result.value <= 1)).all()


def test_masked_policy_assigns_probability_only_to_legal_moves() -> None:
    from lfu_tree_self_play.model import masked_policy_logits

    game = ReversiPyGame()
    first = game.initial_state()
    second = game.step(first, 19)
    legal_mask = torch.from_numpy(
        np.stack([state.observation[3].reshape(64) for state in (first, second)])
    ).bool()
    logits = torch.zeros((2, 64))

    probabilities = masked_policy_logits(logits, legal_mask).softmax(dim=-1)

    assert torch.all(probabilities[~legal_mask] == 0)
    assert torch.all(probabilities[legal_mask] > 0)
    torch.testing.assert_close(probabilities.sum(dim=-1), torch.ones(2))
    torch.testing.assert_close(probabilities[0, 19], torch.tensor(0.25))
    torch.testing.assert_close(probabilities[1, 18], torch.tensor(1 / 3))


def test_masked_policy_rejects_terminal_position() -> None:
    from lfu_tree_self_play.model import masked_policy_logits

    with pytest.raises(ValueError, match="legal move"):
        masked_policy_logits(
            torch.zeros((1, 64)), torch.zeros((1, 64), dtype=torch.bool)
        )


def test_policy_and_value_heads_both_receive_gradients() -> None:
    from lfu_tree_self_play.model import PolicyValueModel

    model = PolicyValueModel()
    observation = torch.from_numpy(
        np.stack([ReversiPyGame().initial_state().observation])
    )

    output = model(observation)
    (output.policy_logits.sum() + output.value.sum()).backward()

    assert model.policy_head.weight.grad is not None
    assert torch.any(model.policy_head.weight.grad != 0)
    assert model.value_head[2].weight.grad is not None
    assert torch.any(model.value_head[2].weight.grad != 0)


def test_saved_model_reloads_with_identical_outputs(tmp_path) -> None:
    from lfu_tree_self_play.model import PolicyValueModel, load_model, save_model

    model = PolicyValueModel().eval()
    observation = torch.from_numpy(
        np.stack([ReversiPyGame().initial_state().observation])
    )
    before = model(observation)
    checkpoint = tmp_path / "policy-value.pt"

    save_model(model, checkpoint)
    reloaded = load_model(checkpoint).eval()
    after = reloaded(observation)

    torch.testing.assert_close(
        after.policy_logits, before.policy_logits, rtol=0, atol=0
    )
    torch.testing.assert_close(after.value, before.value, rtol=0, atol=0)
