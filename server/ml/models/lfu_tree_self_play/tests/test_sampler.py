"""Acceptance tests for temperature-one, replacement policy sampling."""

import math
import random
from collections import Counter
from copy import deepcopy

import pytest
import torch

from lfu_tree_self_play.game import ReversiPyGame
from lfu_tree_self_play.model import PolicyValueConfig, PolicyValueModel
from lfu_tree_self_play.rng import create_rng_streams


def distribution():
    logits = torch.full((64,), 1000.0, dtype=torch.float64)
    logits[19], logits[26] = 0.0, math.log(3)
    mask = torch.zeros(64, dtype=torch.bool)
    mask[[19, 26]] = True
    return logits, mask


def test_frequency_matches_masked_policy_and_retains_duplicates():
    from lfu_tree_self_play.sampler import sample_policy

    logits, mask = distribution()
    draws = sample_policy(logits, mask, random.Random(83), count=20_000)
    counts = Counter(draw.action for draw in draws)

    assert set(counts) == {19, 26}
    assert abs(counts[19] / len(draws) - 0.25) < 0.015
    assert abs(counts[26] / len(draws) - 0.75) < 0.015
    assert len(draws) == 20_000
    for draw in draws:
        expected = {19: 0.25, 26: 0.75}[draw.action]
        assert draw.old_log_probability == pytest.approx(math.log(expected))
        assert math.exp(math.log(expected) - draw.old_log_probability) == pytest.approx(
            1
        )
        assert draw.legal_mask == tuple(mask.tolist())


def test_single_legal_action_is_sampled_repeatedly_with_zero_log_probability():
    from lfu_tree_self_play.sampler import sample_policy

    mask = torch.zeros(64, dtype=torch.bool)
    mask[7] = True
    draws = sample_policy(torch.zeros(64), mask, random.Random(1), count=4)

    assert [draw.action for draw in draws] == [7, 7, 7, 7]
    assert [draw.old_log_probability for draw in draws] == [0.0] * 4
    mask.zero_()
    assert all(draw.legal_mask[7] for draw in draws)


def test_sampling_reproduces_after_rng_restore_and_ignores_other_streams():
    from lfu_tree_self_play.sampler import sample_policy

    logits, mask = distribution()
    first, second = create_rng_streams(83), create_rng_streams(83)
    for _ in range(100):
        first.branch_position.random()
    draws = sample_policy(logits, mask, first.action_sampling, count=40)
    assert draws == sample_policy(logits, mask, second.action_sampling, count=40)
    checkpoint = first.action_sampling.getstate()
    continuation = sample_policy(logits, mask, first.action_sampling, count=40)
    first.action_sampling.setstate(checkpoint)
    assert continuation == sample_policy(logits, mask, first.action_sampling, count=40)


def test_batched_draws_preserve_sequential_draw_order_and_rng_position():
    from lfu_tree_self_play.sampler import sample_policy

    logits, mask = distribution()
    batch_rng, sequential_rng = random.Random(83), random.Random(83)
    batch = sample_policy(logits, mask, batch_rng, count=40)
    sequential = tuple(
        sample_policy(logits, mask, sequential_rng)[0] for _ in range(40)
    )
    assert batch == sequential
    assert batch_rng.getstate() == sequential_rng.getstate()


def test_nonfinite_illegal_logits_are_excluded():
    from lfu_tree_self_play.sampler import sample_policy

    logits, mask = distribution()
    logits[~mask] = torch.nan
    draws = sample_policy(logits, mask, random.Random(83), count=4)
    assert all(draw.action in (19, 26) for draw in draws)
    assert all(math.isfinite(draw.old_log_probability) for draw in draws)


@pytest.mark.parametrize("count", [0, -1, True, 1.5])
def test_invalid_sample_count_does_not_consume_rng(count):
    from lfu_tree_self_play.sampler import sample_policy

    logits, mask = distribution()
    rng = random.Random(1)
    before = rng.getstate()
    with pytest.raises(ValueError):
        sample_policy(logits, mask, rng, count=count)
    assert rng.getstate() == before


@pytest.mark.parametrize(
    "case", ["terminal", "shape", "mask_type", "nan", "inf", "integer"]
)
def test_invalid_policy_inputs_do_not_consume_rng(case):
    from lfu_tree_self_play.sampler import sample_policy

    logits, mask = distribution()
    if case == "terminal":
        mask.zero_()
    elif case == "shape":
        logits = logits[:63]
    elif case == "mask_type":
        mask = mask.float()
    elif case == "integer":
        logits = logits.long()
    else:
        logits[19] = float(case)
    rng = random.Random(1)
    before = rng.getstate()
    with pytest.raises(ValueError):
        sample_policy(logits, mask, rng)
    assert rng.getstate() == before


def test_extreme_logits_produce_finite_logs_without_gradients():
    from lfu_tree_self_play.sampler import sample_policy

    logits, mask = distribution()
    logits[19], logits[26] = -10_000, 10_000
    logits.requires_grad_()
    draws = sample_policy(logits, mask, random.Random(1), count=4)
    assert all(draw.action == 26 and draw.old_log_probability == 0 for draw in draws)
    assert logits.grad is None


def test_frozen_sampler_isolated_from_training_model_and_rng_globals():
    from lfu_tree_self_play.sampler import FrozenPolicySampler

    torch.manual_seed(83)
    model = PolicyValueModel(PolicyValueConfig(width=8, num_blocks=1)).train()
    state = ReversiPyGame().initial_state()
    sampler = FrozenPolicySampler(model)
    rng = random.Random(83)
    rng_checkpoint = rng.getstate()
    torch_checkpoint = torch.random.get_rng_state().clone()
    python_checkpoint = random.getstate()
    model_checkpoint = {key: value.clone() for key, value in model.state_dict().items()}
    before = sampler.sample(state, rng, count=12)

    assert model.training
    assert all(parameter.requires_grad for parameter in model.parameters())
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, model_checkpoint[key], rtol=0, atol=0)
    assert torch.equal(torch.random.get_rng_state(), torch_checkpoint)
    assert random.getstate() == python_checkpoint
    assert all(draw.action in state.legal_actions for draw in before)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(10)
        for buffer in model.buffers():
            buffer.fill_(100)
    rng.setstate(rng_checkpoint)
    assert sampler.sample(state, rng, count=12) == before
    assert all(parameter.grad is None for parameter in model.parameters())


def test_frozen_sampler_stored_logs_match_snapshot_policy_at_temperature_one():
    from lfu_tree_self_play.model import masked_policy_logits
    from lfu_tree_self_play.sampler import FrozenPolicySampler

    model = PolicyValueModel(PolicyValueConfig(width=8, num_blocks=1)).train()
    reference = deepcopy(model).eval()
    state = ReversiPyGame().initial_state()
    observations = torch.from_numpy(state.observation).unsqueeze(0)
    with torch.no_grad():
        logits = reference(observations).policy_logits.double()
        mask = observations[:, 3].flatten(1).bool()
        logs = masked_policy_logits(logits, mask).log_softmax(-1)[0]
    draws = FrozenPolicySampler(model).sample(state, random.Random(83), count=8)
    for draw in draws:
        assert math.exp(
            float(logs[draw.action]) - draw.old_log_probability
        ) == pytest.approx(1)


def test_frozen_sampler_rejects_terminal_without_advancing_rng():
    from lfu_tree_self_play.sampler import FrozenPolicySampler

    game = ReversiPyGame()
    state = game.initial_state()
    while not state.is_terminal:
        state = game.step(state, state.legal_actions[0])
    sampler = FrozenPolicySampler(
        PolicyValueModel(PolicyValueConfig(width=8, num_blocks=1))
    )
    rng = random.Random(83)
    checkpoint = rng.getstate()
    with pytest.raises(ValueError, match="terminal"):
        sampler.sample(state, rng)
    assert rng.getstate() == checkpoint


def test_evaluate_and_sample_returns_detached_value_and_matching_draws():
    from lfu_tree_self_play.sampler import FrozenPolicySampler

    torch.manual_seed(17)
    model = PolicyValueModel(PolicyValueConfig(width=8, num_blocks=1)).train()
    reference = deepcopy(model).eval()
    state = ReversiPyGame().initial_state()
    observations = torch.from_numpy(state.observation).unsqueeze(0)
    with torch.no_grad():
        expected_value = float(reference(observations).value[0].double().item())

    sampler = FrozenPolicySampler(model)
    evaluation = sampler.evaluate_and_sample(state, random.Random(83), count=4)
    assert type(evaluation.old_value) is float
    assert -1.0 <= evaluation.old_value <= 1.0
    assert evaluation.old_value == pytest.approx(expected_value)
    assert evaluation.samples == sampler.sample(state, random.Random(83), count=4)
