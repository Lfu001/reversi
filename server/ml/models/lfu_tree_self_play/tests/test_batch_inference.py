"""Batch inference keeps model ownership and sampling semantics intact."""

import random
from copy import deepcopy

import pytest
import torch

from lfu_tree_self_play.batch_inference import (
    EvaluationRequest,
    FrozenBatchEvaluator,
    evaluate_model,
)
from lfu_tree_self_play.game import ReversiPyGame
from lfu_tree_self_play.model import PolicyValueConfig, PolicyValueModel
from lfu_tree_self_play.sampler import sample_policy
from lfu_tree_self_play.verification import endgame_position


def model():
    torch.manual_seed(88)
    return PolicyValueModel(PolicyValueConfig(width=8, num_blocks=1))


def test_batch_output_order_and_masked_sampling():
    evaluator = FrozenBatchEvaluator(model(), model_generation=2)
    states = [
        ReversiPyGame().initial_state(),
        endgame_position(1),
        endgame_position(-1),
    ]
    requests = [EvaluationRequest(index, state) for index, state in enumerate(states)]
    outputs = evaluator.evaluate(requests)
    assert [output.token for output in outputs] == [0, 1, 2]
    for request, output in zip(requests, outputs, strict=True):
        reference = evaluator.evaluate([request])[0]
        torch.testing.assert_close(
            output.policy_logits, reference.policy_logits, atol=1e-6, rtol=0
        )
        assert abs(output.old_value - reference.old_value) <= 1e-6
        assert output.policy_logits.device.type == "cpu"
        assert output.policy_logits.dtype == torch.float64
        mask = torch.from_numpy(request.state.observation[3]).flatten().bool()
        samples = sample_policy(output.policy_logits, mask, random.Random(1), count=4)
        assert all(sample.action in request.state.legal_actions for sample in samples)
        assert output.inference_seconds >= 0


def test_empty_batch_skips_forward():
    assert FrozenBatchEvaluator(model(), model_generation=0).evaluate([]) == ()


@pytest.mark.parametrize("mode", ["training", "mixed", "eval"])
def test_evaluate_model_uses_eval_without_changing_caller_state(mode):
    caller = model()
    if mode != "training":
        caller.eval()
    if mode == "mixed":
        caller.stem[1].train()
    modes = [module.training for module in caller.modules()]
    before = {name: value.clone() for name, value in caller.state_dict().items()}
    request = EvaluationRequest(0, endgame_position(1))
    with torch.no_grad():
        expected = deepcopy(caller).eval()(
            torch.from_numpy(request.state.observation).unsqueeze(0)
        )
    actual = evaluate_model(caller, [request])[0]
    torch.testing.assert_close(
        actual.policy_logits, expected.policy_logits[0].double(), atol=0, rtol=0
    )
    assert actual.old_value == float(expected.value[0])
    assert [module.training for module in caller.modules()] == modes
    for name, value in caller.state_dict().items():
        assert torch.equal(value, before[name])
    assert all(parameter.requires_grad for parameter in caller.parameters())


def test_evaluate_model_restores_mixed_modes_when_forward_fails():
    caller = model().train()
    caller.stem[1].eval()
    modes = [module.training for module in caller.modules()]

    def fail_forward(module, args):
        assert not any(child.training for child in module.modules())
        assert torch.is_inference_mode_enabled()
        raise RuntimeError("forward failed")

    handle = caller.register_forward_pre_hook(fail_forward)
    try:
        with pytest.raises(RuntimeError, match="forward failed"):
            evaluate_model(caller, [EvaluationRequest(0, endgame_position(1))])
    finally:
        handle.remove()
    assert [module.training for module in caller.modules()] == modes


def test_training_can_continue_after_evaluation():
    caller = model().train()
    optimizer = torch.optim.SGD(caller.parameters(), lr=0.1)
    request = EvaluationRequest(0, endgame_position(1))
    observations = torch.from_numpy(request.state.observation).unsqueeze(0)
    for _ in range(2):
        optimizer.zero_grad()
        output = caller(observations)
        (output.value.sum() + output.policy_logits.square().mean()).backward()
        optimizer.step()
        gradients = [parameter.grad.clone() for parameter in caller.parameters()]
        evaluate_model(caller, [request])
        assert all(module.training for module in caller.modules())
        for parameter, gradient in zip(caller.parameters(), gradients, strict=True):
            assert torch.equal(parameter.grad, gradient)


def test_snapshot_does_not_mutate_caller_or_global_rng():
    caller = model().train()
    evaluator = FrozenBatchEvaluator(caller, model_generation=0)
    request = EvaluationRequest(0, endgame_position(1))
    python_state, torch_state = random.getstate(), torch.get_rng_state().clone()
    first = evaluator.evaluate([request])[0]
    with torch.no_grad():
        for parameter in caller.parameters():
            parameter.add_(30)
    second = evaluator.evaluate([request])[0]
    assert caller.training and all(p.requires_grad for p in caller.parameters())
    assert all(p.grad is None for p in caller.parameters())
    assert random.getstate() == python_state
    assert torch.equal(torch.get_rng_state(), torch_state)
    assert first.old_value == second.old_value
    assert torch.equal(first.policy_logits, second.policy_logits)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires MPS")
def test_cpu_then_float64_transfer_preserves_values():
    logits = torch.zeros(64)
    logits[26] = 2
    mask = torch.zeros(64, dtype=torch.bool)
    mask[[19, 26]] = True
    cpu = sample_policy(logits, mask, random.Random(88), count=100)
    gpu = sample_policy(logits.to("mps"), mask.to("mps"), random.Random(88), count=100)
    assert cpu == gpu
