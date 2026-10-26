"""Temperature-one sampling from a fixed, legally masked old policy."""

import math
import random
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass

import torch
from torch import Tensor

from .batch_inference import EvaluationOutput, EvaluationRequest, evaluate_model
from .game import GameState
from .model import PolicyValueModel, masked_policy_logits


@dataclass(frozen=True)
class ActionSample:
    action: int
    legal_mask: tuple[bool, ...]
    old_log_probability: float


@dataclass(frozen=True)
class PolicyEvaluation:
    old_value: float
    samples: tuple[ActionSample, ...]


def sample_policy(
    policy_logits: Tensor,
    legal_mask: Tensor,
    rng: random.Random,
    *,
    count: int = 1,
) -> tuple[ActionSample, ...]:
    """Draw with replacement, retaining order, mask, and detached log probabilities.

    Inputs describe one position. The caller supplies the action or continuation
    RNG and logits from the fixed collection policy; no global RNG is consumed.
    """
    if type(count) is not int or count < 1:
        raise ValueError("count must be a positive integer")
    if (
        policy_logits.shape != (64,)
        or not policy_logits.is_floating_point()
        or legal_mask.shape != (64,)
        or legal_mask.dtype != torch.bool
    ):
        raise ValueError("expected 64 floating-point logits and a 64-element bool mask")
    logits = policy_logits.detach().cpu().to(dtype=torch.float64)
    mask = legal_mask.detach().to(device="cpu")
    if not torch.isfinite(logits[mask]).all():
        raise ValueError("legal policy logits must be finite")
    logs = masked_policy_logits(logits.unsqueeze(0), mask.unsqueeze(0)).log_softmax(-1)[
        0
    ]
    weights = logs.exp().tolist()
    actions = rng.choices(range(64), weights=weights, k=count)
    saved_mask = tuple(mask.tolist())
    return tuple(
        ActionSample(action, saved_mask, float(logs[action])) for action in actions
    )


class FrozenPolicySampler:
    """Own a round-start model snapshot isolated from later training updates."""

    def __init__(self, model: PolicyValueModel) -> None:
        self._model = deepcopy(model).eval().requires_grad_(False)

    def evaluate(
        self, requests: Sequence[EvaluationRequest]
    ) -> tuple[EvaluationOutput, ...]:
        """Evaluate the same snapshot for a collector's resumable session."""
        return evaluate_model(self._model, requests)

    def evaluate_and_sample(
        self, state: GameState, rng: random.Random, *, count: int = 1
    ) -> PolicyEvaluation:
        """Infer a nonterminal position once and draw from its temperature-one policy."""
        if state.is_terminal:
            raise ValueError("terminal position has no legal move")
        if type(count) is not int or count < 1:
            raise ValueError("count must be a positive integer")
        parameter = next(self._model.parameters())
        observations = (
            torch.from_numpy(state.observation)
            .unsqueeze(0)
            .to(device=parameter.device, dtype=parameter.dtype)
        )
        mask = observations[0, 3].flatten().bool()
        with torch.no_grad():
            output = self._model(observations)
        old_value = float(output.value[0].detach().cpu().to(dtype=torch.float64).item())
        if not math.isfinite(old_value) or not -1.0 <= old_value <= 1.0:
            raise ValueError("old_value must be finite within [-1, 1]")
        samples = sample_policy(output.policy_logits[0], mask, rng, count=count)
        return PolicyEvaluation(old_value=old_value, samples=samples)

    def sample(
        self, state: GameState, rng: random.Random, *, count: int = 1
    ) -> tuple[ActionSample, ...]:
        """Infer a nonterminal position and draw from its temperature-one policy."""
        return self.evaluate_and_sample(state, rng, count=count).samples
