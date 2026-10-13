"""Temperature-one sampling from a fixed, legally masked old policy."""

import random
from copy import deepcopy
from dataclasses import dataclass

import torch
from torch import Tensor

from .game import GameState
from .model import PolicyValueModel, masked_policy_logits


@dataclass(frozen=True)
class ActionSample:
    action: int
    legal_mask: tuple[bool, ...]
    old_log_probability: float


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
    logits = policy_logits.detach().to(device="cpu", dtype=torch.float64)
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

    def sample(
        self, state: GameState, rng: random.Random, *, count: int = 1
    ) -> tuple[ActionSample, ...]:
        """Infer a nonterminal position and draw from its temperature-one policy."""
        if state.is_terminal:
            raise ValueError("terminal position has no legal move")
        parameter = next(self._model.parameters())
        observations = (
            torch.from_numpy(state.observation)
            .unsqueeze(0)
            .to(device=parameter.device, dtype=parameter.dtype)
        )
        mask = observations[0, 3].flatten().bool()
        with torch.no_grad():
            logits = self._model(observations).policy_logits[0]
        return sample_policy(logits, mask, rng, count=count)
