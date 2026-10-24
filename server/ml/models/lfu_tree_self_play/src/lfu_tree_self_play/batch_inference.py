"""Frozen model evaluation shared by sequential and batched collection."""

import math
import time
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import torch

from .game import GameState
from .model import PolicyValueModel


@dataclass(frozen=True)
class EvaluationRequest:
    token: int
    state: GameState


@dataclass(frozen=True)
class EvaluationOutput:
    token: int
    policy_logits: torch.Tensor
    old_value: float
    inference_seconds: float = 0.0


class FrozenBatchEvaluator:
    """Own one round-start snapshot; leave RNG and sampling to collectors."""

    def __init__(self, model: PolicyValueModel, *, model_generation: int) -> None:
        if type(model) is not PolicyValueModel:
            raise ValueError("model must be a PolicyValueModel")
        if type(model_generation) is not int or model_generation < 0:
            raise ValueError("model_generation must be a nonnegative integer")
        self._model = deepcopy(model).eval().requires_grad_(False)
        self.model_generation = model_generation

    @property
    def device(self) -> torch.device:
        return next(self._model.parameters()).device

    def evaluate(
        self, requests: Sequence[EvaluationRequest]
    ) -> tuple[EvaluationOutput, ...]:
        return evaluate_model(self._model, requests)


def evaluate_model(
    model: PolicyValueModel, requests: Sequence[EvaluationRequest]
) -> tuple[EvaluationOutput, ...]:
    """Evaluate an eval-mode snapshot without changing the caller's model mode.

    Snapshot owners set eval() at construction. inference_mode() disables
    autograd bookkeeping, but does not disable dropout or BatchNorm updates.
    """
    if not requests:
        return ()
    if any(module.training for module in model.modules()):
        raise ValueError("model and all submodules must be in eval mode")
    if any(request.state.is_terminal for request in requests):
        raise ValueError("terminal position has no legal move")
    start = time.perf_counter()
    parameter = next(model.parameters())
    observations = torch.from_numpy(
        np.stack([request.state.observation for request in requests])
    ).to(device=parameter.device, dtype=parameter.dtype)
    with torch.inference_mode():
        output = model(observations)
    # MPS cannot convert to float64 on device. Transfer before converting.
    logits = output.policy_logits.detach().cpu().to(torch.float64)
    values = output.value.detach().cpu().to(torch.float64)
    if logits.shape != (len(requests), 64) or values.shape != (len(requests),):
        raise ValueError("invalid model output shape")
    mask = observations[:, 3].flatten(1).bool().cpu()
    if not bool(torch.isfinite(logits[mask]).all()):
        raise ValueError("legal policy logits must be finite")
    if any(
        not math.isfinite(value) or not -1 <= value <= 1 for value in values.tolist()
    ):
        raise ValueError("old_value must be finite within [-1, 1]")
    seconds = (time.perf_counter() - start) / len(requests)
    return tuple(
        EvaluationOutput(request.token, row.clone(), float(value), seconds)
        for request, row, value in zip(requests, logits, values, strict=True)
    )
