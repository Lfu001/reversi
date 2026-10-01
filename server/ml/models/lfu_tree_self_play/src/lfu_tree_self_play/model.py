"""Shared policy/value network for Reversi positions."""

from pathlib import Path
from typing import NamedTuple

import torch
from torch import Tensor, nn
from transformers.models.resnet.modeling_resnet import ResNetBasicLayer


class PolicyValueOutput(NamedTuple):
    policy_logits: Tensor
    value: Tensor


class PolicyValueModel(nn.Module):
    """Predict 64 placement logits and a mover-perspective value."""

    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.residual_blocks = nn.Sequential(
            *(ResNetBasicLayer(64, 64, stride=1) for _ in range(4))
        )
        self.policy_head = nn.Conv2d(64, 1, kernel_size=1)
        self.value_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, 1), nn.Tanh()
        )

    def forward(self, observations: Tensor) -> PolicyValueOutput:
        features = self.residual_blocks(self.stem(observations))
        return PolicyValueOutput(
            policy_logits=self.policy_head(features).flatten(start_dim=1),
            value=self.value_head(features).squeeze(-1),
        )


def masked_policy_logits(policy_logits: Tensor, legal_mask: Tensor) -> Tensor:
    """Keep legal placement logits; softmax then assigns illegal moves zero mass."""
    if (
        policy_logits.ndim != 2
        or policy_logits.shape[1] != 64
        or legal_mask.shape != policy_logits.shape
        or legal_mask.dtype != torch.bool
    ):
        raise ValueError(
            "policy logits and legal mask must be matching (batch, 64) tensors"
        )
    if not legal_mask.any(dim=-1).all():
        raise ValueError("each position must have at least one legal move")
    return policy_logits.masked_fill(~legal_mask, -torch.inf)


def save_model(model: PolicyValueModel, path: str | Path) -> None:
    """Save the network weights for inference or later training."""
    torch.save(model.state_dict(), path)


def load_model(path: str | Path) -> PolicyValueModel:
    """Load network weights onto the CPU without accepting pickled objects."""
    model = PolicyValueModel()
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return model
