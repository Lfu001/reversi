"""Shared policy/value network for Reversi positions."""

from typing import NamedTuple

import torch
from torch import Tensor, nn
from transformers import PreTrainedConfig, PreTrainedModel
from transformers.models.resnet.modeling_resnet import ResNetBasicLayer


class PolicyValueOutput(NamedTuple):
    policy_logits: Tensor
    value: Tensor


class PolicyValueConfig(PreTrainedConfig):
    """Architecture parameters saved alongside policy/value weights."""

    model_type = "lfu_reversi_policy_value"

    def __init__(self, width: int = 64, num_blocks: int = 4, **kwargs) -> None:
        super().__init__(**kwargs)
        self.width = width
        self.num_blocks = num_blocks


class PolicyValueModel(PreTrainedModel):
    """Predict 64 placement logits and a mover-perspective value."""

    config_class = PolicyValueConfig
    main_input_name = "observations"

    def __init__(self, config: PolicyValueConfig | None = None) -> None:
        config = config or PolicyValueConfig()
        super().__init__(config)
        self.stem = nn.Sequential(
            nn.Conv2d(4, config.width, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(config.width),
            nn.ReLU(),
        )
        self.residual_blocks = nn.Sequential(
            *(
                ResNetBasicLayer(config.width, config.width, stride=1)
                for _ in range(config.num_blocks)
            )
        )
        self.policy_head = nn.Conv2d(config.width, 1, kernel_size=1)
        self.value_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(config.width, 1),
            nn.Tanh(),
        )
        self.post_init()

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
