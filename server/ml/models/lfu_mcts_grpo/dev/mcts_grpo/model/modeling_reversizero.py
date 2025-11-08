import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel

from .configuration_reversizero import ReversiZeroConfig

__all__ = ["ReversiZeroModel"]


class ResidualBlock(nn.Module):
    def __init__(self, num_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(
            num_channels, num_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(
            num_channels, num_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return F.relu(out)


class ReversiZeroModel(PreTrainedModel):
    config_class = ReversiZeroConfig

    def __init__(self, config: ReversiZeroConfig):
        super().__init__(config)
        self.config = config

        self.initial_conv = nn.Sequential(
            nn.Conv2d(4, config.num_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(config.num_channels),
            nn.ReLU(),
        )

        self.residual_blocks = nn.ModuleList(
            [
                ResidualBlock(config.num_channels)
                for _ in range(config.num_residual_blocks)
            ]
        )

        # Policy Head
        self.policy_head_conv = nn.Conv2d(
            config.num_channels, 2, kernel_size=1, bias=False
        )
        self.policy_head_bn = nn.BatchNorm2d(2)
        self.policy_head_fc = nn.Linear(2 * 8 * 8, 8 * 8)

        # Value Head
        self.value_head_conv = nn.Conv2d(
            config.num_channels, 1, kernel_size=1, bias=False
        )
        self.value_head_bn = nn.BatchNorm2d(1)
        self.value_head_fc1 = nn.Linear(1 * 8 * 8, 256)
        self.value_head_fc2 = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Input x shape: [batch, 4, 8, 8]
        out = self.initial_conv(x)
        for block in self.residual_blocks:
            out = block(out)

        # Policy head
        policy_out = self.policy_head_conv(out)
        policy_out = F.relu(self.policy_head_bn(policy_out))
        policy_out = policy_out.view(policy_out.size(0), -1)
        policy_logits = self.policy_head_fc(policy_out)  # [batch, 64]

        # Value head
        value_out = self.value_head_conv(out)
        value_out = F.relu(self.value_head_bn(value_out))
        value_out = value_out.view(value_out.size(0), -1)
        value_out = F.relu(self.value_head_fc1(value_out))
        value = torch.tanh(self.value_head_fc2(value_out))  # [batch, 1]

        return policy_logits.view(-1, 8, 8), value.squeeze(-1)  # [batch, 8, 8], [batch]
