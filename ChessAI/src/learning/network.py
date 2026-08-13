"""PyTorch chess neural network (AlphaZero-style).

    Position (18 x 8 x 8 planes)
        -> convolutional trunk + residual blocks
        -> policy head (4096 move logits)
        -> value head (scalar in [-1, 1] from the side-to-move's perspective)

This is the model consumed by the policy/value heads for future MCTS; the
current infrastructure trains it directly from self-play policy/value targets.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(x + out)


class ChessNet(nn.Module):
    """Position encoder + policy head + value head."""

    def __init__(self, in_channels: int = 18, channels: int = 64, res_blocks: int = 2) -> None:
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )
        self.res_blocks = nn.ModuleList([ResidualBlock(channels) for _ in range(res_blocks)])

        # Policy head.
        self.policy_conv = nn.Sequential(
            nn.Conv2d(channels, 32, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.policy_fc = nn.Linear(32 * 64, 4096)

        # Value head.
        self.value_conv = nn.Sequential(
            nn.Conv2d(channels, 1, 1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
        )
        self.value_fc1 = nn.Linear(64, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.conv_block(x)
        for block in self.res_blocks:
            x = block(x)

        policy = self.policy_conv(x)
        policy = policy.view(policy.size(0), -1)
        policy_logits = self.policy_fc(policy)

        value = self.value_conv(x)
        value = value.view(value.size(0), -1)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))

        return policy_logits, value

    def policy_distribution(self, x: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        logits, _ = self.forward(x)
        return F.softmax(logits / max(temperature, 1e-9), dim=-1)


def create_chess_net(in_channels: int = 18, channels: int = 64, res_blocks: int = 2) -> ChessNet:
    """Factory for the default network."""
    return ChessNet(in_channels=in_channels, channels=channels, res_blocks=res_blocks)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)