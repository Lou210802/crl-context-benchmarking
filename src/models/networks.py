"""
Policy and Value Network Architectures for Reinforcement Learning.

This module defines PyTorch neural network modules for policy (actor) logits
and value function (critic) scalar estimations using a standard 2-layer MLP architecture
with orthogonal weight initialization for stable PPO optimization.
"""

from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Linear:
    """Helper function to apply orthogonal initialization to PyTorch linear layers."""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Policy(nn.Module):
    """
    Multi-Layer Perceptron (MLP) Discrete Policy Network (2 hidden layers).
    """

    def __init__(self, input_dim: int, action_dim: int, hidden_size: int = 128) -> None:
        super().__init__()
        self.fc1 = layer_init(nn.Linear(input_dim, hidden_size))
        self.fc2 = layer_init(nn.Linear(hidden_size, hidden_size))
        self.fc_out = layer_init(nn.Linear(hidden_size, action_dim), std=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = x.view(x.size(0), -1)
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return self.fc_out(x)


class ContinuousPolicy(nn.Module):
    """
    Multi-Layer Perceptron (MLP) Continuous Gaussian Policy Network (2 hidden layers).
    """

    def __init__(self, input_dim: int, action_dim: int, hidden_size: int = 128) -> None:
        super().__init__()
        self.fc1 = layer_init(nn.Linear(input_dim, hidden_size))
        self.fc2 = layer_init(nn.Linear(hidden_size, hidden_size))
        self.fc_mean = layer_init(nn.Linear(hidden_size, action_dim), std=0.01)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = x.view(x.size(0), -1)
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        mean = self.fc_mean(x)
        std = torch.exp(self.log_std)
        return mean, std


class ValueNetwork(nn.Module):
    """
    Multi-Layer Perceptron (MLP) Value Network (2 hidden layers).
    """

    def __init__(self, input_dim: int, hidden_size: int = 128) -> None:
        super().__init__()
        self.fc1 = layer_init(nn.Linear(input_dim, hidden_size))
        self.fc2 = layer_init(nn.Linear(hidden_size, hidden_size))
        self.fc_out = layer_init(nn.Linear(hidden_size, 1), std=1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = x.view(x.size(0), -1)
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return self.fc_out(x).squeeze(-1)