"""
Policy and Value Network Architectures for Reinforcement Learning.

This module defines PyTorch neural network modules for policy (actor) logits
and value function (critic) scalar estimations.
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class Policy(nn.Module):
    """
    Multi-Layer Perceptron (MLP) Policy Network mapping state observations
    (or state concatenated with context vectors) to unnormalized action logits.
    """

    def __init__(self, input_dim: int, action_dim: int, hidden_size: int = 128) -> None:
        """
        Initializes linear layers for the policy MLP network.
        """
        super().__init__()
        # Input state projection layer
        self.fc1 = nn.Linear(input_dim, hidden_size)
        # Action logits output layer
        self.fc2 = nn.Linear(hidden_size, action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs forward pass to compute unnormalized action logits for input states.
        """
        # Ensure 2D tensor shape (batch_size, input_dim) for single state inputs
        if x.dim() == 1:
            x = x.unsqueeze(0)

        # Flatten input tensor features
        x = x.view(x.size(0), -1)

        # First hidden layer with ReLU activation
        x = F.relu(self.fc1(x))

        # Output linear layer returning action logits
        return self.fc2(x)


class ContinuousPolicy(nn.Module):
    """
    Multi-Layer Perceptron (MLP) Gaussian Policy Network mapping state observations
    (or state concatenated with context vectors) to continuous action means and learnable log standard deviations.
    """

    def __init__(self, input_dim: int, action_dim: int, hidden_size: int = 128) -> None:
        """
        Initializes linear layers for continuous Gaussian policy MLP network.
        """
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.fc_mean = nn.Linear(hidden_size, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs forward pass to compute action mean and standard deviation tensors.
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        mean = self.fc_mean(x)
        std = torch.exp(self.log_std)
        return mean, std


class ValueNetwork(nn.Module):
    """
    Multi-Layer Perceptron (MLP) Value Network mapping state observations
    (or state concatenated with context vectors) to scalar state-value estimates.
    """

    def __init__(self, input_dim: int, hidden_size: int = 128) -> None:
        """
        Initializes linear layers for the value function MLP network.
        """
        super().__init__()
        # Input state projection layer
        self.fc1 = nn.Linear(input_dim, hidden_size)
        # Scalar value estimate output layer
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs forward pass to compute scalar value predictions for input states.
        """
        # Ensure 2D tensor shape (batch_size, input_dim) for single state inputs
        if x.dim() == 1:
            x = x.unsqueeze(0)

        # Flatten input tensor features
        x = x.view(x.size(0), -1)

        # First hidden layer with ReLU activation
        x = F.relu(self.fc1(x))

        # Output linear layer squeezed to 1D tensor of shape (batch_size,)
        return self.fc2(x).squeeze(-1)