"""
Recurrent History Encoder for Context Representation Learning.

This module provides the RNNHistoryEncoder class which uses a Gated Recurrent Unit (GRU)
to map agent interaction histories (state, action, reward sequences) to compact latent
context representations z_t.
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class RNNHistoryEncoder(nn.Module):
    """
    Recurrent Neural Network (GRU) Encoder mapping history sequences of shape
    (batch_size, seq_len, feature_dim) into a compact latent context representation z_t.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        latent_dim: int = 8,
        hidden_dim: int = 64,
        num_layers: int = 1,
    ) -> None:
        """
        Initializes GRU recurrent layers and latent feature projections.

        Args:
            state_dim: Dimension of state observation vector.
            action_dim: Dimension of action representation (continuous dim or discrete one-hot dim).
            latent_dim: Dimension of output latent context vector z_t (default: 8).
            hidden_dim: Hidden dimension of GRU recurrent layer (default: 64).
            num_layers: Number of stacked GRU recurrent layers (default: 1).
        """
        super().__init__()
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.feature_dim = int(state_dim + action_dim + 1)  # state + action + reward
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)

        # Recurrent GRU backbone network
        self.gru = nn.GRU(
            input_size=self.feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )

        # Projection head mapping GRU hidden state to latent context vector z_t
        self.fc_latent = nn.Linear(hidden_dim, latent_dim)

    def forward(
        self, history_sequence: torch.Tensor, h_0: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs forward pass through GRU backbone to produce latent context vector z_t.

        Args:
            history_sequence: Tensor of shape (batch_size, seq_len, feature_dim).
            h_0: Optional initial GRU hidden state tensor of shape (num_layers, batch_size, hidden_dim).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - z_t: Latent context representation tensor of shape (batch_size, latent_dim).
                - h_n: Final GRU hidden state tensor of shape (num_layers, batch_size, hidden_dim).
        """
        # Pass sequence through GRU
        output, h_n = self.gru(history_sequence, h_0)

        # Extract last time step hidden representation output[:, -1, :]
        last_out = output[:, -1, :]

        # Project to latent context representation z_t
        z_t = self.fc_latent(F.relu(last_out))
        return z_t, h_n
