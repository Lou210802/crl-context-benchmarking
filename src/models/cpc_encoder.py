"""
Contrastive Predictive Coding (CPC) Context Representation Learning Module.

This module provides the CPCContextEncoder class, which uses a recurrent GRU backbone
identical in architecture to the VAE encoder backbone to map interaction histories h_t to
a compact context vector z_t, and uses InfoNCE contrastive predictive loss over future
environment transition dynamics across multiple prediction steps k in {1, ..., K}.
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CPCContextEncoder(nn.Module):
    """
    Contrastive Predictive Coding (CPC) encoder for context representation learning in changing environments.

    Encoder: g_enc(h_t) -> compact latent context vector z_t
    Predictor heads: W_k z_t -> predicts target embeddings e_{t+k} across horizon steps k=1..K
    Loss: Multi-step InfoNCE contrastive loss distinguishing true future targets from negative batch distractors.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        latent_dim: int = 8,
        hidden_dim: int = 64,
        predict_horizon: int = 3,
        temperature: float = 0.1,
        device: Optional[torch.device] = None,
    ) -> None:
        """
        Initializes CPC Context Encoder network components.

        Args:
            state_dim: Dimension of state observation vector.
            action_dim: Dimension of action vector.
            latent_dim: Dimension of latent context vector z_t (default: 8).
            hidden_dim: Hidden dimension of GRU recurrent backbone (default: 64).
            predict_horizon: Maximum future prediction horizon steps K (default: 3).
            temperature: Softmax temperature parameter tau for InfoNCE score scaling (default: 0.1).
            device: Computing device (CPU/GPU).
        """
        super().__init__()
        self.device = device if device is not None else torch.device("cpu")
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.feature_dim = int(state_dim + action_dim + 1)  # state + action + reward
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)
        self.predict_horizon = max(1, int(predict_horizon))
        self.temperature = float(temperature)

        # -------------------------------------------------------------
        # 1. Recurrent GRU Backbone (identical architecture to VAE backbone)
        # -------------------------------------------------------------
        self.gru = nn.GRU(
            input_size=self.feature_dim,
            hidden_size=self.hidden_dim,
            batch_first=True,
        )

        # Projection head mapping GRU hidden representation to latent context vector z_t
        self.fc_latent = nn.Linear(self.hidden_dim, self.latent_dim)

        # -------------------------------------------------------------
        # 2. Target Feature Extractor & Bilinear Predictor Heads W_k
        # -------------------------------------------------------------
        target_dim = self.state_dim + 1  # next_state + reward
        self.target_encoder = nn.Sequential(
            nn.Linear(target_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.latent_dim),
        )

        # Bilinear / Linear projection matrices W_k for predicting k steps into the future
        self.predictors = nn.ModuleList(
            [nn.Linear(self.latent_dim, self.latent_dim, bias=False) for _ in range(self.predict_horizon)]
        )

        self.to(self.device)

    def encode(self, history_sequence: torch.Tensor) -> torch.Tensor:
        """
        Encodes trajectory history sequence into a compact latent context representation vector z_t.

        Args:
            history_sequence: History tensor of shape (batch_size, seq_len, feature_dim).

        Returns:
            torch.Tensor: Context vector z_t of shape (batch_size, latent_dim).
        """
        history_sequence = history_sequence.to(self.device)
        output, _ = self.gru(history_sequence)
        last_hidden = F.relu(output[:, -1, :])
        z_t = self.fc_latent(last_hidden)
        return z_t

    def forward(self, history_sequence: torch.Tensor) -> torch.Tensor:
        """Forward passAlias returning latent context z_t."""
        return self.encode(history_sequence)

    def compute_loss(
        self,
        history_sequence: torch.Tensor,
        state: torch.Tensor,
        action: torch.Tensor,
        target_next_state: torch.Tensor,
        target_reward: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Computes multi-step InfoNCE contrastive predictive loss across batch samples.

        Args:
            history_sequence: Tensor of shape (batch_size, seq_len, feature_dim).
            state: Current state tensor of shape (batch_size, state_dim).
            action: Action tensor of shape (batch_size, action_dim).
            target_next_state: Target next state tensor of shape (batch_size, state_dim).
            target_reward: Target reward tensor of shape (batch_size, 1).

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing total 'loss' and 'cpc_loss'.
        """
        history_sequence = history_sequence.to(self.device)
        state = state.to(self.device)
        action = action.to(self.device)
        target_next_state = target_next_state.to(self.device)
        target_reward = target_reward.to(self.device)

        batch_size = history_sequence.size(0)
        z_t = self.encode(history_sequence)  # (batch_size, latent_dim)

        # Build target representation e_target from target next state and reward
        target_input = torch.cat([target_next_state, target_reward], dim=-1)
        target_e = self.target_encoder(target_input)  # (batch_size, latent_dim)
        target_e = F.normalize(target_e, p=2, dim=-1)

        total_infonce_loss = torch.tensor(0.0, device=self.device)

        for k in range(self.predict_horizon):
            pred_k = self.predictors[k](z_t)  # (batch_size, latent_dim)
            pred_k = F.normalize(pred_k, p=2, dim=-1)

            # Similarity logits matrix: (batch_size, batch_size)
            # Entry (i, j) is dot product between prediction for sample i and target for sample j
            logits = torch.matmul(pred_k, target_e.T) / self.temperature

            # Ground truth targets for InfoNCE contrastive loss: positive pair is on diagonal i == j
            labels = torch.arange(batch_size, device=self.device, dtype=torch.long)

            loss_k = F.cross_entropy(logits, labels)
            total_infonce_loss = total_infonce_loss + loss_k

        mean_cpc_loss = total_infonce_loss / float(self.predict_horizon)

        return {
            "loss": mean_cpc_loss,
            "cpc_loss": mean_cpc_loss,
        }
