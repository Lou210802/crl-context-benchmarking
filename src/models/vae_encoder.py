"""
Variational Autoencoder (VAE) Context Representation Learning Module with GPU Support.

This module provides the VAEContextEncoder class, which uses a recurrent GRU backbone
with the reparameterization trick to map interaction histories h_t to latent Gaussian context
distributions q_phi(z_t | h_t), and a decoder network p_theta(s_{t+1}, r_t | s_t, a_t, z_t)
that reconstructs environment transition dynamics and rewards on GPU/CPU devices.
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class VAEContextEncoder(nn.Module):
    """
    Variational Autoencoder (VAE) for context representation learning in changing environments.
    
    Encoder: q_phi(z_t | h_t) -> outputs mean mu_z and log_var_z
    Decoder: p_theta(s_{t+1}, r_t | s_t, a_t, z_t) -> predicts next_state and reward
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        latent_dim: int = 8,
        hidden_dim: int = 64,
        predict_horizon: int = 3,
        kl_weight: float = 1e-4,
        reward_weight: float = 10.0,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.device = device if device is not None else torch.device("cpu")
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.feature_dim = int(state_dim + action_dim + 1)  # state + action + reward
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)
        self.predict_horizon = max(1, int(predict_horizon))
        self.kl_weight = float(kl_weight)
        self.reward_weight = float(reward_weight)

        self.feature_norm = nn.LayerNorm(self.feature_dim)
        self.layer_norm = nn.LayerNorm(self.hidden_dim)

        # -------------------------------------------------------------
        # 1. Recurrent GRU Encoder Backbone
        # -------------------------------------------------------------
        self.gru = nn.GRU(
            input_size=self.feature_dim,
            hidden_size=self.hidden_dim,
            batch_first=True,
        )

        # Heads for Gaussian latent distribution parameters (mu, log_var)
        self.fc_mu = nn.Linear(self.hidden_dim, self.latent_dim)
        self.fc_log_var = nn.Linear(self.hidden_dim, self.latent_dim)

        # -------------------------------------------------------------
        # 2. Transition Dynamics & Reward Decoder Network
        # -------------------------------------------------------------
        decoder_input_dim = self.state_dim + self.action_dim + self.latent_dim
        self.decoder = nn.Sequential(
            nn.Linear(decoder_input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.state_dim + 1),
        )

        self.to(self.device)

    def encode(self, history_sequence: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encodes trajectory history sequence into Gaussian distribution parameters (mu, log_var).
        """
        history_sequence = self.feature_norm(history_sequence.to(self.device))
        output, _ = self.gru(history_sequence)
        last_hidden = self.layer_norm(output[:, -1, :])
        mu_z = self.fc_mu(last_hidden)
        log_var_z = self.fc_log_var(last_hidden)
        return mu_z, log_var_z

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        Applies the reparameterization trick z = mu + std * epsilon.
        """
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def decode(
        self, state: torch.Tensor, action: torch.Tensor, z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decodes state delta s_{t+1} - s_t and reward r_t from [state, action, latent z].
        """
        state = state.to(self.device)
        action = action.to(self.device)
        z = z.to(self.device)
        decoder_input = torch.cat([state, action, z], dim=-1)
        pred_out = self.decoder(decoder_input)
        pred_delta_state = pred_out[:, : self.state_dim]
        pred_reward = pred_out[:, self.state_dim :]
        pred_next_state = state + pred_delta_state
        return pred_next_state, pred_reward

    def forward(
        self,
        history_sequence: torch.Tensor,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu_z, log_var_z = self.encode(history_sequence)
        z_t = self.reparameterize(mu_z, log_var_z)
        pred_next_state, pred_reward = self.decode(state, action, z_t)
        return z_t, pred_next_state, pred_reward, mu_z, log_var_z

    def compute_loss(
        self,
        history_sequence: torch.Tensor,
        state: torch.Tensor,
        action: torch.Tensor,
        target_next_state: torch.Tensor,
        target_reward: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Computes Multi-Step Autoregressive Residual VAE Loss over prediction horizon K.
        Supports both 2D single-step tensors (B, dim) and 3D multi-step horizon tensors (B, K, dim).
        """
        history_sequence = history_sequence.to(self.device)
        state = state.to(self.device)
        action = action.to(self.device)
        target_next_state = target_next_state.to(self.device)
        target_reward = target_reward.to(self.device)

        if state.ndim == 2:
            state = state.unsqueeze(1)
            action = action.unsqueeze(1)
            target_next_state = target_next_state.unsqueeze(1)
            target_reward = target_reward.unsqueeze(1)

        batch_size, horizon, _ = state.shape
        mu_z, log_var_z = self.encode(history_sequence)
        z_t = self.reparameterize(mu_z, log_var_z)

        curr_state = state[:, 0, :]
        total_state_loss = torch.tensor(0.0, device=self.device)
        total_reward_loss = torch.tensor(0.0, device=self.device)

        for k in range(horizon):
            act_k = action[:, k, :]
            target_next_s_k = target_next_state[:, k, :]
            target_r_k = target_reward[:, k, :]

            dec_in = torch.cat([curr_state, act_k, z_t], dim=-1)
            pred_out = self.decoder(dec_in)

            pred_delta_s = pred_out[:, : self.state_dim]
            pred_r = pred_out[:, self.state_dim :]

            pred_next_s = curr_state + pred_delta_s
            target_delta_s = target_next_s_k - curr_state

            total_state_loss = total_state_loss + F.mse_loss(pred_delta_s, target_delta_s)
            total_reward_loss = total_reward_loss + F.mse_loss(pred_r, target_r_k)

            # Autoregressive rollout step
            curr_state = pred_next_s

        state_loss = total_state_loss / float(horizon)
        reward_loss = total_reward_loss / float(horizon)
        kl_loss = -0.5 * torch.sum(1 + log_var_z - mu_z.pow(2) - log_var_z.exp(), dim=-1).mean()

        total_loss = state_loss + self.reward_weight * reward_loss + self.kl_weight * kl_loss

        return {
            "loss": total_loss,
            "state_loss": state_loss,
            "reward_loss": reward_loss,
            "kl_loss": kl_loss,
        }
