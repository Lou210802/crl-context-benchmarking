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

        self.feature_norm = nn.LayerNorm(self.feature_dim)
        self.layer_norm = nn.LayerNorm(self.hidden_dim)

        # Predictor Heads W_k mapping latent context z_t to target transition embeddings
        pred_in_dim = self.latent_dim
        self.predictors = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(pred_in_dim, self.hidden_dim),
                    nn.ReLU(),
                    nn.Linear(self.hidden_dim, self.latent_dim),
                )
                for _ in range(self.predict_horizon)
            ]
        )

        # Auxiliary Dynamics MSE Decoder Head (reconstructs delta_s and reward directly for continuous supervision)
        dyn_in_dim = self.state_dim + self.action_dim + self.latent_dim
        self.dyn_decoder = nn.Sequential(
            nn.Linear(dyn_in_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.state_dim + 1),
        )

        self.to(self.device)

    def encode(self, history_sequence: torch.Tensor) -> torch.Tensor:
        """
        Encodes trajectory history sequence into a compact latent context vector z_t.
        """
        history_sequence = self.feature_norm(history_sequence.to(self.device))
        output, _ = self.gru(history_sequence)
        last_hidden = self.layer_norm(output[:, -1, :])
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
        pos_history_sequence: Optional[torch.Tensor] = None,
        ep_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Computes Multi-Step or Inter-Instance InfoNCE contrastive predictive loss.
        Applies false-negative masking to ignore distractors from the same environment context episode,
        and detaches target embeddings for stable InfoNCE gradients.
        """
        history_sequence = history_sequence.to(self.device)
        batch_size = history_sequence.size(0)
        z_t = self.encode(history_sequence)  # (batch_size, latent_dim)
        z_t_norm = F.normalize(z_t, p=2, dim=-1)

        # Supervised Context Contrastive (SupCon) Loss over same-context history pairs
        supcon_loss = torch.tensor(0.0, device=self.device)
        if ep_ids is not None and batch_size > 1:
            ep_ids = ep_ids.to(self.device)
            eye = torch.eye(batch_size, device=self.device, dtype=torch.bool)
            same_ctx = (ep_ids.unsqueeze(0) == ep_ids.unsqueeze(1)) & (~eye)
            diff_ctx = (ep_ids.unsqueeze(0) != ep_ids.unsqueeze(1))

            sim_matrix = torch.matmul(z_t_norm, z_t_norm.T) / self.temperature
            max_sim, _ = torch.max(sim_matrix, dim=1, keepdim=True)
            logits = sim_matrix - max_sim.detach()
            exp_logits = torch.exp(logits)

            sup_losses = []
            for i in range(batch_size):
                pos_idx = torch.where(same_ctx[i])[0]
                if len(pos_idx) > 0:
                    neg_sum = torch.sum(exp_logits[i, diff_ctx[i]])
                    pos_exp = exp_logits[i, pos_idx]
                    loss_i = -torch.log(pos_exp / (pos_exp + neg_sum + 1e-8)).mean()
                    sup_losses.append(loss_i)

            if sup_losses:
                supcon_loss = torch.stack(sup_losses).mean()

        state = state.to(self.device)
        action = action.to(self.device)
        target_next_state = target_next_state.to(self.device)
        target_reward = target_reward.to(self.device)

        if target_next_state.ndim == 2:
            target_next_state = target_next_state.unsqueeze(1)
            target_reward = target_reward.unsqueeze(1)

        horizon = target_next_state.size(1)
        total_infonce_loss = torch.tensor(0.0, device=self.device)
        total_aux_loss = torch.tensor(0.0, device=self.device)
        eval_horizon = min(self.predict_horizon, horizon)

        neg_same_ep_mask = None
        if ep_ids is not None:
            ep_ids = ep_ids.to(self.device)
            same_ep_mask = (ep_ids.unsqueeze(0) == ep_ids.unsqueeze(1))
            eye_mask = torch.eye(batch_size, device=self.device, dtype=torch.bool)
            neg_same_ep_mask = same_ep_mask & (~eye_mask)

        for k in range(eval_horizon):
            target_next_s_k = target_next_state[:, k, :]
            target_r_k = target_reward[:, k, :]
            curr_s_k = state[:, k, :] if state.shape[1] > k else state[:, 0, :]
            delta_s_k = target_next_s_k - curr_s_k

            target_input_k = torch.cat([delta_s_k, target_r_k], dim=-1)
            target_e_k = self.target_encoder(target_input_k)
            target_e_k = F.normalize(target_e_k, p=2, dim=-1).detach()

            pred_in_k = z_t_norm
            pred_k = self.predictors[k](pred_in_k)  # (batch_size, latent_dim)
            pred_k = F.normalize(pred_k, p=2, dim=-1)

            logits = torch.matmul(pred_k, target_e_k.T) / self.temperature
            labels = torch.arange(batch_size, device=self.device, dtype=torch.long)
            loss_k = F.cross_entropy(logits, labels)
            total_infonce_loss = total_infonce_loss + loss_k

            # Auxiliary Dynamics MSE Loss for continuous parameter regression
            act_k = action[:, k, :] if action.shape[1] > k else action[:, 0, :]
            dyn_pred_k = self.dyn_decoder(torch.cat([curr_s_k, act_k, z_t], dim=-1))
            dyn_target_k = torch.cat([delta_s_k, target_r_k], dim=-1)
            aux_loss_k = F.mse_loss(dyn_pred_k, dyn_target_k)
            total_aux_loss = total_aux_loss + aux_loss_k

        mean_infonce_loss = total_infonce_loss / float(eval_horizon)
        mean_aux_loss = total_aux_loss / float(eval_horizon)
        combined_loss = supcon_loss + mean_infonce_loss + mean_aux_loss

        return {
            "loss": combined_loss,
            "cpc_loss": mean_infonce_loss,
            "supcon_loss": supcon_loss,
            "aux_dyn_loss": mean_aux_loss,
        }

