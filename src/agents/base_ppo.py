"""
Base Proximal Policy Optimization (PPO) Agent Implementation with GPU Support.

This module contains the BasePPOAgent class implementing clipped surrogate PPO updates
and Generalized Advantage Estimation (GAE) for both discrete and continuous action space environments,
fully accelerated on CUDA GPU devices when available.
"""

from typing import Any, List, Tuple, Union, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, Normal

from src.models.networks import Policy, ContinuousPolicy, ValueNetwork


class BasePPOAgent:
    """
    PPO Agent using separate policy (actor) and value (critic) network architectures,
    clipped surrogate objectives, entropy regularization, and Generalized Advantage Estimation (GAE).
    Supports both discrete (Categorical) and continuous (Gaussian) action spaces on GPU/CPU devices.
    """

    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        is_continuous: bool = False,
        lr_actor: float = 3e-4,
        lr_critic: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        epochs: int = 10,
        batch_size: int = 64,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        hidden_size: int = 128,
        device: Optional[torch.device] = None,
    ) -> None:
        """
        Initializes hyperparameters, neural network models on target GPU/CPU device, and Adam optimizer.
        """
        self.device = device if device is not None else torch.device("cpu")
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.epochs = epochs
        self.batch_size = batch_size
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.is_continuous = is_continuous

        # Instantiate policy network (actor) and value network (critic)
        if self.is_continuous:
            self.policy = ContinuousPolicy(input_dim, action_dim, hidden_size).to(self.device)
        else:
            self.policy = Policy(input_dim, action_dim, hidden_size).to(self.device)
        self.value_fn = ValueNetwork(input_dim, hidden_size).to(self.device)

        # Set up Adam optimizer for both networks
        self.optimizer = optim.Adam(
            [
                {"params": self.policy.parameters(), "lr": lr_actor},
                {"params": self.value_fn.parameters(), "lr": lr_critic},
            ],
            eps=1e-5,
        )

    def get_dist(self, states: torch.Tensor) -> Union[Categorical, Normal]:
        """
        Constructs action distribution (Categorical for discrete, Normal for continuous) for states.
        """
        if self.is_continuous:
            mean, std = self.policy(states)
            return Normal(mean, std)
        else:
            logits = self.policy(states)
            return Categorical(logits=logits)

    def predict(
        self, state: np.ndarray
    ) -> Tuple[Any, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes action prediction, log probability, entropy, and state value estimate for a single observation.
        """
        t = torch.from_numpy(state).float().to(self.device)

        with torch.no_grad():
            dist = self.get_dist(t)
            val = self.value_fn(t)
            action = dist.sample()

            if self.is_continuous:
                log_prob = dist.log_prob(action).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1)
                act_out = action.squeeze(0).cpu().numpy()
            else:
                log_prob = dist.log_prob(action)
                entropy = dist.entropy()
                act_out = int(action.item())

        return (
            act_out,
            log_prob,
            entropy,
            val,
        )

    def compute_gae(
        self,
        rewards: List[float],
        values: torch.Tensor,
        next_values: torch.Tensor,
        terms: torch.Tensor,
        dones: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes Generalized Advantage Estimation (GAE) and target returns backwards through time on device.
        """
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        deltas = rewards_t + self.gamma * next_values * (1 - terms) - values

        advantages = torch.zeros_like(deltas)
        gae = 0.0
        for i in reversed(range(deltas.shape[0])):
            gae = deltas[i] + self.gamma * self.gae_lambda * gae * (1 - dones[i])
            advantages[i] = gae

        returns = advantages + values
        return advantages.detach(), returns.detach()

    def update(self, trajectory: List[Any]) -> Tuple[float, float, float]:
        """
        Updates policy and value network parameters over multiple epochs using minibatch PPO clipped objective.
        """
        states = torch.stack([torch.from_numpy(t[0]).float() for t in trajectory]).to(self.device)
        if self.is_continuous:
            actions = torch.stack([torch.from_numpy(np.array(t[1], dtype=np.float32)).float() for t in trajectory]).to(self.device)
            if actions.dim() == 1:
                actions = actions.unsqueeze(-1)
        else:
            actions = torch.tensor([t[1] for t in trajectory], dtype=torch.long, device=self.device)

        rewards = [t[4] for t in trajectory]
        terms = torch.tensor([t[5] for t in trajectory], dtype=torch.float32, device=self.device)
        truncs = torch.tensor([t[6] for t in trajectory], dtype=torch.float32, device=self.device)
        dones = torch.clamp(terms + truncs, 0.0, 1.0)

        with torch.no_grad():
            values = self.value_fn(states)
            next_states = torch.stack([torch.from_numpy(t[7]).float() for t in trajectory]).to(self.device)
            next_values = self.value_fn(next_states)

            old_dist = self.get_dist(states)
            if self.is_continuous:
                old_logps = old_dist.log_prob(actions).sum(dim=-1).detach()
            else:
                old_logps = old_dist.log_prob(actions).detach()

        advantages, returns = self.compute_gae(rewards, values, next_values, terms, dones)

        dataset = torch.utils.data.TensorDataset(
            states, actions, old_logps, advantages, returns
        )
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )

        p_loss_epoch = 0.0
        v_loss_epoch = 0.0
        e_loss_epoch = 0.0

        for _ in range(self.epochs):
            for b_states, b_actions, b_oldlogp, b_adv, b_ret in loader:
                b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

                dist = self.get_dist(b_states)
                if self.is_continuous:
                    new_logp = dist.log_prob(b_actions).sum(dim=-1)
                    entropy_loss = -dist.entropy().sum(dim=-1).mean()
                else:
                    new_logp = dist.log_prob(b_actions)
                    entropy_loss = -dist.entropy().mean()

                ratio = torch.exp(new_logp - b_oldlogp)
                surr1 = ratio * b_adv
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * b_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                value_preds = self.value_fn(b_states)
                value_loss = F.mse_loss(value_preds, b_ret)

                loss = policy_loss + self.vf_coef * value_loss + self.ent_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.policy.parameters()) + list(self.value_fn.parameters()),
                    max_norm=0.5,
                )
                self.optimizer.step()

                p_loss_epoch += policy_loss.item()
                v_loss_epoch += value_loss.item()
                e_loss_epoch += entropy_loss.item()

        num_updates = self.epochs * len(loader)
        return p_loss_epoch / num_updates, v_loss_epoch / num_updates, e_loss_epoch / num_updates

    def save_checkpoint(self, filepath: str) -> None:
        """Saves policy, value function, and optimizer weights to disk."""
        torch.save(
            {
                "policy_state_dict": self.policy.state_dict(),
                "value_fn_state_dict": self.value_fn.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "is_continuous": self.is_continuous,
            },
            filepath,
        )

    def load_checkpoint(self, filepath: str) -> None:
        """Loads policy, value function, and optimizer weights from disk."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.value_fn.load_state_dict(checkpoint["value_fn_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])