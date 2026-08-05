"""
Unified Training Pipeline for Contextual Reinforcement Learning Benchmarking.

Supports training across benchmark algorithms:
  - context_free: PPO on state observation only
  - oracle: PPO on state observation + ground-truth context parameters
  - vae: PPO augmented with Variational Autoencoder (VAE) representation
"""

from datetime import datetime
import os
import sys
from typing import Any, Dict, List, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.agents.base_ppo import BasePPOAgent
from src.models.vae_encoder import VAEContextEncoder
from src.utils.env_utils import (
    make_env,
    sample_contexts,
    extract_input,
    extract_raw_state,
    resolve_path,
)
from src.utils.experiment_logger import ExperimentLogger
from src.utils.history_buffer import HistoryBuffer
from src.utils.plotting import Plotter


def train_single_run(
    mode: str = "context_free",
    env_name: str = "CARLPendulum",
    total_steps: int = 51200,
    rollout_steps: int = 2048,
    seed: int = 0,
    num_contexts: int = 100,
    train_context_seed: int = 42,
    latent_dim: int = 8,
    max_history_len: int = 25,
    exp_tag: str = "",
    output_dir: str = "results",
) -> str:
    """
    Trains a single RL model checkpoint on training context environments.

    Args:
        mode: Algorithm mode ('context_free', 'oracle', 'vae', 'cpc').
        env_name: Environment name (e.g., 'CARLPendulum', 'CARLCartPole').
        total_steps: Total interaction steps.
        rollout_steps: PPO rollout step size.
        seed: Model random seed.
        num_contexts: Number of training context instances (default: 100).
        train_context_seed: Context sampling seed for training set (default: 42).
        latent_dim: Latent context dimension z_t (default: 8).
        max_history_len: History buffer sequence length (default: 25).
        exp_tag: Optional run tag.
        output_dir: Base directory to save run folder (default: 'results').

    Returns:
        str: Absolute path to the created run directory.
    """
    output_dir = resolve_path(output_dir)

    # Restrict PyTorch thread count per worker process to 1 thread to avoid CPU thread contention
    torch.set_num_threads(1)

    # Set seeds
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Make training environment
    env = make_env(env_name, seed=seed, num_contexts=num_contexts, context_seed=train_context_seed)
    is_continuous = isinstance(env.action_space, gym.spaces.Box)
    action_dim = env.action_space.shape[0] if is_continuous else env.action_space.n

    # Setup deterministic run directory inside results/ (allows overwriting)
    tag_str = f"_{exp_tag}" if exp_tag else ""
    folder_name = f"{total_steps}steps_{env_name.lower().replace('-', '_')}_{mode}_seed{seed}{tag_str}"
    run_dir = os.path.join(output_dir, folder_name)
    os.makedirs(run_dir, exist_ok=True)
    exp_name = f"{mode}_seed{seed}"
    logger = ExperimentLogger(output_dir=run_dir, experiment_name=exp_name)

    # Determine state and input dimensions
    obs, _ = env.reset(seed=seed)
    raw_state = extract_raw_state(obs)
    raw_state_dim = len(raw_state)

    history_buf: Optional[HistoryBuffer] = None
    vae_encoder: Optional[VAEContextEncoder] = None
    encoder_optimizer: Optional[torch.optim.Optimizer] = None

    device = torch.device("cpu")
    device_name = "CPU"

    if mode in ["context_free", "oracle"]:
        initial_input = extract_input(obs, env, mode)
        input_dim = len(initial_input)
    else:
        history_buf = HistoryBuffer(
            max_history_len=max_history_len,
            state_dim=raw_state_dim,
            action_dim=action_dim,
            is_continuous=is_continuous,
        )
        input_dim = raw_state_dim + latent_dim

        if mode == "vae":
            vae_encoder = VAEContextEncoder(
                state_dim=raw_state_dim,
                action_dim=action_dim,
                latent_dim=latent_dim,
                hidden_dim=64,
                kl_weight=1e-3,
                device=device,
            )
            encoder_optimizer = torch.optim.Adam(vae_encoder.parameters(), lr=3e-4, eps=1e-5)

    agent = BasePPOAgent(input_dim=input_dim, action_dim=action_dim, is_continuous=is_continuous, device=device)

    # Save hyperparameters
    hyperparams = {
        "env_name": env_name,
        "mode": mode,
        "seed": seed,
        "train_context_seed": train_context_seed,
        "num_contexts": num_contexts,
        "total_steps": total_steps,
        "rollout_steps": rollout_steps,
        "latent_dim": latent_dim,
        "max_history_len": max_history_len,
        "is_continuous": is_continuous,
        "raw_state_dim": raw_state_dim,
        "input_dim": input_dim,
        "action_dim": action_dim,
    }
    logger.save_config(hyperparams)

    global_step = 0
    ep_return = 0.0
    completed_returns: List[float] = []

    print(f"\n[TRAIN] Mode: {mode.upper()} | Env: {env_name} | Seed: {seed} | Device: {device_name} | Input Dim: {input_dim}")

    curr_obs = obs
    curr_hist_tensor = history_buf.get_tensor(pad_to_max=True) if history_buf else None
    curr_z_np = None
    if mode == "vae" and vae_encoder is not None and curr_hist_tensor is not None:
        with torch.no_grad():
            mu_z, _ = vae_encoder.encode(curr_hist_tensor)
            curr_z_np = mu_z.squeeze(0).cpu().numpy()

    while global_step < total_steps:
        trajectory = []

        for _ in range(rollout_steps):
            if mode in ["context_free", "oracle"]:
                policy_input = extract_input(curr_obs, env, mode)
                action, logp, ent, val = agent.predict(policy_input)
                next_obs, reward, term, trunc, _ = env.step(action)
                done = term or trunc
                next_policy_input = extract_input(next_obs, env, mode)

                trajectory.append((policy_input, action, logp, ent, reward, term, trunc, next_policy_input, None))
            else:
                raw_state = extract_raw_state(curr_obs)
                policy_input = np.concatenate([raw_state, curr_z_np], axis=0)

                action, logp, ent, val = agent.predict(policy_input)
                next_obs, reward, term, trunc, _ = env.step(action)
                done = term or trunc

                history_buf.add(raw_state, action, float(reward))
                next_raw_state = extract_raw_state(next_obs)

                next_hist_tensor = history_buf.get_tensor(pad_to_max=True)
                with torch.no_grad():
                    if mode == "vae" and vae_encoder is not None:
                        next_mu, _ = vae_encoder.encode(next_hist_tensor)
                        next_z_np = next_mu.squeeze(0).cpu().numpy()
                    else:
                        next_z_np = curr_z_np

                next_policy_input = np.concatenate([next_raw_state, next_z_np], axis=0)
                trajectory.append((
                    policy_input,
                    action,
                    logp,
                    ent,
                    reward,
                    term,
                    trunc,
                    next_policy_input,
                    curr_hist_tensor,
                    raw_state,
                    next_raw_state,
                ))

                curr_hist_tensor = next_hist_tensor
                curr_z_np = next_z_np

            ep_return += float(reward)
            global_step += 1

            if done:
                completed_returns.append(ep_return)
                ep_return = 0.0
                if history_buf:
                    history_buf.reset()
                    curr_hist_tensor = history_buf.get_tensor(pad_to_max=True)
                    if mode == "vae" and vae_encoder is not None:
                        with torch.no_grad():
                            mu_z, _ = vae_encoder.encode(curr_hist_tensor)
                            curr_z_np = mu_z.squeeze(0).cpu().numpy()
                curr_obs, _ = env.reset()
            else:
                curr_obs = next_obs

        # Update representation encoder network (VAE)
        vae_loss_val = 0.0
        if mode == "vae" and vae_encoder is not None and encoder_optimizer is not None:
            hist_batch = torch.cat([step[8] for step in trajectory], dim=0)
            state_batch = torch.tensor(np.array([step[9] for step in trajectory]), dtype=torch.float32)

            raw_actions = [step[1] for step in trajectory]
            if is_continuous:
                action_batch = torch.tensor(np.array(raw_actions), dtype=torch.float32)
            else:
                action_batch = F.one_hot(torch.tensor(raw_actions, dtype=torch.long), num_classes=action_dim).float()

            next_state_batch = torch.tensor(np.array([step[10] for step in trajectory]), dtype=torch.float32)
            reward_batch = torch.tensor(np.array([step[4] for step in trajectory]), dtype=torch.float32).unsqueeze(-1)

            encoder_optimizer.zero_grad()
            vae_loss_dict = vae_encoder.compute_loss(hist_batch, state_batch, action_batch, next_state_batch, reward_batch)
            vae_loss = vae_loss_dict["loss"]
            vae_loss.backward()
            encoder_optimizer.step()
            vae_loss_val = vae_loss.item()

        # Update PPO Agent
        ppo_traj = [(s, a, lp, e, r, tm, tr, ns, h) for (s, a, lp, e, r, tm, tr, ns, h, *_) in trajectory]
        p_loss, v_loss, e_loss = agent.update(ppo_traj)

        # Log training metrics
        recent_returns = completed_returns[-10:] if completed_returns else [0.0]
        mean_ret = float(np.mean(recent_returns))
        log_dict = {
            "step": global_step,
            "mean_return": mean_ret,
            "policy_loss": p_loss,
            "value_loss": v_loss,
            "entropy_loss": e_loss,
        }
        if mode == "vae":
            log_dict["vae_loss"] = vae_loss_val
        logger.log(log_dict)

        loss_str = f"VAE Loss: {vae_loss_val:6.3f} | " if mode == "vae" else ""
        print(
            f"Step {global_step:7d}/{total_steps:7d} | "
            f"Mean Return (last 10 eps): {mean_ret:6.1f} | "
            f"{loss_str}"
            f"Value Loss: {v_loss:7.2f}"
        )

    # Save metrics CSV and model weights inside run_dir
    logger.save()
    agent.save_checkpoint(os.path.join(run_dir, "agent.pt"))

    if vae_encoder:
        torch.save(vae_encoder.state_dict(), os.path.join(run_dir, "vae_encoder.pt"))

    print(f"[SUCCESS] Saved trained model checkpoint to '{run_dir}'")
    return run_dir
