"""
Representation Learning Training Script for Contextual RL Benchmarking.

This script trains PPO policies augmented with a Recurrent History Encoder (GRU)
that converts an agent's trajectory history of states, actions, and rewards into a compact
context vector z_t, preparing the foundation for VAE and CPC self-supervised representation learning.
"""

import argparse
from datetime import datetime
import os
import sys
from typing import List
import numpy as np
import torch
import gymnasium as gym

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.agents.base_ppo import BasePPOAgent
from src.models.history_encoder import RNNHistoryEncoder
from src.utils.env_utils import (
    make_env,
    extract_raw_state,
    resolve_path,
)
from src.utils.experiment_logger import ExperimentLogger
from src.utils.history_buffer import HistoryBuffer
from src.utils.plotting import Plotter


def run_representation_experiment(
    method: str = "rnn_history",
    env_name: str = "CARLPendulum",
    total_steps: int = 51200,
    rollout_steps: int = 2048,
    seed: int = 0,
    num_contexts: int = 100,
    latent_dim: int = 8,
    max_history_len: int = 25,
    exp_tag: str = "",
    output_dir: str = "scripts/results",
) -> str:
    """
    Executes a single representation learning training run (PPO + Recurrent History Encoder).

    Args:
        method: Representation method ('rnn_history', and later 'vae', 'cpc').
        env_name: Name of the Gymnasium or CARL environment.
        total_steps: Total environment interaction steps for training.
        rollout_steps: Number of interaction steps per PPO update buffer.
        seed: Random seed for reproducibility.
        num_contexts: Number of varying context instances to sample (default: 100).
        latent_dim: Dimension of latent context vector z_t (default: 8).
        max_history_len: Maximum trajectory history sliding window length.
        exp_tag: Optional custom experiment tag for run directory naming.
        output_dir: Directory where results CSV files will be saved.

    Returns:
        str: File path to the saved CSV results file.
    """
    output_dir = resolve_path(output_dir)

    # Set seed for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Initialize environment
    env = make_env(env_name, seed=seed, num_contexts=num_contexts)

    # Determine action space properties
    is_continuous = isinstance(env.action_space, gym.spaces.Box)
    if is_continuous:
        action_dim = env.action_space.shape[0]
    else:
        action_dim = env.action_space.n

    # Reset environment and get initial raw state
    obs, _ = env.reset(seed=seed)
    raw_state = extract_raw_state(obs)
    raw_state_dim = len(raw_state)

    # Initialize HistoryBuffer and RNNHistoryEncoder
    history_buf = HistoryBuffer(
        max_history_len=max_history_len,
        state_dim=raw_state_dim,
        action_dim=action_dim,
        is_continuous=is_continuous,
    )
    history_encoder = RNNHistoryEncoder(
        state_dim=raw_state_dim,
        action_dim=action_dim,
        latent_dim=latent_dim,
        hidden_dim=64,
    )

    # Combined input dimension: raw state dim + latent context dim z_t
    input_dim = raw_state_dim + latent_dim

    # Instantiate PPO Agent
    agent = BasePPOAgent(input_dim=input_dim, action_dim=action_dim, is_continuous=is_continuous)

    # Include history encoder parameters in joint Adam optimizer
    encoder_optimizer = torch.optim.Adam(history_encoder.parameters(), lr=3e-4, eps=1e-5)

    # Setup dedicated run directory and logger
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag_str = f"_{exp_tag}" if exp_tag else ""
    folder_name = f"{timestamp}_{env_name.lower().replace('-', '_')}_{method}_seed{seed}{tag_str}"
    run_dir = os.path.join(output_dir, folder_name)
    exp_name = f"{method}_seed{seed}"
    logger = ExperimentLogger(output_dir=run_dir, experiment_name=exp_name)

    # Record hyperparameter configuration to config.yaml
    hyperparams = {
        "env_name": env_name,
        "method": method,
        "seed": seed,
        "total_steps": total_steps,
        "rollout_steps": rollout_steps,
        "num_contexts": num_contexts,
        "latent_dim": latent_dim,
        "max_history_len": max_history_len,
        "is_continuous": is_continuous,
        "raw_state_dim": raw_state_dim,
        "input_dim": input_dim,
        "action_dim": action_dim,
        "gamma": agent.gamma,
        "gae_lambda": agent.gae_lambda,
        "clip_eps": agent.clip_eps,
        "epochs": agent.epochs,
        "batch_size": agent.batch_size,
    }
    logger.save_config(hyperparams)

    global_step = 0
    ep_return = 0.0
    completed_returns: List[float] = []

    print(f"\n[INFO] Starting Representation Training | Method: {method.upper()} | Seed: {seed} | Raw State Dim: {raw_state_dim} | Latent Dim: {latent_dim} | Input Dim: {input_dim}")

    # Main training loop
    while global_step < total_steps:
        trajectory = []

        for _ in range(rollout_steps):
            # Compute latent context vector z_t from history sequence
            hist_tensor = history_buf.get_tensor()
            with torch.no_grad():
                z_t, _ = history_encoder(hist_tensor)
                z_np = z_t.squeeze(0).cpu().numpy()

            # Concatenate raw state observation and latent context vector [s_t, z_t]
            policy_input = np.concatenate([raw_state, z_np], axis=0)

            # Sample action from policy
            action, logp, ent, val = agent.predict(policy_input)

            # Execute environment step
            next_obs, reward, term, trunc, _ = env.step(action)
            done = term or trunc

            # Update history buffer
            history_buf.add(raw_state, action, float(reward))
            next_raw_state = extract_raw_state(next_obs)

            # Compute next latent context vector z_{t+1} from updated history buffer
            next_hist_tensor = history_buf.get_tensor()
            with torch.no_grad():
                next_z_t, _ = history_encoder(next_hist_tensor)
                next_z_np = next_z_t.squeeze(0).cpu().numpy()

            next_policy_input = np.concatenate([next_raw_state, next_z_np], axis=0)

            # Store step in trajectory buffer
            trajectory.append((policy_input, action, logp, ent, reward, term, trunc, next_policy_input, hist_tensor))

            ep_return += float(reward)
            global_step += 1

            if done:
                completed_returns.append(ep_return)
                ep_return = 0.0
                history_buf.reset()
                obs, _ = env.reset()
                raw_state = extract_raw_state(obs)
            else:
                raw_state = next_raw_state

        # Update PPO policy and value networks
        p_loss, v_loss, e_loss = agent.update(trajectory)

        # Log metrics
        recent_returns = completed_returns[-10:] if completed_returns else [0.0]
        mean_ret = float(np.mean(recent_returns))
        logger.log(
            {
                "step": global_step,
                "mean_return": mean_ret,
                "policy_loss": p_loss,
                "value_loss": v_loss,
                "entropy_loss": e_loss,
            }
        )

        print(
            f"Step {global_step:7d}/{total_steps:7d} | "
            f"Mean Return (last 10 eps): {mean_ret:6.1f} | "
            f"Policy Loss: {p_loss:6.3f} | "
            f"Value Loss: {v_loss:7.2f}"
        )

    # Save metrics CSV
    csv_filepath = logger.save()
    return csv_filepath


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train PPO augmented with Recurrent History Encoder on Contextual RL environments."
    )
    parser.add_argument(
        "--env",
        type=str,
        default="CARLPendulum",
        help="Gymnasium or CARL environment name (default: CARLPendulum).",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="rnn_history",
        choices=["rnn_history"],
        help="Representation learning method (default: rnn_history).",
    )
    parser.add_argument(
        "--total_steps",
        type=int,
        default=51200,
        help="Total training steps (default: 51200).",
    )
    parser.add_argument(
        "--rollout_steps",
        type=int,
        default=2048,
        help="PPO rollout buffer step size (default: 2048).",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0],
        help="List of random seeds (default: 0).",
    )
    parser.add_argument(
        "--num_contexts",
        type=int,
        default=100,
        help="Number of varying context instances to sample for CARL envs (default: 100).",
    )
    parser.add_argument(
        "--latent_dim",
        type=int,
        default=8,
        help="Dimension of latent context vector z_t (default: 8).",
    )
    parser.add_argument(
        "--max_history_len",
        type=int,
        default=25,
        help="Maximum history window length (default: 25).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="scripts/results",
        help="Directory where results CSV files and plots will be saved.",
    )
    parser.add_argument(
        "--output_plot",
        type=str,
        default=None,
        help="Optional custom filename for generated plots.",
    )
    parser.add_argument(
        "--exp_tag",
        type=str,
        default="",
        help="Optional custom tag for run directory naming.",
    )

    args = parser.parse_args()
    output_dir = resolve_path(args.output_dir)

    generated_filepaths: List[str] = []
    for seed in args.seeds:
        filepath = run_representation_experiment(
            method=args.method,
            env_name=args.env,
            total_steps=args.total_steps,
            rollout_steps=args.rollout_steps,
            seed=seed,
            num_contexts=args.num_contexts,
            latent_dim=args.latent_dim,
            max_history_len=args.max_history_len,
            exp_tag=args.exp_tag,
            output_dir=output_dir,
        )
        generated_filepaths.append(filepath)

    env_slug = args.env.lower().replace("-", "_")
    return_plot_filename = args.output_plot if args.output_plot else f"{args.method}_{env_slug}_mean_return.png"
    value_loss_plot_filename = f"{args.method}_{env_slug}_value_loss.png"

    return_plot_path = os.path.join(output_dir, return_plot_filename)
    value_loss_plot_path = os.path.join(output_dir, value_loss_plot_filename)

    print(f"\n[INFO] Generating representation learning plots...")
    plotter = Plotter()
    plotter.plot_multiple_csv_runs(
        filepaths=generated_filepaths,
        title=f"Representation PPO ({args.method}) Mean Return on {args.env}",
        output_path=return_plot_path,
        metric_col="mean_return",
    )
    plotter.plot_multiple_csv_runs(
        filepaths=generated_filepaths,
        title=f"Representation PPO ({args.method}) Critic Value Loss on {args.env}",
        output_path=value_loss_plot_path,
        metric_col="value_loss",
    )
    print(f"[SUCCESS] Experiments completed!")
    print(f"  - Mean Return Plot: '{return_plot_path}'")
    print(f"  - Value Loss Plot:  '{value_loss_plot_path}'")


if __name__ == "__main__":
    main()
