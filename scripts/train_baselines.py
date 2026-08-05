"""
Baseline Training and Benchmark Evaluation Script for Contextual Reinforcement Learning.

This script trains baseline agents (e.g., Context-Free or Oracle) on Gymnasium environments,
logs training metrics, and generates reliable evaluation plots using RLiable.

Usage Examples:
    # Run a single experiment mode via CLI arguments:
    python scripts/train_baselines.py --mode context_free --total_steps 50000 --seeds 0

    # Run multiple seeds for both baseline modes and generate comparative RLiable plots:
    python scripts/train_baselines.py --mode all --total_steps 50000 --seeds 0 1 2
"""

import warnings

# Suppress ConfigSpace division warnings when context lower/upper bounds match
warnings.filterwarnings("ignore", category=RuntimeWarning, module="ConfigSpace")
# Suppress CARL optional dependency import warnings (Box2D, brax, dm_control, etc.)
warnings.filterwarnings("ignore", category=UserWarning)

import argparse
import os
from typing import List

import gymnasium as gym
import numpy as np

from src.agents.base_ppo import BasePPOAgent
from src.utils.env_utils import (
    make_env,
    extract_input,
    resolve_path,
)
from src.utils.experiment_logger import ExperimentLogger
from src.utils.plotting import Plotter


def run_baseline_experiment(
    mode: str = "context_free",
    env_name: str = "CARLPendulum",
    total_steps: int = 51200,
    rollout_steps: int = 2048,
    seed: int = 0,
    num_contexts: int = 100,
    exp_tag: str = "",
    output_dir: str = "scripts/results",
) -> str:
    """
    Executes a single PPO baseline training run for a given mode and random seed.

    Args:
        mode: Training mode ('context_free' or 'oracle').
        env_name: Name of the Gymnasium or CARL environment.
        total_steps: Total environment interaction steps for training.
        rollout_steps: Number of interaction steps per PPO update buffer.
        seed: Random seed for reproducibility.
        num_contexts: Number of varying context instances to sample (default: 100).
        exp_tag: Optional custom experiment tag for run directory naming.
        output_dir: Directory where results CSV files will be saved.

    Returns:
        str: Absolute or relative file path to the saved CSV results file.
    """
    from datetime import datetime

    output_dir = resolve_path(output_dir)

    # Set seed for reproducibility
    np.random.seed(seed)

    # Initialize Gymnasium or CARL environment
    env = make_env(env_name, seed=seed, num_contexts=num_contexts)

    # Reset environment and set initial agent state
    state, _ = env.reset(seed=seed)
    agent_state = extract_input(state, env, mode)
    input_dim = len(agent_state)

    is_continuous = isinstance(env.action_space, gym.spaces.Box)
    if is_continuous:
        action_dim = env.action_space.shape[0]
    else:
        action_dim = env.action_space.n

    # Instantiate PPO Agent
    agent = BasePPOAgent(input_dim=input_dim, action_dim=action_dim, is_continuous=is_continuous)
    exp_name = f"{mode}_seed{seed}"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag_str = f"_{exp_tag}" if exp_tag else ""
    folder_name = f"{timestamp}_{env_name.lower().replace('-', '_')}_{mode}_seed{seed}{tag_str}"
    run_dir = os.path.join(output_dir, folder_name)
    logger = ExperimentLogger(output_dir=run_dir, experiment_name=exp_name)

    # Record and save experiment hyperparameters to config.yaml
    hyperparams = {
        "env_name": env_name,
        "mode": mode,
        "seed": seed,
        "total_steps": total_steps,
        "rollout_steps": rollout_steps,
        "num_contexts": num_contexts,
        "is_continuous": is_continuous,
        "input_dim": input_dim,
        "action_dim": action_dim,
        "gamma": agent.gamma,
        "gae_lambda": agent.gae_lambda,
        "clip_eps": agent.clip_eps,
        "epochs": agent.epochs,
        "batch_size": agent.batch_size,
        "ent_coef": agent.ent_coef,
        "vf_coef": agent.vf_coef,
    }
    logger.save_config(hyperparams)

    global_step = 0
    ep_return = 0.0
    completed_returns: List[float] = []

    print(f"\n[INFO] Starting Baseline Training | Mode: {mode.upper()} | Seed: {seed} | Input Dim: {input_dim}")

    # Main training loop across total environment steps
    while global_step < total_steps:
        trajectory = []

        # Collect rollout trajectory buffer of size `rollout_steps`
        for _ in range(rollout_steps):
            # Sample action, log probability, entropy, and value estimate from agent policy
            action, logp, ent, val = agent.predict(agent_state)

            # Step environment forward using sampled action
            next_state, reward, term, trunc, _ = env.step(action)
            done = term or trunc

            ep_return += float(reward)
            global_step += 1

            next_agent_state = extract_input(next_state, env, mode)
            trajectory.append((agent_state, action, logp, ent, reward, float(term), float(trunc), next_agent_state))

            if done:
                # Store completed episode return and reset environment for next episode
                completed_returns.append(ep_return)
                ep_return = 0.0
                state, _ = env.reset()
                agent_state = extract_input(state, env, mode)
            else:
                agent_state = next_agent_state

        # Update PPO policy and value network parameters using collected rollout trajectory
        p_loss, v_loss, e_loss = agent.update(trajectory)

        # Compute moving average return over the last 10 completed episodes and log metrics
        mean_ret = float(np.mean(completed_returns[-10:])) if completed_returns else 0.0
        logger.log({
            "step": global_step,
            "mean_return": mean_ret,
            "policy_loss": p_loss,
            "value_loss": v_loss,
        })

        print(
            f"Step {global_step:6d}/{total_steps} | "
            f"Mean Return (last 10 eps): {mean_ret:5.1f} | "
            f"Policy Loss: {p_loss:6.3f} | Value Loss: {v_loss:6.2f}"
        )

    # Save logged training data to CSV file
    results_path = logger.save()
    return results_path


def main():
    """
    Parses command-line arguments and executes specified baseline experiments.
    """
    parser = argparse.ArgumentParser(
        description="Run Contextual RL Baseline Experiments with RLiable Plotting."
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="context_free",
        choices=["context_free", "oracle", "all"],
        help="Baseline mode to train: 'context_free', 'oracle', or 'all'.",
    )
    parser.add_argument(
        "--env",
        type=str,
        default="CARLPendulum",
        help="Gymnasium environment ID (default: 'CARLPendulum').",
    )
    parser.add_argument(
        "--total_steps",
        type=int,
        default=51200,
        help="Total training environment steps (default: 51200).", # should be a multiple of rollout_steps
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
        help="List of random seeds to run (e.g. --seeds 0 1 2).",
    )
    parser.add_argument(
        "--num_contexts",
        type=int,
        default=100,
        help="Number of varying context instances to sample for CARL envs (0 for static default context).",
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
        help="Optional custom filename for the generated plot image (e.g. 'custom_curve.png').",
    )
    parser.add_argument(
        "--exp_tag",
        type=str,
        default="",
        help="Optional custom tag for run directory naming (e.g. 'lr1e-3' or 'baseline_v1').",
    )

    args = parser.parse_args()

    # Resolve output directory relative to project root
    output_dir = resolve_path(args.output_dir)

    # Determine modes to run based on command-line selection
    modes_to_run = ["context_free", "oracle"] if args.mode == "all" else [args.mode]
    generated_filepaths: List[str] = []

    # Run experiments across selected modes and seeds
    for mode in modes_to_run:
        for seed in args.seeds:
            filepath = run_baseline_experiment(
                mode=mode,
                env_name=args.env,
                total_steps=args.total_steps,
                rollout_steps=args.rollout_steps,
                seed=seed,
                num_contexts=args.num_contexts,
                exp_tag=args.exp_tag,
                output_dir=output_dir,
            )
            generated_filepaths.append(filepath)

    # Dynamically construct plot filenames based on environment name and metric
    env_slug = args.env.lower().replace("-", "_")
    return_plot_filename = args.output_plot if args.output_plot else f"{env_slug}_mean_return.png"
    value_loss_plot_filename = f"{env_slug}_value_loss.png"

    return_plot_path = os.path.join(output_dir, return_plot_filename)
    value_loss_plot_path = os.path.join(output_dir, value_loss_plot_filename)

    # Generate Mean Return and Critic Value Loss Plots
    print(f"\n[INFO] Generating learning & value loss plots ({return_plot_filename}, {value_loss_plot_filename})...")
    plotter = Plotter()
    plotter.plot_multiple_csv_runs(
        filepaths=generated_filepaths,
        title=f"PPO Mean Return on {args.env}",
        output_path=return_plot_path,
        metric_col="mean_return",
    )
    plotter.plot_multiple_csv_runs(
        filepaths=generated_filepaths,
        title=f"PPO Critic Value Loss on {args.env}",
        output_path=value_loss_plot_path,
        metric_col="value_loss",
    )
    print(f"[SUCCESS] All experiments completed!")
    print(f"  - Mean Return Plot: '{return_plot_path}'")
    print(f"  - Value Loss Plot:  '{value_loss_plot_path}'")


if __name__ == "__main__":
    main()