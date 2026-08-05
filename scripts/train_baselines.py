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

import argparse
import os
from typing import List
import gymnasium as gym
import numpy as np

from src.agents.base_ppo import BasePPOAgent
from src.utils.experiment_logger import ExperimentLogger
from src.utils.plotting import Plotter

# Absolute path anchoring relative to script location to ensure consistent output locations
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))


def resolve_path(path: str) -> str:
    """
    Resolves relative path against project root to ensure consistent output file locations
    regardless of the current working directory from which the script is launched.
    """
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(PROJECT_ROOT, path))


def extract_input(state: np.ndarray, env: gym.Env, mode: str) -> np.ndarray:
    """
    Extracts the input vector for the agent based on the specified baseline mode.

    Args:
        state: Raw observation array from the environment.
        env: The Gymnasium environment instance.
        mode: Baseline mode ('context_free' or 'oracle').

    Returns:
        np.ndarray: Input vector for the agent policy and value network.
    """
    if mode == "context_free":
        # Ignore context features and return raw environment state only
        return state
    elif mode == "oracle":
        # Concatenate context vector to the state observation if available in environment
        context_dict = getattr(env, "context", {})
        if context_dict:
            context_vec = np.array(list(context_dict.values()), dtype=np.float32)
            return np.concatenate([state, context_vec])
        return state
    else:
        raise ValueError(f"Unknown mode '{mode}'. Supported modes are 'context_free' and 'oracle'.")


def run_baseline_experiment(
    mode: str = "context_free",
    env_name: str = "CartPole-v1",
    total_steps: int = 51200,
    rollout_steps: int = 2048,
    seed: int = 0,
    output_dir: str = "scripts/results",
) -> str:
    """
    Executes a single PPO baseline training run for a given mode and random seed.

    Args:
        mode: Training mode ('context_free' or 'oracle').
        env_name: Name of the Gymnasium environment.
        total_steps: Total environment interaction steps for training.
        rollout_steps: Number of interaction steps per PPO update buffer.
        seed: Random seed for reproducibility.
        output_dir: Directory where results CSV files will be saved.

    Returns:
        str: Absolute or relative file path to the saved CSV results file.
    """
    output_dir = resolve_path(output_dir)

    # Set seed for reproducibility
    np.random.seed(seed)

    # Initialize Gymnasium environment
    env = gym.make(env_name)
    env.action_space.seed(seed)

    # Determine input dimensions based on state observation and context vector length
    raw_obs_dim = env.observation_space.shape[0]
    context_dim = len(getattr(env, "context", {})) if mode == "oracle" else 0
    input_dim = raw_obs_dim + context_dim
    action_dim = env.action_space.n

    # Instantiate PPO Agent and Experiment Logger
    agent = BasePPOAgent(input_dim=input_dim, action_dim=action_dim)
    exp_name = f"{mode}_seed{seed}"
    logger = ExperimentLogger(output_dir=output_dir, experiment_name=exp_name)

    # Reset environment and set initial agent state
    state, _ = env.reset(seed=seed)
    agent_state = extract_input(state, env, mode)

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
        default="CartPole-v1",
        help="Gymnasium environment ID (default: 'CartPole-v1').",
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
                output_dir=output_dir,
            )
            generated_filepaths.append(filepath)

    # Dynamically construct plot filename based on mode and environment name if not explicitly set
    if args.output_plot:
        plot_filename = args.output_plot
    else:
        env_slug = args.env.lower().replace("-", "_")
        plot_filename = f"{args.mode}_{env_slug}_learning_curve.png"

    plot_output_path = os.path.join(output_dir, plot_filename)

    # Plot
    print(f"\n[INFO] Generating plots ({plot_filename})...")
    plotter = Plotter()
    plotter.plot_learning_curves(
        filepaths=generated_filepaths,
        title=f"PPO Performance on {args.env}",
        output_path=plot_output_path,
    )
    print(f"[SUCCESS] All experiments completed! Plot saved to '{plot_output_path}'.")


if __name__ == "__main__":
    main()