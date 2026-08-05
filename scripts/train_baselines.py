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
from typing import Any, Dict, List, Optional

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


def sample_contexts(env_name: str, num_contexts: int = 100, seed: int = 42) -> Optional[Dict[int, Dict[str, Any]]]:
    """
    Samples a set of varying physical context instances for a CARL environment across episodes.
    """
    try:
        import carl.envs as carl_envs
        if hasattr(carl_envs, env_name):
            env_cls = getattr(carl_envs, env_name)
            default_ctx = env_cls.get_default_context()
            rng = np.random.RandomState(seed)
            contexts = {}
            for i in range(num_contexts):
                ctx = default_ctx.copy()
                if "gravity" in ctx:
                    ctx["gravity"] = float(rng.uniform(3.0, 20.0))
                if "g" in ctx:
                    ctx["g"] = float(rng.uniform(3.0, 20.0))
                if "masspole" in ctx:
                    ctx["masspole"] = float(rng.uniform(0.05, 0.5))
                if "length" in ctx:
                    ctx["length"] = float(rng.uniform(0.2, 1.5))
                if "l" in ctx:
                    ctx["l"] = float(rng.uniform(0.5, 2.0))
                if "masscart" in ctx:
                    ctx["masscart"] = float(rng.uniform(0.5, 2.0))
                if "m" in ctx:
                    ctx["m"] = float(rng.uniform(0.5, 2.0))
                contexts[i] = ctx
            return contexts
    except (ImportError, Exception):
        pass
    return None


def make_env(env_name: str, seed: Optional[int] = None, num_contexts: int = 100) -> gym.Env:
    """
    Instantiates either a standard Gymnasium environment or a CARL benchmark environment.
    If num_contexts > 0, samples varying context instances across episodes.

    Args:
        env_name: Name of the environment (e.g., 'CartPole-v1', 'CARLCartPole', 'CARLPendulum').
        seed: Optional random seed.
        num_contexts: Number of varying context instances to sample (0 for default static context).

    Returns:
        gym.Env: Instantiated Gymnasium or CARL environment instance.
    """
    env = None
    try:
        import carl.envs as carl_envs
        if hasattr(carl_envs, env_name):
            env_cls = getattr(carl_envs, env_name)
            if num_contexts > 0:
                contexts = sample_contexts(env_name, num_contexts=num_contexts, seed=seed if seed is not None else 42)
                env = env_cls(contexts=contexts)
            else:
                env = env_cls()
    except (ImportError, Exception):
        pass

    if env is None:
        env = gym.make(env_name)

    if seed is not None and hasattr(env.action_space, "seed"):
        env.action_space.seed(seed)

    return env


def extract_input(state: Any, env: gym.Env, mode: str, normalize_context: bool = True) -> np.ndarray:
    """
    Extracts the input vector for the agent based on the specified baseline mode.
    Handles standard Gymnasium environments as well as CARL environments where
    observations are returned as dictionaries containing 'obs' and 'context'.

    Args:
        state: Observation dictionary or array from environment reset/step.
        env: The Gymnasium or CARL environment instance.
        mode: Baseline mode ('context_free' or 'oracle').
        normalize_context: Whether to scale context features relative to default values (default: True).

    Returns:
        np.ndarray: Input vector for the agent policy and value network.
    """
    if isinstance(state, dict) and "obs" in state:
        raw_obs = np.array(state["obs"], dtype=np.float32).flatten()
        context_dict = state.get("context", {})
    else:
        raw_obs = np.array(state, dtype=np.float32).flatten()
        context_dict = getattr(env, "context", getattr(getattr(env, "unwrapped", None), "context", {}))

    if mode == "context_free":
        return raw_obs
    elif mode == "oracle":
        if context_dict:
            context_vals = np.array(list(context_dict.values()), dtype=np.float32).flatten()
            if normalize_context:
                default_fn = getattr(env, "get_default_context", None)
                if default_fn is None and hasattr(env, "unwrapped"):
                    default_fn = getattr(env.unwrapped, "get_default_context", None)
                default_ctx = default_fn() if default_fn else {}
                if default_ctx and len(default_ctx) == len(context_vals):
                    default_vals = np.array(list(default_ctx.values()), dtype=np.float32).flatten()
                    diff_std = np.where(default_vals == 0.0, 1e-8, np.abs(default_vals))
                    context_vals = context_vals / diff_std
                else:
                    ctx_std = np.std(context_vals) + 1e-8
                    context_vals = (context_vals - np.mean(context_vals)) / ctx_std
            return np.concatenate([raw_obs, context_vals])
        return raw_obs
    else:
        raise ValueError(f"Unknown mode '{mode}'. Supported modes are 'context_free' and 'oracle'.")


def run_baseline_experiment(
    mode: str = "context_free",
    env_name: str = "CARLPendulum",
    total_steps: int = 51200,
    rollout_steps: int = 2048,
    seed: int = 0,
    num_contexts: int = 100,
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
        output_dir: Directory where results CSV files will be saved.

    Returns:
        str: Absolute or relative file path to the saved CSV results file.
    """
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

    # Instantiate PPO Agent and Experiment Logger
    agent = BasePPOAgent(input_dim=input_dim, action_dim=action_dim, is_continuous=is_continuous)
    exp_name = f"{mode}_seed{seed}"
    logger = ExperimentLogger(output_dir=output_dir, experiment_name=exp_name)

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