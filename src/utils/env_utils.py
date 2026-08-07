"""
Environment Utilities for Gymnasium and CARL Contextual RL Benchmarks.

This module provides shared helper functions for environment creation, dynamic context
sampling across episodes, state/context feature extraction, and path resolution.
"""

import os
import sys
import warnings
from typing import Any, Dict, Optional, Tuple, List
import numpy as np
import gymnasium as gym

# Configure global NumPy and Warning filters to prevent third-party library noise (CARL / ConfigSpace)
np.seterr(divide="ignore", invalid="ignore")
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*invalid value encountered in.*divide.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*Module .* not found.*")

# Absolute path anchoring relative to project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def resolve_path(path: str) -> str:
    """
    Resolves relative path against project root to ensure consistent output file locations
    regardless of the current working directory from which a script is launched.
    """
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(PROJECT_ROOT, path))


def sample_contexts(
    env_name: str,
    num_contexts: int = 100,
    seed: int = 42,
    vary_contexts: Optional[List[str]] = None,
) -> Optional[Dict[int, Dict[str, Any]]]:
    """
    Samples a set of varying physical context instances for a CARL environment across episodes.

    Args:
        env_name: Name of CARL environment.
        num_contexts: Number of context instances to sample.
        seed: Random seed for reproducibility.
        vary_contexts: Optional list of context parameter names to vary (e.g., ['gravity'], ['g'], ['g', 'l']).
                        If None or empty, all available context parameters for the environment are varied.
    """
    try:
        with warnings.catch_warnings(record=False):
            warnings.simplefilter("ignore")
            import carl.envs as carl_envs
            if hasattr(carl_envs, env_name):
                env_cls = getattr(carl_envs, env_name)
                default_ctx = env_cls.get_default_context()
                rng = np.random.RandomState(seed)

                vary_set = set(k.lower() for k in vary_contexts) if vary_contexts else None

                def should_vary(key: str, aliases: List[str]) -> bool:
                    if vary_set is None:
                        return True
                    for name in [key] + aliases:
                        if name.lower() in vary_set:
                            return True
                    return False

                contexts = {}
                for i in range(num_contexts):
                    ctx = default_ctx.copy()
                    if "gravity" in ctx and should_vary("gravity", ["g"]):
                        ctx["gravity"] = float(rng.uniform(3.0, 20.0))
                    if "g" in ctx and should_vary("g", ["gravity"]):
                        ctx["g"] = float(rng.uniform(3.0, 20.0))
                    if "masspole" in ctx and should_vary("masspole", ["m", "mass"]):
                        ctx["masspole"] = float(rng.uniform(0.05, 0.5))
                    if "length" in ctx and should_vary("length", ["l"]):
                        ctx["length"] = float(rng.uniform(0.2, 1.5))
                    if "l" in ctx and should_vary("l", ["length"]):
                        ctx["l"] = float(rng.uniform(0.5, 2.0))
                    if "masscart" in ctx and should_vary("masscart", ["m", "mass"]):
                        ctx["masscart"] = float(rng.uniform(0.5, 2.0))
                    if "m" in ctx and should_vary("m", ["mass"]):
                        ctx["m"] = float(rng.uniform(0.5, 2.0))
                    contexts[i] = ctx
                return contexts
    except (ImportError, Exception):
        pass
    return None


def make_env(
    env_name: str,
    seed: Optional[int] = None,
    num_contexts: int = 100,
    context_seed: int = 42,
    vary_contexts: Optional[List[str]] = None,
) -> gym.Env:
    """
    Instantiates either a standard Gymnasium environment or a CARL benchmark environment.
    If num_contexts > 0, samples varying context instances across episodes.

    Args:
        env_name: Name of the environment (e.g., 'CartPole-v1', 'CARLCartPole', 'CARLPendulum').
        seed: Optional action space random seed.
        num_contexts: Number of varying context instances to sample (0 for default static context).
        context_seed: Seed for sampling context variations (42 for training, 9999 for held-out testing).
        vary_contexts: Optional list of context parameter names to vary (e.g. ['gravity'], ['g'], ['g', 'l']).

    Returns:
        gym.Env: Instantiated Gymnasium or CARL environment instance.
    """
    env = None
    try:
        with warnings.catch_warnings(record=False):
            warnings.simplefilter("ignore")
            import carl.envs as carl_envs
            if hasattr(carl_envs, env_name):
                env_cls = getattr(carl_envs, env_name)
                if num_contexts > 0:
                    contexts = sample_contexts(
                        env_name,
                        num_contexts=num_contexts,
                        seed=context_seed,
                        vary_contexts=vary_contexts,
                    )
                    env = env_cls(contexts=contexts)
                else:
                    env = env_cls()
    except (ImportError, Exception):
        pass

    if env is None:
        with warnings.catch_warnings(record=False):
            warnings.simplefilter("ignore")
            env = gym.make(env_name)

    if seed is not None and hasattr(env.action_space, "seed"):
        env.action_space.seed(seed)

    return env


def extract_raw_state(obs: Any) -> np.ndarray:
    """
    Extracts flat raw observation state vector from standard Gymnasium or CARL dictionary observations.
    """
    if isinstance(obs, dict) and "obs" in obs:
        return np.array(obs["obs"], dtype=np.float32).flatten()
    return np.array(obs, dtype=np.float32).flatten()


def extract_input(state: Any, env: gym.Env, mode: str, normalize_context: bool = True) -> np.ndarray:
    """
    Extracts state observation vector (or state concatenated with normalized context features for Oracle).

    Args:
        state: Observation returned by env.reset() or env.step().
        env: The Gymnasium or CARL environment instance.
        mode: Training mode ('context_free' or 'oracle').
        normalize_context: Whether to scale raw context values by default magnitudes.

    Returns:
        np.ndarray: Input feature vector for PPO policy and value networks.
    """
    if isinstance(state, dict) and "obs" in state:
        obs_vals = np.array(state["obs"], dtype=np.float32).flatten()
        if mode == "oracle":
            ctx = state["context"]
            context_vals = np.array(list(ctx.values()), dtype=np.float32).flatten()

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

            return np.concatenate([obs_vals, context_vals])
        return obs_vals

    return np.array(state, dtype=np.float32).flatten()
