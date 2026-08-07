"""
Bayesian Optimization (BO) Hyperparameter Tuning Module.

Uses Optuna (Tree-structured Parzen Estimator / TPE Bayesian Optimization sampler)
to automatically discover optimal hyperparameters for representation learning encoders
(VAE and CPC) prior to final benchmark evaluation on unseen test seeds.
"""

import os
import sys
import tempfile
import yaml
from typing import Dict, Any, Optional

import optuna

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.pipeline.train import train_single_run
from src.pipeline.evaluate import evaluate_run_directory
from src.utils.env_utils import resolve_path

# Suppress verbose optuna logging during optimization steps
optuna.logging.set_verbosity(optuna.logging.WARNING)


def run_bayesian_optimization(
    mode: str = "vae",
    env_name: str = "CARLPendulum",
    n_trials: int = 10,
    bo_steps: int = 20480,
    val_episodes: int = 10,
    val_context_seed: int = 8888,
    seed: int = 42,
    output_dir: str = "results",
) -> Dict[str, Any]:
    """
    Executes Bayesian Optimization (BO) search over hyperparameter space for VAE or CPC context encoders.

    Args:
        mode: Representation mode ('vae' or 'cpc').
        env_name: Benchmark environment name.
        n_trials: Number of BO trials (default: 10).
        bo_steps: Training step budget per optimization trial (default: 20480).
        val_episodes: Number of validation episodes per trial evaluation.
        val_context_seed: Context seed for validation set.
        seed: Random seed for BO study sampler.
        output_dir: Base directory to save BO study log and best config.

    Returns:
        Dict[str, Any]: Best hyperparameter dictionary found by Bayesian Optimization.
    """
    if mode not in ["vae", "cpc"]:
        raise ValueError(f"Bayesian Optimization search is only configured for 'vae' and 'cpc' modes, got '{mode}'.")

    output_dir = resolve_path(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    configs_dir = resolve_path("configs")
    os.makedirs(configs_dir, exist_ok=True)

    print(f"\n[BO SEARCH] Starting Bayesian Optimization for '{mode.upper()}' ({n_trials} trials, {bo_steps} steps/trial)...")

    def objective(trial: optuna.Trial) -> float:
        # Sample hyperparameters from defined search space
        latent_dim = trial.suggest_categorical("latent_dim", [4, 8, 16])
        hidden_dim = trial.suggest_categorical("hidden_dim", [32, 64, 128])
        encoder_lr = trial.suggest_float("encoder_lr", 1e-4, 1e-2, log=True)
        max_history_len = trial.suggest_categorical("max_history_len", [10, 25, 50])

        kl_weight = 1e-3
        cpc_temperature = 0.1
        predict_horizon = 3

        if mode == "vae":
            kl_weight = trial.suggest_float("kl_weight", 1e-4, 1e-1, log=True)
        elif mode == "cpc":
            cpc_temperature = trial.suggest_float("cpc_temperature", 0.05, 0.5, log=True)
            predict_horizon = trial.suggest_int("predict_horizon", 1, 5)

        # Train trial model inside a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = train_single_run(
                mode=mode,
                env_name=env_name,
                total_steps=bo_steps,
                rollout_steps=1024,
                seed=seed,
                latent_dim=latent_dim,
                hidden_dim=hidden_dim,
                encoder_lr=encoder_lr,
                kl_weight=kl_weight,
                cpc_temperature=cpc_temperature,
                predict_horizon=predict_horizon,
                max_history_len=max_history_len,
                exp_tag=f"bo_trial_{trial.number}",
                output_dir=temp_dir,
            )

            # Evaluate trial on validation context seed
            eval_res = evaluate_run_directory(
                run_dir=run_dir,
                eval_episodes=val_episodes,
                eval_context_seed=val_context_seed,
                num_eval_contexts=10,
            )

            val_return = eval_res["eval_mean_return"]
            probing_r2 = eval_res.get("probing_r2", 0.0)

            # Optimization objective score (primarily validation return with minor probing reward)
            score = float(val_return + 0.1 * probing_r2)
            return score

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_params
    best_value = study.best_value

    print(f"\n[BO SUCCESS] Optimization finished for '{mode.upper()}'. Best Score: {best_value:.3f}")
    print("Best Hyperparameters:")
    for k, v in best_params.items():
        print(f"  - {k}: {v}")

    # Save best hyperparameter YAML file in configs/
    best_yaml_path = os.path.join(configs_dir, f"best_hyperparams_{mode}.yaml")
    best_config_data = {
        "mode": mode,
        "env_name": env_name,
        "best_score": float(best_value),
        "hyperparameters": best_params,
    }
    with open(best_yaml_path, "w") as f:
        yaml.dump(best_config_data, f, default_flow_style=False)

    # Save full BO study log
    study_yaml_path = os.path.join(output_dir, f"bo_study_{mode}.yaml")
    trials_summary = [
        {"trial": t.number, "value": t.value, "params": t.params} for t in study.trials if t.value is not None
    ]
    with open(study_yaml_path, "w") as f:
        yaml.dump({"best_params": best_params, "trials": trials_summary}, f, default_flow_style=False)

    print(f"[INFO] Best hyperparameters saved to '{best_yaml_path}'")
    return best_params
