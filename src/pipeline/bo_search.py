"""
Bayesian Optimization (BO) Hyperparameter Tuning Module.

Uses Optuna (Tree-structured Parzen Estimator / TPE Bayesian Optimization sampler)
to automatically discover optimal hyperparameters for representation learning encoders
(VAE and CPC) prior to final benchmark evaluation on unseen test seeds.

SEARCH SPACE (matches the proposal, kept low-dimensional given the small trial budget):
    VAE: encoder_lr, latent_dim, kl_weight (beta)
    CPC: encoder_lr, latent_dim, cpc_temperature
    ORACLE / CONTEXT_FREE (not in the proposal -- team decision): lr_actor, lr_critic, ent_coef.
        These two modes have no encoder, so there's nothing in the proposal's search space to
        tune; oracle/context_free previously always ran with fixed PPO defaults while vae/cpc
        got BO-tuned encoders, which wasn't a fair comparison. Kept deliberately small (3 PPO
        hyperparameters) to match the encoder search spaces' dimensionality. lr_actor/lr_critic
        search range: 1e-4 to 3e-3 (log); ent_coef: 1e-3 to 5e-2 (log) -- narrowed from a wider
        1e-5/1e-2 and 1e-4/1e-1 range after that wider range let a 30-trial/3-seed BO land on an
        extreme, unstable combo for oracle.
hidden_dim, max_history_len, and predict_horizon are intentionally NOT tuned here -- they
stay fixed at train_single_run()'s defaults. CPC's aux_weight/supcon_weight are fixed
constants (see CPCContextEncoder), not BO-tuned.

OBJECTIVE: the BO score is held-out `eval_mean_return`, averaged over multiple training
seeds per trial (see trial_seeds below). Linear context-probing R^2 is still computed and
logged per trial (trial.set_user_attr), but is NOT part of the score -- per the supervisor
feedback, probing R^2 is an *insight* metric for explaining which encoder captured context,
not a target to optimize directly. Optimizing R^2 directly would reward representations
that are easy to linearly decode even if they don't help the PPO policy, which isn't the
actual research question.

FAIRNESS NOTE (see supervisor feedback): VAE and CPC have different hyperparameters and
therefore different search-space sizes/difficulty. Running the same n_trials for both does
NOT guarantee a matched tuning budget in any strict sense -- state this honestly.

RELIABILITY: not in the proposal's Algorithm 1 (single seed per trial). Added per automl.org RL
best practices ("enhance reliability through multiple seeds or episodes") -- each trial now
trains and evaluates trial_seeds (default: 3 seeds) independently and averages eval_mean_return
across them, instead of a single fixed seed, so a trial's score isn't just seed luck. Triples
the training cost per trial.
"""

import os
import sys
import tempfile
import yaml
from typing import Dict, Any, Optional, List

import numpy as np
import optuna
import matplotlib.pyplot as plt
import seaborn as sns

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
    n_trials: int = 30,  # Proposal: 12 BO trials (Algorithm 1). Raised to 30 for a more reliable optimum -- tutor also noted 12 trials aren't many.
    bo_steps: int = 25000,  # Matches total_steps (25k) so BO isn't tuned on a shorter horizon than the final training run (was 15000).
    val_episodes: int = 10,
    val_context_seed: int = 8888,
    vary_contexts: Optional[List[str]] = None,
    seed: int = 42,
    num_contexts: int = 30,  # Sopt (optimization contexts, proposal's 10-50 range; was 10). Raised to reduce context-sampling noise per tutor feedback on seed variance.
    trial_seeds: Optional[List[int]] = None,  # Not in the proposal (single seed per trial). Averaged over 3 seeds per automl.org RL best practices to reduce seed-luck noise in each trial's score.
    output_dir: str = "results",
) -> Dict[str, Any]:
    """
    Executes Bayesian Optimization (BO) search over hyperparameter space for VAE or CPC context encoders.

    Args:
        mode: Mode to tune ('vae', 'cpc', 'oracle', or 'context_free').
        env_name: Benchmark environment name.
        n_trials: Number of BO trials (default: 30).
        bo_steps: Training step budget per optimization trial (default: 25000).
        val_episodes: Number of validation episodes per trial evaluation.
        val_context_seed: Context seed for validation set.
        vary_contexts: Optional list of context parameters to vary during BO search.
        seed: Random seed for the BO study's TPE sampler (search reproducibility only --
            no longer used as a training seed, see trial_seeds).
        num_contexts: Number of training context instances ("optimization seeds" Sopt in the
            proposal) each trial is trained on (default: 30).
        trial_seeds: Training seeds averaged per BO trial for a more reliable score (default:
            [100, 200, 300] -- deliberately disjoint from the final benchmark's seeds 0-15 in
            main.py, so no agent seed used to pick hyperparameters is also among the seeds
            used to report final results).
        output_dir: Base directory to save BO study log and best config.

    Returns:
        Dict[str, Any]: Best hyperparameter dictionary found by Bayesian Optimization.
    """
    if mode not in ["vae", "cpc", "oracle", "context_free"]:
        raise ValueError(f"Bayesian Optimization search is only configured for 'vae', 'cpc', 'oracle', and 'context_free' modes, got '{mode}'.")

    trial_seeds = trial_seeds if trial_seeds is not None else [100, 200, 300]  # disjoint from the final benchmark's seeds 0-15 (main.py)

    output_dir = resolve_path(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    configs_dir = resolve_path("configs")
    os.makedirs(configs_dir, exist_ok=True)

    print(f"\n[BO SEARCH] Starting Bayesian Optimization for '{mode.upper()}' ({n_trials} trials, {bo_steps} steps/trial, {len(trial_seeds)} seeds/trial averaged)...")

    def objective(trial: optuna.Trial) -> float:
        # Sample hyperparameters from the proposal's search space only (kept low-dimensional
        # given the small trial budget -- see supervisor feedback on HPO fairness/honesty).

        # NOT tuned -- fixed at train_single_run() defaults.
        latent_dim = 8
        encoder_lr = 3e-4
        hidden_dim = 64
        max_history_len = 25
        predict_horizon = 3
        kl_weight = 1e-3
        cpc_temperature = 0.1
        lr_actor = 3e-4
        lr_critic = 3e-4
        ent_coef = 0.01

        if mode == "vae":
            # Bug fix: latent_dim/encoder_lr used to be sampled unconditionally above (for every
            # mode), wasting 2 of oracle/context_free's 5 search dimensions on parameters that have
            # zero effect for them (no encoder). Moved here so each mode only searches its own
            # relevant dimensions.
            latent_dim = trial.suggest_categorical("latent_dim", [4, 8, 16])
            encoder_lr = trial.suggest_float("encoder_lr", 1e-4, 1e-2, log=True)
            kl_weight = trial.suggest_float("kl_weight", 1e-5, 1e-1, log=True)  # beta
        elif mode == "cpc":
            latent_dim = trial.suggest_categorical("latent_dim", [4, 8, 16])
            encoder_lr = trial.suggest_float("encoder_lr", 1e-4, 1e-2, log=True)
            cpc_temperature = trial.suggest_float("cpc_temperature", 0.05, 0.5, log=True)
        elif mode in ("oracle", "context_free"):
            # No encoder to tune -- tune PPO itself instead (see SEARCH SPACE note in module docstring).
            # Range narrowed from (1e-5, 1e-2)/(1e-4, 1e-1) after the first 25k/100k full-BO runs
            # landed oracle on an extreme, value-function-starving combo (lr_actor=0.0097,
            # lr_critic=6.4e-5) that made context_free implausibly beat oracle -- 30 trials x 3 seeds
            # isn't enough to reliably cover a full 3-decade log-range, so we tightened it around
            # commonly-stable PPO values instead.
            lr_actor = trial.suggest_float("lr_actor", 1e-4, 3e-3, log=True)
            lr_critic = trial.suggest_float("lr_critic", 1e-4, 3e-3, log=True)
            ent_coef = trial.suggest_float("ent_coef", 1e-3, 5e-2, log=True)

        # Train + evaluate this hyperparameter config once per seed in trial_seeds and average
        # the results, instead of a single fixed training seed (see RELIABILITY note above).
        seed_returns: List[float] = []
        seed_probing_r2s: List[Optional[float]] = []
        for s in trial_seeds:
            with tempfile.TemporaryDirectory() as temp_dir:
                run_dir = train_single_run(
                    mode=mode,
                    env_name=env_name,
                    total_steps=bo_steps,
                    rollout_steps=1024,
                    seed=s,
                    num_contexts=num_contexts,
                    vary_contexts=vary_contexts,
                    latent_dim=latent_dim,
                    hidden_dim=hidden_dim,
                    encoder_lr=encoder_lr,
                    kl_weight=kl_weight,
                    cpc_temperature=cpc_temperature,
                    predict_horizon=predict_horizon,
                    max_history_len=max_history_len,
                    lr_actor=lr_actor,
                    lr_critic=lr_critic,
                    ent_coef=ent_coef,
                    exp_tag=f"bo_trial_{trial.number}_seed{s}",
                    output_dir=temp_dir,
                    use_bo_config=False,
                )

                # Evaluate trial on validation context seed
                eval_res = evaluate_run_directory(
                    run_dir=run_dir,
                    eval_episodes=val_episodes,
                    eval_context_seed=val_context_seed,
                    num_eval_contexts=25,  # validation contexts (val_context_seed, not the final Stest) -- was 10, raised to reduce noise in the BO score itself
                )
                seed_returns.append(eval_res["eval_mean_return"])
                seed_probing_r2s.append(eval_res.get("probing_r2", None))

        mean_return = float(np.mean(seed_returns))
        std_return = float(np.std(seed_returns))
        valid_r2s = [r for r in seed_probing_r2s if r is not None]
        mean_probing_r2 = float(np.mean(valid_r2s)) if valid_r2s else None

        # BO objective: mean held-out task performance across trial_seeds (eval_mean_return).
        # Probing R^2 is logged as an insight/diagnostic metric per the supervisor feedback
        # ("which encoder captured context, not just which got a higher return") but is NOT
        # part of the optimized score -- optimizing R^2 directly would reward representations
        # that are easy to linearly decode even if they don't actually help the PPO policy.
        trial.set_user_attr("probing_r2", mean_probing_r2)
        trial.set_user_attr("eval_std_across_seeds", std_return)
        trial.set_user_attr("per_seed_returns", seed_returns)
        return mean_return

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_params
    best_value = study.best_value
    best_probing_r2 = study.best_trial.user_attrs.get("probing_r2")
    best_std_across_seeds = study.best_trial.user_attrs.get("eval_std_across_seeds")

    print(f"\n[BO SUCCESS] Optimization finished for '{mode.upper()}'. Best eval_mean_return: {best_value:.3f}"
          f"{f' (± {best_std_across_seeds:.3f} across {len(trial_seeds)} seeds)' if best_std_across_seeds is not None else ''}"
          f"{f' | probing R^2: {best_probing_r2:.3f}' if best_probing_r2 is not None else ''}")
    print("Best Hyperparameters:")
    for k, v in best_params.items():
        print(f"  - {k}: {v}")

    # Save best hyperparameter YAML file in configs/
    best_yaml_path = os.path.join(configs_dir, f"best_hyperparams_{mode}.yaml")
    best_config_data = {
        "mode": mode,
        "env_name": env_name,
        "best_eval_mean_return": float(best_value),
        "best_eval_std_across_seeds": best_std_across_seeds,
        "best_probing_r2": best_probing_r2,
        "trial_seeds": trial_seeds,
        "hyperparameters": best_params,
    }
    with open(best_yaml_path, "w") as f:
        yaml.dump(best_config_data, f, default_flow_style=False)

    # Save full BO study log (value = eval_mean_return, the optimized objective, now averaged
    # over trial_seeds; eval_std_across_seeds/per_seed_returns show how noisy that average was;
    # probing_r2 is logged alongside as an insight metric, not part of the score)
    study_yaml_path = os.path.join(output_dir, f"bo_study_{mode}.yaml")
    trials_summary = [
        {
            "trial": t.number,
            "eval_mean_return": t.value,
            "eval_std_across_seeds": t.user_attrs.get("eval_std_across_seeds"),
            "per_seed_returns": t.user_attrs.get("per_seed_returns"),
            "probing_r2": t.user_attrs.get("probing_r2"),
            "params": t.params,
        }
        for t in study.trials
        if t.value is not None
    ]
    with open(study_yaml_path, "w") as f:
        yaml.dump({"best_params": best_params, "trial_seeds": trial_seeds, "trials": trials_summary}, f, default_flow_style=False)

    print(f"[INFO] Best hyperparameters saved to '{best_yaml_path}'")

    plot_bo_study(mode=mode, output_dir=output_dir)

    return best_params


def plot_bo_study(mode: str, output_dir: str = "results") -> str:
    """
    Plots BO trial history from an already-saved bo_study_<mode>.yaml (no re-optimization
    needed): eval_mean_return per trial with the running best, and probing R^2 per trial
    as a separate panel (insight metric, not the optimized objective).

    Args:
        mode: 'vae' or 'cpc' -- selects which bo_study_<mode>.yaml to read.
        output_dir: Directory containing bo_study_<mode>.yaml (default: 'results').

    Returns:
        str: Absolute path to the saved plot image.
    """
    output_dir = resolve_path(output_dir)
    study_yaml_path = os.path.join(output_dir, f"bo_study_{mode}.yaml")
    if not os.path.exists(study_yaml_path):
        raise FileNotFoundError(f"No BO study file found at '{study_yaml_path}'. Run optimize first.")

    with open(study_yaml_path, "r") as f:
        data = yaml.safe_load(f)

    trials = sorted(data.get("trials", []), key=lambda t: t["trial"])
    if not trials:
        raise ValueError(f"'{study_yaml_path}' contains no completed trials to plot.")

    trial_nums = [t["trial"] for t in trials]
    returns = [t["eval_mean_return"] for t in trials]
    r2_vals = [t.get("probing_r2") for t in trials]
    running_best = np.maximum.accumulate(returns)

    sns.set_theme(style="whitegrid", palette="muted")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

    ax = axes[0]
    ax.scatter(trial_nums, returns, color="steelblue", alpha=0.8, zorder=3, label="Trial value")
    ax.plot(trial_nums, running_best, color="firebrick", linewidth=2, zorder=2, label="Best so far")
    ax.set_title(f"BO Optimization History ({mode.upper()})", fontsize=12, fontweight="bold")
    ax.set_xlabel("Trial", fontweight="bold")
    ax.set_ylabel("Eval Mean Return", fontweight="bold")
    ax.legend()
    sns.despine(ax=ax, top=True, right=True)

    ax2 = axes[1]
    if any(v is not None for v in r2_vals):
        ax2.bar(trial_nums, [v if v is not None else 0.0 for v in r2_vals], color="seagreen", alpha=0.85, edgecolor="black")
    ax2.set_title(f"Linear Context-Probing R² per Trial ({mode.upper()})", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Trial", fontweight="bold")
    ax2.set_ylabel("Probing R² (insight metric, not optimized)", fontweight="bold")
    sns.despine(ax=ax2, top=True, right=True)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"bo_study_{mode}_plot.png")
    plt.savefig(plot_path, dpi=300)
    plt.close(fig)

    print(f"[SUCCESS] BO study plot saved to '{plot_path}'")
    return plot_path
