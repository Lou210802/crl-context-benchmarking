"""
Pilot Run Step-Budget Analysis for Contextual RL Benchmarking.

Addresses supervisor feedback: joint training of a representation encoder and PPO can
require more environment interaction steps to converge than the context-free/oracle
baselines, and this difference must not get lost in seed variance. This module trains
short pilot runs across a range of `total_steps` budgets (each repeated over multiple
seeds) and analyzes convergence PER MODE separately -- context_free/oracle and vae/cpc
are expected to plateau at different step counts, so there is no single "optimal step
size" for the whole benchmark, only a per-mode recommendation.

Kept as a separate, standalone step, run BEFORE both the full benchmark (`train`/`run-all`)
and BO hyperparameter tuning (`optimize`). Once you've picked a final `total_steps` per
mode from this analysis, scale `bo_steps` (src/pipeline/bo_search.py) accordingly rather
than leaving it fixed at its current default -- a BO search tuned on a much shorter
horizon than the final training run does not necessarily transfer its optimum.

Usage:
    from src.pipeline.pilot_runs import run_pilot_step_analysis
    run_pilot_step_analysis(
        modes=["context_free", "oracle", "vae", "cpc"],
        env_name="CARLPendulum",
        step_candidates=[5000, 10000, 15000, 25000, 40000],
        seeds=[0, 1, 2],
    )
"""

import os
import sys
from typing import List, Optional

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.pipeline.train import train_single_run
from src.utils.env_utils import resolve_path
from src.utils.experiment_logger import ExperimentLogger


def _final_mean_return(run_dir: str, mode: str, seed: int) -> float:
    """
    Reads back a completed run's logged metrics CSV and returns the last logged
    `mean_return` value (already a trailing average over the last 10 completed
    episodes -- see the log_dict built in train_single_run).
    """
    csv_path = os.path.join(run_dir, f"{mode}_seed{seed}_results.csv")
    df = ExperimentLogger.load(csv_path)
    return float(df["mean_return"].iloc[-1])


def run_pilot_step_analysis(
    modes: List[str],
    env_name: str = "CARLPendulum",
    step_candidates: Optional[List[int]] = None,
    seeds: Optional[List[int]] = None,
    num_contexts: int = 30,  # Sopt (proposal's 10-50 range; was 10), kept consistent with the rest of the pipeline. Note: pilot_runs/ data predating this change was generated with the old default of 10.
    vary_contexts: Optional[List[str]] = None,
    output_dir: str = "results/pilot_runs",
) -> str:
    """
    Trains modes x step_candidates x seeds pilot runs and analyzes final-return convergence
    PER MODE, to help choose a sufficient `total_steps` for each mode independently before
    the full benchmark / BO runs. VAE and CPC auto-load already-tuned hyperparameters from
    configs/best_hyperparams_<mode>.yaml if present (train_single_run's use_bo_config=True
    default), so this reflects the config you'd actually train with.

    Args:
        modes: Modes to include (e.g. ['context_free', 'oracle', 'vae', 'cpc']).
        env_name: Gymnasium/CARL environment name.
        step_candidates: Total-step budgets to compare (default: [5000, 10000, 15000, 25000, 40000]).
        seeds: Seeds to repeat each budget over, to separate step-budget effects from seed
            variance (default: [0, 1, 2]).
        num_contexts: Number of training context instances (default: 10).
        vary_contexts: Optional list of context parameters to vary during training.
        output_dir: Directory to save pilot run folders and the summary CSV/plot.

    Returns:
        str: Absolute path to the output directory containing all pilot artifacts.
    """
    step_candidates = sorted(step_candidates or [5000, 10000, 15000, 25000, 40000])
    seeds = seeds or [0, 1, 2]
    output_dir = resolve_path(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    records = []
    for mode in modes:
        for steps in step_candidates:
            for seed in seeds:
                print(f"\n[PILOT] mode={mode} total_steps={steps} seed={seed}")
                run_dir = train_single_run(
                    mode=mode,
                    env_name=env_name,
                    total_steps=steps,
                    seed=seed,
                    num_contexts=num_contexts,
                    vary_contexts=vary_contexts,
                    exp_tag=f"pilot_{steps}steps",
                    output_dir=output_dir,
                )
                final_return = _final_mean_return(run_dir, mode, seed)
                records.append(
                    {
                        "mode": mode,
                        "total_steps": steps,
                        "seed": seed,
                        "final_mean_return": final_return,
                    }
                )

    df = pd.DataFrame(records)
    csv_path = os.path.join(output_dir, "pilot_step_analysis.csv")
    df.to_csv(csv_path, index=False)

    plot_path = os.path.join(output_dir, "pilot_step_convergence.png")
    _plot_convergence(df, plot_path)

    _print_per_mode_summary(df)

    print(f"\n[SUCCESS] Pilot step-budget analysis saved to '{output_dir}'")
    print(f"          -> Summary table: {csv_path}")
    print(f"          -> Convergence plot: {plot_path}")
    print(
        "\n[REMINDER] Pick a total_steps value PER MODE from the table/plot above (where the "
        "curve plateaus). Once decided, scale --bo_steps in `optimize` to a comparable "
        "fraction of the chosen total_steps rather than leaving it at its current default -- "
        "hyperparameters tuned on a much shorter horizon don't necessarily transfer."
    )
    return output_dir


def _print_per_mode_summary(df: pd.DataFrame) -> None:
    """Prints a per-mode table of mean +/- std final return across step budgets, plus the
    relative improvement between consecutive budgets, to make convergence easy to read
    without opening the plot. Modes are expected to plateau at different step counts
    (context_free/oracle typically earlier than vae/cpc)."""
    print("\n" + "=" * 70)
    print("      PER-MODE STEP-BUDGET CONVERGENCE SUMMARY")
    print("=" * 70)
    for mode, grp in df.groupby("mode"):
        agg = grp.groupby("total_steps")["final_mean_return"].agg(["mean", "std"]).reset_index()
        agg = agg.sort_values("total_steps").reset_index(drop=True)
        print(f"\n  [{mode.upper()}]")
        prev_mean = None
        for _, row in agg.iterrows():
            rel_change_str = ""
            if prev_mean is not None and abs(prev_mean) > 1e-8:
                rel_change = (row["mean"] - prev_mean) / abs(prev_mean)
                rel_change_str = f"  (Δ {rel_change:+.1%} vs. previous budget)"
            std_val = row["std"] if not pd.isna(row["std"]) else 0.0
            print(f"    total_steps={int(row['total_steps']):>7d} | return={row['mean']:9.2f} ± {std_val:6.2f}{rel_change_str}")
            prev_mean = row["mean"]
    print("\n" + "=" * 70)


def _plot_convergence(df: pd.DataFrame, output_path: str) -> str:
    """Plots final mean return (mean +/- std across seeds) vs. total_steps, one line per mode."""
    sns.set_theme(style="whitegrid", palette="muted")
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)

    for mode, grp in df.groupby("mode"):
        agg = grp.groupby("total_steps")["final_mean_return"].agg(["mean", "std"]).reset_index()
        ax.errorbar(
            agg["total_steps"],
            agg["mean"],
            yerr=agg["std"].fillna(0.0),
            marker="o",
            capsize=4,
            linewidth=2,
            label=mode.upper(),
        )

    ax.set_title("Pilot Runs: Final Return vs. Training Step Budget (per Mode)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Total Training Steps", fontsize=11, fontweight="bold")
    ax.set_ylabel("Final Mean Return (last 10 episodes, mean ± std over seeds)", fontsize=10, fontweight="bold")
    ax.legend(title="Mode")
    sns.despine(ax=ax, top=True, right=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path
