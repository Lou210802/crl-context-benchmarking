"""
Unified Contextual RL Benchmarking Pipeline CLI.

Commands:
    # 1. Train models across algorithms and seeds in parallel:
    python main.py train --modes context_free oracle vae --seeds 0 1 2 3 4 5 6 7 8 9 --total_steps 200000 --num_workers 8

    # 2. Evaluate trained models on held-out unseen physical contexts:
    python main.py eval --dir results

    # 3. Generate every metric-comparison plot for all runs in a directory:
    python main.py plot --dir results

    # 4. Full end-to-end benchmark (BO -> Train -> Evaluate -> Plot). BO runs for every mode in
    #    --modes by default (encoder search space for vae/cpc, small PPO search space for
    #    oracle/context_free), tuned at --bo_steps (defaults to --total_steps); pass --skip_bo
    #    to reuse an existing configs/best_hyperparams_<mode>.yaml instead:
    python main.py run-all --env CARLPendulum --seeds 0 1 2 --total_steps 51200 --num_workers 4

    # 5. Pilot runs to pick a sufficient total_steps PER MODE (run before optimize/run-all):
    python main.py pilot --modes context_free oracle vae cpc --step_candidates 5000 10000 15000 25000 40000
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import glob
import os
import sys
import warnings
from typing import List, Tuple, Any, Optional
import numpy as np

# Bug fix (not in proposal): force the headless 'Agg' backend before any matplotlib.pyplot import
# happens anywhere downstream (train/evaluate/probing/bo_search/pilot_runs all plot to PNG files,
# never plt.show()). Without this, matplotlib auto-picks an interactive backend (TkAgg on Windows
# when tkinter is present), which causes spurious "RuntimeError: main thread is not in main loop"
# errors from Tk's image garbage collector during long eval loops with many sequential plots. The
# PNGs still saved correctly either way -- this just removes the noisy, harmless-looking errors.
import matplotlib
matplotlib.use("Agg")

# Restrict OpenMP, MKL, OpenBLAS, etc. to 1 thread per worker to prevent CPU thread thrashing across workers
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# Configure global NumPy and Warning filters to prevent third-party library noise (CARL / ConfigSpace)
np.seterr(divide="ignore", invalid="ignore")
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*invalid value encountered in.*divide.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*Module .* not found.*")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.pipeline.train import train_single_run
from src.pipeline.evaluate import evaluate_run_directory, plot_evaluation_comparison_bar_chart
from src.pipeline.bo_search import run_bayesian_optimization
from src.pipeline.pilot_runs import run_pilot_step_analysis
from src.utils.env_utils import resolve_path
from scripts.plot_results import generate_all_plots


def _init_worker():
    """Initializer for child CPU worker processes to enforce single-threaded PyTorch execution."""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    try:
        import torch
        torch.set_num_threads(1)
    except ImportError:
        pass


def _worker_train_task(task_args: Tuple[str, str, int, int, int, Optional[List[str]], str, str]) -> str:
    mode, env_name, total_steps, seed, num_contexts, vary_contexts, exp_tag, output_dir = task_args
    return train_single_run(
        mode=mode,
        env_name=env_name,
        total_steps=total_steps,
        seed=seed,
        num_contexts=num_contexts,
        vary_contexts=vary_contexts,
        exp_tag=exp_tag,
        output_dir=output_dir,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Unified Contextual RL Benchmarking Pipeline CLI."
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to execute.")

    # -------------------------------------------------------------
    # 1. TRAIN Command
    # -------------------------------------------------------------
    train_parser = subparsers.add_parser("train", help="Train baseline and representation models.")
    train_parser.add_argument(
        "--modes",
        type=str,
        nargs="+",
        default=["context_free", "oracle", "vae", "cpc"],
        help="List of algorithms to train ('context_free', 'oracle', 'vae', 'cpc', 'all').",
    )
    train_parser.add_argument(
        "--env",
        type=str,
        default="CARLPendulum",
        help="Gymnasium or CARL environment name (default: CARLPendulum).",
    )
    train_parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],  # Proposal doesn't specify a training-seed count. Tutor feedback: don't let encoder differences get lost in seed variance -- train each mode across multiple seeds (was 8; doubled to 16 after cpc showed high across-seed variance at 100k steps).
        help="List of random seeds (default: 0-15).",
    )
    train_parser.add_argument(
        "--total_steps",
        type=int,
        default=25000,  # Proposal: 10k-15k steps (was 15000). Tutor feedback: joint encoder+policy training needs more steps so encoder differences don't get lost in seed noise -- raised to 25k.
        help="Total environment interaction steps per run (default: 25000).",
    )
    train_parser.add_argument(
        "--num_contexts",
        type=int,
        default=30,  # Sopt (optimization contexts, proposal's 10-50 range; was 10). Raised to reduce context-sampling noise per tutor feedback on seed variance.
        help="Number of training context instances (default: 30).",
    )
    train_parser.add_argument(
        "--vary_contexts",
        type=str,
        nargs="+",
        default=None,
        help="Specific physical context parameters to vary (e.g., 'gravity' or 'g' 'l'). Default: all.",
    )
    train_parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of parallel CPU worker processes for multi-seed/mode runs (default: 4).",
    )
    train_parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Directory to save experiment run folders (default: 'results').",
    )
    train_parser.add_argument(
        "--exp_tag",
        type=str,
        default="",
        help="Optional custom experiment tag.",
    )

    # -------------------------------------------------------------
    # 2. EVAL Command
    # -------------------------------------------------------------
    eval_parser = subparsers.add_parser("eval", help="Evaluate trained models on held-out unseen contexts.")
    eval_parser.add_argument(
        "--dir",
        type=str,
        default="results",
        help="Directory containing trained run folders (default: 'results').",
    )
    eval_parser.add_argument(
        "--eval_episodes",
        type=int,
        default=20,
        help="Number of evaluation episodes per run (default: 20).",
    )
    eval_parser.add_argument(
        "--eval_context_seed",
        type=int,
        default=9999,
        help="Seed for sampling held-out evaluation contexts (default: 9999).",
    )
    eval_parser.add_argument(
        "--num_eval_contexts",
        type=int,
        default=25,  # Stest (held-out test contexts) -- proposal doesn't give a count (was 10). Raised to reduce generalization-gap noise.
        help="Number of held-out test context instances (default: 25).",
    )
    eval_parser.add_argument(
        "--vary_contexts",
        type=str,
        nargs="+",
        default=None,
        help="Specific physical context parameters to vary during evaluation (e.g., 'gravity' or 'g' 'l'). Default: uses training config.",
    )

    # -------------------------------------------------------------
    # 3. PLOT Command
    # -------------------------------------------------------------
    plot_parser = subparsers.add_parser("plot", help="Interactive terminal plotting menu.")
    plot_parser.add_argument(
        "--dir",
        type=str,
        default="results",
        help="Directory containing result CSV files (default: 'results').",
    )

    # -------------------------------------------------------------
    # 4. OPTIMIZE Command (Bayesian Optimization)
    # -------------------------------------------------------------
    opt_parser = subparsers.add_parser("optimize", help="Run Bayesian Optimization (BO) hyperparameter search.")
    opt_parser.add_argument(
        "--modes",
        type=str,
        nargs="+",
        default=["vae", "cpc"],
        help="Modes to optimize ('vae', 'cpc', 'oracle', 'context_free', or 'all'). vae/cpc tune "
        "the encoder; oracle/context_free tune PPO itself (lr_actor, lr_critic, ent_coef).",
    )
    opt_parser.add_argument(
        "--env",
        type=str,
        default="CARLPendulum",
        help="Environment name.",
    )
    opt_parser.add_argument(
        "--n_trials",
        type=int,
        default=30,  # Proposal: 12 BO trials (Algorithm 1). Raised to 30 for a more reliable optimum -- tutor also noted 12 trials aren't many.
        help="Number of BO trials per mode (default: 30).",
    )
    opt_parser.add_argument(
        "--bo_steps",
        type=int,
        default=25000,  # Matches total_steps (25k) so BO isn't tuned on a shorter horizon than the final training run (was 15000).
        help="Training step budget per BO trial (default: 25000).",
    )
    opt_parser.add_argument(
        "--num_contexts",
        type=int,
        default=30,  # Sopt (optimization contexts, proposal's 10-50 range; was 10). Raised to reduce context-sampling noise per tutor feedback on seed variance.
        help="Number of training context instances per BO trial (default: 30).",
    )
    opt_parser.add_argument(
        "--vary_contexts",
        type=str,
        nargs="+",
        default=None,
        help="Specific physical context parameters to vary during BO search.",
    )
    opt_parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Output directory for BO study results.",
    )

    # -------------------------------------------------------------
    # 5. PILOT Command (step-budget convergence analysis, per mode)
    # -------------------------------------------------------------
    pilot_parser = subparsers.add_parser(
        "pilot", help="Pilot runs to pick a sufficient total_steps PER MODE (run before optimize/run-all)."
    )
    pilot_parser.add_argument(
        "--modes",
        type=str,
        nargs="+",
        default=["context_free", "oracle", "vae", "cpc"],
        help="Modes to include in the step-budget comparison.",
    )
    pilot_parser.add_argument(
        "--env",
        type=str,
        default="CARLPendulum",
        help="Gymnasium or CARL environment name (default: CARLPendulum).",
    )
    pilot_parser.add_argument(
        "--step_candidates",
        type=int,
        nargs="+",
        default=[5000, 10000, 15000, 25000, 40000],
        help="Total-step budgets to compare (default: 5000 10000 15000 25000 40000).",
    )
    pilot_parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2],
        help="Seeds to repeat each step budget over (default: 0 1 2).",
    )
    pilot_parser.add_argument(
        "--num_contexts",
        type=int,
        default=30,  # Sopt (proposal's 10-50 range; was 10), kept consistent with the rest of the pipeline. Note: pilot_runs/ data predating this change was generated with the old default of 10.
        help="Number of training context instances (default: 30).",
    )
    pilot_parser.add_argument(
        "--vary_contexts",
        type=str,
        nargs="+",
        default=None,
        help="Specific physical context parameters to vary during training.",
    )
    pilot_parser.add_argument(
        "--output_dir",
        type=str,
        default="results/pilot_runs",
        help="Directory to save pilot run folders and the summary CSV/plot (default: 'results/pilot_runs').",
    )

    # -------------------------------------------------------------
    # 6. RUN-ALL Command
    # -------------------------------------------------------------
    runall_parser = subparsers.add_parser("run-all", help="Full end-to-end benchmark (BO -> Train -> Eval -> Plot).")
    runall_parser.add_argument(
        "--env",
        type=str,
        default="CARLPendulum",
        help="Environment name.",
    )
    runall_parser.add_argument(
        "--modes",
        type=str,
        nargs="+",
        default=["context_free", "oracle", "vae", "cpc"],
        help="Algorithms to benchmark.",
    )
    runall_parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],  # Proposal doesn't specify a training-seed count. Tutor feedback: don't let encoder differences get lost in seed variance -- train each mode across multiple seeds (was 8; doubled to 16 after cpc showed high across-seed variance at 100k steps).
        help="Model seeds (default: 0-15).",
    )
    runall_parser.add_argument(
        "--vary_contexts",
        type=str,
        nargs="+",
        default=None,
        help="Specific physical context parameters to vary (e.g., 'gravity' or 'g' 'l'). Default: all.",
    )
    runall_parser.add_argument(
        "--skip_bo",
        action="store_true",
        help="Skip the BO step and train vae/cpc with whatever configs/best_hyperparams_<mode>.yaml "
        "already exists (falls back to untuned defaults if none exists). Use this when you already "
        "tuned separately (e.g. via 'optimize') and want run-all to reuse that config as-is.",
    )
    runall_parser.add_argument(
        "--n_trials",
        type=int,
        default=30,  # Proposal: 12 BO trials (Algorithm 1). Raised to 30 for a more reliable optimum -- tutor also noted 12 trials aren't many.
        help="Number of BO trials per mode, if BO runs (default: 30).",
    )
    runall_parser.add_argument(
        "--bo_steps",
        type=int,
        default=None,
        help="Training step budget per BO trial, if BO runs. Defaults to --total_steps so BO isn't "
        "tuned on a shorter horizon than the final training run.",
    )
    runall_parser.add_argument(
        "--total_steps",
        type=int,
        default=25000,  # Proposal: 10k-15k steps (was 15000). Tutor feedback: joint encoder+policy training needs more steps so encoder differences don't get lost in seed noise -- raised to 25k.
        help="Training steps (default: 25000).",
    )
    runall_parser.add_argument(
        "--num_contexts",
        type=int,
        default=30,  # Sopt (optimization contexts, proposal's 10-50 range; was 10). Raised to reduce context-sampling noise per tutor feedback on seed variance.
        help="Number of training context instances (default: 30).",
    )
    runall_parser.add_argument(
        "--num_eval_contexts",
        type=int,
        default=25,  # Stest (held-out test contexts) -- proposal doesn't give a count (was 10). Raised to reduce generalization-gap noise.
        help="Number of held-out test context instances (default: 25).",
    )
    runall_parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of parallel CPU worker processes (default: 4).",
    )
    runall_parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Output directory (default: 'results').",
    )

    args = parser.parse_args()

    if args.command == "train":
        modes = args.modes
        if "all" in modes:
            modes = ["context_free", "oracle", "vae", "cpc"]

        train_tasks = [
            (mode, args.env, args.total_steps, seed, args.num_contexts, args.vary_contexts, args.exp_tag, args.output_dir)
            for mode in modes
            for seed in args.seeds
        ]

        if args.num_workers > 1 and len(train_tasks) > 1:
            print(f"\n[INFO] Launching {len(train_tasks)} training run(s) across {args.num_workers} parallel CPU workers...")
            with ProcessPoolExecutor(max_workers=args.num_workers, initializer=_init_worker) as executor:
                list(executor.map(_worker_train_task, train_tasks))
        else:
            for task in train_tasks:
                _worker_train_task(task)

    elif args.command == "eval":
        target_dir = resolve_path(args.dir)
        run_dirs = [
            d for d in glob.glob(os.path.join(target_dir, "*")) if os.path.isdir(d) and os.path.exists(os.path.join(d, "config.yaml"))
        ]
        if not run_dirs:
            print(f"[ERROR] No valid trained run directories found in '{target_dir}'.")
            return

        print(f"\n[INFO] Found {len(run_dirs)} run folder(s) for evaluation.")
        eval_summaries = []
        for r_dir in sorted(run_dirs):
            res = evaluate_run_directory(
                run_dir=r_dir,
                eval_episodes=args.eval_episodes,
                eval_context_seed=args.eval_context_seed,
                num_eval_contexts=args.num_eval_contexts,
                vary_contexts=args.vary_contexts,
            )
            eval_summaries.append(res)

        if eval_summaries:
            plot_evaluation_comparison_bar_chart(
                eval_summaries,
                output_path=os.path.join(target_dir, "eval_returns_comparison.png"),
            )

    elif args.command == "plot":
        generate_all_plots(resolve_path(args.dir))

    elif args.command == "optimize":
        modes = args.modes
        if "all" in modes:
            modes = ["context_free", "oracle", "vae", "cpc"]

        for m in modes:
            if m in ["vae", "cpc", "oracle", "context_free"]:
                run_bayesian_optimization(
                    mode=m,
                    env_name=args.env,
                    n_trials=args.n_trials,
                    bo_steps=args.bo_steps,
                    num_contexts=args.num_contexts,
                    vary_contexts=args.vary_contexts,
                    output_dir=args.output_dir,
                )

    elif args.command == "pilot":
        modes = args.modes
        if "all" in modes:
            modes = ["context_free", "oracle", "vae", "cpc"]

        run_pilot_step_analysis(
            modes=modes,
            env_name=args.env,
            step_candidates=args.step_candidates,
            seeds=args.seeds,
            num_contexts=args.num_contexts,
            vary_contexts=args.vary_contexts,
            output_dir=args.output_dir,
        )

    elif args.command == "run-all":
        output_dir = resolve_path(args.output_dir)
        modes = args.modes
        if "all" in modes:
            modes = ["context_free", "oracle", "vae", "cpc"]

        if not args.skip_bo:
            bo_modes = [m for m in modes if m in ("vae", "cpc", "oracle", "context_free")]
            bo_steps = args.bo_steps if args.bo_steps is not None else args.total_steps
            for m in bo_modes:
                print("\n" + "=" * 65)
                print(f"      BAYESIAN OPTIMIZATION -- {m.upper()}")
                print("=" * 65)
                run_bayesian_optimization(
                    mode=m,
                    env_name=args.env,
                    n_trials=args.n_trials,
                    bo_steps=bo_steps,
                    num_contexts=args.num_contexts,
                    vary_contexts=args.vary_contexts,
                    output_dir=output_dir,
                )

        train_tasks = [
            (mode, args.env, args.total_steps, seed, args.num_contexts, args.vary_contexts, "", output_dir)
            for mode in modes
            for seed in args.seeds
        ]

        if args.num_workers > 1 and len(train_tasks) > 1:
            print(f"\n[INFO] Launching {len(train_tasks)} training run(s) across {args.num_workers} parallel CPU workers...")
            with ProcessPoolExecutor(max_workers=args.num_workers, initializer=_init_worker) as executor:
                trained_run_dirs = list(executor.map(_worker_train_task, train_tasks))
        else:
            trained_run_dirs = [_worker_train_task(t) for t in train_tasks]

        print("\n" + "=" * 65)
        print("      EVALUATING ALL TRAINED MODELS ON HELD-OUT CONTEXTS")
        print("=" * 65)
        eval_summaries = []
        for r_dir in trained_run_dirs:
            res = evaluate_run_directory(
                run_dir=r_dir,
                num_eval_contexts=args.num_eval_contexts,
                vary_contexts=args.vary_contexts,
            )
            eval_summaries.append(res)

        if eval_summaries:
            plot_evaluation_comparison_bar_chart(
                eval_summaries,
                output_path=os.path.join(output_dir, "eval_returns_comparison.png"),
            )

        print("\n[INFO] Launching Interactive Plotter...")
        generate_all_plots(output_dir)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()

