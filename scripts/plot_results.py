"""
Interactive Plotting CLI script for Contextual Reinforcement Learning Results.

This script provides an interactive terminal menu with arrow-key navigation,
spacebar checkbox toggling [x], and multi-metric plot generation.

Usage Examples:
    # Launch interactive terminal selection window:
    python scripts/plot_results.py

    # Non-interactive CLI mode:
    python scripts/plot_results.py --files scripts/results/*/oracle_seed0_results.csv --metric_col mean_return
"""

import argparse
import glob
import os
import sys
from typing import List, Dict, Any, Optional
import pandas as pd

# Bug fix (not in proposal): force the headless 'Agg' backend before any matplotlib.pyplot import
# happens anywhere downstream (this script's own plotting.py import, plus train/evaluate/probing/
# bo_search/pilot_runs when invoked through main.py). Without this, matplotlib auto-picks an
# interactive backend (TkAgg on Windows when tkinter is present), which causes spurious
# "RuntimeError: main thread is not in main loop" errors from Tk's image garbage collector during
# long eval loops with many sequential plots. The PNGs still saved correctly either way -- this
# just removes the noisy, harmless-looking errors.
import matplotlib
matplotlib.use("Agg")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.env_utils import resolve_path
from src.utils.plotting import Plotter


def discover_csv_files(results_dir: str) -> List[str]:
    """
    Recursively discovers all result CSV files under results_dir.
    """
    pattern = os.path.join(results_dir, "**", "*.csv")
    files = glob.glob(pattern, recursive=True)
    valid_files = []
    for f in sorted(files):
        try:
            df = pd.read_csv(f, nrows=2)
            if "step" in df.columns or "Step" in df.columns:
                valid_files.append(f)
        except Exception:
            pass
    return sorted(valid_files)


def get_available_metrics(csv_files: List[str]) -> List[str]:
    """
    Extracts common numeric metric column names from selected CSV files.
    """
    metrics_set = set()
    for f in csv_files:
        try:
            df = pd.read_csv(f, nrows=2)
            for col in df.columns:
                if col.lower() not in ["step", "unnamed: 0", "index"]:
                    metrics_set.add(col)
        except Exception:
            pass

    preferred_order = ["mean_return", "value_loss", "policy_loss", "entropy_loss"]
    ordered_metrics = [m for m in preferred_order if m in metrics_set]
    remaining = sorted(list(metrics_set - set(ordered_metrics)))
    return ordered_metrics + remaining


def generate_all_plots(results_dir: str) -> None:
    """
    Discovers all result CSVs and metrics under results_dir and plots every metric-comparison
    chart automatically, with no interactive prompts.
    """
    results_dir = os.path.abspath(results_dir)
    print("\n" + "=" * 65)
    print("      Contextual RL Plotting")
    print("=" * 65)

    csv_files = discover_csv_files(results_dir)
    if not csv_files:
        print(f"\n[ERROR] No result CSV files found in '{results_dir}'. Run a training script first.")
        return

    print(f"\n[INFO] Found {len(csv_files)} run file(s).")

    available_metrics = get_available_metrics(csv_files)
    if not available_metrics:
        print("[ERROR] No numeric metric columns found in the discovered CSV files.")
        return

    print(f"[INFO] Plotting {len(available_metrics)} metric(s): {', '.join(available_metrics)}")

    print("\n" + "-" * 65)
    print("      GENERATING PLOTS")
    print("-" * 65)

    plotter = Plotter()
    for metric in available_metrics:
        metric_title = f"PPO {metric.replace('_', ' ').title()} Comparison"
        output_filename = f"{metric}_comparison.png"
        output_path = os.path.join(results_dir, output_filename)

        plotter.plot_multiple_csv_runs(
            filepaths=csv_files,
            title=metric_title,
            output_path=output_path,
            metric_col=metric,
        )
        rel_output = os.path.relpath(output_path, PROJECT_ROOT)
        print(f"  [SUCCESS] Created {metric.replace('_', ' ').title()} Plot: '{rel_output}'")

    print("\n[SUCCESS] All plot images generated successfully!\n")


def main():
    parser = argparse.ArgumentParser(
        description="Plot results across multiple CSV runs via interactive menu or CLI arguments."
    )
    parser.add_argument(
        "--files",
        type=str,
        nargs="+",
        default=None,
        help="List of CSV file paths to plot.",
    )
    parser.add_argument(
        "--dir",
        type=str,
        default="results",
        help="Directory to search for result CSV files (default: 'results').",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Custom title for the plot.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path for the plot image.",
    )
    parser.add_argument(
        "--metric_col",
        type=str,
        default=None,
        help="CSV column name to plot (e.g. 'mean_return', 'value_loss').",
    )

    args = parser.parse_args()

    if not args.files and args.metric_col is None and args.output is None:
        generate_all_plots(args.dir)
    else:
        filepaths: List[str] = []
        if args.files:
            filepaths.extend(args.files)
        else:
            filepaths = discover_csv_files(resolve_path(args.dir))

        if not filepaths:
            print("[ERROR] No CSV files specified or found.")
            return

        metric = args.metric_col if args.metric_col else "mean_return"
        title = args.title if args.title else f"{metric.replace('_', ' ').title()} Comparison"
        output = args.output if args.output else os.path.join(resolve_path(args.dir), f"{metric}_comparison.png")

        print(f"[INFO] Plotting {len(filepaths)} CSV run file(s) for metric '{metric}':")
        for f in filepaths:
            print(f"  - {os.path.relpath(f, PROJECT_ROOT)}")

        plotter = Plotter()
        plotter.plot_multiple_csv_runs(
            filepaths=filepaths,
            title=title,
            output_path=output,
            metric_col=metric,
        )
        print(f"\n[SUCCESS] Plot saved to '{output}'.")


if __name__ == "__main__":
    main()
