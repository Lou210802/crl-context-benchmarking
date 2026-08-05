"""
Plotting CLI script for Contextual Reinforcement Learning CSV Results.

Usage Examples:
    # Plot all CSV files in a results directory:
    python scripts/plot_results.py --dir scripts/results --output scripts/results/comparison.png

    # Plot specific CSV files:
    python scripts/plot_results.py --files scripts/results/context_free_seed0_results.csv scripts/results/oracle_seed0_results.csv --title "CARL CartPole Baselines"
"""

import argparse
import glob
import os
from typing import List

from src.utils.plotting import Plotter


def main():
    parser = argparse.ArgumentParser(
        description="Plot mean return learning curves across multiple CSV result runs."
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
        default=None,
        help="Directory containing CSV result files to plot.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Baseline Mean Return Comparison",
        help="Title for the generated plot.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="scripts/results/mean_return_comparison.png",
        help="Output file path for the plot image.",
    )
    parser.add_argument(
        "--metric_col",
        type=str,
        default="mean_return",
        help="CSV column name for mean return values.",
    )
    parser.add_argument(
        "--step_col",
        type=str,
        default="step",
        help="CSV column name for environmental steps.",
    )

    args = parser.parse_args()

    filepaths: List[str] = []
    if args.files:
        filepaths.extend(args.files)
    if args.dir:
        pattern = os.path.join(args.dir, "*.csv")
        filepaths.extend(sorted(glob.glob(pattern)))

    if not filepaths:
        print("[ERROR] No CSV files specified or found. Provide --files or --dir.")
        return

    print(f"[INFO] Plotting {len(filepaths)} CSV run file(s):")
    for f in filepaths:
        print(f"  - {f}")

    plotter = Plotter()
    plotter.plot_multiple_csv_runs(
        filepaths=filepaths,
        title=args.title,
        output_path=args.output,
        metric_col=args.metric_col,
        step_col=args.step_col,
    )
    print(f"\n[SUCCESS] Plot saved to '{args.output}'.")


if __name__ == "__main__":
    main()
