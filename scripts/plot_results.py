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
import termios
import tty
from typing import List, Dict, Any, Optional
import pandas as pd

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


def select_interactive(title: str, options: List[str], multi_select: bool = True) -> List[int]:
    """
    Renders an interactive terminal checklist menu using ANSI escape sequences.
    User navigates with Up/Down arrow keys, toggles with Spacebar, and confirms with Enter.

    Args:
        title: Header title for the selection prompt.
        options: List of string labels to display.
        multi_select: Whether multiple items can be checked with Spacebar.

    Returns:
        List[int]: List of selected zero-based option indices.
    """
    if not options:
        return []

    if not sys.stdin.isatty():
        return list(range(len(options)))

    selected = [True] * len(options) if multi_select else [False] * len(options)
    if not multi_select and options:
        selected[0] = True

    cursor = 0
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)

    def draw_menu(is_initial: bool = False):
        if not is_initial:
            lines_to_clear = len(options) + 2
            sys.stdout.write(f"\033[{lines_to_clear}F")

        sys.stdout.write("\033[J")  # Clear from cursor down
        sys.stdout.write(f"\033[1;36m{title}\033[0m\r\n")
        sys.stdout.write("\033[90m(Nav: UP/DOWN arrows | Toggle: SPACE [x] | All: 'a' | Confirm: ENTER)\033[0m\r\n")

        for idx, option in enumerate(options):
            is_cursor = (idx == cursor)
            prefix = "\033[1;33m> \033[0m" if is_cursor else "  "
            chk = "\033[1;32m[x]\033[0m" if selected[idx] else "\033[90m[ ]\033[0m"
            label = f"\033[1;37m{option}\033[0m" if is_cursor else f"\033[37m{option}\033[0m"
            sys.stdout.write(f"{prefix}{chk} {label}\r\n")

        sys.stdout.flush()

    try:
        tty.setraw(fd)
        sys.stdout.write("\033[?25l")  # Hide cursor
        draw_menu(is_initial=True)

        while True:
            ch = sys.stdin.read(1)
            if ch == "\x1b":
                next1 = sys.stdin.read(1)
                next2 = sys.stdin.read(1)
                if next1 == "[":
                    if next2 == "A":  # Up arrow
                        cursor = (cursor - 1) % len(options)
                    elif next2 == "B":  # Down arrow
                        cursor = (cursor + 1) % len(options)
            elif ch == " ":  # Spacebar
                if multi_select:
                    selected[cursor] = not selected[cursor]
                else:
                    selected = [False] * len(options)
                    selected[cursor] = True
            elif ch.lower() == "a" and multi_select:
                all_val = not all(selected)
                selected = [all_val] * len(options)
            elif ch in ["\r", "\n"]:  # Enter
                break
            elif ch == "\x03":  # Ctrl+C
                sys.exit(0)

            draw_menu(is_initial=False)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        sys.stdout.write("\033[?25h\n")  # Show cursor

    return [idx for idx, sel in enumerate(selected) if sel]


def interactive_menu(results_dir: str) -> None:
    """
    Launches an interactive terminal checklist menu to select runs and metrics for multi-plot generation.
    """
    results_dir = os.path.abspath(results_dir)
    print("\n" + "=" * 65)
    print("      Contextual RL Interactive Plotting CLI")
    print("=" * 65)

    csv_files = discover_csv_files(results_dir)
    if not csv_files:
        print(f"\n[ERROR] No result CSV files found in '{results_dir}'. Run a training script first.")
        return

    rel_paths = [os.path.relpath(f, PROJECT_ROOT) for f in csv_files]

    # 1. Interactive Run Selection
    selected_run_indices = select_interactive(
        title=f"SELECT EXPERIMENT RUNS TO PLOT ({len(csv_files)} runs found):",
        options=rel_paths,
        multi_select=True,
    )

    if not selected_run_indices:
        print("[ERROR] No runs selected.")
        return

    selected_files = [csv_files[i] for i in selected_run_indices]
    print(f"\n[INFO] Selected {len(selected_files)} run file(s).")

    # 2. Interactive Metric Selection
    available_metrics = get_available_metrics(selected_files)
    if not available_metrics:
        print("[ERROR] No numeric metric columns found in selected CSV files.")
        return

    formatted_metrics = [f"{m:15s} ({m.replace('_', ' ').title()})" for m in available_metrics]
    selected_metric_indices = select_interactive(
        title="SELECT METRICS TO PLOT (Multiple selection supported):",
        options=formatted_metrics,
        multi_select=True,
    )

    if not selected_metric_indices:
        selected_metric_indices = [0]

    selected_metrics = [available_metrics[i] for i in selected_metric_indices]
    print(f"[INFO] Selected {len(selected_metrics)} metric(s): {', '.join(selected_metrics)}")

    # 3. Generate All Selected Plots
    print("\n" + "-" * 65)
    print("      GENERATING PLOTS")
    print("-" * 65)

    plotter = Plotter()
    for metric in selected_metrics:
        metric_title = f"PPO {metric.replace('_', ' ').title()} Comparison"
        output_filename = f"{metric}_comparison.png"
        output_path = os.path.join(results_dir, output_filename)

        plotter.plot_multiple_csv_runs(
            filepaths=selected_files,
            title=metric_title,
            output_path=output_path,
            metric_col=metric,
        )
        rel_output = os.path.relpath(output_path, PROJECT_ROOT)
        print(f"  [SUCCESS] Created {metric.replace('_', ' ').title()} Plot: '{rel_output}'")

    print("\n[SUCCESS] All selected plot images generated successfully!\n")


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
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Force launch interactive terminal menu mode.",
    )

    args = parser.parse_args()

    if args.interactive or (not args.files and args.metric_col is None and args.output is None):
        interactive_menu(args.dir)
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
