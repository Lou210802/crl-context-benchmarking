import os
from typing import List, Dict, Optional, Union, Tuple, Callable
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from rliable import library, metrics, plot_utils


class Plotter:
    """
    Plotter class utilizing Google's RLiable library for reliable evaluation
    and visualization of reinforcement learning algorithm benchmarks.
    """

    def __init__(
        self,
        title_fontsize: int = 16,
        axis_fontsize: int = 14,
        tick_fontsize: int = 12,
        legend_fontsize: int = 12,
        reps: int = 2000,
        confidence_interval_size: float = 0.95,
    ):
        self.title_fontsize = title_fontsize
        self.axis_fontsize = axis_fontsize
        self.tick_fontsize = tick_fontsize
        self.legend_fontsize = legend_fontsize
        self.reps = reps
        self.confidence_interval_size = confidence_interval_size
        sns.set_style("whitegrid")

    def plot_learning_curves(
        self,
        filepaths: List[str],
        title: str,
        output_path: Optional[str] = None,
        metric_col: str = "mean_return",
        step_col: str = "step",
    ) -> plt.Axes:
        """
        Plots sample efficiency learning curves with RLiable stratified bootstrap 95% CIs.

        Args:
            filepaths: List of file paths to CSV results.
            title: Title of the plot.
            output_path: Optional file path to save the plot image.
            metric_col: Column name in CSV representing return/score (default: "mean_return").
            step_col: Column name in CSV representing steps (default: "step").

        Returns:
            matplotlib.axes.Axes object.
        """
        algo_runs: Dict[str, List[np.ndarray]] = {}
        frames = None

        for filepath in filepaths:
            df = pd.read_csv(filepath)
            mode = df.attrs.get("mode")
            if not mode:
                base = os.path.basename(filepath)
                mode = base.replace("_results.csv", "").replace(".csv", "")

            if mode not in algo_runs:
                algo_runs[mode] = []

            val_col = metric_col if metric_col in df.columns else df.columns[1]
            s_col = step_col if step_col in df.columns else df.columns[0]

            algo_runs[mode].append(df[val_col].values)
            if frames is None:
                frames = df[s_col].values

        score_dict: Dict[str, np.ndarray] = {}
        for algo, runs in algo_runs.items():
            arr = np.array(runs)  # shape: (num_runs, num_steps)
            score_dict[algo] = np.expand_dims(arr, axis=1)  # shape: (num_runs, 1, num_steps)

        iqm_func = lambda x: np.array([metrics.aggregate_iqm(x[..., i]) for i in range(x.shape[-1])])
        point_estimates, interval_estimates = library.get_interval_estimates(
            score_dict,
            iqm_func,
            reps=self.reps,
            confidence_interval_size=self.confidence_interval_size,
        )

        fig, ax = plt.subplots(figsize=(10, 6))
        plot_utils.plot_sample_efficiency_curve(
            frames,
            point_estimates,
            interval_estimates,
            algorithms=list(score_dict.keys()),
            ax=ax,
        )

        ax.set_title(title, fontsize=self.title_fontsize)
        ax.set_xlabel("Step", fontsize=self.axis_fontsize)
        ax.set_ylabel("Mean Return (IQM)", fontsize=self.axis_fontsize)
        ax.tick_params(axis="both", labelsize=self.tick_fontsize)
        if ax.get_legend():
            plt.setp(ax.get_legend().get_texts(), fontsize=self.legend_fontsize)

        plt.tight_layout()
        if output_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            plt.savefig(output_path, dpi=300)
            print(f"Plot saved to {output_path}")

        return ax

    def plot_aggregate_metrics(
        self,
        score_dict: Dict[str, np.ndarray],
        metric_funcs: Optional[Dict[str, Callable]] = None,
        title: str = "Aggregate Performance",
        output_path: Optional[str] = None,
    ) -> plt.Axes:
        """
        Plots aggregate metrics (IQM, Mean, Median) with bootstrap CIs using RLiable.

        Args:
            score_dict: Dictionary mapping algorithm to score array of shape (num_runs, num_tasks).
            metric_funcs: Optional dictionary mapping metric name to evaluation function.
            title: Title for the plot.
            output_path: Optional file path to save the plot image.

        Returns:
            matplotlib.axes.Axes object.
        """
        if metric_funcs is None:
            metric_funcs = {
                "IQM": metrics.aggregate_iqm,
                "Mean": metrics.aggregate_mean,
                "Median": metrics.aggregate_median,
            }

        metric_names = list(metric_funcs.keys())
        funcs_list = list(metric_funcs.values())
        agg_func = lambda x: np.array([fn(x) for fn in funcs_list])

        point_estimates, interval_estimates = library.get_interval_estimates(
            score_dict,
            agg_func,
            reps=self.reps,
            confidence_interval_size=self.confidence_interval_size,
        )

        fig, ax = plt.subplots(figsize=(10, 6))
        plot_utils.plot_interval_estimates(
            point_estimates,
            interval_estimates,
            metric_names=metric_names,
            algorithms=list(score_dict.keys()),
            ax=ax,
        )

        ax.set_title(title, fontsize=self.title_fontsize)
        ax.tick_params(axis="both", labelsize=self.tick_fontsize)
        plt.tight_layout()

        if output_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            plt.savefig(output_path, dpi=300)
            print(f"Aggregate metrics plot saved to {output_path}")

        return ax

    def plot_performance_profiles(
        self,
        score_dict: Dict[str, np.ndarray],
        tau_grid: Optional[np.ndarray] = None,
        title: str = "Performance Profiles",
        output_path: Optional[str] = None,
    ) -> plt.Axes:
        """
        Plots score distribution performance profiles with bootstrap CIs using RLiable.

        Args:
            score_dict: Dictionary mapping algorithm to score array of shape (num_runs, num_tasks).
            tau_grid: Optional 1D array of score thresholds.
            title: Title for the plot.
            output_path: Optional file path to save plot image.

        Returns:
            matplotlib.axes.Axes object.
        """
        if tau_grid is None:
            min_val = min(np.min(v) for v in score_dict.values())
            max_val = max(np.max(v) for v in score_dict.values())
            tau_grid = np.linspace(min_val, max_val, 101)

        profiles, profile_cis = library.create_performance_profile(
            score_dict,
            tau_grid,
            reps=self.reps,
            confidence_interval_size=self.confidence_interval_size,
        )

        fig, ax = plt.subplots(figsize=(10, 6))
        plot_utils.plot_performance_profiles(
            profiles,
            tau_grid,
            performance_profile_cis=profile_cis,
            ax=ax,
        )

        ax.set_title(title, fontsize=self.title_fontsize)
        ax.tick_params(axis="both", labelsize=self.tick_fontsize)
        plt.tight_layout()

        if output_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            plt.savefig(output_path, dpi=300)
            print(f"Performance profiles plot saved to {output_path}")

        return ax

    def plot_probability_of_improvement(
        self,
        score_dict: Dict[str, np.ndarray],
        pairs: List[Tuple[str, str]],
        title: str = "Probability of Improvement",
        output_path: Optional[str] = None,
    ) -> plt.Axes:
        """
        Plots pairwise probability of improvement P(X > Y) with bootstrap CIs using RLiable.

        Args:
            score_dict: Dictionary mapping algorithm name to score array of shape (num_runs, num_tasks).
            pairs: List of algorithm name pairs (algo_a, algo_b) to evaluate P(algo_a > algo_b).
            title: Title for the plot.
            output_path: Optional file path to save plot image.

        Returns:
            matplotlib.axes.Axes object.
        """
        pair_dict = {
            f"{algo_a},{algo_b}": (score_dict[algo_a], score_dict[algo_b])
            for algo_a, algo_b in pairs
        }
        prob_func = lambda x, y: np.array([metrics.probability_of_improvement(x, y)])

        pair_point, pair_interval = library.get_interval_estimates(
            pair_dict,
            prob_func,
            reps=self.reps,
            confidence_interval_size=self.confidence_interval_size,
        )

        fig, ax = plt.subplots(figsize=(8, 5))
        plot_utils.plot_probability_of_improvement(
            pair_point,
            pair_interval,
            ax=ax,
        )

        ax.set_title(title, fontsize=self.title_fontsize)
        ax.tick_params(axis="both", labelsize=self.tick_fontsize)
        plt.tight_layout()

        if output_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            plt.savefig(output_path, dpi=300)
            print(f"Probability of improvement plot saved to {output_path}")

        return ax