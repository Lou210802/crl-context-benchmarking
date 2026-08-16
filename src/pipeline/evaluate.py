"""
Evaluation Module for Contextual Reinforcement Learning Models.

Loads trained model checkpoints (agent.pt, vae_encoder.pt) and evaluates performance
on held-out, unseen physical context instances (eval_context_seed = 9999).
Includes linear context probing (R^2 determination metric) and 2D PCA latent visualization.
"""

import glob
import os
import sys
import yaml
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import torch
import gymnasium as gym
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.agents.base_ppo import BasePPOAgent
from src.models.vae_encoder import VAEContextEncoder
from src.models.cpc_encoder import CPCContextEncoder
from src.pipeline.probing import (
    probe_latent_context,
    visualize_latent_space_pca,
    visualize_pca_analysis,
    visualize_probing_scatter,
)
from src.utils.env_utils import (
    make_env,
    extract_input,
    extract_raw_state,
    resolve_path,
)
from src.utils.history_buffer import HistoryBuffer


def plot_run_training_curves(run_dir: str) -> Optional[str]:
    """
    Plots training metrics curves (Mean Return, Policy Loss, Value Loss, Representation Loss)
    from results CSV file found inside run_dir.
    Saves 'training_curves.png' directly inside run_dir.
    """
    csv_files = glob.glob(os.path.join(run_dir, "*_results.csv"))
    if not csv_files:
        return None

    csv_path = csv_files[0]
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None

    if "step" not in df.columns:
        return None

    steps = df["step"].values
    metric_cols = [c for c in df.columns if c.lower() not in ["step", "unnamed: 0", "index"]]
    if not metric_cols:
        return None

    num_metrics = len(metric_cols)
    cols_per_row = 2
    rows = int(np.ceil(num_metrics / cols_per_row))

    sns.set_theme(style="whitegrid", palette="muted")
    fig, axes = plt.subplots(rows, cols_per_row, figsize=(11, 3.8 * rows), dpi=300)
    if num_metrics == 1:
        axes_list = [axes]
    else:
        axes_list = axes.flatten()

    for idx, col in enumerate(metric_cols):
        ax = axes_list[idx]
        color = "#1f77b4" if "return" in col.lower() else "#ff7f0e" if "loss" in col.lower() else "#2ca02c"
        ax.plot(steps, df[col].values, linewidth=2.0, color=color)
        ax.set_title(col.replace("_", " ").title(), fontsize=12, fontweight="bold")
        ax.set_xlabel("Environment Step", fontsize=10, fontweight="bold")
        ax.set_ylabel("Value", fontsize=10, fontweight="bold")
        sns.despine(ax=ax, top=True, right=True)

    for idx in range(num_metrics, len(axes_list)):
        fig.delaxes(axes_list[idx])

    plt.tight_layout()
    output_path = os.path.join(run_dir, "training_curves.png")
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"  [SUCCESS] Training Curves Plot saved to '{output_path}'")
    return output_path


def evaluate_run_directory(
    run_dir: str,
    eval_episodes: int = 20,
    eval_context_seed: int = 9999,
    num_eval_contexts: int = 20,
    vary_contexts: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Evaluates a trained model checkpoint on held-out physical context instances.
    Performs linear context probing (R^2 metrics), 2D PCA visualization, and training curves plotting.
    """
    run_dir = resolve_path(run_dir)
    config_path = os.path.join(run_dir, "config.yaml")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found in '{run_dir}'. Cannot evaluate.")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    env_name = config["env_name"]
    mode = config.get("mode", config.get("method", "context_free"))
    input_dim = config["input_dim"]
    action_dim = config["action_dim"]
    is_continuous = config["is_continuous"]
    raw_state_dim = config["raw_state_dim"]
    latent_dim = config.get("latent_dim", 8)
    hidden_dim = config.get("hidden_dim", 64)
    predict_horizon = config.get("predict_horizon", 3)
    cpc_temperature = config.get("cpc_temperature", 0.1)
    max_history_len = config.get("max_history_len", 25)
    if vary_contexts is None:
        vary_contexts = config.get("vary_contexts", None)

    device = torch.device("cpu")

    # 1. Instantiate Agent and load weights
    agent = BasePPOAgent(input_dim=input_dim, action_dim=action_dim, is_continuous=is_continuous, device=device)
    agent_ckpt = os.path.join(run_dir, "agent.pt")
    if os.path.exists(agent_ckpt):
        agent.load_checkpoint(agent_ckpt)

    # 2. Instantiate and load encoder if representation mode
    vae_encoder: Optional[VAEContextEncoder] = None
    cpc_encoder: Optional[CPCContextEncoder] = None
    history_buf: Optional[HistoryBuffer] = None

    if mode not in ["context_free", "oracle"]:
        history_buf = HistoryBuffer(
            max_history_len=max_history_len,
            state_dim=raw_state_dim,
            action_dim=action_dim,
            is_continuous=is_continuous,
        )
        if mode == "vae":
            vae_encoder = VAEContextEncoder(
                state_dim=raw_state_dim,
                action_dim=action_dim,
                latent_dim=latent_dim,
                hidden_dim=hidden_dim,
                device=device,
            )
            vae_ckpt = os.path.join(run_dir, "vae_encoder.pt")
            if os.path.exists(vae_ckpt):
                vae_encoder.load_state_dict(torch.load(vae_ckpt, map_location=device))
            vae_encoder.eval()
        elif mode == "cpc":
            cpc_encoder = CPCContextEncoder(
                state_dim=raw_state_dim,
                action_dim=action_dim,
                latent_dim=latent_dim,
                hidden_dim=hidden_dim,
                predict_horizon=predict_horizon,
                temperature=cpc_temperature,
                device=device,
            )
            cpc_ckpt = os.path.join(run_dir, "cpc_encoder.pt")
            if os.path.exists(cpc_ckpt):
                try:
                    cpc_encoder.load_state_dict(torch.load(cpc_ckpt, map_location=device), strict=False)
                except Exception as e:
                    print(f"[WARNING] Skipping legacy incompatible CPC checkpoint in '{run_dir}': {e}")
            cpc_encoder.eval()

    # 3. Instantiate evaluation environment on held-out context seed
    eval_env = make_env(
        env_name=env_name,
        num_contexts=num_eval_contexts,
        context_seed=eval_context_seed,
        vary_contexts=vary_contexts,
    )

    episode_returns: List[float] = []
    episode_lengths: List[int] = []

    # Buffers to store collected latent context vectors z_t and ground-truth contexts c_t
    collected_z_vectors: List[np.ndarray] = []
    collected_true_contexts: List[np.ndarray] = []
    context_keys: List[str] = []

    print(f"\n[EVAL] Evaluating '{mode.upper()}' on HELD-OUT contexts ({num_eval_contexts} contexts, seed={eval_context_seed})...")

    for ep in range(eval_episodes):
        obs, _ = eval_env.reset(seed=eval_context_seed + ep)
        if history_buf:
            history_buf.reset()

        ep_return = 0.0
        ep_len = 0
        done = False

        while not done:
            # Extract ground-truth physical context values for probing
            ctx_dict = None
            if isinstance(obs, dict) and "context" in obs:
                ctx_dict = obs["context"]
            elif hasattr(eval_env, "context"):
                ctx_dict = getattr(eval_env, "context")
            elif hasattr(eval_env, "unwrapped") and hasattr(eval_env.unwrapped, "context"):
                ctx_dict = getattr(eval_env.unwrapped, "context")

            if ctx_dict:
                if vary_contexts:
                    v_set = set(v.lower() for v in vary_contexts)
                    filtered_dict = {k: v for k, v in ctx_dict.items() if k.lower() in v_set}
                    if not filtered_dict:
                        filtered_dict = ctx_dict
                else:
                    filtered_dict = ctx_dict

                if not context_keys:
                    context_keys = list(filtered_dict.keys())
                true_c = np.array([filtered_dict[k] for k in context_keys], dtype=np.float32)
            else:
                true_c = np.array([0.0], dtype=np.float32)

            if mode in ["context_free", "oracle"]:
                policy_input = extract_input(obs, eval_env, mode)
            else:
                hist_tensor = history_buf.get_tensor(pad_to_max=True)
                with torch.no_grad():
                    if mode == "vae" and vae_encoder is not None:
                        mu_z, _ = vae_encoder.encode(hist_tensor)
                        z_t = mu_z
                    elif mode == "cpc" and cpc_encoder is not None:
                        z_t = cpc_encoder.encode(hist_tensor)
                    else:
                        z_t = torch.zeros((1, latent_dim), dtype=torch.float32)
                    z_np = z_t.squeeze(0).cpu().numpy()

                raw_state = extract_raw_state(obs)
                policy_input = np.concatenate([raw_state, z_np], axis=0)

                collected_z_vectors.append(z_np)
                collected_true_contexts.append(true_c)

            action, _, _, _ = agent.predict(policy_input)
            next_obs, reward, term, trunc, _ = eval_env.step(action)
            done = term or trunc

            if history_buf:
                raw_state = extract_raw_state(obs)
                history_buf.add(raw_state, action, float(reward))

            ep_return += float(reward)
            ep_len += 1
            obs = next_obs

        episode_returns.append(ep_return)
        episode_lengths.append(ep_len)

    eval_results: Dict[str, Any] = {
        "mode": mode,
        "env_name": env_name,
        "eval_context_seed": eval_context_seed,
        "num_eval_contexts": num_eval_contexts,
        "eval_episodes": eval_episodes,
        "eval_mean_return": float(np.mean(episode_returns)),
        "eval_std_return": float(np.std(episode_returns)),
        "eval_mean_length": float(np.mean(episode_lengths)),
    }

    # 4. Perform Linear Context Probing & Visualizations
    if collected_z_vectors and collected_true_contexts:
        z_arr = np.array(collected_z_vectors)
        c_arr = np.array(collected_true_contexts)

        # Linear Context Probing
        probing_results = probe_latent_context(z_arr, c_arr, context_keys=context_keys)
        eval_results["probing_r2"] = probing_results["mean_r2"]
        eval_results["non_linear_probing_r2"] = probing_results["non_linear_r2"]
        eval_results["probing_mse"] = probing_results["probing_mse"]

        # PCA Visualization (2D Scatter)
        pca_plot_path = os.path.join(run_dir, "latent_space_pca.png")
        primary_color_vals = c_arr[:, 0]
        color_label = context_keys[0].capitalize() if context_keys else "Context Value"
        visualize_latent_space_pca(
            latent_vectors=z_arr,
            color_values=primary_color_vals,
            color_label=color_label,
            title=f"Latent Context Space ({mode.upper()}) PCA on {env_name}",
            output_path=pca_plot_path,
        )

        # Multi-panel PCA Analysis Plot (Scree Plot & Probing Projections)
        pca_analysis_path = os.path.join(run_dir, "pca_analysis.png")
        visualize_pca_analysis(
            latent_vectors=z_arr,
            color_values=primary_color_vals,
            pred_color_values=probing_results.get("non_linear_pred_contexts"),
            color_label=color_label,
            title=f"Latent Context PCA Analysis ({mode.upper()}) on {env_name}",
            output_path=pca_analysis_path,
        )

        # Probing Prediction Scatter Plot (Linear)
        prob_plot_path = os.path.join(run_dir, "probing_scatter.png")
        visualize_probing_scatter(
            true_contexts=c_arr,
            pred_contexts=probing_results["pred_contexts"],
            r2_score=probing_results["mean_r2"],
            mode=mode,
            probe_type="Linear",
            context_label=color_label,
            output_path=prob_plot_path,
        )

        # Probing Prediction Scatter Plot (Non-Linear)
        non_linear_prob_plot_path = os.path.join(run_dir, "non_linear_probing_scatter.png")
        visualize_probing_scatter(
            true_contexts=c_arr,
            pred_contexts=probing_results["non_linear_pred_contexts"],
            r2_score=probing_results["non_linear_r2"],
            mode=mode,
            probe_type="Non-Linear",
            context_label=color_label,
            output_path=non_linear_prob_plot_path,
        )

    # 5. Automatically plot training curves image inside run_dir
    plot_run_training_curves(run_dir)

    # Save evaluation summary to YAML
    eval_yaml_path = os.path.join(run_dir, "evaluation_summary.yaml")
    with open(eval_yaml_path, "w") as f:
        yaml.dump(eval_results, f, default_flow_style=False)

    prob_str = ""
    if "probing_r2" in eval_results:
        nl_r2 = eval_results.get("non_linear_probing_r2", eval_results["probing_r2"])
        prob_str = f" | Linear R^2: {eval_results['probing_r2']:5.3f} | Non-Linear R^2: {nl_r2:5.3f}"
    print(
        f"[EVAL RESULT] {mode.upper():12s} | "
        f"Mean Return (Unseen Contexts): {eval_results['eval_mean_return']:6.1f} ± {eval_results['eval_std_return']:4.1f}"
        f"{prob_str}"
    )
    return eval_results



def plot_evaluation_comparison_bar_chart(
    eval_summaries: List[Dict[str, Any]],
    output_path: str = "results/eval_returns_comparison.png",
) -> str:
    """
    Generates comparative bar chart plots comparing held-out evaluation returns AND linear probing R^2 scores across algorithms.

    Args:
        eval_summaries: List of evaluation summary result dictionaries.
        output_path: Output image path.

    Returns:
        str: Absolute file path to saved plot image.
    """
    output_path = resolve_path(output_path)

    # Group mean and std returns by mode
    grouped_data: Dict[str, List[float]] = {}
    r2_data: Dict[str, List[float]] = {}
    nl_r2_data: Dict[str, List[float]] = {}

    for res in eval_summaries:
        m = res["mode"].lower()
        if m not in grouped_data:
            grouped_data[m] = []
        grouped_data[m].append(res["eval_mean_return"])
        if "probing_r2" in res:
            if m not in r2_data:
                r2_data[m] = []
            r2_data[m].append(res["probing_r2"])
        if "non_linear_probing_r2" in res:
            if m not in nl_r2_data:
                nl_r2_data[m] = []
            nl_r2_data[m].append(res["non_linear_probing_r2"])

    modes = list(grouped_data.keys())
    means = [np.mean(grouped_data[m]) for m in modes]
    stds = [np.std(grouped_data[m]) if len(grouped_data[m]) > 1 else 0.0 for m in modes]

    sns.set_theme(style="whitegrid", palette="muted")
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)

    bars = ax.bar(
        [m.upper() for m in modes],
        means,
        yerr=stds,
        capsize=6,
        color=sns.color_palette("muted", len(modes)),
        edgecolor="black",
        linewidth=1.2,
        alpha=0.85,
    )

    ax.set_title("Held-Out Evaluation Return Comparison (Unseen Contexts)", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Mean Evaluation Return", fontsize=11, fontweight="bold")
    ax.set_xlabel("Algorithm", fontsize=11, fontweight="bold")

    # Add numeric value labels above positive bars / below negative bars
    for bar, mean_val, std_val in zip(bars, means, stds):
        height = bar.get_height()
        if height < 0:
            y_pos = height - std_val
            xy_text = (0, -12)
            va_align = "top"
        else:
            y_pos = height + std_val
            xy_text = (0, 4)
            va_align = "bottom"

        ax.annotate(
            f"{mean_val:.1f}",
            xy=(bar.get_x() + bar.get_width() / 2, y_pos),
            xytext=xy_text,
            textcoords="offset points",
            ha="center",
            va=va_align,
            fontsize=10,
            fontweight="bold",
        )

    sns.despine(ax=ax, top=True, right=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)

    # 2. Also generate Probing R^2 Comparison Bar Chart (Linear vs Non-Linear) if representation modes are present
    if r2_data:
        r2_modes = list(r2_data.keys())
        r2_lin_means = [np.mean(r2_data[m]) for m in r2_modes]
        r2_lin_stds = [np.std(r2_data[m]) if len(r2_data[m]) > 1 else 0.0 for m in r2_modes]

        has_nl = bool(nl_r2_data)
        if has_nl:
            r2_nl_means = [np.mean(nl_r2_data.get(m, r2_data[m])) for m in r2_modes]
            r2_nl_stds = [np.std(nl_r2_data[m]) if m in nl_r2_data and len(nl_r2_data[m]) > 1 else 0.0 for m in r2_modes]

        r2_output_path = os.path.join(os.path.dirname(output_path), "probing_r2_comparison.png")
        fig2, ax2 = plt.subplots(figsize=(9, 5.5), dpi=300)

        x = np.arange(len(r2_modes))
        width = 0.35 if has_nl else 0.5

        if has_nl:
            rects1 = ax2.bar(
                x - width / 2,
                r2_lin_means,
                width,
                yerr=r2_lin_stds,
                capsize=5,
                label="Linear Probe R²",
                color="#1f77b4",
                edgecolor="black",
                alpha=0.85,
            )
            rects2 = ax2.bar(
                x + width / 2,
                r2_nl_means,
                width,
                yerr=r2_nl_stds,
                capsize=5,
                label="Non-Linear Probe R² (RF)",
                color="#ff7f0e",
                edgecolor="black",
                alpha=0.85,
            )

            for rect in rects1:
                height = rect.get_height()
                ax2.annotate(
                    f"{height:.3f}",
                    xy=(rect.get_x() + rect.get_width() / 2, height + 0.02),
                    xytext=(0, 2),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="bold",
                )
            for rect in rects2:
                height = rect.get_height()
                ax2.annotate(
                    f"{height:.3f}",
                    xy=(rect.get_x() + rect.get_width() / 2, height + 0.02),
                    xytext=(0, 2),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="bold",
                )

            ax2.set_xticks(x)
            ax2.set_xticklabels([m.upper() for m in r2_modes], fontsize=11, fontweight="bold")
            ax2.legend(loc="upper left", fontsize=10, frameon=True)
        else:
            bars2 = ax2.bar(
                [m.upper() for m in r2_modes],
                r2_lin_means,
                yerr=r2_lin_stds,
                capsize=5,
                color="#1f77b4",
                edgecolor="black",
                alpha=0.85,
            )
            for rect in bars2:
                height = rect.get_height()
                ax2.annotate(
                    f"{height:.3f}",
                    xy=(rect.get_x() + rect.get_width() / 2, height + 0.02),
                    xytext=(0, 2),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="bold",
                )

        ax2.set_title("Context Probing R² Score Comparison (Linear vs Non-Linear)", fontsize=13, fontweight="bold", pad=12)
        ax2.set_ylabel("Probing R² Score (Higher is Better)", fontsize=11, fontweight="bold")
        ax2.set_xlabel("Representation Algorithm", fontsize=11, fontweight="bold")
        ax2.set_ylim(0.0, 1.12)

        sns.despine(ax=ax2, top=True, right=True)
        plt.tight_layout()
        plt.savefig(r2_output_path, dpi=300)
        plt.close(fig2)
        print(f"[SUCCESS] Probing R^2 comparison plot saved to '{r2_output_path}'")

    return output_path
