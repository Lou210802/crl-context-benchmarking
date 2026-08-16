"""
Latent Context Probing and Visualization Module for Contextual RL Benchmarking.

This module provides tools to scientifically evaluate whether representation encoders (VAE, CPC)
have successfully captured and disentangled the environment's true physical context parameters:
  1. Linear Context Probing: Fits linear regression W * z_t + b -> c_true to measure R^2 score.
  2. Latent PCA Scatter Visualization: Projects 8D latent context space z_t to 2D PCA colored by physical context.
  3. Probing Prediction Scatter Plot: Plots True Context vs. Linearly Predicted Context (Option 4 insight plot).
"""

import os
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def probe_latent_context(
    latent_vectors: np.ndarray,
    true_contexts: np.ndarray,
    context_keys: Optional[List[str]] = None,
) -> Dict[str, Any]:
    num_samples, latent_dim = latent_vectors.shape
    if true_contexts.ndim == 1:
        true_contexts = true_contexts.reshape(-1, 1)
    num_samples, context_dim = true_contexts.shape

    # 1. Linear Probe
    z_augmented = np.hstack([latent_vectors, np.ones((num_samples, 1))])
    weights, _, _, _ = np.linalg.lstsq(z_augmented, true_contexts, rcond=None)
    pred_contexts = z_augmented @ weights

    ss_res = np.sum((true_contexts - pred_contexts) ** 2, axis=0)
    ss_tot = np.sum((true_contexts - np.mean(true_contexts, axis=0)) ** 2, axis=0)
    varying_mask = ss_tot > 1e-5
    mean_r2 = float(np.mean(1.0 - (ss_res[varying_mask] / ss_tot[varying_mask]))) if np.any(varying_mask) else 1.0

    per_key_r2 = {}
    if context_keys:
        for idx, key in enumerate(context_keys):
            if idx < context_dim:
                if ss_tot[idx] > 1e-5:
                    per_key_r2[f"r2_{key}"] = float(1.0 - (ss_res[idx] / ss_tot[idx]))
                else:
                    per_key_r2[f"r2_{key}"] = 1.0

    # 2. Non-Linear Probing (Random Forest Probe)
    non_linear_r2 = mean_r2
    non_linear_pred_contexts = pred_contexts.copy()
    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import r2_score
        from sklearn.model_selection import train_test_split

        # Striker 80/20 Train-Test-Split gegen Overfitting
        X_train, X_test, y_train, y_test = train_test_split(
            latent_vectors, true_contexts, test_size=0.2, random_state=42
        )

        # Flachere Bäume (max_depth=5), damit er nicht Rauschen auswendig lernt
        rf = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train.ravel() if context_dim == 1 else y_train)

        # Echte Out-of-Sample Evaluation:
        rf_preds_full = rf.predict(latent_vectors)
        rf_preds_test = rf.predict(X_test)

        if context_dim == 1:
            non_linear_pred_contexts = rf_preds_full.reshape(-1, 1)
            non_linear_r2 = float(r2_score(y_test, rf_preds_test))
        else:
            non_linear_pred_contexts = rf_preds_full
            non_linear_r2 = float(r2_score(y_test, rf_preds_test))

        if context_keys:
            for idx, key in enumerate(context_keys):
                if idx < context_dim:
                    if ss_tot[idx] > 1e-5:
                        y_t_sub = y_test[:, idx] if context_dim > 1 else y_test
                        y_p_sub = rf_preds_test[:, idx] if context_dim > 1 else rf_preds_test
                        per_key_r2[f"non_linear_r2_{key}"] = float(r2_score(y_t_sub, y_p_sub))
                    else:
                        per_key_r2[f"non_linear_r2_{key}"] = 1.0
    except Exception as e:
        print(f"[WARNING] Non-linear probing failed: {e}")

    results = {
        "mean_r2": mean_r2,
        "non_linear_r2": non_linear_r2,
        "probing_mse": float(np.mean((true_contexts - pred_contexts) ** 2)),
        "non_linear_probing_mse": float(np.mean((true_contexts - non_linear_pred_contexts) ** 2)),
        "pred_contexts": pred_contexts,
        "non_linear_pred_contexts": non_linear_pred_contexts,
    }
    results.update(per_key_r2)
    return results


def visualize_latent_space_pca(
    latent_vectors: np.ndarray,
    color_values: np.ndarray,
    color_label: str = "Gravity",
    title: str = "Latent Context Representation PCA",
    output_path: str = "latent_space_pca.png",
) -> str:
    """
    Projects 8D latent context vectors z_t down to 2D using SVD-based PCA and generates a scatter plot
    colored by ground-truth physical context values.
    """
    output_path = os.path.abspath(output_path)

    # Center latent vectors
    z_mean = np.mean(latent_vectors, axis=0)
    z_centered = latent_vectors - z_mean

    # Compute SVD for 2D PCA projection
    _, _, vt = np.linalg.svd(z_centered, full_matrices=False)
    z_pca = z_centered @ vt[:2].T

    # Generate Seaborn scatter plot
    sns.set_theme(style="whitegrid", palette="muted")
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

    scatter = ax.scatter(
        z_pca[:, 0],
        z_pca[:, 1],
        c=color_values,
        cmap="viridis",
        alpha=0.85,
        edgecolor="k",
        linewidth=0.5,
        s=45,
    )
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label(color_label, fontsize=11, fontweight="bold")

    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("PCA Component 1", fontsize=11, fontweight="bold")
    ax.set_ylabel("PCA Component 2", fontsize=11, fontweight="bold")

    sns.despine(ax=ax, top=True, right=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)

    print(f"  [SUCCESS] Latent PCA Visualization saved to '{output_path}'")
    return output_path


def visualize_pca_analysis(
    latent_vectors: np.ndarray,
    color_values: np.ndarray,
    pred_color_values: Optional[np.ndarray] = None,
    color_label: str = "Context Value",
    title: str = "Latent Context Representation PCA Analysis",
    output_path: str = "pca_analysis.png",
) -> str:
    """
    Generates a detailed multi-panel PCA Analysis Plot for latent context representations z_t.
    Includes:
      1. Scree Plot / Cumulative & Individual Explained Variance Ratio across principal components.
      2. 2D PCA Scatter Plot colored by true physical context values.
      3. 2D PCA Scatter Plot colored by non-linear probing predicted context values.
    """
    output_path = os.path.abspath(output_path)

    z_mean = np.mean(latent_vectors, axis=0)
    z_centered = latent_vectors - z_mean

    _, s, vt = np.linalg.svd(z_centered, full_matrices=False)
    z_pca = z_centered @ vt[:2].T

    variance_explained = (s ** 2) / np.sum(s ** 2)
    cum_variance = np.cumsum(variance_explained)
    n_components = len(variance_explained)

    sns.set_theme(style="whitegrid", palette="muted")
    has_pred = pred_color_values is not None
    num_panels = 3 if has_pred else 2
    fig, axes = plt.subplots(1, num_panels, figsize=(5.5 * num_panels, 4.8), dpi=300)

    # Panel 1: Scree Plot & Cumulative Variance
    ax1 = axes[0]
    comps = np.arange(1, n_components + 1)
    ax1.bar(comps, variance_explained * 100, color="#1f77b4", alpha=0.7, label="Individual Var %")
    ax1.plot(comps, cum_variance * 100, color="#d62728", marker="o", linewidth=2.0, label="Cumulative Var %")
    ax1.set_title("PCA Scree Plot & Variance Explained", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Principal Component Index", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Variance Explained (%)", fontsize=10, fontweight="bold")
    ax1.set_ylim(0, 105)
    ax1.set_xticks(comps)
    ax1.legend(loc="center right", fontsize=9)
    sns.despine(ax=ax1, top=True, right=True)

    # Panel 2: PCA 2D Scatter (True Context)
    ax2 = axes[1]
    sc2 = ax2.scatter(
        z_pca[:, 0],
        z_pca[:, 1],
        c=color_values,
        cmap="viridis",
        alpha=0.85,
        edgecolor="k",
        linewidth=0.5,
        s=40,
    )
    cb2 = fig.colorbar(sc2, ax=ax2)
    cb2.set_label(f"True {color_label}", fontsize=9, fontweight="bold")
    pc1_pct = variance_explained[0] * 100
    pc2_pct = variance_explained[1] * 100 if n_components > 1 else 0.0
    ax2.set_title(f"2D PCA Projection (True {color_label})", fontsize=11, fontweight="bold")
    ax2.set_xlabel(f"PC1 ({pc1_pct:.1f}% var)", fontsize=10, fontweight="bold")
    ax2.set_ylabel(f"PC2 ({pc2_pct:.1f}% var)", fontsize=10, fontweight="bold")
    sns.despine(ax=ax2, top=True, right=True)

    # Panel 3: PCA 2D Scatter (Non-Linear Predicted Context)
    if has_pred:
        ax3 = axes[2]
        if pred_color_values.ndim > 1:
            pred_vals = pred_color_values[:, 0]
        else:
            pred_vals = pred_color_values
        sc3 = ax3.scatter(
            z_pca[:, 0],
            z_pca[:, 1],
            c=pred_vals,
            cmap="viridis",
            alpha=0.85,
            edgecolor="k",
            linewidth=0.5,
            s=40,
        )
        cb3 = fig.colorbar(sc3, ax=ax3)
        cb3.set_label(f"Non-Linear Pred {color_label}", fontsize=9, fontweight="bold")
        ax3.set_title(f"2D PCA Projection (Non-Linear Pred {color_label})", fontsize=11, fontweight="bold")
        ax3.set_xlabel(f"PC1 ({pc1_pct:.1f}% var)", fontsize=10, fontweight="bold")
        ax3.set_ylabel(f"PC2 ({pc2_pct:.1f}% var)", fontsize=10, fontweight="bold")
        sns.despine(ax=ax3, top=True, right=True)

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"  [SUCCESS] PCA Analysis Plot saved to '{output_path}'")
    return output_path




def visualize_latent_space_pca(
    latent_vectors: np.ndarray,
    color_values: np.ndarray,
    color_label: str = "Gravity",
    title: str = "Latent Context Representation PCA",
    output_path: str = "latent_space_pca.png",
) -> str:
    """
    Projects 8D latent context vectors z_t down to 2D using SVD-based PCA and generates a scatter plot
    colored by ground-truth physical context values.
    """
    output_path = os.path.abspath(output_path)

    # Center latent vectors
    z_mean = np.mean(latent_vectors, axis=0)
    z_centered = latent_vectors - z_mean

    # Compute SVD for 2D PCA projection
    _, _, vt = np.linalg.svd(z_centered, full_matrices=False)
    z_pca = z_centered @ vt[:2].T

    # Generate Seaborn scatter plot
    sns.set_theme(style="whitegrid", palette="muted")
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

    scatter = ax.scatter(
        z_pca[:, 0],
        z_pca[:, 1],
        c=color_values,
        cmap="viridis",
        alpha=0.85,
        edgecolor="k",
        linewidth=0.5,
        s=45,
    )
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label(color_label, fontsize=11, fontweight="bold")

    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("PCA Component 1", fontsize=11, fontweight="bold")
    ax.set_ylabel("PCA Component 2", fontsize=11, fontweight="bold")

    sns.despine(ax=ax, top=True, right=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)

    print(f"  [SUCCESS] Latent PCA Visualization saved to '{output_path}'")
    return output_path


def visualize_probing_scatter(
    true_contexts: np.ndarray,
    pred_contexts: np.ndarray,
    r2_score: float,
    mode: str = "CPC",
    probe_type: str = "Linear",
    context_label: str = "Physical Parameter",
    output_path: str = "probing_scatter.png",
) -> str:
    """
    Generates a True Context vs. Predicted Context Scatter Plot.
    Compares predicted context parameters from probing to ground truth with an ideal y = x line.
    """
    output_path = os.path.abspath(output_path)

    if true_contexts.ndim > 1:
        y_true = true_contexts[:, 0]
        y_pred = pred_contexts[:, 0]
    else:
        y_true = true_contexts
        y_pred = pred_contexts

    sns.set_theme(style="whitegrid", palette="muted")
    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)

    ax.scatter(y_pred, y_true, alpha=0.7, color="#1f77b4" if mode.lower()=="cpc" else "#e377c2", edgecolor="k", linewidth=0.5, s=40)

    # Plot ideal identity diagonal y = x
    min_val = min(float(np.min(y_true)), float(np.min(y_pred)))
    max_val = max(float(np.max(y_true)), float(np.max(y_pred)))
    ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2.0, label="Ideal Recovery ($y = x$)")

    ax.set_title(f"{probe_type} Context Probing ({mode.upper()}) | R² = {r2_score:.3f}", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel(f"{probe_type} Predicted Context (z_t → ĉ)", fontsize=11, fontweight="bold")
    ax.set_ylabel(f"True Context ({context_label})", fontsize=11, fontweight="bold")
    ax.legend(loc="upper left", fontsize=10)

    sns.despine(ax=ax, top=True, right=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)

    print(f"  [SUCCESS] {probe_type} Probing Scatter Plot saved to '{output_path}'")
    return output_path

