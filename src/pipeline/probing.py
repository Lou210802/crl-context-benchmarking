"""
Latent Context Probing and Visualization Module for Contextual RL Benchmarking.

This module provides tools to scientifically evaluate whether representation encoders (VAE, CPC)
have successfully captured and disentangled the environment's true physical context parameters:
  1. Linear Context Probing: Fits linear regression W * z_t + b -> c_true to measure R^2 score.
  2. Latent PCA Scatter Visualization: Projects 8D latent context space z_t to 2D PCA colored by physical context.
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
    """
    Fits a linear regression model mapping latent context vectors z_t to true physical context parameters c_t.

    Args:
        latent_vectors: Array of shape (num_samples, latent_dim).
        true_contexts: Array of shape (num_samples, context_dim).
        context_keys: Optional names of physical context parameters (e.g. ['gravity', 'mass']).

    Returns:
        Dict[str, Any]: Dictionary containing overall mean R^2 score, probing MSE, and per-parameter R^2 scores.
    """
    num_samples, latent_dim = latent_vectors.shape
    if true_contexts.ndim == 1:
        true_contexts = true_contexts.reshape(-1, 1)
    num_samples, context_dim = true_contexts.shape

    # Augment latent vectors with bias column: [z, 1]
    z_augmented = np.hstack([latent_vectors, np.ones((num_samples, 1))])

    # Fit linear least-squares regression weights
    weights, _, _, _ = np.linalg.lstsq(z_augmented, true_contexts, rcond=None)
    pred_contexts = z_augmented @ weights

    # Compute Probing MSE
    probing_mse = float(np.mean((true_contexts - pred_contexts) ** 2))

    # Compute R^2 determination score per context dimension
    ss_res = np.sum((true_contexts - pred_contexts) ** 2, axis=0)
    ss_tot = np.sum((true_contexts - np.mean(true_contexts, axis=0)) ** 2, axis=0)
    r2_per_dim = 1.0 - (ss_res / (ss_tot + 1e-8))

    mean_r2 = float(np.mean(r2_per_dim))

    results = {
        "mean_r2": mean_r2,
        "probing_mse": probing_mse,
        "r2_per_dim": r2_per_dim.tolist(),
    }

    if context_keys and len(context_keys) == context_dim:
        for idx, key in enumerate(context_keys):
            results[f"r2_{key}"] = float(r2_per_dim[idx])

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

    Args:
        latent_vectors: Array of shape (num_samples, latent_dim).
        color_values: Continuous values used to color scatter points (e.g. gravity or mass).
        color_label: Label for colorbar legend.
        title: Title for the generated plot.
        output_path: Output file path for the plot image.

    Returns:
        str: Absolute output file path of the saved scatter plot image.
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
