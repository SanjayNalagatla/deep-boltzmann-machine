"""
visualize.py — Visualisation utilities for the DBM experiment
--------------------------------------------------------------
Produces rich plots for:
  • Dataset samples
  • Layer-wise reconstruction errors
  • Learned weight filters (receptive fields)
  • t-SNE plots of hierarchical representations
  • Reconstruction quality
  • Generated (sampled) images
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator

matplotlib.rcParams.update({
    "figure.facecolor": "#0e0e14",
    "axes.facecolor":   "#16161f",
    "axes.edgecolor":   "#3a3a55",
    "axes.labelcolor":  "#c8c8e8",
    "text.color":       "#c8c8e8",
    "xtick.color":      "#8888aa",
    "ytick.color":      "#8888aa",
    "grid.color":       "#2a2a3a",
    "grid.linestyle":   "--",
    "grid.alpha":       0.5,
    "lines.linewidth":  2.0,
    "font.family":      "sans-serif",
    "font.size":        10,
})

# Colour palette
LAYER_COLORS = ["#7c6af7", "#f7716a", "#59d6c0", "#f7c659", "#a6e87c"]


# ─────────────────────────────────────────────────────────────────────────────

def _imshow_grid(images, n_cols, ax, img_shape, cmap="inferno", title=""):
    """Tile images into a grid on the given axis."""
    n = len(images)
    n_rows = (n + n_cols - 1) // n_cols
    grid = np.zeros((n_rows * img_shape[0], n_cols * img_shape[1]))
    for i, img in enumerate(images):
        r, c = divmod(i, n_cols)
        patch = img.reshape(img_shape)
        patch = (patch - patch.min()) / (patch.max() - patch.min() + 1e-8)
        grid[r * img_shape[0]:(r + 1) * img_shape[0],
             c * img_shape[1]:(c + 1) * img_shape[1]] = patch
    ax.imshow(grid, cmap=cmap, aspect="auto", interpolation="nearest")
    ax.set_title(title, fontsize=11, pad=6, fontweight="bold")
    ax.axis("off")


# ─────────────────────────────────────────────────────────────────────────────

def plot_dataset_samples(X, y, img_shape, n_samples=60,
                         n_cols=12, save_path="fig_dataset_samples.png"):
    """Plot a grid of raw dataset samples."""
    fig, ax = plt.subplots(figsize=(14, 6))
    fig.suptitle("Dataset Samples", fontsize=14, fontweight="bold", y=1.01)
    _imshow_grid(X[:n_samples], n_cols, ax, img_shape, cmap="viridis",
                 title=f"{n_samples} raw samples from the dataset")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved → {save_path}")


def plot_training_curves(rbms, ft_errors, save_path="fig_training_curves.png"):
    """
    Plot reconstruction error curves for each RBM pre-training phase and
    the DBM fine-tuning phase.
    """
    n_rbms = len(rbms)
    n_cols = n_rbms + 1
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 4))
    fig.suptitle("Training Loss Curves", fontsize=14, fontweight="bold")

    for i, rbm in enumerate(rbms):
        ax = axes[i]
        errors = rbm.reconstruction_errors
        ax.plot(errors, color=LAYER_COLORS[i % len(LAYER_COLORS)], alpha=0.8)
        ax.set_title(f"RBM Layer {i + 1}\n"
                     f"({rbm.n_visible}→{rbm.n_hidden})", fontsize=10)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("MSE" if i == 0 else "")
        ax.grid(True)
        ax.xaxis.set_major_locator(MaxNLocator(5))

    # Fine-tuning
    ax = axes[-1]
    ax.plot(ft_errors, color="#f7c659", marker="o", markersize=4, alpha=0.85)
    ax.set_title("DBM Fine-tuning", fontsize=10)
    ax.set_xlabel("Epoch")
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved → {save_path}")


def plot_weight_filters(rbms, img_shape, n_filters=64,
                        save_path="fig_weight_filters.png"):
    """
    Visualise the learned weight filters for each RBM layer.
    Layer 1 filters are shown in pixel space.
    Layer 2 filters are compositions and shown as weighted combinations.
    """
    n_layers = len(rbms)
    fig, axes = plt.subplots(1, n_layers, figsize=(7 * n_layers, 7))
    if n_layers == 1:
        axes = [axes]

    fig.suptitle("Learned Weight Filters per Layer",
                 fontsize=14, fontweight="bold")

    for i, (rbm, ax) in enumerate(zip(rbms, axes)):
        # For layer 1: each column of W is a filter in pixel space
        # For deeper layers: project back to pixel space via chained weights
        W = rbm.W.T   # (n_hidden, n_visible)
        if i > 0:
            # Project to pixel space through earlier layers
            W_proj = W
            for j in range(i - 1, -1, -1):
                W_proj = W_proj @ rbms[j].W.T
            filters = W_proj[:n_filters]
        else:
            filters = W[:n_filters]

        _imshow_grid(filters, n_cols=8, ax=ax, img_shape=img_shape,
                     cmap="RdBu_r",
                     title=f"Layer {i + 1} Filters\n"
                           f"({rbm.n_visible} → {rbm.n_hidden})")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved → {save_path}")


def plot_reconstructions(X_orig, X_recon, img_shape, n_samples=10,
                         save_path="fig_reconstructions.png"):
    """Side-by-side original vs DBM reconstruction."""
    n = min(n_samples, len(X_orig))
    fig, axes = plt.subplots(2, n, figsize=(n * 1.5, 3.5))
    fig.suptitle("Original  vs  DBM Reconstruction",
                 fontsize=13, fontweight="bold")

    cmaps = ["viridis", "inferno"]
    row_labels = ["Original", "Reconstruction"]

    for row in range(2):
        imgs = X_orig if row == 0 else X_recon
        ax_row = axes[row]
        if row == 0:
            ax_row[0].set_ylabel(row_labels[row], fontsize=9, color="#c8c8e8")
        else:
            ax_row[0].set_ylabel(row_labels[row], fontsize=9, color="#f7716a")

        for col in range(n):
            ax = ax_row[col]
            img = imgs[col].reshape(img_shape)
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            ax.imshow(img, cmap=cmaps[row], aspect="auto",
                      interpolation="nearest")
            ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved → {save_path}")


def plot_tsne_representations(representations, y, layer_names=None,
                              save_path="fig_tsne_representations.png"):
    """t-SNE 2-D plots of features at each hidden layer."""
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        print("  ⚠ scikit-learn not found — skipping t-SNE plot.")
        return

    n_layers = len(representations)
    if layer_names is None:
        layer_names = [f"Hidden Layer {i + 1}" for i in range(n_layers)]

    fig, axes = plt.subplots(1, n_layers, figsize=(6 * n_layers, 5))
    if n_layers == 1:
        axes = [axes]

    fig.suptitle("t-SNE Visualisation of Hierarchical Representations",
                 fontsize=14, fontweight="bold")

    n_classes = len(np.unique(y))
    import matplotlib
    cmap = matplotlib.colormaps["tab10"].resampled(n_classes)

    for i, (rep, ax) in enumerate(zip(representations, axes)):
        print(f"  Running t-SNE for layer {i + 1}…")
        perp = min(30, rep.shape[0] // 4)
        tsne = TSNE(n_components=2, perplexity=perp,
                    random_state=42, max_iter=1000, learning_rate="auto",
                    init="random")
        emb = tsne.fit_transform(rep)

        for cls in range(n_classes):
            mask = y == cls
            ax.scatter(emb[mask, 0], emb[mask, 1],
                       c=[cmap(cls)], label=f"Class {cls}",
                       s=18, alpha=0.75, edgecolors="none")

        ax.set_title(layer_names[i], fontsize=11, fontweight="bold")
        ax.set_xlabel("t-SNE dim 1")
        ax.set_ylabel("t-SNE dim 2" if i == 0 else "")
        ax.grid(True)
        ax.legend(fontsize=7, markerscale=1.5,
                  framealpha=0.3, edgecolor="#3a3a55")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved → {save_path}")


def plot_generated_samples(samples, img_shape, n_cols=8,
                           save_path="fig_generated_samples.png"):
    """Plot samples generated by running the model's Gibbs chains."""
    fig, ax = plt.subplots(figsize=(12, 4))
    fig.suptitle("DBM Generated Samples  (Ancestral Gibbs Sampling)",
                 fontsize=13, fontweight="bold")
    _imshow_grid(samples, n_cols=n_cols, ax=ax, img_shape=img_shape,
                 cmap="hot", title="")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved → {save_path}")


def plot_hidden_activation_heatmap(representations, layer_names=None,
                                   n_samples=50,
                                   save_path="fig_activation_heatmap.png"):
    """
    Show a heatmap of hidden unit activations (mean across samples)
    for the first n_samples data points.
    """
    n_layers = len(representations)
    if layer_names is None:
        layer_names = [f"Layer {i + 1}" for i in range(n_layers)]

    fig, axes = plt.subplots(1, n_layers, figsize=(6 * n_layers, 4))
    if n_layers == 1:
        axes = [axes]

    fig.suptitle("Hidden Unit Activation Heatmap",
                 fontsize=14, fontweight="bold")

    for i, (rep, ax) in enumerate(zip(representations, axes)):
        data = rep[:n_samples]
        im = ax.imshow(data.T, aspect="auto", cmap="plasma",
                       interpolation="nearest", vmin=0, vmax=1)
        ax.set_title(f"{layer_names[i]}\n({rep.shape[1]} units)", fontsize=10)
        ax.set_xlabel(f"Sample index (first {n_samples})")
        ax.set_ylabel("Hidden unit" if i == 0 else "")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                     label="Activation")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved → {save_path}")
