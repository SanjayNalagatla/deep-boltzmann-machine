"""
train.py — End-to-end DBM experiment on a small dataset
---------------------------------------------------------
Dataset : MNIST (digits 0–4, 2 000 samples) — automatically downloaded.
          Falls back to a synthetic bars-and-stripes dataset if MNIST
          is unavailable.

Workflow:
  1. Load & normalise data
  2. Greedy layer-wise pre-training (3 hidden layers)
  3. Global mean-field + PCD fine-tuning
  4. Evaluation : reconstruction MSE
  5. Visualisations (saved as PNG files in the working directory)
"""

import os
import sys
import time
import numpy as np

# Ensure UTF-8 output on all platforms without charmap encoding errors
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# ─── optional progress bar ────────────────────────────────────────────────────
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

from dbm import DeepBoltzmannMachine
import visualize as viz


# ─────────────────────────────────────────────────────────────────────────────
#  Dataset helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_mnist_subset(n_samples: int = 2000, n_classes: int = 5):
    """
    Download MNIST via torchvision or sklearn and return a small subset.
    Falls back to synthetic data if neither is available.
    """
    img_shape = (28, 28)
    # 1. Try torchvision first (cached locally)
    try:
        from torchvision import datasets
        print("  Loading MNIST via torchvision ...")
        mnist_ds = datasets.MNIST(root="./data", train=True, download=True)
        X_raw = mnist_ds.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
        y_raw = mnist_ds.targets.numpy()
        mask = y_raw < n_classes
        X_all, y_all = X_raw[mask], y_raw[mask]
        idx = np.random.RandomState(42).choice(len(X_all), min(n_samples, len(X_all)), replace=False)
        return X_all[idx], y_all[idx], img_shape, "MNIST"
    except Exception as e_tv:
        pass

    # 2. Try scikit-learn
    try:
        from sklearn.datasets import fetch_openml
        print("  Loading MNIST via scikit-learn ...")
        mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
        X_all = mnist.data.astype(np.float32) / 255.0
        y_all = mnist.target.astype(int)
        mask = y_all < n_classes
        X_all, y_all = X_all[mask], y_all[mask]
        idx = np.random.RandomState(42).choice(len(X_all), n_samples, replace=False)
        return X_all[idx], y_all[idx], img_shape, "MNIST"
    except Exception as e_sk:
        print(f"  [i] Note: using synthetic Bars & Stripes dataset.")
        return make_bars_stripes(n_samples)


def make_bars_stripes(n_samples: int = 2000,
                      grid: int = 8,
                      random_state: int = 42):
    """
    Synthetic bars-and-stripes dataset (classic RBM benchmark).
    Each image is a grid x grid binary image containing either
    horizontal bars or vertical stripes.
    """
    rng = np.random.RandomState(random_state)
    imgs, labels = [], []
    for _ in range(n_samples):
        img = np.zeros((grid, grid), dtype=np.float32)
        label = rng.randint(0, 2)   # 0 = bars, 1 = stripes
        if label == 0:              # horizontal bars
            n_bars = rng.randint(1, grid // 2 + 1)
            rows = rng.choice(grid, n_bars, replace=False)
            img[rows, :] = 1.0
        else:                       # vertical stripes
            n_stripes = rng.randint(1, grid // 2 + 1)
            cols = rng.choice(grid, n_stripes, replace=False)
            img[:, cols] = 1.0
        imgs.append(img.ravel())
        labels.append(label)

    X = np.array(imgs, dtype=np.float32)
    y = np.array(labels, dtype=int)
    return X, y, (grid, grid), "Bars-and-Stripes"


def binarize(X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Convert continuous pixel values to binary."""
    return (X > threshold).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
#  Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(dbm: DeepBoltzmannMachine, X: np.ndarray):
    """Compute mean squared reconstruction error."""
    X_recon = dbm.reconstruct(X)
    mse = float(np.mean((X - X_recon) ** 2))
    return mse, X_recon


# ─────────────────────────────────────────────────────────────────────────────
#  Main experiment
# ─────────────────────────────────────────────────────────────────────────────

def main():
    np.random.seed(42)

    # ── Hyper-parameters (overridable via env vars from dashboard) ───────────
    N_SAMPLES       = int(os.environ.get("DBM_N_SAMPLES",       2000))
    N_CLASSES       = 5
    PRETRAIN_EPOCHS = int(os.environ.get("DBM_PRETRAIN_EPOCHS", 25))
    FINETUNE_EPOCHS = int(os.environ.get("DBM_FINETUNE_EPOCHS", 15))
    BATCH_SIZE      = int(os.environ.get("DBM_BATCH_SIZE",      64))
    LEARNING_RATE   = float(os.environ.get("DBM_LEARNING_RATE", 0.01))
    MOMENTUM        = 0.5
    WEIGHT_DECAY    = 1e-4
    K_CD            = 1        # CD-k steps in pre-training
    K_PCD           = 5        # Gibbs steps in PCD fine-tuning
    # Hidden layer size overrides (0 = auto)
    _H1             = int(os.environ.get("DBM_H1", 0))
    _H2             = int(os.environ.get("DBM_H2", 0))
    _H3             = int(os.environ.get("DBM_H3", 0))

    print("\n" + "=" * 60)
    print("       Deep Boltzmann Machine - Feature Learning Demo")
    print("=" * 60)

    # ── 1. Data ──────────────────────────────────────────────────────────────
    print("\n[1/5] Loading dataset …")
    X, y, img_shape, dataset_name = load_mnist_subset(N_SAMPLES, N_CLASSES)
    X = binarize(X)

    n_visible = X.shape[1]
    print(f"  Dataset   : {dataset_name}")
    print(f"  Samples   : {X.shape[0]}")
    print(f"  Visible   : {n_visible}  ({img_shape[0]}×{img_shape[1]})")
    print(f"  Classes   : {np.unique(y)}")

    # Determine hidden layer sizes relative to input (env override wins if != 0)
    h1 = _H1 if _H1 > 0 else max(64, n_visible // 4)
    h2 = _H2 if _H2 > 0 else max(32, n_visible // 8)
    h3 = _H3 if _H3 > 0 else max(16, n_visible // 16)
    LAYER_SIZES = [n_visible, h1, h2, h3]
    print(f"  Architecture: {' -> '.join(map(str, LAYER_SIZES))}")

    # Train / validation split
    split = int(0.9 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    # ── 2. Build DBM ─────────────────────────────────────────────────────────
    print("\n[2/5] Building DBM …")
    dbm = DeepBoltzmannMachine(
        layer_sizes=LAYER_SIZES,
        learning_rate=LEARNING_RATE,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
        k_cd=K_CD,
        k_pcd=K_PCD,
        random_state=42,
    )

    # ── 3. Pre-train ──────────────────────────────────────────────────────────
    print("\n[3/5] Greedy layer-wise pre-training …")
    t0 = time.time()
    dbm.pretrain(X_train, epochs=PRETRAIN_EPOCHS,
                 batch_size=BATCH_SIZE, verbose=True)
    print(f"\n  Pre-training time : {time.time() - t0:.1f}s")

    # Validation MSE after pre-training
    mse_pre, X_recon_pre = evaluate(dbm, X_val)
    print(f"  Validation MSE (post pre-train) : {mse_pre:.6f}")

    # ── 4. Fine-tune ──────────────────────────────────────────────────────────
    print("\n[4/5] Fine-tuning DBM (mean-field + PCD) …")
    t1 = time.time()
    dbm.finetune(X_train, epochs=FINETUNE_EPOCHS,
                 batch_size=BATCH_SIZE, verbose=True)
    print(f"\n  Fine-tuning time : {time.time() - t1:.1f}s")

    mse_post, X_recon_post = evaluate(dbm, X_val)
    print(f"  Validation MSE (post fine-tune) : {mse_post:.6f}")
    print(f"  MSE improvement : {100 * (mse_pre - mse_post) / mse_pre:+.2f}%")

    # ── 5. Visualise ──────────────────────────────────────────────────────────
    print("\n[5/5] Generating visualisations …")

    viz.plot_dataset_samples(X_val, y_val, img_shape,
                             n_samples=60, n_cols=12)

    viz.plot_training_curves(dbm.rbms, dbm.ft_errors)

    viz.plot_weight_filters(dbm.rbms, img_shape, n_filters=64)

    viz.plot_reconstructions(X_val, X_recon_post, img_shape, n_samples=12)

    representations = dbm.get_layer_representations(X_val)
    layer_names = [f"Hidden Layer {i+1}  ({s} units)"
                   for i, s in enumerate(LAYER_SIZES[1:])]

    viz.plot_tsne_representations(representations, y_val,
                                  layer_names=layer_names)

    viz.plot_hidden_activation_heatmap(representations,
                                       layer_names=layer_names, n_samples=50)

    print("\n  Generating samples from the DBM …")
    samples = dbm.sample(n_samples=32, n_gibbs=500)
    viz.plot_generated_samples(samples, img_shape, n_cols=8)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Experiment complete! Summary:")
    print(f"    Dataset        : {dataset_name}")
    print(f"    Architecture   : {' -> '.join(map(str, LAYER_SIZES))}")
    print(f"    Pre-train MSE  : {mse_pre:.6f}")
    print(f"    Fine-tune MSE  : {mse_post:.6f}")
    print(f"    Improvement    : {100 * (mse_pre - mse_post) / mse_pre:+.2f}%")
    print("  Figures saved in current directory.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
