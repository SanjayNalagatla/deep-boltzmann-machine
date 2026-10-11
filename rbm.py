"""
rbm.py — Restricted Boltzmann Machine (RBM)
---------------------------------------------
Single-layer energy-based model that forms the building block of the DBM.
Training is done via k-step Contrastive Divergence (CD-k).
"""

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
#  Helper utilities
# ─────────────────────────────────────────────────────────────────────────────

def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid with bounded exponentiation."""
    x_clipped = np.clip(x, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-x_clipped))


def sample_binary(probs: np.ndarray) -> np.ndarray:
    """Sample binary states from Bernoulli probabilities."""
    return (np.random.rand(*probs.shape) < probs).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
#  Restricted Boltzmann Machine
# ─────────────────────────────────────────────────────────────────────────────

class RBM:
    """
    Restricted Boltzmann Machine with binary visible and hidden units.

    Energy:
        E(v, h) = -v^T W h - b^T v - c^T h

    Parameters
    ----------
    n_visible : int
        Number of visible (input) units.
    n_hidden : int
        Number of hidden units.
    learning_rate : float
        Step size for parameter updates.
    momentum : float
        Momentum coefficient for SGD with momentum.
    weight_decay : float
        L2 regularisation coefficient.
    k : int
        Number of Gibbs sampling steps in CD-k.
    """

    def __init__(
        self,
        n_visible: int,
        n_hidden: int,
        learning_rate: float = 0.01,
        momentum: float = 0.5,
        weight_decay: float = 1e-4,
        k: int = 1,
        random_state: int = 42,
    ):
        rng = np.random.RandomState(random_state)

        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.lr = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.k = k

        # ── Parameters ──────────────────────────────────────────────────────
        self.W = rng.normal(0, 0.01, size=(n_visible, n_hidden)).astype(np.float32)
        self.b = np.zeros(n_visible, dtype=np.float32)   # visible bias
        self.c = np.zeros(n_hidden,  dtype=np.float32)   # hidden  bias

        # ── Velocity buffers (momentum) ──────────────────────────────────────
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self.dc = np.zeros_like(self.c)

        # ── Training history ────────────────────────────────────────────────
        self.reconstruction_errors: list[float] = []

    # ── Forward / backward pass ─────────────────────────────────────────────

    def v_to_h(self, v: np.ndarray):
        """Compute P(h=1|v) and a binary sample."""
        p_h = sigmoid(v @ self.W + self.c)
        return p_h, sample_binary(p_h)

    def h_to_v(self, h: np.ndarray):
        """Compute P(v=1|h) and a binary sample."""
        p_v = sigmoid(h @ self.W.T + self.b)
        return p_v, sample_binary(p_v)

    def free_energy(self, v: np.ndarray) -> float:
        """Compute the free energy of the visible configuration."""
        vbias_term = v @ self.b
        wx_b = v @ self.W + self.c
        hidden_term = np.sum(np.log(1 + np.exp(wx_b)), axis=1)
        return float(np.mean(-vbias_term - hidden_term))

    # ── Training (CD-k) ─────────────────────────────────────────────────────

    def contrastive_divergence(self, v0: np.ndarray):
        """
        One CD-k update.

        Returns
        -------
        recon_error : float
            Mean squared reconstruction error for monitoring.
        """
        # ── Positive phase ───────────────────────────────────────────────────
        ph0, h0 = self.v_to_h(v0)

        # ── Negative phase (k Gibbs steps) ──────────────────────────────────
        vk, hk = v0.copy(), h0.copy()
        for _ in range(self.k):
            pvk, vk = self.h_to_v(hk)
            phk, hk = self.v_to_h(vk)

        batch = v0.shape[0]

        # ── Gradients ────────────────────────────────────────────────────────
        pos = (v0.T @ ph0) / batch
        neg = (vk.T @ phk) / batch

        grad_W = pos - neg - self.weight_decay * self.W
        grad_b = np.mean(v0 - vk, axis=0)
        grad_c = np.mean(ph0 - phk, axis=0)

        # ── Momentum SGD update ──────────────────────────────────────────────
        self.dW = self.momentum * self.dW + self.lr * grad_W
        self.db = self.momentum * self.db + self.lr * grad_b
        self.dc = self.momentum * self.dc + self.lr * grad_c

        self.W += self.dW
        self.b += self.db
        self.c += self.dc

        recon_error = float(np.mean((v0 - pvk) ** 2))
        self.reconstruction_errors.append(recon_error)
        return recon_error

    def fit(self, X: np.ndarray, epochs: int = 20, batch_size: int = 64, verbose: bool = True):
        """Greedy layer-wise pre-training on dataset X."""
        n = X.shape[0]
        for epoch in range(1, epochs + 1):
            # increase momentum after 5 epochs (common schedule)
            self.momentum = 0.9 if epoch > 5 else 0.5

            idx = np.random.permutation(n)
            epoch_errors = []
            for start in range(0, n, batch_size):
                batch = X[idx[start:start + batch_size]]
                err = self.contrastive_divergence(batch)
                epoch_errors.append(err)

            if verbose:
                print(f"  Epoch {epoch:>3}/{epochs}  |  Recon Error: {np.mean(epoch_errors):.6f}")

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Pass data through the RBM to get hidden probabilities (soft features)."""
        p_h, _ = self.v_to_h(X)
        return p_h

    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        """Encode then decode to get reconstructed visible units."""
        _, h = self.v_to_h(X)
        p_v, _ = self.h_to_v(h)
        return p_v
