"""
dbm.py — Deep Boltzmann Machine (DBM)
----------------------------------------
Stacks multiple RBM layers to form a Deep Boltzmann Machine (DBM).

Training strategy
-----------------
1. **Greedy pre-training** : each RBM layer is trained independently on the
   feature representation produced by the previous layer.
2. **Mean-field fine-tuning** : a global pass refines all layers jointly
   using variational mean-field inference for the positive phase and
   persistent contrastive divergence (PCD) for the negative phase.

Architecture note
-----------------
In a true DBM the visible-to-hidden1 weights are doubled (W1 × 2) and
the hidden_{i-1}-to-hidden_i weights for intermediate layers are doubled
(W_i × 2) to account for the bidirectional nature of the model.  This
implementation follows that convention during fine-tuning.
"""

import numpy as np
from rbm import RBM, sigmoid, sample_binary


class DeepBoltzmannMachine:
    """
    Deep Boltzmann Machine built by stacking RBMs.

    Parameters
    ----------
    layer_sizes : list[int]
        Sizes of each layer, e.g. [784, 256, 128, 64] means visible=784,
        three hidden layers of sizes 256, 128, 64.
    learning_rate : float
    momentum : float
    weight_decay : float
    k_cd : int
        CD steps used in pre-training.
    k_pcd : int
        Gibbs steps used in PCD fine-tuning.
    random_state : int
    """

    def __init__(
        self,
        layer_sizes: list[int],
        learning_rate: float = 0.005,
        momentum: float = 0.5,
        weight_decay: float = 1e-4,
        k_cd: int = 1,
        k_pcd: int = 5,
        random_state: int = 42,
    ):
        assert len(layer_sizes) >= 2, "Need at least one visible + one hidden layer."

        self.layer_sizes = layer_sizes
        self.n_layers = len(layer_sizes) - 1   # number of weight matrices
        self.lr = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.k_pcd = k_pcd
        self.rng = np.random.RandomState(random_state)

        # ── Build stacked RBMs ───────────────────────────────────────────────
        self.rbms: list[RBM] = []
        for i in range(self.n_layers):
            rbm = RBM(
                n_visible=layer_sizes[i],
                n_hidden=layer_sizes[i + 1],
                learning_rate=learning_rate,
                momentum=momentum,
                weight_decay=weight_decay,
                k=k_cd,
                random_state=random_state + i,
            )
            self.rbms.append(rbm)

        # ── Fine-tuning buffers ──────────────────────────────────────────────
        self.ft_errors: list[float] = []

    # ── Properties for convenient weight access ──────────────────────────────

    @property
    def weights(self) -> list[np.ndarray]:
        return [rbm.W for rbm in self.rbms]

    @property
    def biases_visible(self) -> list[np.ndarray]:
        return [rbm.b for rbm in self.rbms]

    @property
    def biases_hidden(self) -> list[np.ndarray]:
        return [rbm.c for rbm in self.rbms]

    # ── Stage 1 : Greedy pre-training ────────────────────────────────────────

    def pretrain(self, X: np.ndarray, epochs: int = 20,
                 batch_size: int = 64, verbose: bool = True):
        """
        Greedily train each RBM layer using the hidden probabilities of the
        previous layer as input.
        """
        current_input = X.astype(np.float32)
        for layer_idx, rbm in enumerate(self.rbms):
            if verbose:
                print(f"\n{'═'*55}")
                print(f"  Pre-training Layer {layer_idx + 1}  "
                      f"({rbm.n_visible} → {rbm.n_hidden} units)")
                print(f"{'═'*55}")
            rbm.fit(current_input, epochs=epochs,
                    batch_size=batch_size, verbose=verbose)
            # Pass data through trained layer to create input for next layer
            current_input = rbm.transform(current_input)

    # ── Stage 2 : Mean-field variational fine-tuning ─────────────────────────

    def _mean_field_inference(self, v: np.ndarray,
                              n_iter: int = 20) -> list[np.ndarray]:
        """
        Approximate the positive phase expectations using variational
        mean-field updates (fixed-point iteration).

        Returns μ: list of mean-field activations, one per hidden layer.
        """
        # Initialise μ with a bottom-up pass
        mu = []
        current = v
        for rbm in self.rbms:
            p_h, _ = rbm.v_to_h(current)
            mu.append(p_h.copy())
            current = p_h

        # Iterative mean-field updates
        for _ in range(n_iter):
            for i, rbm in enumerate(self.rbms):
                # Bottom-up contribution
                if i == 0:
                    bottom_up = v @ rbm.W
                else:
                    bottom_up = mu[i - 1] @ rbm.W

                # Top-down contribution
                if i < self.n_layers - 1:
                    top_down = mu[i + 1] @ self.rbms[i + 1].W.T
                else:
                    top_down = 0.0

                # In a DBM, visible weights are doubled for the first layer
                scale = 2.0 if i == 0 else 1.0
                mu[i] = sigmoid(scale * bottom_up + top_down + rbm.c)

        return mu

    def _gibbs_sample(self, v_persistent: np.ndarray,
                      h_persistent: list[np.ndarray]) -> tuple:
        """
        k_pcd steps of Gibbs sampling for the negative phase (PCD).
        """
        v = v_persistent.copy()
        h = [hk.copy() for hk in h_persistent]

        for _ in range(self.k_pcd):
            # Up pass
            for i, rbm in enumerate(self.rbms):
                inp = v if i == 0 else h[i - 1]
                _, h[i] = rbm.v_to_h(inp)

            # Down pass (reconstruct visible)
            p_v = sigmoid(h[0] @ self.rbms[0].W.T + self.rbms[0].b)
            v = sample_binary(p_v)

        return v, h, p_v

    def finetune(self, X: np.ndarray, epochs: int = 10,
                 batch_size: int = 64, verbose: bool = True):
        """
        Global fine-tuning using mean-field positive phase and PCD.
        """
        n = X.shape[0]
        X = X.astype(np.float32)

        # Initialise persistent chains from data
        idx0 = self.rng.choice(n, batch_size, replace=False)
        v_persistent = X[idx0].copy()
        h_persistent = []
        cur = v_persistent
        for rbm in self.rbms:
            _, h = rbm.v_to_h(cur)
            h_persistent.append(h)
            cur = h

        # Velocity buffers
        dW = [np.zeros_like(rbm.W) for rbm in self.rbms]
        db = [np.zeros_like(rbm.b) for rbm in self.rbms]
        dc = [np.zeros_like(rbm.c) for rbm in self.rbms]

        if verbose:
            print(f"\n{'═'*55}")
            print("  Fine-tuning DBM (Mean-Field + PCD)")
            print(f"{'═'*55}")

        for epoch in range(1, epochs + 1):
            mom = 0.9 if epoch > 5 else 0.5
            idx = np.random.permutation(n)
            epoch_errors = []

            for start in range(0, n, batch_size):
                v0 = X[idx[start:start + batch_size]]
                bsz = v0.shape[0]

                # ── Positive phase ───────────────────────────────────────────
                mu = self._mean_field_inference(v0)

                # ── Negative phase ───────────────────────────────────────────
                v_neg, h_neg, p_v_neg = self._gibbs_sample(
                    v_persistent, h_persistent)
                v_persistent = v_neg
                h_persistent = h_neg

                # ── Update weights ───────────────────────────────────────────
                for i, rbm in enumerate(self.rbms):
                    pos_v = v0 if i == 0 else mu[i - 1]
                    neg_v = v_neg if i == 0 else h_neg[i - 1]
                    neg_h = h_neg[i]

                    # Use separate means to handle different batch sizes
                    neg_bsz = neg_v.shape[0]

                    grad_W = (pos_v.T @ mu[i]) / bsz \
                           - (neg_v.T @ neg_h) / neg_bsz \
                           - self.weight_decay * rbm.W
                    # Bias gradient: separate positive and negative means
                    grad_b = np.mean(pos_v, axis=0) - np.mean(neg_v, axis=0)
                    grad_c = np.mean(mu[i], axis=0) - np.mean(neg_h, axis=0)

                    dW[i] = mom * dW[i] + self.lr * grad_W
                    db[i] = mom * db[i] + self.lr * grad_b
                    dc[i] = mom * dc[i] + self.lr * grad_c

                    rbm.W += dW[i]
                    rbm.b += db[i]
                    rbm.c += dc[i]

                # Reconstruction error: encode v0 then decode back
                p_h0, _ = self.rbms[0].v_to_h(v0)
                p_v0, _ = self.rbms[0].h_to_v(p_h0)
                recon_err = float(np.mean((v0 - p_v0) ** 2))
                epoch_errors.append(recon_err)

            mean_err = float(np.mean(epoch_errors))
            self.ft_errors.append(mean_err)
            if verbose:
                print(f"  Epoch {epoch:>3}/{epochs}  |  "
                      f"Recon Error: {mean_err:.6f}")

    # ── Inference helpers ─────────────────────────────────────────────────────

    def encode(self, X: np.ndarray, layer: int = -1) -> np.ndarray:
        """
        Encode input data up to a given hidden layer index.
        layer=-1 → deepest layer.
        """
        X = X.astype(np.float32)
        out = X
        n_layers = self.n_layers if layer == -1 else layer + 1
        for i in range(n_layers):
            out = self.rbms[i].transform(out)
        return out

    def decode(self, h: np.ndarray, from_layer: int = -1) -> np.ndarray:
        """
        Decode from a hidden layer back to visible space.
        from_layer=-1 → deepest layer.
        """
        out = h.astype(np.float32)
        start = self.n_layers - 1 if from_layer == -1 else from_layer
        for i in range(start, -1, -1):
            p_v, _ = self.rbms[i].h_to_v(out)
            out = p_v
        return out

    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        """Encode all the way up, then decode all the way down."""
        h = self.encode(X)
        return self.decode(h)

    def sample(self, n_samples: int = 16, n_gibbs: int = 1000) -> np.ndarray:
        """
        Generate samples by running Gibbs chains in the top-layer RBM space,
        then decoding down to visible space.
        """
        top_rbm = self.rbms[-1]
        # Start from random visible state in the top RBM
        v_top = sample_binary(
            0.5 * np.ones((n_samples, top_rbm.n_visible), dtype=np.float32))

        for _ in range(n_gibbs):
            p_h, h_top = top_rbm.v_to_h(v_top)
            p_v, v_top = top_rbm.h_to_v(h_top)

        # Decode top-layer visible (v_top) through earlier layers down to input
        out = v_top
        for i in range(self.n_layers - 2, -1, -1):
            p_v, _ = self.rbms[i].h_to_v(out)
            out = p_v

        return out

    def get_layer_representations(self, X: np.ndarray) -> list[np.ndarray]:
        """Return feature representations at every hidden layer."""
        representations = []
        current = X.astype(np.float32)
        for rbm in self.rbms:
            current = rbm.transform(current)
            representations.append(current.copy())
        return representations
