"""
agi_core.py — AGI Integration Core
====================================
Three-stage pipeline matching the paper's methodology (§3):

  Stage 1 — Theoretic training:
    Train a standard FC network in the electronic domain using ADAM + NMSE.
    Architecture from paper Table 1: [4 → 6 → 3] for MFR task.

  Stage 2 — Optical weight transfer:
    Map trained weights → FiberNeuralNetwork.
    Encode as synaptic optical parameters via RecursiveMasterPrompt.

  Stage 3 — Recursive self-expansion:
    Evaluate optical NN performance.
    If insufficient → RecursiveMasterPrompt.self_expand() inserts a new
    fiber loop and the cycle repeats (AGI self-improvement).

MFR Dataset (paper §2):
  3 formats: OOK, PAM-4, PSK
  4 features: algebraic mean, variance, IQR, geometric mean
  150 samples, 80/20 split
"""

import numpy as np
from fiber_nn import FiberNeuralNetwork, minmax_norm, softmax
from recursive_prompt import RecursiveMasterPrompt


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class TheoreticalTrainer:
    """
    Electronic-domain FC trainer — pure numpy, no dependencies.

    Loss:       NMSE = Σ(ŷ - y)² / Σy²         (paper §3)
    Optimiser:  ADAM with lr=1e-3               (paper Table 1)
    Architecture: [4, 6, 3] fully connected     (paper Table 1)
    """

    def __init__(self, layer_sizes, lr=1e-3):
        self.sizes = layer_sizes
        self.lr    = lr
        self.rng   = np.random.default_rng(7)

        self.W, self.b = [], []
        for i in range(len(layer_sizes) - 1):
            n_in, n_out = layer_sizes[i], layer_sizes[i + 1]
            scale = np.sqrt(2.0 / n_in)
            self.W.append(self.rng.normal(0, scale, (n_out, n_in)))
            self.b.append(np.zeros(n_out))

        # ADAM state
        self.mW = [np.zeros_like(w) for w in self.W]
        self.vW = [np.zeros_like(w) for w in self.W]
        self.mb = [np.zeros_like(b) for b in self.b]
        self.vb = [np.zeros_like(b) for b in self.b]
        self.t  = 0
        self.b1, self.b2, self.eps = 0.9, 0.999, 1e-8

    # ── Activations ───────────────────────────────────────────────────────
    @staticmethod
    def _relu(x):      return np.maximum(0.0, x)
    @staticmethod
    def _relu_d(x):    return (x > 0).astype(float)
    @staticmethod
    def _softmax(x):   e = np.exp(x - x.max()); return e / e.sum()

    # ── Forward ───────────────────────────────────────────────────────────
    def forward(self, x):
        self._a = [x]
        self._z = []
        h = x
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = W @ h + b
            self._z.append(z)
            h = self._relu(z) if i < len(self.W) - 1 else self._softmax(z)
            self._a.append(h)
        return h

    # ── NMSE loss ─────────────────────────────────────────────────────────
    @staticmethod
    def nmse(y_hat, y):
        return float(np.sum((y_hat - y) ** 2) / (np.sum(y ** 2) + 1e-12))

    # ── Backward ──────────────────────────────────────────────────────────
    def backward(self, y_true):
        n  = len(self.W)
        gW = [None] * n
        gb = [None] * n
        y_hat  = self._a[-1]
        denom  = np.sum(y_true ** 2) + 1e-12
        delta  = 2.0 * (y_hat - y_true) / denom   # NMSE gradient at output

        for i in reversed(range(n)):
            if i < n - 1:
                delta *= self._relu_d(self._z[i])
            gW[i]  = np.outer(delta, self._a[i])
            gb[i]  = delta.copy()
            if i > 0:
                delta = self.W[i].T @ delta
        return gW, gb

    # ── ADAM update ───────────────────────────────────────────────────────
    def _adam(self, gW, gb):
        self.t += 1
        b1, b2, eps = self.b1, self.b2, self.eps
        lr_t = self.lr * np.sqrt(1 - b2 ** self.t) / (1 - b1 ** self.t)

        for i in range(len(self.W)):
            self.mW[i] = b1 * self.mW[i] + (1 - b1) * gW[i]
            self.vW[i] = b2 * self.vW[i] + (1 - b2) * gW[i] ** 2
            self.W[i] -= lr_t * self.mW[i] / (np.sqrt(self.vW[i]) + eps)

            self.mb[i] = b1 * self.mb[i] + (1 - b1) * gb[i]
            self.vb[i] = b2 * self.vb[i] + (1 - b2) * gb[i] ** 2
            self.b[i]  -= lr_t * self.mb[i] / (np.sqrt(self.vb[i]) + eps)

    # ── Train ─────────────────────────────────────────────────────────────
    def train(self, X, Y, epochs=500, verbose=True):
        losses = []
        for ep in range(epochs):
            idx = np.random.default_rng(ep).permutation(len(X))
            ep_loss = 0.0
            for i in idx:
                x_n = minmax_norm(X[i])
                y_hat = self.forward(x_n)
                ep_loss += self.nmse(y_hat, Y[i])
                gW, gb = self.backward(Y[i])
                self._adam(gW, gb)
            ep_loss /= len(X)
            losses.append(ep_loss)
            if verbose and (ep % 100 == 0 or ep == epochs - 1):
                print(f"    Epoch {ep+1:>4}/{epochs}   NMSE = {ep_loss:.6f}")
        return losses

    def evaluate(self, X, Y):
        correct = 0
        for x, y in zip(X, Y):
            x_n = minmax_norm(np.asarray(x, dtype=float))
            pred = self.forward(x_n)
            if np.argmax(pred) == np.argmax(y):
                correct += 1
        return correct / len(X)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def generate_mfr_dataset(n_samples=150, noise=0.04, seed=42):
    """
    Synthetic MFR dataset — 3 formats × 4 statistical features.
    Reproduces the paper's task (§2) without requiring physical lab data.

    OOK  : binary  {0, 1}  — high mean, moderate variance
    PAM-4: 4-level {0,1/3,2/3,1}  — flatter amplitude histogram
    PSK  : constant amplitude, phase-coded  — low variance in intensity

    Features per sample:
      [algebraic mean, variance, IQR, geometric mean]
    """
    rng = np.random.default_rng(seed)
    X, Y = [], []
    n = n_samples // 3

    def features(sig):
        sig = np.clip(sig, 0, None)
        alg  = float(np.mean(sig))
        var  = float(np.var(sig))
        iqr  = float(np.percentile(sig, 75) - np.percentile(sig, 25))
        gm   = float(np.exp(np.mean(np.log(sig + 1e-9))))
        return np.array([alg, var, iqr, gm])

    for _ in range(n):   # OOK
        s = rng.choice([0.0, 1.0], 2000) + rng.normal(0, noise, 2000)
        X.append(features(s)); Y.append([1, 0, 0])

    for _ in range(n):   # PAM-4
        s = rng.choice([0.0, 1/3, 2/3, 1.0], 2000) + rng.normal(0, noise, 2000)
        X.append(features(s)); Y.append([0, 1, 0])

    for _ in range(n):   # PSK (constant-amplitude phase keying)
        phi = rng.choice([0, np.pi/2, np.pi, 3*np.pi/2], 2000)
        s   = 0.5 * (1 + np.cos(phi)) + rng.normal(0, noise, 2000)
        X.append(features(s)); Y.append([0, 0, 1])

    X, Y = np.array(X), np.array(Y)
    idx  = rng.permutation(len(X))
    return X[idx], Y[idx]


def generate_mfr5_dataset(n_samples=250, noise=0.04, seed=42):
    """
    Extended 5-class MFR dataset for recursive development Level 2.

    Formats:
      0  OOK   — binary {0,1},            high mean, moderate var
      1  PAM-4 — 4-level {0,1/3,2/3,1},   mid mean, low var
      2  PAM-8 — 8-level {0,1/7,...,1},   mid mean, very low var
      3  QPSK  — 4 phase states,           constant amplitude ≈ 0.5
      4  QAM16 — 16 constellation points, mixed amplitude

    Features (8 now — richer for 5-class):
      [mean, variance, IQR, geometric_mean,
       skewness, kurtosis, peak_ratio, zero_crossing_rate]
    """
    rng = np.random.default_rng(seed)
    X, Y = [], []
    n = n_samples // 5

    def features(sig):
        sig = np.clip(sig, 0, None)
        mn   = float(np.mean(sig))
        var  = float(np.var(sig))
        iqr  = float(np.percentile(sig, 75) - np.percentile(sig, 25))
        gm   = float(np.exp(np.mean(np.log(sig + 1e-9))))
        # Extra features for 5-class discrimination
        std  = float(np.std(sig)) + 1e-9
        sk   = float(np.mean(((sig - mn) / std) ** 3))   # skewness
        ku   = float(np.mean(((sig - mn) / std) ** 4))   # kurtosis
        pk   = float(np.max(sig) / (mn + 1e-9))           # peak ratio
        zc   = float(np.mean(np.diff(sig > mn) != 0))    # zero-crossing
        return np.array([mn, var, iqr, gm, sk, ku, pk, zc])

    for _ in range(n):   # OOK
        s = rng.choice([0.0, 1.0], 2000) + rng.normal(0, noise, 2000)
        X.append(features(s)); Y.append([1,0,0,0,0])

    for _ in range(n):   # PAM-4
        s = rng.choice([0.0,1/3,2/3,1.0], 2000) + rng.normal(0, noise, 2000)
        X.append(features(s)); Y.append([0,1,0,0,0])

    for _ in range(n):   # PAM-8
        lvls = np.linspace(0, 1, 8)
        s = rng.choice(lvls, 2000) + rng.normal(0, noise*0.5, 2000)
        X.append(features(s)); Y.append([0,0,1,0,0])

    for _ in range(n):   # QPSK
        phi = rng.choice([0, np.pi/2, np.pi, 3*np.pi/2], 2000)
        s   = 0.5*(1 + np.cos(phi)) + rng.normal(0, noise, 2000)
        X.append(features(s)); Y.append([0,0,0,1,0])

    for _ in range(n):   # QAM-16
        re = rng.choice([-3,-1,1,3], 2000) / 3.0
        im = rng.choice([-3,-1,1,3], 2000) / 3.0
        s  = np.abs(re + 1j*im) + rng.normal(0, noise, 2000)
        X.append(features(s)); Y.append([0,0,0,0,1])

    X, Y = np.array(X, dtype=float), np.array(Y, dtype=float)
    idx  = rng.permutation(len(X))
    return X[idx], Y[idx]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class AGIFiberCore:
    """
    Unified AGI orchestration layer.

    Owns:
      • TheoreticalTrainer      — electronic domain learning
      • FiberNeuralNetwork      — optical domain inference
      • RecursiveMasterPrompt   — synaptic parameter generation + self-expansion

    The recursive loop:
      train → embed → evaluate → [expand?] → recursive_unfold → repeat
    """

    CLASS_NAMES = ['OOK', 'PAM', 'PSK']

    def __init__(self, layer_sizes=None, seed_dim=64, max_depth=8, noisy=False):
        self.layer_sizes    = layer_sizes or [4, 6, 3]
        self.noisy          = noisy
        self.master_prompt  = RecursiveMasterPrompt(seed_dim, max_depth)
        self.trainer        = TheoreticalTrainer(self.layer_sizes)
        self.fiber_nn       = None
        self.expansion_log  = []
        self.perf_history   = []

    # ── Full pipeline ─────────────────────────────────────────────────────
    def run(self, X_train, Y_train, X_test, Y_test,
            epochs=500, verbose=True, expand_threshold=0.95):
        """
        Execute the complete 3-stage AGI pipeline.
        Returns final fiber NN accuracy.
        """
        _bar = "═" * 62
        print(f"\n{_bar}")
        print("  VIRTUAL FIBER OPTIC NEURAL NETWORK  ·  AGI CORE")
        print(f"  Recursive Master Prompt  ·  Synaptic Optical Embedding")
        print(_bar)

        # ── Stage 1: Theoretic training ───────────────────────────────────
        print(f"\n  [1/3] Theoretic training  {self.layer_sizes}")
        print(f"        Optimiser: ADAM  |  Loss: NMSE  |  lr=1e-3")
        losses = self.trainer.train(
            X_train, Y_train, epochs=epochs, verbose=verbose
        )
        tr_acc = self.trainer.evaluate(X_train, Y_train)
        te_acc = self.trainer.evaluate(X_test,  Y_test)
        print(f"        Train acc: {tr_acc*100:.1f}%   Test acc: {te_acc*100:.1f}%")

        # ── Stage 2: Weight transfer → optical frame ──────────────────────
        print(f"\n  [2/3] Transferring weights → optical fiber NN frame")
        self.fiber_nn = FiberNeuralNetwork(
            self.layer_sizes, noisy=self.noisy
        )
        self.fiber_nn.load_trained_weights(self.trainer.W, self.trainer.b)

        shapes   = [(self.layer_sizes[i+1], self.layer_sizes[i])
                    for i in range(len(self.layer_sizes) - 1)]
        _, encs  = self.master_prompt.recursive_unfold(shapes)
        self.fiber_nn.optical_encodings = encs
        self._print_optical_table()

        # ── Evaluate optical NN ────────────────────────────────────────────
        fiber_acc = self.fiber_nn.evaluate(X_test, Y_test)
        self.perf_history.append(fiber_acc)
        print(f"\n        Fiber NN test accuracy: {fiber_acc*100:.1f}%")

        # ── Stage 3: Recursive self-expansion ─────────────────────────────
        print(f"\n  [3/3] Recursive self-expansion check")
        new_shapes, expanded = self.master_prompt.self_expand(
            shapes, fiber_acc, threshold=expand_threshold
        )
        if expanded:
            old = self.layer_sizes[:]
            new_sizes = [new_shapes[0][1]] + [s[0] for s in new_shapes]
            self.layer_sizes = new_sizes
            self.expansion_log.append({
                'from': old, 'to': new_sizes, 'accuracy': fiber_acc
            })
            print(f"        Architecture expanded: {old} → {new_sizes}")
            print(f"        (New fiber loop inserted at bottleneck)")
        else:
            print(f"        No expansion needed  ({fiber_acc*100:.1f}% ≥ threshold)")

        return losses, fiber_acc

    # ── Inference ─────────────────────────────────────────────────────────
    def infer(self, x_raw, verbose=True):
        """Optical inference on a single sample."""
        if self.fiber_nn is None:
            raise RuntimeError("Call run() before infer().")
        out  = self.fiber_nn.forward(x_raw)
        pred = int(np.argmax(out))
        if verbose:
            print(f"    Input:     {np.round(x_raw, 4)}")
            print(f"    Out:       [{', '.join(f'{v:.4f}' for v in out)}]")
            print(f"    Predicted: {self.CLASS_NAMES[pred]}")
        return pred, out

    # ── Reporting ─────────────────────────────────────────────────────────
    def _print_optical_table(self):
        rows = self.fiber_nn.optical_state_table()
        print(f"\n        {'Lyr':>3}  {'Shape':>10}  {'‖W‖':>7}  "
              f"{'Sparse':>7}  {'λ-ch':>5}")
        print(f"        {'-'*42}")
        for r in rows:
            print(f"        {r['layer']:>3}  "
                  f"{str(r['shape']):>10}  "
                  f"{r['W_norm']:>7.3f}  "
                  f"{r['sparsity']:>6.1%}  "
                  f"{r['lambda_channels']:>5}")

    def summary(self):
        print(f"\n{'═'*62}")
        print(f"  SYSTEM SUMMARY")
        print(f"{'═'*62}")
        print(f"  Architecture     : {self.layer_sizes}")
        print(f"  Ring loops       : {self.fiber_nn.ring_loops() if self.fiber_nn else 'N/A'}")
        print(f"  Recursive depth  : {len(self.master_prompt.history)}")
        print(f"  Expansions       : {len(self.expansion_log)}")
        if self.perf_history:
            print(f"  Peak fiber acc   : {max(self.perf_history)*100:.1f}%")
        n_lambda = sum(
            e['wavelengths'].shape[0]
            for e in self.fiber_nn.optical_encodings
            if e is not None
        ) if self.fiber_nn else 0
        print(f"  WDM λ-channels   : {n_lambda}")
        print(f"\n  The fiber IS the network.")
        print(f"  Light propagation IS computation.")
        print(f"  Synaptic weights ARE photon amplitudes.")
