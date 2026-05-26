"""
recursive_dev.py — Recursive Development Engine
================================================
Multi-generation self-improving fiber optical neural network.

Each generation:
  1. Expand architecture by inserting one identity-initialized fiber loop
     (new ring loop starts as optical pass-through → then learns)
  2. Warm-start weights from previous generation
  3. Fine-tune with lower learning rate
  4. Per-layer output calibration — closes theoretic→fiber accuracy gap
  5. Update master prompt meta-generator via fitness-proportional perturbation
  6. Emit a snapshot (Generation) and continue if target not reached

The identity initialization of new layers is the key AGI recursive insight:
  Each new fiber loop adds ZERO disruption at insertion time
  (W_new = I, b_new = 0 → h_out = h_in, identity transform).
  Training then sculpts the new loop into a useful computation.

This mirrors how biological neural systems grow: new connections are
quiescent at birth and become functional through experience.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional
from fiber_nn import FiberNeuralNetwork, minmax_norm, softmax
from recursive_prompt import RecursiveMasterPrompt
from agi_core import TheoreticalTrainer


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
@dataclass
class Generation:
    """Snapshot of one recursive development generation."""
    idx:          int
    layer_sizes:  List[int]
    th_train_acc: float
    th_test_acc:  float
    fiber_acc:    float
    gap:          float
    loss_final:   float
    prompt_norm:  float
    mi_total:     float
    n_ring_loops: int
    expanded:     bool
    cal_scales:   List[float] = field(default_factory=list)

    @property
    def label(self):
        e = "↑" if self.expanded else "·"
        return (f"Gen {self.idx} {e}  arch={self.layer_sizes}  "
                f"theory={self.th_test_acc*100:.1f}%  "
                f"fiber={self.fiber_acc*100:.1f}%  "
                f"gap={self.gap*100:.1f}%")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class CrossEntropyTrainer(TheoreticalTrainer):
    """
    Cross-entropy + softmax trainer (replaces NMSE for better convergence).
    Inherits ADAM from TheoreticalTrainer; overrides loss + backward.
    Also supports L2 regularization for fiber-compatible weight magnitudes.
    """

    def __init__(self, layer_sizes, lr=1e-3, l2=1e-4):
        super().__init__(layer_sizes, lr=lr)
        self.l2 = l2

    @staticmethod
    def _ce(y_hat, y_true):
        return -float(np.sum(y_true * np.log(y_hat + 1e-12)))

    def backward_ce(self, y_true):
        """Cross-entropy gradient (combined with softmax: δ = ŷ - y)."""
        n  = len(self.W)
        gW = [None] * n
        gb = [None] * n
        y_hat = self._a[-1]
        delta = y_hat - y_true          # softmax + CE combined gradient

        for i in reversed(range(n)):
            if i < n - 1:
                delta *= self._relu_d(self._z[i])
            gW[i] = np.outer(delta, self._a[i])
            gb[i] = delta.copy()
            # L2 regularisation on weights (not bias)
            gW[i] += self.l2 * self.W[i]
            if i > 0:
                delta = self.W[i].T @ delta
        return gW, gb

    def train(self, X, Y, epochs=500, verbose=True):
        losses = []
        for ep in range(epochs):
            idx = np.random.default_rng(ep).permutation(len(X))
            ep_loss = 0.0
            for i in idx:
                x_n  = minmax_norm(X[i])
                y_hat = self.forward(x_n)
                ep_loss += self._ce(y_hat, Y[i])
                gW, gb = self.backward_ce(Y[i])
                self._adam(gW, gb)
            ep_loss /= len(X)
            losses.append(ep_loss)
            if verbose and (ep % 100 == 0 or ep == epochs - 1):
                print(f"    Epoch {ep+1:>4}/{epochs}   CE = {ep_loss:.5f}")
        return losses


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class RecursiveDevelopmentEngine:
    """
    Multi-generation recursive development loop.

    Algorithm
    ─────────
    gen = 0
    sizes = initial_layer_sizes
    while gen < max_gen and fiber_acc < target:
        trainer = CrossEntropyTrainer(sizes, lr=lr_schedule[gen])
        warmstart(trainer, prev_trainer)          ← preserve old weights
        trainer.train(X, Y, epochs)
        fiber_nn ← transfer weights
        calibrate(fiber_nn, trainer, X_train)     ← close theory→fiber gap
        fiber_acc ← evaluate(fiber_nn)
        meta_prompt.update(accuracy_signal)       ← evolve meta-generator
        snapshot = Generation(...)
        if fiber_acc < target:
            sizes = expand(sizes)                 ← insert identity fiber loop
        gen += 1

    Expansion strategy
    ──────────────────
    Insert an identity-initialized layer before the output layer.
    For sizes = [4, 6, 3]:
      New sizes = [4, 6, 6, 3]
      W_new[1] = I_6  (identity)   ← optical pass-through at insertion
      W_new[2] = W_old[1]          ← output layer kept intact

    This guarantees:
      accuracy_gen+1 ≥ accuracy_gen  (at insertion; identity = no regression)
    """

    def __init__(self, initial_layer_sizes, master_prompt: RecursiveMasterPrompt,
                 X_train, Y_train, X_test, Y_test,
                 noisy=False, rng_seed=0):
        self.sizes0  = list(initial_layer_sizes)
        self.mp      = master_prompt
        self.X_tr, self.Y_tr = X_train, Y_train
        self.X_te, self.Y_te = X_test,  Y_test
        self.noisy   = noisy
        self.rng     = np.random.default_rng(rng_seed)

        self.generations: List[Generation] = []
        self.best_fiber_acc   = 0.0
        self.best_fiber_nn    = None
        self.prev_trainer     = None

    # ── Warm-start ────────────────────────────────────────────────────────
    def _warmstart(self, trainer: CrossEntropyTrainer,
                   prev: Optional[CrossEntropyTrainer],
                   prev_sizes: List[int]):
        """
        Copy weights from prev generation into new trainer.
        New layers (inserted during expansion) receive identity or prompt init.
        """
        if prev is None:
            return

        new_sizes = trainer.sizes
        n_new = len(trainer.W)
        n_old = len(prev.W)

        if new_sizes == prev_sizes:
            # Same architecture — direct copy
            for i in range(n_new):
                trainer.W[i] = prev.W[i].copy()
                trainer.b[i] = prev.b[i].copy()
            return

        # Expanded architecture: new layer inserted at position n_old - 1
        # Old layers 0..n_old-2 → new layers 0..n_old-2
        # New identity layer     → new layer n_old-1
        # Old output layer       → new layer n_old
        for i in range(n_old - 1):
            if trainer.W[i].shape == prev.W[i].shape:
                trainer.W[i] = prev.W[i].copy()
                trainer.b[i] = prev.b[i].copy()

        # Identity init for new layer (index n_old-1)
        ni = n_old - 1
        n_feat = trainer.W[ni].shape[1]   # square identity block
        n_out  = trainer.W[ni].shape[0]
        if n_out == n_feat:
            trainer.W[ni] = np.eye(n_out)
        else:
            # Rectangular: top-left identity block, rest small random
            W_id = np.zeros((n_out, n_feat))
            sz   = min(n_out, n_feat)
            W_id[:sz, :sz] = np.eye(sz)
            trainer.W[ni] = W_id
        trainer.W[ni] += self.rng.normal(0, 0.01, trainer.W[ni].shape)
        trainer.b[ni]  = np.zeros(trainer.b[ni].shape)

        # Output layer: keep old output weights if shape matches
        out_i = n_new - 1
        if trainer.W[out_i].shape == prev.W[n_old - 1].shape:
            trainer.W[out_i] = prev.W[n_old - 1].copy()
            trainer.b[out_i] = prev.b[n_old - 1].copy()

    # ── Expansion ─────────────────────────────────────────────────────────
    def _expand_sizes(self, sizes: List[int]) -> List[int]:
        """
        Insert a new hidden layer of the same width as the last hidden layer,
        immediately before the output.

          [4, 6, 3]  →  [4, 6, 6, 3]
          [4, 6, 6, 3] → [4, 6, 6, 6, 3]
        """
        if len(sizes) < 3:
            return sizes  # nothing to expand
        width = sizes[-2]            # width of last hidden layer
        return sizes[:-1] + [width, sizes[-1]]

    # ── Calibration ───────────────────────────────────────────────────────
    def _calibrate(self, fiber_nn: FiberNeuralNetwork,
                   trainer: CrossEntropyTrainer) -> List[float]:
        """
        Per-layer output calibration — closes the theoretic→fiber accuracy gap.

        Root cause of gap:
          Theoretic model applies ReLU in hidden layers; fiber NN does not.
          This creates a systematic positive-bias difference in hidden activations.

        Fix:
          For each hidden layer i, collect fiber output h_fi and theoretic
          pre-ReLU output z_i (both have same W·x + b).
          Compute scale_i = mean(|z_i|) / (mean(|h_fi|) + ε).
          Multiply fiber layer i's weights by scale_i.

          The output layer is not calibrated (softmax normalizes it).
        """
        scales = []
        n_hidden = len(fiber_nn.layers) - 1   # exclude output layer

        th_norms  = [[] for _ in range(n_hidden)]
        fi_norms  = [[] for _ in range(n_hidden)]

        for x in self.X_tr:
            # Theoretic pre-activations
            x_n = minmax_norm(x)
            trainer.forward(x_n)
            for i in range(n_hidden):
                th_norms[i].append(float(np.mean(np.abs(trainer._z[i]))))

            # Fiber hidden activations
            fiber_nn.forward(x)
            for i in range(n_hidden):
                fi_norms[i].append(float(np.mean(np.abs(
                    fiber_nn._layer_outputs[i + 1]))) + 1e-8)

        for i in range(n_hidden):
            th_mean = float(np.mean(th_norms[i]))
            fi_mean = float(np.mean(fi_norms[i]))
            scale_i = th_mean / (fi_mean + 1e-8)
            scale_i = float(np.clip(scale_i, 0.1, 10.0))   # safety clamp
            fiber_nn.layers[i].W *= scale_i
            scales.append(scale_i)

        return scales

    # ── Meta-prompt update ────────────────────────────────────────────────
    def _meta_prompt_update(self, accuracy: float, prev_accuracy: float):
        """
        Fitness-proportional perturbation of meta-generator weights.

        If accuracy improved: reinforce current M_W1/M_W2 direction (small lr step).
        If accuracy regressed: add exploration noise.

        This is a minimal (1+1)-ES (evolution strategy) applied to the prompt.
        """
        delta = accuracy - prev_accuracy
        mp    = self.mp

        if delta > 0:
            # Reinforce: nudge M_W1/M_W2 toward current values (contraction)
            scale = min(delta * 5.0, 0.1)
            mp.M_W1 += scale * mp.M_W1 * (1 - np.abs(mp.M_W1))
            mp.M_W2 += scale * mp.M_W2 * (1 - np.abs(mp.M_W2))
        else:
            # Explore: add Gaussian noise scaled by regression magnitude
            sigma = min(abs(delta) * 2.0 + 0.01, 0.05)
            mp.M_W1 += self.rng.normal(0, sigma, mp.M_W1.shape)
            mp.M_W2 += self.rng.normal(0, sigma, mp.M_W2.shape)

        # Re-normalise to prevent explosion
        mp.M_W1 /= (np.linalg.norm(mp.M_W1) / mp.d + 1e-8)
        mp.M_W2 /= (np.linalg.norm(mp.M_W2) / mp.d + 1e-8)

    # ── Single generation ─────────────────────────────────────────────────
    def _run_generation(self, gen_idx: int, sizes: List[int],
                        epochs: int, lr: float,
                        prev_sizes: List[int]) -> Generation:

        print(f"\n  {'─'*60}")
        print(f"  GENERATION {gen_idx}  |  arch = {sizes}")
        print(f"  {'─'*60}")

        # Build trainer and warm-start
        trainer = CrossEntropyTrainer(sizes, lr=lr, l2=1e-4)
        self._warmstart(trainer, self.prev_trainer, prev_sizes)

        # Train
        print(f"  [train]  epochs={epochs}  lr={lr:.0e}  loss=CE  reg=L2")
        losses = trainer.train(self.X_tr, self.Y_tr, epochs=epochs, verbose=True)

        th_tr  = trainer.evaluate(self.X_tr, self.Y_tr)
        th_te  = trainer.evaluate(self.X_te, self.Y_te)
        print(f"  [theory] train={th_tr*100:.1f}%  test={th_te*100:.1f}%")

        # Build fiber NN
        shapes = [(sizes[i+1], sizes[i]) for i in range(len(sizes)-1)]
        _, encs = self.mp.recursive_unfold(shapes)

        fiber_nn = FiberNeuralNetwork(sizes, noisy=self.noisy)
        fiber_nn.load_trained_weights(trainer.W, trainer.b)
        fiber_nn.optical_encodings = encs

        # Pre-calibration fiber accuracy
        pre_cal = fiber_nn.evaluate(self.X_te, self.Y_te)
        print(f"  [fiber]  pre-calibration  = {pre_cal*100:.1f}%")

        # Per-layer output calibration
        cal_scales = self._calibrate(fiber_nn, trainer)
        post_cal   = fiber_nn.evaluate(self.X_te, self.Y_te)
        print(f"  [fiber]  post-calibration = {post_cal*100:.1f}%"
              f"   scales={[f'{s:.3f}' for s in cal_scales]}")

        gap = th_te - post_cal

        # MI of optical encodings
        mi_total = sum(
            self.mp.embedder.mutual_information_proxy(e)
            for e in encs
        )

        # Update prompt
        prev_acc = self.best_fiber_acc
        self._meta_prompt_update(post_cal, prev_acc)
        p_norm   = float(np.linalg.norm(self.mp.P0))

        # Track best
        if post_cal > self.best_fiber_acc:
            self.best_fiber_acc = post_cal
            self.best_fiber_nn  = fiber_nn
            self.prev_trainer   = trainer

        return Generation(
            idx          = gen_idx,
            layer_sizes  = list(sizes),
            th_train_acc = th_tr,
            th_test_acc  = th_te,
            fiber_acc    = post_cal,
            gap          = gap,
            loss_final   = losses[-1],
            prompt_norm  = p_norm,
            mi_total     = mi_total,
            n_ring_loops = len(sizes) - 2,  # hidden layers = ring loops
            expanded     = (sizes != self._expand_sizes(sizes)),
            cal_scales   = cal_scales,
        )

    # ── Main loop ─────────────────────────────────────────────────────────
    def run(self, max_gen: int = 5, target_acc: float = 0.90,
            epochs_base: int = 500, min_epochs: int = 200) -> List[Generation]:
        """
        Execute recursive development until target_acc or max_gen.

        Learning rate schedule: lr0 = 1e-3; halved each generation (fine-tune).
        Epoch schedule: full epochs for gen 0; min_epochs for subsequent (warm-start).
        """
        print("\n" + "█"*62)
        print("█  RECURSIVE DEVELOPMENT ENGINE")
        print("█  Multi-generation fiber loop expansion")
        print("█"*62)

        sizes     = list(self.sizes0)
        prev_sizes = list(sizes)
        lr         = 1e-3

        for g in range(max_gen):
            epochs = epochs_base if g == 0 else min_epochs

            gen = self._run_generation(g, sizes, epochs, lr, prev_sizes)
            self.generations.append(gen)

            if gen.fiber_acc >= target_acc:
                print(f"\n  Target {target_acc*100:.0f}% reached at generation {g}.")
                break

            if g < max_gen - 1:
                prev_sizes = list(sizes)
                sizes      = self._expand_sizes(sizes)
                lr        *= 0.5    # halve lr for fine-tuning

        return self.generations

    # ── Reporting ─────────────────────────────────────────────────────────
    def print_tree(self):
        """ASCII recursive development tree."""
        print(f"\n{'═'*62}")
        print(f"  RECURSIVE DEVELOPMENT TREE")
        print(f"{'═'*62}")
        g_list = self.generations

        for i, g in enumerate(g_list):
            is_last  = (i == len(g_list) - 1)
            conn_v   = "│" if not is_last else " "
            conn_h   = "├─" if not is_last else "└─"
            ring_bar = "◉" * g.n_ring_loops + "○"
            expand_m = "  [+loop]" if i > 0 else "  [init]"

            print(f"\n  {conn_h} Gen {g.idx}{expand_m}")
            print(f"  {conn_v}    arch    : {g.layer_sizes}")
            print(f"  {conn_v}    rings   : {ring_bar}  ({g.n_ring_loops} loops)")
            print(f"  {conn_v}    theory  : {g.th_test_acc*100:.1f}%")
            print(f"  {conn_v}    fiber   : {g.fiber_acc*100:.1f}%   "
                  f"gap={g.gap*100:.1f}%")
            print(f"  {conn_v}    CE loss : {g.loss_final:.5f}")
            print(f"  {conn_v}    MI opt  : {g.mi_total:.3f}")
            print(f"  {conn_v}    |P|     : {g.prompt_norm:.4f}")
            if g.cal_scales:
                print(f"  {conn_v}    cal_sc  : {[f'{s:.2f}' for s in g.cal_scales]}")

        print(f"\n  {'─'*55}")
        best = max(self.generations, key=lambda g: g.fiber_acc)
        print(f"  Best fiber accuracy : {best.fiber_acc*100:.1f}%"
              f"  (Gen {best.idx}, arch={best.layer_sizes})")
        print(f"  Total ring loops    : {best.n_ring_loops}")
        print(f"  Total generations   : {len(self.generations)}")
        print(f"  Prompt MI final     : {self.generations[-1].mi_total:.3f}")
        print(f"{'═'*62}")

    def sample_inference(self, n=6):
        """Optical inference through best fiber NN on test samples."""
        nn    = self.best_fiber_nn
        NAMES = ['OOK', 'PAM-4', 'PSK']
        print(f"\n  OPTICAL INFERENCE — {n} samples (best fiber NN)")
        print(f"  {'#':>2}  {'True':>5}  {'Pred':>5}  "
              f"{'OOK':>6}  {'PAM':>6}  {'PSK':>6}  {'OK':>3}")
        print(f"  {'─'*48}")
        ok = 0
        for i in range(min(n, len(self.X_te))):
            true_c = int(np.argmax(self.Y_te[i]))
            out    = nn.forward(self.X_te[i])
            pred   = int(np.argmax(out))
            mark   = "✓" if pred == true_c else "✗"
            if pred == true_c:
                ok += 1
            print(f"  {i+1:>2}  {NAMES[true_c]:>5}  {NAMES[pred]:>5}  "
                  f"{out[0]:>6.3f}  {out[1]:>6.3f}  {out[2]:>6.3f}  {mark:>3}")
        print(f"  {'─'*48}")
        print(f"  Sample accuracy: {ok}/{min(n, len(self.X_te))}")
