"""
recursive_dev_v2.py — Recursive Development Engine V2
======================================================
Level 2 of recursive development. Three upgrades over V1:

  1. Coherent optical neurons (CoherentFiberNetwork)
       Single-pass IQ modulation + homodyne detection.
       No pos/neg decomposition → no approximation gap.
       Phase-encoded weights: w<0 → π phase shift.

  2. Meta-meta prompt (3-level hierarchy)
       P₀ (seed) → Ψ → (M_W1, M_W2) → W
       The generator is itself generated.
       phi_1 (optical memory) accumulates across generations.

  3. 5-class MFR: OOK / PAM-4 / PAM-8 / QPSK / QAM-16
       8 features, harder classification, more optical ring loops needed.

Expansion strategy (V2)
────────────────────────
  V1 expanded width by doubling hidden units.
  V2 expands BOTH width AND depth:
    - Every even generation: add 1 fiber loop (depth)
    - Every odd  generation: widen the narrowest hidden layer (width)
  This gives a more balanced architecture growth.

  Architecture growth example (5-class, n_in=8, n_out=5):
    Gen 0: [8, 12, 5]
    Gen 1: [8, 16, 5]       (width ↑)
    Gen 2: [8, 16, 12, 5]   (depth ↑, new loop)
    Gen 3: [8, 16, 16, 5]   (width ↑)
    Gen 4: [8, 16, 16, 12, 5] (depth ↑)
    ...
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from coherent_nn import (CoherentFiberNetwork, minmax_norm)
from recursive_prompt import MetaMetaPrompt
from recursive_dev import CrossEntropyTrainer


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
@dataclass
class GenerationV2:
    idx:          int
    layer_sizes:  List[int]
    th_train_acc: float
    th_test_acc:  float
    fiber_acc:    float
    gap:          float
    loss_final:   float
    prompt_norm:  float
    phi1_norm:    float           # level-1 memory norm
    mi_total:     float
    n_ring_loops: int
    expand_type:  str             # 'init' | 'depth' | 'width'
    lambda_total: int
    cal_scales:   List[float] = field(default_factory=list)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class RecursiveDevelopmentEngineV2:
    """
    V2 recursive development with coherent neurons + meta-meta prompt.

    Key differences from V1:
      • Uses CoherentFiberNetwork (no decomposition gap)
      • Uses MetaMetaPrompt (3-level hierarchy)
      • Alternates depth/width expansion
      • Per-neuron output calibration (finer than per-layer)
      • LR schedule: cosine decay within each generation
    """

    CLASS_NAMES = ['OOK', 'PAM-4', 'PAM-8', 'QPSK', 'QAM-16']

    def __init__(self, initial_layer_sizes: List[int],
                 meta_prompt: MetaMetaPrompt,
                 X_train, Y_train, X_test, Y_test,
                 noisy: bool = False,
                 rng_seed: int = 0):
        self.sizes0   = list(initial_layer_sizes)
        self.mp       = meta_prompt
        self.X_tr     = X_train
        self.Y_tr     = Y_train
        self.X_te     = X_test
        self.Y_te     = Y_test
        self.noisy    = noisy
        self.rng      = np.random.default_rng(rng_seed)

        self.generations: List[GenerationV2] = []
        self.best_fiber_acc  = 0.0
        self.best_fiber_nn: Optional[CoherentFiberNetwork] = None
        self.prev_trainer: Optional[CrossEntropyTrainer] = None

    # ── Architecture expansion ────────────────────────────────────────────
    def _expand_depth(self, sizes: List[int]) -> Tuple[List[int], str]:
        """Insert a new hidden layer (same width as last hidden) before output."""
        if len(sizes) < 3:
            return sizes, 'depth'
        width = sizes[-2]
        return sizes[:-1] + [width, sizes[-1]], 'depth'

    def _expand_width(self, sizes: List[int]) -> Tuple[List[int], str]:
        """Widen the narrowest hidden layer by 50% (rounded up)."""
        if len(sizes) < 3:
            return sizes, 'width'
        hiddens = sizes[1:-1]
        idx_min = int(np.argmin(hiddens)) + 1   # +1 for input offset
        new_sizes = list(sizes)
        new_sizes[idx_min] = max(new_sizes[idx_min] + new_sizes[idx_min] // 2, 8)
        # Fix adjacent layer's input dim
        if idx_min + 1 < len(new_sizes):
            # Next layer must accept widened output
            pass   # weight warm-start handles dimension mismatch
        return new_sizes, 'width'

    def _next_expansion(self, gen_idx: int,
                        sizes: List[int]) -> Tuple[List[int], str]:
        """Alternate depth/width. Even gens → depth, odd → width."""
        if gen_idx % 2 == 0:
            return self._expand_depth(sizes)
        else:
            return self._expand_width(sizes)

    # ── Warm-start ────────────────────────────────────────────────────────
    def _warmstart(self, trainer: CrossEntropyTrainer,
                   prev: Optional[CrossEntropyTrainer],
                   prev_sizes: List[int]):
        if prev is None:
            return
        new_sizes  = trainer.sizes
        n_new      = len(trainer.W)
        n_old      = len(prev.W)

        if new_sizes == prev_sizes:
            for i in range(n_new):
                trainer.W[i] = prev.W[i].copy()
                trainer.b[i] = prev.b[i].copy()
            return

        # General warm-start: copy matching dims, identity/random for new
        for i in range(min(n_new, n_old)):
            Wn_shape = trainer.W[i].shape
            Wo       = prev.W[i]
            bn_shape = trainer.b[i].shape
            bo       = prev.b[i]

            # Copy overlapping block
            r_copy = min(Wn_shape[0], Wo.shape[0])
            c_copy = min(Wn_shape[1], Wo.shape[1])
            trainer.W[i][:r_copy, :c_copy] = Wo[:r_copy, :c_copy]
            # New rows/cols: identity block + tiny noise
            if Wn_shape[0] > Wo.shape[0]:
                extra = Wn_shape[0] - Wo.shape[0]
                new_rows = np.zeros((extra, Wn_shape[1]))
                sz = min(extra, Wn_shape[1])
                new_rows[:sz, :sz] = np.eye(sz) * 0.1
                trainer.W[i][Wo.shape[0]:, :] = new_rows
            if Wn_shape[1] > Wo.shape[1]:
                extra = Wn_shape[1] - Wo.shape[1]
                trainer.W[i][:, Wo.shape[1]:] = self.rng.normal(
                    0, 0.01, (Wn_shape[0], extra))
            trainer.W[i] += self.rng.normal(0, 1e-4, Wn_shape)

            b_copy = min(bn_shape[0], bo.shape[0])
            trainer.b[i][:b_copy] = bo[:b_copy]

        # New layers (inserted): identity or master-prompt init
        for i in range(n_old, n_new):
            n_o, n_i = trainer.W[i].shape
            # Identity block init — optical pass-through at insertion
            W_id = np.zeros((n_o, n_i))
            sz   = min(n_o, n_i)
            W_id[:sz, :sz] = np.eye(sz) * 0.1
            trainer.W[i] = W_id + self.rng.normal(0, 0.01, (n_o, n_i))
            trainer.b[i] = np.zeros(n_i if len(trainer.b[i]) == n_i
                                    else n_o)

    # ── Calibration (per-neuron) ──────────────────────────────────────────
    def _calibrate(self, fiber_nn: CoherentFiberNetwork,
                   trainer: CrossEntropyTrainer) -> List[float]:
        """
        Per-neuron output calibration for coherent fiber NN.

        CoherentFiberNeuron computes exact dot product analytically, so
        the gap vs theoretic comes only from:
          • ReLU in theoretic (non-linearity) vs identity in fiber
          • Different input normalisation paths

        Calibration: for each hidden layer, scale W so that mean
        absolute hidden activation matches theoretic mean absolute pre-activation.
        """
        scales = []
        n_hidden = len(fiber_nn.layers) - 1

        th_z = [[] for _ in range(n_hidden)]
        fi_h = [[] for _ in range(n_hidden)]

        for x in self.X_tr:
            x_n = minmax_norm(x)
            trainer.forward(x_n)
            for i in range(n_hidden):
                th_z[i].append(float(np.mean(np.abs(trainer._z[i]))))
            fiber_nn.forward(x)
            for i in range(n_hidden):
                fi_h[i].append(float(np.mean(np.abs(
                    fiber_nn._layer_outputs[i + 1]))) + 1e-8)

        for i in range(n_hidden):
            th_mean = float(np.mean(th_z[i]))
            fi_mean = float(np.mean(fi_h[i]))
            sc = float(np.clip(th_mean / (fi_mean + 1e-8), 0.1, 8.0))
            fiber_nn.layers[i].W *= sc
            scales.append(sc)
        return scales

    # ── Meta-prompt update ────────────────────────────────────────────────
    def _meta_update(self, acc: float, prev_acc: float):
        """
        Fitness-proportional (1+1)-ES on M_W1/M_W2 of the meta-meta prompt.
        Larger improvement → larger reinforcement step.
        Regression → guided exploration (perturb toward higher-MI direction).
        """
        delta = acc - prev_acc
        mp    = self.mp

        if delta >= 0:
            step = min(delta * 8.0, 0.15)
            # Reinforce: move M_W1/M_W2 toward their current gradient
            mp.M_W1 = np.tanh(mp.M_W1 * (1 + step))
            mp.M_W2 = np.tanh(mp.M_W2 * (1 + step))
        else:
            # Explore: perturb in direction of P₀ (back toward universal prior)
            sigma = min(abs(delta) * 3.0 + 0.02, 0.08)
            mp.M_W1 += self.rng.normal(0, sigma, mp.M_W1.shape)
            mp.M_W2 += self.rng.normal(0, sigma, mp.M_W2.shape)

        # Renormalise
        mp.M_W1 /= (np.linalg.norm(mp.M_W1) / mp.d + 1e-8)
        mp.M_W2 /= (np.linalg.norm(mp.M_W2) / mp.d + 1e-8)

    # ── Cosine LR schedule ────────────────────────────────────────────────
    @staticmethod
    def _cosine_lr(lr_max: float, lr_min: float, epoch: int, T: int) -> float:
        return lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos(np.pi * epoch / T))

    # ── Single generation ─────────────────────────────────────────────────
    def _run_generation(self, gen_idx: int, sizes: List[int],
                        epochs: int, lr_max: float,
                        prev_sizes: List[int],
                        expand_type: str) -> GenerationV2:

        n_classes = sizes[-1]
        print(f"\n  {'─'*60}")
        print(f"  GENERATION {gen_idx}  [{expand_type}]  arch = {sizes}")
        print(f"  {'─'*60}")

        # ── Train ─────────────────────────────────────────────────────────
        trainer = CrossEntropyTrainer(sizes, lr=lr_max, l2=5e-5)
        self._warmstart(trainer, self.prev_trainer, prev_sizes)

        print(f"  [train]  epochs={epochs}  lr_max={lr_max:.0e}  "
              f"cosine-decay  CE+L2")
        losses = []
        best_loss = float('inf')
        for ep in range(epochs):
            # Cosine LR decay
            lr_ep = self._cosine_lr(lr_max, lr_max * 0.01, ep, epochs)
            trainer.lr = lr_ep
            idx = np.random.default_rng(ep + gen_idx * 10000).permutation(
                len(self.X_tr))
            ep_loss = 0.0
            for i in idx:
                x_n   = minmax_norm(self.X_tr[i])
                y_hat = trainer.forward(x_n)
                ep_loss += trainer._ce(y_hat, self.Y_tr[i])
                gW, gb = trainer.backward_ce(self.Y_tr[i])
                trainer._adam(gW, gb)
            ep_loss /= len(self.X_tr)
            losses.append(ep_loss)
            if ep_loss < best_loss:
                best_loss = ep_loss
            if (ep % max(1, epochs // 5) == 0) or ep == epochs - 1:
                print(f"    Epoch {ep+1:>4}/{epochs}  CE={ep_loss:.5f}"
                      f"  lr={lr_ep:.2e}")

        th_tr = trainer.evaluate(self.X_tr, self.Y_tr)
        th_te = trainer.evaluate(self.X_te, self.Y_te)
        print(f"  [theory] train={th_tr*100:.1f}%  test={th_te*100:.1f}%")

        # ── Build coherent fiber NN ────────────────────────────────────────
        shapes = [(sizes[i+1], sizes[i]) for i in range(len(sizes)-1)]
        _, encs = self.mp.recursive_unfold(shapes)

        fiber_nn = CoherentFiberNetwork(sizes, noisy=self.noisy)
        fiber_nn.load_trained_weights(trainer.W, trainer.b)
        fiber_nn.optical_encodings = encs

        pre_cal = fiber_nn.evaluate(self.X_te, self.Y_te)
        print(f"  [fiber]  pre-cal  = {pre_cal*100:.1f}%")

        cal_scales = self._calibrate(fiber_nn, trainer)
        post_cal   = fiber_nn.evaluate(self.X_te, self.Y_te)
        print(f"  [fiber]  post-cal = {post_cal*100:.1f}%"
              f"  scales={[f'{s:.2f}' for s in cal_scales]}")

        gap = th_te - post_cal

        # ── MI of optical encodings ────────────────────────────────────────
        mi_total = sum(self.mp.embedder.mutual_information_proxy(e)
                       for e in encs)

        # ── Meta-prompt update ────────────────────────────────────────────
        self._meta_update(post_cal, self.best_fiber_acc)

        if post_cal > self.best_fiber_acc:
            self.best_fiber_acc = post_cal
            self.best_fiber_nn  = fiber_nn
            self.prev_trainer   = trainer

        return GenerationV2(
            idx          = gen_idx,
            layer_sizes  = list(sizes),
            th_train_acc = th_tr,
            th_test_acc  = th_te,
            fiber_acc    = post_cal,
            gap          = gap,
            loss_final   = losses[-1],
            prompt_norm  = float(np.linalg.norm(self.mp.P0)),
            phi1_norm    = float(np.linalg.norm(self.mp.phi_1)),
            mi_total     = mi_total,
            n_ring_loops = len(sizes) - 2,
            expand_type  = expand_type,
            lambda_total = fiber_nn.total_lambda_channels(),
            cal_scales   = cal_scales,
        )

    # ── Main loop ─────────────────────────────────────────────────────────
    def run(self, max_gen: int = 6, target_acc: float = 0.90,
            epochs_base: int = 600, min_epochs: int = 300,
            lr_base: float = 1e-3) -> List[GenerationV2]:

        print("\n" + "█"*62)
        print("█  RECURSIVE DEVELOPMENT ENGINE  V2")
        print("█  Coherent Optical  ·  Meta-Meta Prompt  ·  5-Class MFR")
        print("█"*62)

        sizes      = list(self.sizes0)
        prev_sizes = list(sizes)
        lr         = lr_base
        expand_type = 'init'

        for g in range(max_gen):
            epochs = epochs_base if g == 0 else min_epochs
            gen = self._run_generation(g, sizes, epochs, lr,
                                       prev_sizes, expand_type)
            self.generations.append(gen)

            if gen.fiber_acc >= target_acc:
                print(f"\n  ✓ Target {target_acc*100:.0f}% reached at Gen {g}.")
                break

            if g < max_gen - 1:
                prev_sizes = list(sizes)
                sizes, expand_type = self._next_expansion(g + 1, sizes)
                lr = lr_base * (0.5 ** min(g + 1, 4))

        return self.generations

    # ── Reporting ─────────────────────────────────────────────────────────
    def print_tree(self):
        gens = self.generations
        print(f"\n{'═'*62}")
        print(f"  RECURSIVE DEVELOPMENT TREE  (V2 · Coherent · MetaMeta)")
        print(f"{'═'*62}")

        SYMBOLS = {'init': '◈', 'depth': '⊕', 'width': '⊞'}

        for i, g in enumerate(gens):
            last   = (i == len(gens) - 1)
            branch = "└─" if last else "├─"
            pad    = "  " if last else "│ "
            sym    = SYMBOLS.get(g.expand_type, '·')
            loops  = "◉" * g.n_ring_loops + "○"

            print(f"\n  {branch} Gen {g.idx} {sym} [{g.expand_type}]")
            print(f"  {pad}   arch     : {g.layer_sizes}")
            print(f"  {pad}   loops    : {loops}  ({g.n_ring_loops} coherent rings)")
            print(f"  {pad}   theory   : train={g.th_train_acc*100:.1f}%"
                  f"  test={g.th_test_acc*100:.1f}%")
            print(f"  {pad}   fiber    : {g.fiber_acc*100:.1f}%"
                  f"   gap={g.gap*100:+.1f}%")
            print(f"  {pad}   CE loss  : {g.loss_final:.5f}")
            print(f"  {pad}   λ-ch     : {g.lambda_total}")
            print(f"  {pad}   MI Σ     : {g.mi_total:.3f}")
            print(f"  {pad}   |φ₁|    : {g.phi1_norm:.4f}  (L1 memory)")
            if g.cal_scales:
                print(f"  {pad}   cal_sc   : {[f'{s:.2f}' for s in g.cal_scales]}")

        print(f"\n  {'─'*55}")
        best = max(gens, key=lambda g: g.fiber_acc)
        final = gens[-1]
        print(f"  Best fiber acc    : {best.fiber_acc*100:.1f}%"
              f"  (Gen {best.idx}  arch={best.layer_sizes})")
        print(f"  Final ring loops  : {final.n_ring_loops}")
        print(f"  Final λ-channels  : {final.lambda_total}")
        print(f"  Total generations : {len(gens)}")
        print(f"{'═'*62}")

    def sample_inference(self, n: int = 8):
        nn    = self.best_fiber_nn
        n_cls = nn.layer_sizes[-1]
        NAMES = self.CLASS_NAMES[:n_cls]
        print(f"\n  OPTICAL INFERENCE  ({n} samples  ·  coherent detection)")
        w = max(len(nm) for nm in NAMES)
        hdr = "  ".join(f"{nm:>{w}}" for nm in NAMES)
        print(f"  {'#':>2}  {'True':>{w}}  {'Pred':>{w}}  {hdr}  OK")
        print(f"  {'─'*60}")
        ok = 0
        for i in range(min(n, len(self.X_te))):
            true_c = int(np.argmax(self.Y_te[i]))
            out    = nn.forward(self.X_te[i])
            pred   = int(np.argmax(out))
            mark   = "✓" if pred == true_c else "✗"
            if pred == true_c:
                ok += 1
            probs  = "  ".join(f"{v:>{w}.3f}" for v in out)
            print(f"  {i+1:>2}  {NAMES[true_c]:>{w}}  {NAMES[pred]:>{w}}  {probs}  {mark}")
        print(f"  {'─'*60}")
        print(f"  Sample accuracy : {ok}/{min(n, len(self.X_te))}")

    def accuracy_table(self):
        print(f"\n{'═'*62}")
        print(f"  GENERATION SUMMARY")
        print(f"{'═'*62}")
        print(f"  {'G':>2}  {'Type':>5}  {'Arch':>26}  "
              f"{'Thy':>6}  {'Fib':>6}  {'Gap':>6}  {'λ':>4}")
        print(f"  {'─'*60}")
        for g in self.generations:
            arch = str(g.layer_sizes)
            print(f"  {g.idx:>2}  {g.expand_type:>5}  {arch:>26}  "
                  f"{g.th_test_acc*100:>5.1f}%  "
                  f"{g.fiber_acc*100:>5.1f}%  "
                  f"{g.gap*100:>+5.1f}%  "
                  f"{g.lambda_total:>4}")
