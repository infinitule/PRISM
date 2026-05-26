"""
recursive_prompt.py — Recursive Master Prompt with Synaptic Optical Embedding
==============================================================================
The unifying idea:

  In a fiber neural network, information IS the light.
  Computation IS propagation through the medium.
  Synaptic weights ARE amplitude modulations encoded on photons.

  This module creates a "master prompt" P₀ — a seed tensor that:
    1. Generates weight matrices for each fiber layer via a meta-network
    2. Encodes those weights as optical parameters (amplitude, phase, λ-channel)
    3. Recursively updates its own context after each layer generation:
           P_{n+1} = f(P_n, W_n)     ← light carries memory forward
    4. Can self-expand the architecture when performance is insufficient

  The recursion is isomorphic to light propagating through successive
  fiber loops: each loop transforms the signal AND is transformed by it.

  AGI relevance:
    Recursive self-referential weight generation → meta-learning
    Optical encoding of parameters → hardware-substrate co-design
    Self-expansion → architectural self-improvement
    Fibonacci seed → broad spectral coverage (maximal information density)
"""

import numpy as np

# WDM channel spacing (Dense WDM, 100 GHz ≈ 0.8 nm @ 1550 nm)
LAMBDA_C      = 1550e-9
DELTA_LAMBDA  = 0.8e-9


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class SynapticEmbedder:
    """
    Maps a synaptic weight matrix W ∈ ℝ^(n_out × n_in) to optical parameters:

      amplitude[i,j]  = |W[i,j]| / max|W|          ∈ [0, 1]
                        → intensity modulation depth on AM

      phase[i,j]      = 0   if W[i,j] ≥ 0           (constructive)
                      = π   if W[i,j]  < 0           (destructive — Eq.5)

      wavelength[i]   = λ_c + i·Δλ                  (WDM, one λ per output neuron)

    Decoding reconstructs W exactly from (amplitude, phase, scale).
    """

    def encode(self, W):
        W = np.asarray(W, dtype=float)
        scale = np.abs(W).max() + 1e-12
        amplitude = np.abs(W) / scale
        phase = np.where(W >= 0, 0.0, np.pi)
        wavelengths = np.array([
            LAMBDA_C + i * DELTA_LAMBDA for i in range(W.shape[0])
        ])
        return {
            'amplitude':   amplitude,
            'phase':       phase,
            'wavelengths': wavelengths,
            'scale':       scale,
            'shape':       W.shape,
        }

    def decode(self, enc):
        sign = np.where(enc['phase'] == 0, 1.0, -1.0)
        return sign * enc['amplitude'] * enc['scale']

    def mutual_information_proxy(self, enc):
        """Shannon entropy of amplitude distribution as MI proxy."""
        p = enc['amplitude'].flatten()
        p = p / (p.sum() + 1e-12)
        return -np.sum(p * np.log(p + 1e-12))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class RecursiveMasterPrompt:
    """
    Recursive prompt engine — the cognitive seed of the fiber AGI.

    State:   P ∈ ℝ^d   (the "master prompt vector")
    Action:  W_n = MetaGen(P_n)           [weight generation]
    Update:  P_{n+1} = Propagate(P_n, W_n) [context propagation]

    MetaGen is a small 2-layer tanh network whose parameters are fixed at
    initialisation from a Fibonacci-derived seed — ensuring broad harmonic
    coverage (analogous to a wideband light source in photonics).

    The Fibonacci seed is chosen because:
      • Its frequency spectrum is maximally spread (golden ratio spacing)
      • It is self-similar under recursion  (Fib_n = Fib_{n-1} + Fib_{n-2})
      • These properties mirror AGI desiderata: generality + self-reference
    """

    def __init__(self, seed_dim=64, max_depth=8, rng_seed=42):
        self.d         = seed_dim
        self.max_depth = max_depth
        self.rng       = np.random.default_rng(rng_seed)
        self.embedder  = SynapticEmbedder()
        self.history   = []          # generation log

        self._init_meta_generator()
        self.P0 = self._fibonacci_seed()  # canonical starting prompt

    # ── Meta-generator initialisation ────────────────────────────────────
    def _init_meta_generator(self):
        d = self.d
        # Two-layer tanh meta-network (fixed, not trained)
        # Weights drawn from orthogonal init for maximal rank
        A = self.rng.normal(0, 1, (d, d))
        Q1, _ = np.linalg.qr(A)
        A = self.rng.normal(0, 1, (d, d))
        Q2, _ = np.linalg.qr(A)
        self.M_W1 = Q1 / np.sqrt(d)
        self.M_b1 = np.zeros(d)
        self.M_W2 = Q2 / np.sqrt(d)
        self.M_b2 = np.zeros(d)

    def _meta_forward(self, p):
        h = np.tanh(self.M_W1 @ p + self.M_b1)
        return np.tanh(self.M_W2 @ h + self.M_b2)

    # ── Fibonacci seed ────────────────────────────────────────────────────
    def _fibonacci_seed(self):
        """
        Seed vector from normalised Fibonacci numbers + sinusoidal phase.
        Provides maximal spectral diversity — analogous to a broadband
        laser source seeding the optical computing ring.
        """
        d = self.d
        fib = [1, 1]
        while len(fib) < d:
            fib.append(fib[-1] + fib[-2])
        fib = np.array(fib[:d], dtype=float)
        fib /= np.linalg.norm(fib)
        phase = np.sin(2 * np.pi * np.arange(d) * (1 + np.sqrt(5)) / 2 / d)
        return np.tanh(fib + 0.15 * phase)

    # ── Weight matrix generation ──────────────────────────────────────────
    def generate_weight_matrix(self, p, n_out, n_in):
        """
        Project prompt vector p onto (n_out × n_in) weight space.

        Method: generate separate row- and column-direction vectors from
        the prompt, then form their outer product.  This preserves the
        low-rank structure found in real photonic weight matrices while
        allowing arbitrary target shapes.
        """
        d = self.d
        # Two independent projections via rolling the prompt vector
        row_seed = self._meta_forward(p)
        col_seed = self._meta_forward(np.roll(p, d // 3))

        # Tile / slice to target dimensions
        def _project(v, n):
            if n <= d:
                return v[:n]
            reps = n // d + 1
            return np.tile(v, reps)[:n]

        row_v = _project(row_seed, n_out)
        col_v = _project(col_seed, n_in)

        W = np.outer(row_v, col_v)
        # Scale to Xavier-equivalent range
        scale = np.sqrt(2.0 / (n_in + n_out))
        W = W / (np.abs(W).max() + 1e-12) * scale
        return W

    # ── Context propagation ───────────────────────────────────────────────
    def propagate(self, p, W):
        """
        Update prompt after generating W — light is transformed by
        every fiber loop it passes through.

          P_next = tanh( MetaGen(P) + summary(W) )

        summary(W) = row-mean vector, padded/clipped to d.
        """
        summary = W.mean(axis=1)
        d = self.d
        if len(summary) < d:
            summary = np.pad(summary, (0, d - len(summary)))
        else:
            summary = summary[:d]
        return np.tanh(self._meta_forward(p) + 0.5 * summary)

    # ── Recursive unfold ──────────────────────────────────────────────────
    def recursive_unfold(self, layer_shapes, P_init=None):
        """
        Unfold the master prompt across all layers of the network.

        layer_shapes : [(n_out_0, n_in_0), (n_out_1, n_in_1), ...]
        Returns      : (list of W matrices, list of optical encodings)

        Each call appends to self.history — the full recursive trace.
        """
        p = P_init if P_init is not None else self.P0.copy()
        weights, encodings = [], []

        for depth, (n_out, n_in) in enumerate(layer_shapes):
            if depth >= self.max_depth:
                break

            W   = self.generate_weight_matrix(p, n_out, n_in)
            enc = self.embedder.encode(W)
            mi  = self.embedder.mutual_information_proxy(enc)

            self.history.append({
                'depth':   depth,
                'P':       p.copy(),
                'P_norm':  float(np.linalg.norm(p)),
                'W':       W,
                'shape':   (n_out, n_in),
                'MI':      mi,
            })

            weights.append(W)
            encodings.append(enc)
            p = self.propagate(p, W)

        return weights, encodings

    # ── Self-expansion ────────────────────────────────────────────────────
    def self_expand(self, layer_shapes, accuracy, threshold=0.95):
        """
        If accuracy < threshold AND depth budget remains, insert a new
        fiber loop (layer) at the network's bottleneck.

        This is the AGI self-improvement mechanism: the system widens
        its own processing pipeline when capacity is insufficient.

        Returns (new_shapes, expanded: bool).
        """
        if accuracy >= threshold or len(layer_shapes) >= self.max_depth:
            return layer_shapes, False

        # Find bottleneck: smallest n_out
        sizes = [s[0] for s in layer_shapes]
        idx   = int(np.argmin(sizes))
        n_expand = max(sizes[idx] * 2, 8)  # at least double

        new_shapes = list(layer_shapes)
        # Insert new layer after bottleneck
        n_in_new  = new_shapes[idx][0]
        n_out_new = n_expand
        new_shapes.insert(idx + 1, (n_out_new, n_in_new))

        # Fix downstream layer's input dimension
        if idx + 2 < len(new_shapes):
            n_out_next = new_shapes[idx + 2][0]
            new_shapes[idx + 2] = (n_out_next, n_out_new)

        return new_shapes, True

    # ── Reporting ─────────────────────────────────────────────────────────
    def print_generation_trace(self):
        print(f"\n{'═'*62}")
        print(f"  RECURSIVE MASTER PROMPT — GENERATION TRACE")
        print(f"{'═'*62}")
        print(f"  {'Depth':>5}  {'Shape':>10}  {'|P|':>8}  {'MI_proxy':>10}")
        print(f"  {'-'*50}")
        for h in self.history:
            print(f"  {h['depth']:>5}  "
                  f"{str(h['shape']):>10}  "
                  f"{h['P_norm']:>8.4f}  "
                  f"{h['MI']:>10.4f}")
        print(f"\n  Seed dim  : {self.d}")
        print(f"  Max depth : {self.max_depth}")
        print(f"  Layers gen: {len(self.history)}")
        print(f"  |P₀|      : {np.linalg.norm(self.P0):.4f}  (Fibonacci seed)")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class MetaMetaPrompt(RecursiveMasterPrompt):
    """
    3-Level Recursive Prompt Hierarchy — the seed generates its own generator.

    Level 0  P₀ ∈ ℝᵈ          Fibonacci seed  (universal prior)
    Level 1  Φ = Ψ(P₀)        Meta-generator parameters  ← NEW
                               Ψ: fixed random projection, P₀ → (M_W1, M_W2)
    Level 2  W = MetaGen(Φ,P)  Weight matrix from generated generator

    In RecursiveMasterPrompt, M_W1/M_W2 are fixed at init.
    Here, M_W1/M_W2 are themselves generated FROM P₀ at each call.
    This means the seed "knows" not just what to compute, but HOW to learn.

    AGI interpretation:
      Level 0: innate prior (instinct)
      Level 1: learning algorithm (meta-cognition)
      Level 2: task-specific knowledge (cognition)

    The three levels correspond to:
      Fiber hardware → optical computing ring → signal processing
    """

    def __init__(self, seed_dim=64, max_depth=12, rng_seed=42):
        super().__init__(seed_dim=seed_dim, max_depth=max_depth,
                         rng_seed=rng_seed)
        d = seed_dim
        rng = np.random.default_rng(rng_seed + 1)

        # Ψ: fixed random projection matrices P₀ → M_W1, M_W2
        # Shape: (d, d²) — maps d-dim seed to d×d matrix flattened
        # Use block-Hadamard structure for spectral richness
        self._Psi_1 = rng.normal(0, 1.0 / d, (d * d, d))
        self._Psi_2 = rng.normal(0, 1.0 / d, (d * d, d))

        # Level-1 context: evolves across generations (optical "memory")
        self.phi_1 = np.zeros(d)   # accumulates meta-generator updates

    def _level1_generate(self, p):
        """
        Ψ(P) → (M_W1, M_W2): generate meta-generator from seed.
        Returns new M_W1, M_W2 shaped (d, d).
        """
        d = self.d
        mw1_flat = np.tanh(self._Psi_1 @ p)   # shape (d²,)
        mw2_flat = np.tanh(self._Psi_2 @ p)
        M_W1 = mw1_flat.reshape(d, d) / np.sqrt(d)
        M_W2 = mw2_flat.reshape(d, d) / np.sqrt(d)
        return M_W1, M_W2

    def _meta_forward_mm(self, p, M_W1, M_W2):
        """Level-2 meta-forward using generated M_W1/M_W2."""
        h = np.tanh(M_W1 @ p + self.M_b1)
        return np.tanh(M_W2 @ h + self.M_b2)

    def generate_weight_matrix(self, p, n_out, n_in):
        """
        3-level generation:
          P₀ → Ψ → (M_W1, M_W2)       [level 1: meta-cognition]
          P  → MetaGen(M_W1,M_W2) → W  [level 2: cognition]
        """
        # Generate meta-generator from seed P₀ (not from current p)
        M_W1, M_W2 = self._level1_generate(self.P0)

        # Update level-1 context (optical memory across generations)
        self.phi_1 = np.tanh(0.9 * self.phi_1 + 0.1 * M_W1.mean(axis=0))

        d = self.d

        def _project(v, n):
            if n <= d:
                return v[:n]
            return np.tile(v, n // d + 1)[:n]

        row_seed = self._meta_forward_mm(p,               M_W1, M_W2)
        col_seed = self._meta_forward_mm(np.roll(p, d//3), M_W1, M_W2)
        # Inject level-1 context into weight generation (meta-memory)
        row_seed = np.tanh(row_seed + 0.3 * self.phi_1)

        W = np.outer(_project(row_seed, n_out), _project(col_seed, n_in))
        scale = np.sqrt(2.0 / (n_in + n_out))
        return W / (np.abs(W).max() + 1e-12) * scale

    def print_meta_hierarchy(self):
        M_W1, M_W2 = self._level1_generate(self.P0)
        print(f"\n{'═'*62}")
        print(f"  META-META PROMPT — 3-LEVEL HIERARCHY")
        print(f"{'═'*62}")
        print(f"  Level 0  P₀ (Fibonacci seed)")
        print(f"           dim={self.d}  |P₀|={np.linalg.norm(self.P0):.4f}")
        print(f"  Level 1  Ψ(P₀) → (M_W1, M_W2)  [meta-generator]")
        print(f"           |M_W1|={np.linalg.norm(M_W1):.4f}  "
              f"|M_W2|={np.linalg.norm(M_W2):.4f}")
        print(f"           |φ₁| (memory) = {np.linalg.norm(self.phi_1):.4f}")
        print(f"  Level 2  MetaGen(Φ, P) → W  [weight matrix]")
        print(f"           layers generated: {len(self.history)}")
        # Spectral analysis of M_W1
        sv = np.linalg.svd(M_W1, compute_uv=False)
        print(f"           M_W1 rank proxy: {(sv > sv[0]*0.01).sum()}/{self.d}")
        print(f"{'═'*62}")
