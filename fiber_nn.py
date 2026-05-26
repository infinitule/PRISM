"""
fiber_nn.py — Virtual Fiber Optical Neural Network
===================================================
Implements the complete FNN scheme from Zang et al. (iOptics 2025):

  Architecture:
    Input branch  → laser + SMF stretch + EDFA
    Computing ring → N loops, each loop = one FC layer:
                      AM1(x) · AM2(W_row) → DCF compression → dot product
    Output branch → PD detection + DSP readout

  Positive/negative weight decomposition (§3, Eq. 5):
    y = (W⁺x⁺ + W⁻x⁻) - (W⁺x⁻ + W⁻x⁺)   ← handles negative weights
    physically: run 4 passes per layer; subtract result pairs at PD output

  Normalization: MinMax → [-1, 1] before entering the ring (§2).
"""

import numpy as np
from fiber_physics import (
    SingleModeFiber, EDFA, AmplitudeModulator,
    DispersionCompensatingFiber, Photodetector,
    SMF_BETA2, SMF_LENGTH, DCF_BETA2, DCF_LENGTH,
)


# ── Utility ──────────────────────────────────────────────────────────────────
def minmax_norm(x, low=-1.0, high=1.0):
    """Map array to [low, high]  (paper §2)."""
    x = np.asarray(x, dtype=float)
    mn, mx = x.min(), x.max()
    if mx == mn:
        return np.zeros_like(x)
    return (x - mn) / (mx - mn) * (high - low) + low


def relu(x):    return np.maximum(0.0, x)
def softmax(x): e = np.exp(x - x.max()); return e / e.sum()
def identity(x): return x


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class FiberNeuron:
    """
    One optical neuron computing a single dot product w·x via:
      1. Laser generates flat-topped stretched pulse (carrier)
      2. AM1 modulates input segments  (x⁺ or x⁻)
      3. AM2 modulates weight segments (w⁺ or w⁻)
      4. DCF compresses → E = γ·Σ(wᵢxᵢ)   (Eq. 7)
      5. PD converts to photocurrent

    Positive/negative decomposition handled by FiberLayer.
    """

    def __init__(self, n_inputs, smf=None, edfa=None, am=None,
                 dcf=None, pd=None):
        self.n   = n_inputs
        self.smf = smf  or SingleModeFiber()
        self.edfa= edfa or EDFA()
        self.am  = am   or AmplitudeModulator()
        self.dcf = dcf  or DispersionCompensatingFiber()
        self.pd  = pd   or Photodetector()

    def _optical_dot(self, w_pos, x_pos, E0=1.0, noisy=False, rng=None):
        """
        Positive-only dot product via fiber optics (Eq. 6 & 7, Zang et al.).

        Physical pipeline:
          Laser → SMF(stretch) → EDFA → AM1(w) · AM2(x) → DCF(compress) → E_out

        Reference-pulse calibration (paper §4):
          The paper measures un-modulated reference pulses and uses them to compute
          γ in Eq. 7, which normalises out E_base (attenuation + gain factors).
          We replicate this by dividing the compressed energy by E_base so that:
              E_out = γ · Σ(wᵢ · xᵢ)   with γ=1 (calibrated units)

        Per-component scale factors:
          AM physically clips to [0, 1], so we normalise w and x before modulation
          and restore scale after DCF compression.
        """
        # Per-component scale factors — AM constraint: signals must be in [0, 1]
        # Scale is restored after DCF (paper §3 mapping procedure)
        w_scale = float(w_pos.max()) + 1e-12
        x_scale = float(x_pos.max()) + 1e-12
        w_norm  = w_pos / w_scale
        x_norm  = x_pos / x_scale

        # SMF attenuation (time-stretch physics; E_base measured as reference)
        E_att  = self.smf.attenuate(complex(E0))

        # EDFA amplification (±ASE noise)
        if noisy and rng is not None:
            E_amp = self.edfa.amplify(E_att, rng=rng, noisy=True)
        else:
            E_amp = self.edfa.amplify_noiseless(E_att)

        # E_base: reference level — calibrated out in paper (Eq. 7, coefficient γ)
        E_base = float(np.abs(E_amp))
        if E_base < 1e-15:
            E_base = 1e-15    # prevent division by zero in calibration step

        # AM modulation: each segment = w_norm[i] · x_norm[i] · E_base  (Eq. 6)
        segments = np.array([
            self.am.modulate(self.am.modulate(E_base, w_norm[i]), x_norm[i])
            for i in range(len(w_pos))
        ])

        # DCF compression: raw = γ · E_base · Σ(w_norm_i · x_norm_i)  (Eq. 7)
        raw = self.dcf.dot_product(segments, np.ones(len(segments)))

        # Reference-pulse calibration: divide by E_base to obtain calibrated dot product
        # Equivalent to computing γ from un-modulated reference pulses (paper Fig. 3d/3e)
        calibrated = raw / E_base

        # Restore original weight and input scales
        return calibrated * w_scale * x_scale

    def compute(self, w, x, E0=1.0, noisy=False, rng=None):
        """
        Full optical dot product with positive/negative decomposition (Eq. 5).
        Returns scalar: γ·Σ(wᵢxᵢ)
        """
        w_pos = np.maximum( w, 0.0)
        w_neg = np.maximum(-w, 0.0)
        x_pos = np.maximum( x, 0.0)
        x_neg = np.maximum(-x, 0.0)

        kw = dict(E0=E0, noisy=noisy, rng=rng)
        pp = self._optical_dot(w_pos, x_pos, **kw)
        nn = self._optical_dot(w_neg, x_neg, **kw)
        pn = self._optical_dot(w_pos, x_neg, **kw)
        np_ = self._optical_dot(w_neg, x_pos, **kw)

        # y = (W⁺x⁺ + W⁻x⁻) - (W⁺x⁻ + W⁻x⁺)  ← Eq. 5
        return (pp + nn) - (pn + np_)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class FiberLayer:
    """
    One loop of the optical computing ring = one fully-connected layer.
    N_out neurons run in parallel; light travels ring once per layer.

    The fiber system becomes distributed computational transmission media
    (as stated in paper §1): compute while transmitting.
    """

    def __init__(self, n_in, n_out, weights=None, bias=None, activation=None):
        self.n_in  = n_in
        self.n_out = n_out
        self.W = (weights if weights is not None
                  else np.random.randn(n_out, n_in) * np.sqrt(2.0 / n_in))
        self.b = bias if bias is not None else np.zeros(n_out)
        self.activation = activation
        self.neurons = [FiberNeuron(n_in) for _ in range(n_out)]

    def forward(self, x, noisy=False, rng=None):
        """Optical matrix-vector multiply: y = activation(W·x + b)."""
        out = np.array([
            self.neurons[i].compute(self.W[i], x, noisy=noisy, rng=rng) + self.b[i]
            for i in range(self.n_out)
        ])
        if self.activation is not None:
            out = self.activation(out)
        return out

    def set_weights(self, W, b=None):
        self.W = np.asarray(W, dtype=float).copy()
        if b is not None:
            self.b = np.asarray(b, dtype=float).copy()

    @property
    def weight_norm(self):      return float(np.linalg.norm(self.W))

    @property
    def sparsity(self):         return float(np.mean(np.abs(self.W) < 1e-4))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class FiberNeuralNetwork:
    """
    Complete virtual Fiber Optical Neural Network.

    Physical pipeline (paper Fig. 2):
      Input branch:    Laser → SMF(stretch) → EDFA
      Computing ring:  [AM1(x), AM2(W)] → DCF → repeat N-1 times
      Output branch:   PD → DSP (softmax)

    The network can operate in two modes:
      • noiseless: ideal optical computation
      • noisy:     EDFA (ASE) + PD (shot/thermal) noise injected

    Weights are loaded from a pre-trained electronic model (TheoreticalTrainer),
    then transferred to the optical frame — mirroring the paper's 3-step pipeline.
    """

    def __init__(self, layer_sizes, activations=None, noisy=False):
        """
        layer_sizes : [n_input, n_hidden..., n_output]
        activations : list of callables, length = number of layers
        noisy       : inject physical fiber noise
        """
        self.layer_sizes = layer_sizes
        self.noisy       = noisy
        self.rng         = np.random.default_rng(0)
        self.optical_encodings = []

        n_layers = len(layer_sizes) - 1

        if activations is None:
            # Paper uses no nonlinear activation for simplicity; we add softmax at output
            activations = [None] * (n_layers - 1) + [softmax]

        self.layers = [
            FiberLayer(layer_sizes[i], layer_sizes[i + 1],
                       activation=activations[i])
            for i in range(n_layers)
        ]

    # ── Forward pass ──────────────────────────────────────────────────────
    def forward(self, x_raw):
        """
        Full optical inference:
          1. MinMax normalise to [-1, 1]
          2. Each ring loop (FiberLayer) computes one FC transform
          3. Output branch: softmax
        """
        x = minmax_norm(np.asarray(x_raw, dtype=float))
        h = x
        self._layer_outputs = [x]
        for layer in self.layers:
            h = layer.forward(h, noisy=self.noisy, rng=self.rng)
            self._layer_outputs.append(h)
        return h

    def predict(self, x_raw):
        return int(np.argmax(self.forward(x_raw)))

    def evaluate(self, X, Y):
        correct = sum(
            self.predict(X[i]) == np.argmax(Y[i])
            for i in range(len(X))
        )
        return correct / len(X)

    # ── Weight transfer (paper §3, final step) ────────────────────────────
    def load_trained_weights(self, weights_list, biases_list=None):
        """
        Map optimised theoretic weights → optical fiber NN frame.
        This is the final step in the paper's 3-step pipeline.
        """
        for i, layer in enumerate(self.layers):
            W = weights_list[i]
            b = biases_list[i] if biases_list else None
            layer.set_weights(W, b)

    # ── Inspection ────────────────────────────────────────────────────────
    def optical_state_table(self):
        rows = []
        for i, layer in enumerate(self.layers):
            enc = self.optical_encodings[i] if i < len(self.optical_encodings) else None
            n_lambda = enc['wavelengths'].shape[0] if enc else 0
            rows.append({
                'layer':    i,
                'shape':    (layer.n_out, layer.n_in),
                'W_norm':   layer.weight_norm,
                'sparsity': layer.sparsity,
                'lambda_channels': n_lambda,
            })
        return rows

    def ring_loops(self):
        """Number of times light travels the optical ring = N_layers - 1."""
        return len(self.layers) - 1
