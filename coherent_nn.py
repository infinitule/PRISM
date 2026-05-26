"""
coherent_nn.py — Coherent Optical Fiber Neural Network
=======================================================
Upgrade from intensity-only (FiberNeuron) to full coherent detection.

Why coherent beats intensity-only
──────────────────────────────────
  FiberNeuron (intensity):
    • Needs 4 passes for one dot product (pos/neg decomposition)
    • Attenuation + EDFA noise accumulate 4× per layer
    • Cannot encode phase information

  CoherentFiberNeuron (this module):
    • 1 pass per dot product — sign encoded as optical PHASE (0 or π)
    • Homodyne detection recovers real-valued dot product directly
    • Phase-diversity receiver separates overlapping modes cleanly
    • Physically: this is how modern coherent optical comms work (IQ modulator)

Physical chain
──────────────
  CW laser (LO)  ──┐
                   ├── 90° hybrid ──► balanced PD → I_out ∝ Σ wᵢxᵢ
  Signal field   ──┘
  E_sig = Σᵢ |wᵢ|·e^(jφ_w,i) · |xᵢ|·e^(jφ_x,i) · E₀

  Phase encoding:
    wᵢ > 0 → φ_w,i = 0     xᵢ > 0 → φ_x,i = 0
    wᵢ < 0 → φ_w,i = π     xᵢ < 0 → φ_x,i = π

  Coherent sum:   E_total = E₀ · Σᵢ wᵢxᵢ   (complex amplitudes add)
  Homodyne:       I = R · Re(E_total · E_LO*)  ∝  Σᵢ wᵢxᵢ

Key result: single-pass, exact linear dot product, no decomposition gap.
"""

import numpy as np
from fiber_physics import (
    SingleModeFiber, EDFA, DispersionCompensatingFiber,
    Photodetector, LAMBDA_C, C_LIGHT,
)

# ── Utility ──────────────────────────────────────────────────────────────────
def minmax_norm(x, low=-1.0, high=1.0):
    x = np.asarray(x, dtype=float)
    mn, mx = x.min(), x.max()
    if mx == mn:
        return np.zeros_like(x)
    return (x - mn) / (mx - mn) * (high - low) + low

def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()

def relu(x):
    return np.maximum(0.0, x)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class IQModulator:
    """
    In-phase / Quadrature optical modulator.
    Encodes a real-valued synaptic weight as amplitude + phase of E field:
        E_out = |w| · e^(j·phase(w)) · E_in
        phase(w) = 0  if w ≥ 0,   π  if w < 0
    """
    def modulate(self, E_in_complex, w_real):
        amplitude = float(abs(w_real))
        phase     = 0.0 if w_real >= 0 else np.pi
        return amplitude * np.exp(1j * phase) * E_in_complex


class HomodyneReceiver:
    """
    90°-hybrid + balanced photodetector pair.
    Recovers Re(E_signal · E_LO*) — the in-phase component.

    I_out = R · |E_LO| · Re(E_signal · e^{-j·φ_LO})
    With φ_LO = 0 (LO in-phase): I_out ∝ Re(E_signal)
    """
    def __init__(self, responsivity=0.9):
        self.R = responsivity

    def detect(self, E_signal_complex, E_LO_amplitude=1.0, noisy=False, rng=None):
        """Returns calibrated Re(E_signal) — the linear dot product."""
        I = self.R * E_LO_amplitude * float(np.real(E_signal_complex))
        if noisy and rng is not None:
            sigma = 1e-4 * abs(E_LO_amplitude)
            I    += rng.normal(0, sigma)
        return I


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class CoherentFiberNeuron:
    """
    Single optical neuron using coherent (IQ) modulation + homodyne detection.

    Computation in one pass (Σ wᵢxᵢ):
      1. Laser generates CW carrier E₀ (reference: E_LO)
      2. IQ modulator encodes each (wᵢ, xᵢ) pair as complex amplitude:
             E_i = |wᵢ| · e^(jφ_w,i) · |xᵢ| · e^(jφ_x,i) · E₀
      3. Coherent combiner (DCF) sums all E_i:
             E_total = E₀ · Σᵢ wᵢxᵢ   (complex amplitudes interfere)
      4. Homodyne receiver: I = R·Re(E_total) = R·E₀·Σᵢ wᵢxᵢ
      5. Normalise by E₀ → exact dot product

    Compared to FiberNeuron: 4× faster (1 pass vs 4), no decomposition noise.
    """

    def __init__(self, n_inputs, smf=None, edfa=None, dcf=None,
                 iq_mod=None, receiver=None):
        self.n        = n_inputs
        self.smf      = smf      or SingleModeFiber()
        self.edfa     = edfa     or EDFA()
        self.dcf      = dcf      or DispersionCompensatingFiber()
        self.iq_mod   = iq_mod   or IQModulator()
        self.receiver = receiver or HomodyneReceiver()

    def compute(self, w, x, E0=1.0, noisy=False, rng=None):
        """
        Coherent optical dot product: returns γ·Σ wᵢxᵢ (single pass).

        Normalisation:
          E_att = E₀ · exp(-α·L)   (SMF attenuation)
          E_amp = √G · E_att         (EDFA gain)
          For calibration: divide by |E_amp| so E_ref = 1.0
        """
        # SMF attenuation
        E_att = self.smf.attenuate(complex(E0))
        # EDFA amplification
        E_amp = (self.edfa.amplify(E_att, rng=rng, noisy=noisy)
                 if (noisy and rng is not None)
                 else self.edfa.amplify_noiseless(E_att))
        E_ref = abs(E_amp)
        if E_ref < 1e-15:
            E_ref = 1e-15

        # Normalise carrier to unit amplitude (reference-pulse calibration)
        E_carrier = E_amp / E_ref   # unit-amplitude complex carrier

        # Coherent modulation: each element encoded as amplitude × phase
        # w normalisation keeps IQ modulator in physical range [0, 1]
        w_scale = float(np.abs(w).max()) + 1e-12
        x_scale = float(np.abs(x).max()) + 1e-12

        E_total = complex(0.0)
        for i in range(len(w)):
            E_w = self.iq_mod.modulate(E_carrier,  w[i] / w_scale)
            E_x = self.iq_mod.modulate(E_w,        x[i] / x_scale)
            E_total += E_x

        # Homodyne detection: I ∝ Re(E_total)
        raw = self.receiver.detect(E_total, E_LO_amplitude=1.0,
                                   noisy=noisy, rng=rng)

        # Restore original scales
        return raw * w_scale * x_scale


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class CoherentFiberLayer:
    """
    One coherent fiber ring loop = one FC layer via IQ modulation.

    All N_out neurons operate simultaneously on different WDM channels
    (Dense WDM, one λ-channel per neuron, 100 GHz spacing).
    """

    def __init__(self, n_in, n_out, weights=None, bias=None, activation=None):
        self.n_in      = n_in
        self.n_out     = n_out
        self.W  = (weights if weights is not None
                   else np.random.randn(n_out, n_in) * np.sqrt(2.0 / n_in))
        self.b  = bias if bias is not None else np.zeros(n_out)
        self.activation = activation
        self.neurons = [CoherentFiberNeuron(n_in) for _ in range(n_out)]
        # WDM wavelengths: λᵢ = 1550 + i·0.8 nm
        self.wavelengths = np.array([LAMBDA_C + i * 0.8e-9
                                     for i in range(n_out)])

    def forward(self, x, noisy=False, rng=None):
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
    def weight_norm(self):  return float(np.linalg.norm(self.W))
    @property
    def sparsity(self):     return float(np.mean(np.abs(self.W) < 1e-4))
    @property
    def n_lambda(self):     return self.n_out


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class CoherentFiberNetwork:
    """
    Full coherent fiber optical neural network.

    Physical pipeline per layer:
      IQ-modulated laser → WDM MUX → [N coherent neurons in parallel]
      → WDM DEMUX → homodyne receivers → DSP

    Advantages over FiberNeuralNetwork:
      • No pos/neg decomposition → no approximation error
      • Exact dot product in one optical pass
      • Phase diversity enables true complex-valued computation
      • WDM parallelism: all neurons simultaneously
    """

    def __init__(self, layer_sizes, activations=None, noisy=False):
        self.layer_sizes = layer_sizes
        self.noisy       = noisy
        self.rng         = np.random.default_rng(0)
        self.optical_encodings = []

        n_layers = len(layer_sizes) - 1
        if activations is None:
            activations = [relu] * (n_layers - 1) + [softmax]

        self.layers = [
            CoherentFiberLayer(layer_sizes[i], layer_sizes[i + 1],
                               activation=activations[i])
            for i in range(n_layers)
        ]

    def forward(self, x_raw):
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
        return sum(self.predict(X[i]) == np.argmax(Y[i])
                   for i in range(len(X))) / len(X)

    def load_trained_weights(self, weights_list, biases_list=None):
        for i, layer in enumerate(self.layers):
            W = weights_list[i]
            b = biases_list[i] if biases_list else None
            layer.set_weights(W, b)

    def optical_state_table(self):
        return [
            {
                'layer':    i,
                'shape':    (l.n_out, l.n_in),
                'W_norm':   l.weight_norm,
                'sparsity': l.sparsity,
                'n_lambda': l.n_lambda,
                'lambda_min_nm': round(l.wavelengths[0] * 1e9, 2),
                'lambda_max_nm': round(l.wavelengths[-1] * 1e9, 2),
            }
            for i, l in enumerate(self.layers)
        ]

    def total_lambda_channels(self):
        return sum(l.n_lambda for l in self.layers)

    def ring_loops(self):
        return len(self.layers) - 1
