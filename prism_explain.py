"""
prism_explain.py — T10: Physical Interpretability Dashboard
============================================================
Explain every classification decision in physical optical terms:
  - Which input features drove the decision (numerical saliency)
  - Which WDM λ-channels were most active per layer
  - What the optical phase pattern implies (constructive/destructive ratio)
  - Layer-by-layer activation flow from input to output

Physical XAI advantage over silicon NNs:
  Every weight W[i,j] maps to a physical optical parameter:
    amplitude_ij = |W[i,j]| / max|W|   → IQ modulator drive voltage (0–1 V)
    phase_ij     = 0 (W≥0) or π (W<0)  → electrode bias (constructive/destructive)
    wavelength_i = 1550 + i·0.8 nm     → WDM channel assigned to neuron i

This module provides:
  explain_decision()   — full single-sample explanation
  saliency_map()       — gradient × input for all features
  wdm_activation_map() — per-layer λ-channel activation analysis
  phase_pattern()      — constructive/destructive weight statistics
  weight_to_optics()   — decode any weight matrix to optical parameters
  batch_saliency()     — aggregate saliency over a dataset
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from coherent_nn import CoherentFiberNetwork


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LAMBDA_C     = 1550e-9   # m
DELTA_LAMBDA = 0.8e-9    # m (100 GHz DWDM)


def _numerical_gradient(nn: CoherentFiberNetwork, x_raw: np.ndarray,
                         class_idx: int, eps_frac: float = 1e-3) -> np.ndarray:
    """
    Compute numerical gradient of class_idx probability wrt each input feature.
    Uses central differences for accuracy.
    """
    grad = np.zeros(len(x_raw))
    for i, v in enumerate(x_raw):
        eps       = eps_frac * (abs(float(v)) + 1e-8)
        x_plus    = x_raw.copy(); x_plus[i]  += eps
        x_minus   = x_raw.copy(); x_minus[i] -= eps
        p_plus    = nn.forward(x_plus)[class_idx]
        p_minus   = nn.forward(x_minus)[class_idx]
        grad[i]   = (p_plus - p_minus) / (2 * eps)
    return grad


def saliency_map(nn: CoherentFiberNetwork, x_raw: np.ndarray,
                 class_idx: int | None = None) -> np.ndarray:
    """
    Gradient × input saliency for each feature.
    Higher absolute value → feature more influential for this prediction.

    If class_idx is None, uses the predicted class.
    Returns array of shape (n_features,).
    """
    probs = nn.forward(x_raw)
    if class_idx is None:
        class_idx = int(np.argmax(probs))
    grad = _numerical_gradient(nn, x_raw, class_idx)
    return np.abs(grad * x_raw)


def wdm_activation_map(nn: CoherentFiberNetwork,
                        x_raw: np.ndarray) -> list[dict]:
    """
    Per-layer WDM channel activation analysis.
    Returns list of dicts (one per layer) with:
      layer_idx, top_k_channels, top_k_wavelengths_nm, activation_norm,
      mean_activation, max_activation
    """
    nn.forward(x_raw)
    result = []
    for l_idx, (layer, h) in enumerate(zip(nn.layers, nn._layer_outputs[1:])):
        h_abs   = np.abs(h)
        top_k   = np.argsort(h_abs)[-3:][::-1]
        lam_nm  = [(LAMBDA_C + i * DELTA_LAMBDA) * 1e9 for i in top_k]
        result.append({
            'layer_idx':            l_idx,
            'shape':                (layer.n_out, layer.n_in),
            'top_k_neuron_idx':     top_k.tolist(),
            'top_k_wavelengths_nm': [round(l, 2) for l in lam_nm],
            'activation_norm':      float(np.linalg.norm(h)),
            'mean_activation':      float(h_abs.mean()),
            'max_activation':       float(h_abs.max()),
            'active_fraction':      float((h_abs > h_abs.mean()).mean()),
        })
    return result


def phase_pattern(nn: CoherentFiberNetwork) -> list[dict]:
    """
    Analyse constructive/destructive phase ratio per layer.
    Returns list of dicts (one per layer) with:
      n_constructive (φ=0), n_destructive (φ=π), ratio, dominant_phase
    """
    result = []
    for l_idx, layer in enumerate(nn.layers):
        W     = layer.W
        pos   = int((W > 0).sum())
        neg   = int((W < 0).sum())
        total = pos + neg
        result.append({
            'layer_idx':       l_idx,
            'n_constructive':  pos,
            'n_destructive':   neg,
            'ratio_pos_neg':   round(pos / max(neg, 1), 3),
            'dominant_phase':  'constructive (φ=0)' if pos >= neg else 'destructive (φ=π)',
            'weight_norm':     round(float(np.linalg.norm(W)), 4),
            'sparsity':        round(float((np.abs(W) < 1e-4).mean()), 4),
        })
    return result


def weight_to_optics(W: np.ndarray) -> dict:
    """
    Decode a weight matrix to optical parameters.

    Returns:
      amplitude[i,j]   : IQ modulator drive voltage (normalised to [0,1])
      phase[i,j]       : optical phase (0 or π radians)
      wavelength_nm[i] : WDM channel wavelength (nm) for output neuron i
      scale            : max |W| (restores original values)
    """
    W      = np.asarray(W, dtype=float)
    scale  = np.abs(W).max() + 1e-12
    amp    = np.abs(W) / scale
    phase  = np.where(W >= 0, 0.0, np.pi)
    lam_nm = np.array([(LAMBDA_C + i * DELTA_LAMBDA) * 1e9
                        for i in range(W.shape[0])])
    return {
        'amplitude':       amp,
        'phase_rad':       phase,
        'wavelength_nm':   lam_nm,
        'scale':           float(scale),
        'shape':           W.shape,
        'n_wdm_channels':  W.shape[0],
    }


def batch_saliency(nn: CoherentFiberNetwork, X: np.ndarray,
                   Y: np.ndarray, feature_names: list[str] | None = None,
                   n_samples: int = 50) -> dict:
    """
    Aggregate saliency over a dataset.
    Returns per-class and global average saliency for each feature.
    """
    rng     = np.random.default_rng(0)
    idx     = rng.choice(len(X), min(n_samples, len(X)), replace=False)
    n_feat  = X.shape[1]
    n_class = Y.shape[1]
    sal_cls = np.zeros((n_class, n_feat))
    cnt_cls = np.zeros(n_class, dtype=int)

    for i in idx:
        c = int(np.argmax(Y[i]))
        s = saliency_map(nn, X[i], class_idx=c)
        sal_cls[c] += s
        cnt_cls[c] += 1

    for c in range(n_class):
        if cnt_cls[c] > 0:
            sal_cls[c] /= cnt_cls[c]

    global_sal = sal_cls.mean(axis=0)
    feat_rank  = np.argsort(global_sal)[::-1]

    result = {
        'per_class_saliency': sal_cls,
        'global_saliency':    global_sal,
        'feature_ranking':    feat_rank.tolist(),
    }
    if feature_names:
        result['ranked_features'] = [feature_names[i] for i in feat_rank]
    return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def explain_decision(nn: CoherentFiberNetwork,
                     x_raw: np.ndarray,
                     class_names: list[str],
                     feature_names: list[str] | None = None,
                     top_k_features: int = 5) -> dict:
    """
    Full physical explanation of one inference decision.

    Prints:
      - Predicted class and confidence
      - Top-k features by gradient×input saliency (bar chart)
      - WDM channel activations per layer
      - Phase pattern (constructive/destructive ratio)
      - Optical weight parameters for input layer

    Returns dict with all analysis data for programmatic use.
    """
    probs    = nn.forward(x_raw)
    pred_idx = int(np.argmax(probs))
    sal      = saliency_map(nn, x_raw, pred_idx)
    wdm      = wdm_activation_map(nn, x_raw)
    phases   = phase_pattern(nn)
    optics   = weight_to_optics(nn.layers[0].W)
    n_feat   = len(x_raw)
    feat_names = feature_names or [f'feat_{i}' for i in range(n_feat)]
    top_k    = min(top_k_features, n_feat)
    top_feat = np.argsort(sal)[-top_k:][::-1]

    print(f"\n  {'═'*58}")
    print(f"  PRISM INFERENCE EXPLANATION")
    print(f"  {'═'*58}")
    print(f"  Predicted : {class_names[pred_idx]}  "
          f"(confidence={probs[pred_idx]*100:.1f}%)")
    print(f"  Substrate : coherent IQ · 1550 nm · homodyne")

    # Probability readout
    print(f"\n  Optical probability readout:")
    w = max(len(c) for c in class_names)
    for i, (c, p) in enumerate(zip(class_names, probs)):
        bar  = '█' * int(p * 28)
        mark = ' ◄' if i == pred_idx else ''
        print(f"    {c:>{w}}  {bar:<28}  {p:.4f}{mark}")

    # Feature saliency
    print(f"\n  Feature importance (gradient × input):")
    wf = max(len(n) for n in feat_names)
    max_sal = max(sal.max(), 1e-12)
    for i in top_feat:
        bar = '█' * int(sal[i] / max_sal * 24)
        print(f"    {feat_names[i]:>{wf}}  {x_raw[i]:>9.3f}  "
              f"|{bar:<24}|  {sal[i]:.4f}")
    if n_feat > top_k:
        others_sal = np.sum([sal[i] for i in range(n_feat) if i not in top_feat])
        print(f"    {'(other ' + str(n_feat - top_k) + ' features)':>{wf}}  "
              f"{'':>9}   {'···':>26}  {others_sal:.4f}")

    # WDM activation map
    print(f"\n  WDM λ-channel activations:")
    for lyr in wdm:
        lam_str = ', '.join(f'{l:.1f}nm' for l in lyr['top_k_wavelengths_nm'])
        print(f"    Layer {lyr['layer_idx']}  "
              f"top-3 λ: [{lam_str}]  "
              f"max={lyr['max_activation']:.4f}  "
              f"active={lyr['active_fraction']*100:.0f}%")

    # Phase pattern
    print(f"\n  Optical phase pattern per layer:")
    for ph in phases:
        dom = '↑ construct.' if 'constructive' in ph['dominant_phase'] else '↓ destruct.'
        print(f"    Layer {ph['layer_idx']}  "
              f"φ=0: {ph['n_constructive']:>4}  "
              f"φ=π: {ph['n_destructive']:>4}  "
              f"ratio={ph['ratio_pos_neg']:.2f}  {dom}")

    # Optical weight decode (input layer only)
    print(f"\n  Input layer optical parameters (layer 0):")
    print(f"    n_wdm_channels : {optics['n_wdm_channels']}")
    print(f"    λ range        : "
          f"{optics['wavelength_nm'].min():.2f} – "
          f"{optics['wavelength_nm'].max():.2f} nm")
    print(f"    amplitude stats: "
          f"mean={optics['amplitude'].mean():.3f}  "
          f"max={optics['amplitude'].max():.3f}")
    print(f"    weight scale   : {optics['scale']:.4f}")
    print(f"  {'═'*58}")

    return {
        'predicted_class': class_names[pred_idx],
        'predicted_idx':   pred_idx,
        'confidence':      float(probs[pred_idx]),
        'probabilities':   {c: float(p) for c, p in zip(class_names, probs)},
        'saliency':        sal.tolist(),
        'top_features':    [(feat_names[i], float(sal[i])) for i in top_feat],
        'wdm_activations': wdm,
        'phase_patterns':  phases,
        'optics_layer0':   {k: v.tolist() if hasattr(v, 'tolist') else v
                             for k, v in optics.items()},
    }
