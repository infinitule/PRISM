"""
use_cases.py — Production Use Cases for the Coherent Fiber Neural Network
==========================================================================
Three real-world optical networking problems solved by the same physical
substrate: coherent IQ modulation + WDM + homodyne detection.

  UC1  Multi-Impairment Classifier (MIC)
         Live diagnosis of the dominant link impairment.
         Input : 8 receiver-side signal quality metrics
         Output: 5 classes  (Normal / CD / PMD / OSNR / Nonlinear)

  UC2  Adaptive Modulation & Coding Controller (AMC)
         Select the highest-throughput MCS the channel can support.
         Input : 8 channel-state parameters
         Output: 6 classes  (OOK / PAM-4 / PAM-8 / QPSK / QAM-16 / QAM-64)

  UC3  OTDR Fault Diagnosis Engine (FDE)
         Identify fault type from optical time-domain reflectometry traces.
         Input : 8 OTDR-derived features
         Output: 5 classes  (Normal / ConnectorLoss / Splice / Break / MacroBend)

Key insight
───────────
  The SAME coherent fiber NN architecture (different trained weights)
  solves all three problems. The optical ring computes Σwᵢxᵢ at the
  speed of light regardless of what those weights represent.
  This is the photonic universal-function-approximation principle.

Physical substrate (unchanged across all use cases):
  1550 nm CW laser  →  IQ modulator (φ=0/π)  →  SMF G.652
  →  EDFA  →  DCF coherent sum  →  90°-hybrid + balanced PD
  Output: I ∝ Re(Σwᵢxᵢ·e^jφ) = Σwᵢxᵢ  (exact, single pass)
"""

import numpy as np
import sys
import os

# ── path fix so we can import from same directory ────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from coherent_nn import CoherentFiberNetwork, minmax_norm
from recursive_prompt import MetaMetaPrompt
from recursive_dev import CrossEntropyTrainer


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SHARED UTILITIES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _train_and_transfer(layer_sizes, X_tr, Y_tr, X_te, Y_te,
                        epochs=600, lr=1e-3, l2=5e-5, seed=42,
                        mp: MetaMetaPrompt = None, verbose=True):
    """
    Train theoretic CE model → transfer weights to coherent fiber NN.
    Returns (fiber_nn, test_accuracy, trainer).
    """
    trainer = CrossEntropyTrainer(layer_sizes, lr=lr, l2=l2)
    trainer.rng = np.random.default_rng(seed)

    losses = []
    best_loss = float('inf')
    step = max(1, epochs // 5)
    for ep in range(epochs):
        # Cosine LR decay
        lr_ep = lr * 0.01 + 0.99 * lr * 0.5 * (1 + np.cos(np.pi * ep / epochs))
        trainer.lr = lr_ep
        idx = np.random.default_rng(ep + seed).permutation(len(X_tr))
        ep_loss = 0.0
        for i in idx:
            x_n = minmax_norm(X_tr[i])
            y_hat = trainer.forward(x_n)
            ep_loss += trainer._ce(y_hat, Y_tr[i])
            gW, gb = trainer.backward_ce(Y_tr[i])
            trainer._adam(gW, gb)
        ep_loss /= len(X_tr)
        losses.append(ep_loss)
        if verbose and (ep % step == 0 or ep == epochs - 1):
            print(f"    ep {ep+1:>4}/{epochs}  CE={ep_loss:.5f}  lr={lr_ep:.2e}")

    # Transfer to coherent fiber NN
    shapes = [(layer_sizes[i+1], layer_sizes[i]) for i in range(len(layer_sizes)-1)]
    if mp is None:
        mp = MetaMetaPrompt(seed_dim=64, max_depth=16, rng_seed=seed)
    _, encs = mp.recursive_unfold(shapes)

    fiber_nn = CoherentFiberNetwork(layer_sizes, noisy=False)
    fiber_nn.load_trained_weights(trainer.W, trainer.b)
    fiber_nn.optical_encodings = encs

    # Per-layer calibration
    n_hidden = len(fiber_nn.layers) - 1
    for i in range(n_hidden):
        th_vals, fi_vals = [], []
        for x in X_tr:
            x_n = minmax_norm(x)
            trainer.forward(x_n)
            th_vals.append(float(np.mean(np.abs(trainer._z[i]))))
            fiber_nn.forward(x)
            fi_vals.append(float(np.mean(np.abs(
                fiber_nn._layer_outputs[i + 1]))) + 1e-8)
        sc = float(np.clip(np.mean(th_vals) / (np.mean(fi_vals) + 1e-8), 0.1, 8.0))
        fiber_nn.layers[i].W *= sc

    acc = fiber_nn.evaluate(X_te, Y_te)
    th_acc = trainer.evaluate(X_te, Y_te)
    return fiber_nn, acc, th_acc, trainer


def _print_conf_matrix(fiber_nn, X_te, Y_te, class_names):
    """Print confusion matrix for a fiber NN."""
    n = len(class_names)
    cm = np.zeros((n, n), dtype=int)
    for x, y in zip(X_te, Y_te):
        true_c = int(np.argmax(y))
        pred_c = fiber_nn.predict(x)
        cm[true_c][pred_c] += 1
    w = max(len(c) for c in class_names) + 1
    print(f"\n  Confusion matrix (row=true, col=pred):")
    print("  " + " " * w + "  " + "  ".join(f"{c:>{w}}" for c in class_names))
    for i, row in enumerate(cm):
        vals = "  ".join(
            f"\033[1m{v:>{w}}\033[0m" if j == i else f"{v:>{w}}"
            for j, v in enumerate(row))
        print(f"  {class_names[i]:>{w}}  {vals}")


def _optical_probe(fiber_nn, x_raw, class_names):
    """Show per-class optical output probabilities as a bar chart."""
    out = fiber_nn.forward(x_raw)
    pred = int(np.argmax(out))
    w    = max(len(c) for c in class_names)
    BAR  = "█"
    print(f"\n  Optical probability readout (homodyne detection):")
    for i, (name, p) in enumerate(zip(class_names, out)):
        bar   = BAR * int(p * 30)
        mark  = " ◄ PREDICTED" if i == pred else ""
        print(f"    {name:>{w}}  {bar:<30}  {p:.4f}{mark}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# USE CASE 1 — MULTI-IMPAIRMENT CLASSIFIER (MIC)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MIC_CLASSES = ['Normal', 'CD-dominant', 'PMD-dominant', 'OSNR-limited', 'Nonlinear']
MIC_FEATURES = ['EVM(%)', 'Q-factor(dB)', 'Eye-open', 'BER(log)',
                 'Spec-skew', 'RMS-jitter(ps)', 'CD-est(ps/nm)', 'DGD(ps)']


def _mic_sample(rng, label):
    """
    Generate one signal quality measurement vector for a given impairment class.
    Values are physically motivated ranges from coherent optical comms.
    """
    noise = lambda s: rng.normal(0, s)

    if label == 0:   # Normal
        evm   = 2.0  + noise(0.3)
        q     = 14.0 + noise(0.5)
        eye   = 0.85 + noise(0.03)
        ber   = -7.0 + noise(0.2)
        skew  = 0.1  + noise(0.05)
        jitt  = 3.0  + noise(0.5)
        cd    = 10.0 + noise(2.0)
        dgd   = 1.0  + noise(0.2)

    elif label == 1:   # CD-dominant (chromatic dispersion)
        evm   = 8.0  + noise(1.0)
        q     = 10.0 + noise(0.8)
        eye   = 0.55 + noise(0.05)
        ber   = -4.5 + noise(0.3)
        skew  = 0.3  + noise(0.1)
        jitt  = 12.0 + noise(2.0)   # jitter high — ISI from CD
        cd    = 800.0+ noise(50.0)  # high residual CD
        dgd   = 1.5  + noise(0.3)

    elif label == 2:   # PMD-dominant (polarisation mode dispersion)
        evm   = 7.0  + noise(1.0)
        q     = 10.5 + noise(0.8)
        eye   = 0.60 + noise(0.05)
        ber   = -4.8 + noise(0.3)
        skew  = 0.15 + noise(0.05)
        jitt  = 8.0  + noise(1.5)
        cd    = 15.0 + noise(5.0)
        dgd   = 25.0 + noise(5.0)   # DGD high — PMD signature

    elif label == 3:   # OSNR-limited (amplifier noise dominant)
        evm   = 12.0 + noise(1.5)
        q     = 8.0  + noise(0.8)
        eye   = 0.40 + noise(0.05)
        ber   = -3.0 + noise(0.3)
        skew  = 0.2  + noise(0.08)
        jitt  = 5.0  + noise(1.0)
        cd    = 12.0 + noise(3.0)
        dgd   = 2.0  + noise(0.5)

    else:              # Nonlinear (SPM/XPM/FWM)
        evm   = 9.0  + noise(1.2)
        q     = 9.5  + noise(0.8)
        eye   = 0.50 + noise(0.05)
        ber   = -3.8 + noise(0.3)
        skew  = 0.8  + noise(0.15)  # spectral skew — nonlinear signature
        jitt  = 6.0  + noise(1.0)
        cd    = 20.0 + noise(5.0)
        dgd   = 1.8  + noise(0.4)

    return np.array([evm, q, eye, ber, skew, jitt, cd, dgd])


def generate_mic_dataset(n=300, seed=42):
    rng   = np.random.default_rng(seed)
    n_cls = len(MIC_CLASSES)
    X, Y  = [], []
    for label in range(n_cls):
        for _ in range(n // n_cls):
            X.append(_mic_sample(rng, label))
            y = np.zeros(n_cls); y[label] = 1.0
            Y.append(y)
    X, Y  = np.array(X), np.array(Y)
    idx   = rng.permutation(len(X))
    return X[idx], Y[idx]


def run_uc1(mp, verbose=True):
    print("\n" + "━"*62)
    print("  USE CASE 1 — MULTI-IMPAIRMENT CLASSIFIER (MIC)")
    print("  Real-time fiber link health monitor (no signal interruption)")
    print("━"*62)
    print(f"  Classes  : {' | '.join(MIC_CLASSES)}")
    print(f"  Features : {', '.join(MIC_FEATURES)}")
    print(f"  Dataset  : 300 samples  ·  80/20 split")
    print(f"  Network  : [8, 20, 10, 5]  coherent fiber NN")

    X, Y  = generate_mic_dataset(n=300)
    split = int(0.8 * len(X))
    X_tr, Y_tr = X[:split], Y[:split]
    X_te, Y_te = X[split:], Y[split:]

    print(f"\n  [Training theoretic model → coherent fiber NN]")
    fiber_nn, acc, th_acc, trainer = _train_and_transfer(
        [8, 20, 10, 5], X_tr, Y_tr, X_te, Y_te,
        epochs=600, lr=1e-3, mp=mp, verbose=verbose)

    print(f"\n  Theory  accuracy : {th_acc*100:.1f}%")
    print(f"  Fiber   accuracy : {acc*100:.1f}%")
    print(f"  WDM λ-channels  : {fiber_nn.total_lambda_channels()}")
    _print_conf_matrix(fiber_nn, X_te, Y_te, MIC_CLASSES)

    # Live probe: synthetic "PMD event" arriving on the network
    print(f"\n  ── Live probe: PMD event entering the optical ring ──")
    probe = _mic_sample(np.random.default_rng(99), label=2)   # PMD
    probe += np.random.default_rng(7).normal(0, 0.5, probe.shape)
    print(f"  Sensor reading: {dict(zip(MIC_FEATURES, np.round(probe, 2)))}")
    _optical_probe(fiber_nn, probe, MIC_CLASSES)

    return fiber_nn, acc


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# USE CASE 2 — ADAPTIVE MODULATION & CODING CONTROLLER (AMC)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

AMC_CLASSES  = ['OOK', 'PAM-4', 'PAM-8', 'QPSK', 'QAM-16', 'QAM-64']
AMC_FEATURES = ['OSNR(dB)', 'PathLoss(dB)', 'Spans(#)',
                 'NoiseFig(dB)', 'ResCDisp(ps/nm)', 'PMD(ps)',
                 'NonlinCoef(1/W/km)', 'Margin(dB)']

# Each class has distinct, well-separated channel-state signatures
# Generated label-first (same strategy as UC1/UC3) for clean boundaries
def _amc_sample(rng, label):
    """
    Generate channel-state vector for a given optimal modulation class.
    Physical interpretation: the channel conditions that make each
    format the best choice are physically distinct enough to classify.

      OOK    → very poor channel (short reach, high loss, low OSNR)
      PAM-4  → moderate loss, some PMD
      PAM-8  → decent OSNR, low PMD, some dispersion
      QPSK   → good OSNR, moderate loss, coherent-friendly
      QAM-16 → high OSNR, low loss, controlled impairments
      QAM-64 → excellent OSNR, low loss, low noise, metro-core
    """
    n = lambda mu, s: float(rng.normal(mu, s))

    if label == 0:    # OOK — very challenged channel
        return np.array([n(9,0.5),  n(28,1.5), n(18,1), n(7.5,0.3),
                         n(180,10), n(9,0.5),  n(2.8,0.1), n(-1.5,0.2)])

    elif label == 1:  # PAM-4 — moderate channel
        return np.array([n(13,0.5), n(22,1.2), n(12,1), n(6.5,0.3),
                         n(120,8),  n(6,0.4),  n(2.0,0.1), n(0.5,0.2)])

    elif label == 2:  # PAM-8 — good channel, intensity-based
        return np.array([n(17,0.5), n(18,1.0), n(8,1),  n(5.5,0.3),
                         n(60,6),   n(3,0.3),  n(1.5,0.1), n(1.5,0.2)])

    elif label == 3:  # QPSK — coherent, tolerates more PMD
        return np.array([n(15,0.5), n(20,1.0), n(10,1), n(6.0,0.3),
                         n(90,8),   n(5,0.4),  n(1.8,0.1), n(1.0,0.2)])

    elif label == 4:  # QAM-16 — high OSNR, metro/regional
        return np.array([n(22,0.5), n(14,0.8), n(5,0.8), n(5.0,0.2),
                         n(25,4),   n(1.5,0.2),n(1.0,0.1), n(3.0,0.2)])

    else:             # QAM-64 — premium metro-core
        return np.array([n(30,0.5), n(10,0.6), n(3,0.5), n(4.2,0.2),
                         n(8,2),    n(0.5,0.1), n(0.7,0.05), n(4.0,0.2)])


def generate_amc_dataset(n=360, seed=42):
    rng   = np.random.default_rng(seed)
    n_cls = len(AMC_CLASSES)
    X, Y  = [], []
    for label in range(n_cls):
        for _ in range(n // n_cls):
            X.append(_amc_sample(rng, label))
            y = np.zeros(n_cls); y[label] = 1.0
            Y.append(y)
    X, Y = np.array(X), np.array(Y)
    idx  = rng.permutation(len(X))
    return X[idx], Y[idx]


def run_uc2(mp, verbose=True):
    print("\n" + "━"*62)
    print("  USE CASE 2 — ADAPTIVE MODULATION & CODING CONTROLLER (AMC)")
    print("  Maximise spectral efficiency from real-time channel state")
    print("━"*62)
    print(f"  Classes  : {' | '.join(AMC_CLASSES)}")
    print(f"  Features : {', '.join(AMC_FEATURES)}")
    print(f"  Dataset  : 360 samples  ·  80/20 split")
    print(f"  Network  : [8, 24, 12, 6]  coherent fiber NN")

    X, Y  = generate_amc_dataset(n=360)
    split = int(0.8 * len(X))
    X_tr, Y_tr = X[:split], Y[:split]
    X_te, Y_te = X[split:], Y[split:]

    print(f"\n  [Training theoretic model → coherent fiber NN]")
    fiber_nn, acc, th_acc, trainer = _train_and_transfer(
        [8, 24, 12, 6], X_tr, Y_tr, X_te, Y_te,
        epochs=600, lr=1e-3, mp=mp, verbose=verbose)

    print(f"\n  Theory  accuracy : {th_acc*100:.1f}%")
    print(f"  Fiber   accuracy : {acc*100:.1f}%")
    print(f"  WDM λ-channels  : {fiber_nn.total_lambda_channels()}")
    _print_conf_matrix(fiber_nn, X_te, Y_te, AMC_CLASSES)

    # Scenario: network operator checks three candidate routes
    rng_probe = np.random.default_rng(77)
    scenarios = [
        ("Short metro link     (QAM-64?)",
         _amc_sample(rng_probe, 5)),    # QAM-64 conditions
        ("Regional backbone   (QAM-16?)",
         _amc_sample(rng_probe, 4)),    # QAM-16 conditions
        ("Long-haul transoce. (PAM-4?)",
         _amc_sample(rng_probe, 1)),    # PAM-4 conditions
    ]
    print(f"\n  ── Route optimisation: 3 candidate links ──")
    for name, x_raw in scenarios:
        out  = fiber_nn.forward(x_raw)
        pred = int(np.argmax(out))
        bps  = [1, 2, 3, 2, 4, 6][pred]   # bits per symbol
        print(f"  {name:<28}  →  {AMC_CLASSES[pred]:<7}"
              f"  ({bps} b/sym)  p={out[pred]:.3f}")

    return fiber_nn, acc


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# USE CASE 3 — OTDR FAULT DIAGNOSIS ENGINE (FDE)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

FDE_CLASSES  = ['Normal', 'ConnectorLoss', 'SpliceLoss', 'FiberBreak', 'MacroBend']
FDE_FEATURES = ['BackscatterSlope(dB/km)', 'EventPower(dB)', 'EventDist(km)',
                 'PulseWidth(ns)', 'DeadZone(m)', 'SNR(dB)',
                 'Reflectance(dB)', 'LossRate(dB/km)']


def _fde_sample(rng, label, noise_std=0.0):
    """Physically motivated OTDR trace feature vector per fault type.

    noise_std adds a small global Gaussian floor across all features,
    forcing the model to learn robust decision boundaries rather than
    memorising individual sample values.
    """
    n = lambda s: rng.normal(0, s)

    if label == 0:   # Normal — clean trace, low event power, no reflectance peak
        v = np.array([0.20+n(0.01), 0.0+n(0.08), 50+n(5), 10+n(1),
                      1.0+n(0.1),  30+n(2),  -60+n(2),  0.20+n(0.01)])

    elif label == 1:   # Connector loss — distinct event + partial Fresnel
        v = np.array([0.20+n(0.01), 1.5+n(0.25), 25+n(3), 10+n(1),
                      1.0+n(0.1),  28+n(2),  -35+n(2.5), 0.20+n(0.01)])

    elif label == 2:   # Splice loss — low power, no air gap → very low reflectance
        v = np.array([0.20+n(0.01), 0.6+n(0.08), 30+n(4), 10+n(1),
                      0.4+n(0.04), 29+n(2),  -70+n(2.5), 0.20+n(0.01)])
                     # key separators vs Normal: EventPower↑, DeadZone↓, Reflectance↓↓

    elif label == 3:   # Fiber break — full Fresnel reflection, loss of backscatter
        v = np.array([0.20+n(0.01), 10.0+n(0.4), 40+n(2), 10+n(1),
                      1.0+n(0.1),  20+n(2),  -14+n(1.5), 0.20+n(0.01)])

    else:              # Macro-bend — elevated slope + loss rate
        v = np.array([0.40+n(0.025), 3.0+n(0.35), 15+n(3), 10+n(1),
                      1.0+n(0.1),   25+n(2),  -55+n(2.5), 0.80+n(0.04)])

    if noise_std > 0.0:
        v += rng.normal(0.0, noise_std, v.shape)
    return v


def generate_fde_dataset(n=300, seed=42, noise_std=0.02):
    rng  = np.random.default_rng(seed)
    n_cls = len(FDE_CLASSES)
    X, Y  = [], []
    for label in range(n_cls):
        for _ in range(n // n_cls):
            X.append(_fde_sample(rng, label, noise_std=noise_std))
            y = np.zeros(n_cls); y[label] = 1.0
            Y.append(y)
    X, Y = np.array(X), np.array(Y)
    idx  = rng.permutation(len(X))
    return X[idx], Y[idx]


def run_uc3(mp, verbose=True):
    print("\n" + "━"*62)
    print("  USE CASE 3 — OTDR FAULT DIAGNOSIS ENGINE (FDE)")
    print("  Sub-second fault identification from OTDR trace features")
    print("━"*62)
    print(f"  Classes  : {' | '.join(FDE_CLASSES)}")
    print(f"  Features : {', '.join(FDE_FEATURES)}")
    print(f"  Dataset  : 300 samples  ·  80/20 split")
    print(f"  Network  : [8, 20, 10, 5]  coherent fiber NN")

    X, Y  = generate_fde_dataset(n=300, noise_std=0.02)
    split = int(0.8 * len(X))
    X_tr, Y_tr = X[:split], Y[:split]
    X_te, Y_te = X[split:], Y[split:]

    print(f"\n  [Training theoretic model → coherent fiber NN]")
    fiber_nn, acc, th_acc, trainer = _train_and_transfer(
        [8, 20, 10, 5], X_tr, Y_tr, X_te, Y_te,
        epochs=800, lr=1e-3, mp=mp, verbose=verbose)

    print(f"\n  Theory  accuracy : {th_acc*100:.1f}%")
    print(f"  Fiber   accuracy : {acc*100:.1f}%")
    print(f"  WDM λ-channels  : {fiber_nn.total_lambda_channels()}")
    _print_conf_matrix(fiber_nn, X_te, Y_te, FDE_CLASSES)

    # Live fault event sequence (simulated NOC alert stream)
    events = [
        ("Alert #1  14:02:31", _fde_sample(np.random.default_rng(10), 3)),  # break
        ("Alert #2  14:02:45", _fde_sample(np.random.default_rng(11), 1)),  # connector
        ("Alert #3  14:03:02", _fde_sample(np.random.default_rng(12), 0)),  # normal
        ("Alert #4  14:03:19", _fde_sample(np.random.default_rng(13), 4)),  # bend
    ]
    print(f"\n  ── NOC alert stream — optical diagnosis in real time ──")
    for label, x_raw in events:
        out  = fiber_nn.forward(x_raw)
        pred = int(np.argmax(out))
        sev  = ['✓ OK', '⚠ WARN', '⚠ WARN', '✗ CRIT', '⚠ WARN'][pred]
        print(f"  {label}  →  {FDE_CLASSES[pred]:<15}  {sev}  "
              f"confidence={out[pred]*100:.1f}%")

    return fiber_nn, acc


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# USE CASE 4 — DATACENTER INTERCONNECT LINK STATE CLASSIFIER (DCI)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DCI_CLASSES  = ['Optimal', 'Degraded', 'Marginal', 'Critical', 'Failed']
DCI_FEATURES = ['TXPower(dBm)', 'RXPower(dBm)', 'PreFEC_BER(log)',
                 'PostFEC_BER(log)', 'CDPenalty(dB)', 'PMDPenalty(dB)',
                 'Temperature(C)', 'WavDrift(pm)']


def _dci_sample(rng, label, noise_std=0.02):
    """
    Generate a DCI transceiver measurement vector per link-state class.
    Models coherent 400ZR/ZR+ pluggable optics on metro/DCI spans.

      Optimal  → clean, well-powered, low penalties, stable wavelength
      Degraded → slight RX power drop, minor BER increase, mild CD/PMD
      Marginal → marginal power budget, elevated BER, thermal stress
      Critical → near-threshold BER, high penalties, wavelength drift
      Failed   → below RX sensitivity, FEC overload, out-of-lock
    """
    n = lambda mu, s: float(rng.normal(mu, s))

    if label == 0:   # Optimal
        v = np.array([n(2.0,0.2),  n(-15.0,0.3), n(-5.5,0.2), n(-12.0,0.3),
                      n(0.5,0.1),  n(0.1,0.03),  n(44.0,1.0), n(2.0,0.3)])

    elif label == 1:  # Degraded
        v = np.array([n(1.0,0.2),  n(-18.5,0.4), n(-4.5,0.2), n(-8.5,0.4),
                      n(1.2,0.15), n(0.4,0.08),  n(47.0,1.0), n(4.5,0.5)])

    elif label == 2:  # Marginal
        v = np.array([n(0.0,0.3),  n(-21.5,0.5), n(-3.5,0.2), n(-4.0,0.5),
                      n(2.5,0.2),  n(0.9,0.1),   n(51.5,1.5), n(9.0,1.0)])

    elif label == 3:  # Critical
        v = np.array([n(-1.5,0.3), n(-25.5,0.6), n(-2.5,0.2), n(-1.0,0.4),
                      n(4.5,0.3),  n(1.8,0.15),  n(57.0,2.0), n(20.0,2.0)])

    else:             # Failed
        v = np.array([n(-6.0,0.5), n(-35.0,1.0), n(-1.0,0.1), n(0.0,0.1),
                      n(9.0,0.5),  n(5.0,0.3),   n(64.0,2.0), n(55.0,5.0)])

    if noise_std > 0.0:
        v += rng.normal(0.0, noise_std, v.shape)
    return v


def generate_dci_dataset(n=300, seed=42):
    rng   = np.random.default_rng(seed)
    n_cls = len(DCI_CLASSES)
    X, Y  = [], []
    for label in range(n_cls):
        for _ in range(n // n_cls):
            X.append(_dci_sample(rng, label))
            y = np.zeros(n_cls); y[label] = 1.0
            Y.append(y)
    X, Y = np.array(X), np.array(Y)
    idx  = rng.permutation(len(X))
    return X[idx], Y[idx]


def run_uc4(mp, verbose=True):
    print("\n" + "━"*62)
    print("  USE CASE 4 — DATACENTER INTERCONNECT LINK STATE (DCI)")
    print("  Classify 400ZR/ZR+ transceiver health from live PM counters")
    print("━"*62)
    print(f"  Classes  : {' | '.join(DCI_CLASSES)}")
    print(f"  Features : {', '.join(DCI_FEATURES)}")
    print(f"  Dataset  : 300 samples  ·  80/20 split")
    print(f"  Network  : [8, 20, 10, 5]  coherent fiber NN")

    X, Y  = generate_dci_dataset(n=300)
    split = int(0.8 * len(X))
    X_tr, Y_tr = X[:split], Y[:split]
    X_te, Y_te = X[split:], Y[split:]

    print(f"\n  [Training theoretic model → coherent fiber NN]")
    fiber_nn, acc, th_acc, trainer = _train_and_transfer(
        [8, 20, 10, 5], X_tr, Y_tr, X_te, Y_te,
        epochs=700, lr=1e-3, mp=mp, verbose=verbose)

    print(f"\n  Theory  accuracy : {th_acc*100:.1f}%")
    print(f"  Fiber   accuracy : {acc*100:.1f}%")
    print(f"  WDM λ-channels  : {fiber_nn.total_lambda_channels()}")
    _print_conf_matrix(fiber_nn, X_te, Y_te, DCI_CLASSES)

    # Cloud operator scenario: real-time fleet health sweep
    rng_probe = np.random.default_rng(201)
    fleet = [
        ("AMS-LON spine  #1",  _dci_sample(rng_probe, 0)),  # Optimal
        ("AMS-LON spine  #2",  _dci_sample(rng_probe, 1)),  # Degraded
        ("FRA-PAR metro  #1",  _dci_sample(rng_probe, 2)),  # Marginal
        ("LHR-DUB access #1",  _dci_sample(rng_probe, 3)),  # Critical
        ("NYC-BOS backup #1",  _dci_sample(rng_probe, 4)),  # Failed
    ]
    actions = {
        'Optimal':  '✓ No action',
        'Degraded': '⚠ Schedule maintenance',
        'Marginal': '⚠ Re-route traffic',
        'Critical': '✗ Immediate failover',
        'Failed':   '✗ Dispatch engineer',
    }
    print(f"\n  ── Cloud operator fleet sweep — optical PM analysis ──")
    for link, x_raw in fleet:
        out  = fiber_nn.forward(x_raw)
        pred = int(np.argmax(out))
        cls  = DCI_CLASSES[pred]
        print(f"  {link:<22}  →  {cls:<8}  {actions[cls]}  "
              f"(p={out[pred]:.3f})")

    return fiber_nn, acc


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    print("\n" + "█"*62)
    print("█                                                          █")
    print("█   COHERENT FIBER NN — USE CASE DEMONSTRATION            █")
    print("█   Universal optical substrate  ·  4 production tasks    █")
    print("█                                                          █")
    print("█"*62)
    print("""
  Same physical hardware — different trained weights:
    Laser 1550 nm  →  IQ mod (φ=0/π)  →  SMF+EDFA+DCF
    →  90°-hybrid  →  balanced PD  →  I = Σwᵢxᵢ
  """)

    # Shared MetaMetaPrompt across all use cases
    mp = MetaMetaPrompt(seed_dim=64, max_depth=16, rng_seed=42)

    results = {}

    # ── UC1 ───────────────────────────────────────────────────────────────
    nn1, acc1 = run_uc1(mp, verbose=True)
    results['MIC'] = acc1

    # ── UC2 ───────────────────────────────────────────────────────────────
    nn2, acc2 = run_uc2(mp, verbose=True)
    results['AMC'] = acc2

    # ── UC3 ───────────────────────────────────────────────────────────────
    nn3, acc3 = run_uc3(mp, verbose=True)
    results['FDE'] = acc3

    # ── UC4 ───────────────────────────────────────────────────────────────
    nn4, acc4 = run_uc4(mp, verbose=True)
    results['DCI'] = acc4

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'█'*62}")
    print(f"  COHERENT FIBER NN — USE CASE SUMMARY")
    print(f"{'━'*62}")
    print(f"  {'Use Case':<38}  {'Fiber Acc':>10}  {'λ-ch':>5}")
    print(f"  {'─'*56}")
    specs = [
        ("UC1  Multi-Impairment Classifier",    nn1, acc1),
        ("UC2  Adaptive Modulation Controller", nn2, acc2),
        ("UC3  OTDR Fault Diagnosis Engine",    nn3, acc3),
        ("UC4  DCI Link State Classifier",      nn4, acc4),
    ]
    for name, nn, acc in specs:
        print(f"  {name:<38}  {acc*100:>9.1f}%  {nn.total_lambda_channels():>5}")
    print(f"{'━'*62}")
    mean_acc = np.mean(list(results.values()))
    print(f"  Mean fiber accuracy  : {mean_acc*100:.1f}%")
    print(f"  Shared substrate     : Coherent IQ  ·  1550 nm  ·  DWDM")
    print(f"  Prompt               : MetaMetaPrompt  (3-level hierarchy)")
    print(f"  Layers generated     : {len(mp.history)}")
    print(f"\n  One photon.  Four minds.  All at light speed.")
    print(f"{'█'*62}\n")


if __name__ == "__main__":
    main()
