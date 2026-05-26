<div align="center">

# ⬡ fiber-agi

### A neural network where every dot product is a pulse of light.

[![tests](https://github.com/infinitule/fiber-agi/actions/workflows/tests.yml/badge.svg)](https://github.com/infinitule/fiber-agi/actions/workflows/tests.yml)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://python.org)
[![NumPy](https://img.shields.io/badge/numpy-%E2%89%A51.24-013243?logo=numpy)](https://numpy.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![no GPU](https://img.shields.io/badge/GPU-not%20required-lightgrey.svg)]()

```
Laser 1550 nm  →  IQ mod (φ=0/π)  →  SMF G.652  →  EDFA  →  DCF
→  90°-hybrid  →  balanced PD  →  I = Σ wᵢxᵢ   (exact, single pass)
```

*Coherent light computes. Phase encodes sign. The seed generates its own generator.*

</div>

---

## What this is

A physically accurate simulation of a **photonic neural network** — the kind research labs are spending millions to build in silicon and glass — running on any laptop, in pure Python, with one dependency.

Every multiply-accumulate operation that a GPU does with transistors, this does with **coherent 1550 nm light** passing through dispersion-compensating fiber. The math is identical. The physics are real.

Built directly from:
> Zang et al., *"Fiber neural networks for intelligent optical fiber communication signal processing"* — iOptics, 2025

---

## Why it matters for AI

| What silicon does | What this does |
|---|---|
| Matrix multiply in SRAM | Dot product via DCF coherent sum: `E = γ Σ wᵢxᵢ` |
| Sign via two's complement | Sign via phase: `w≥0 → φ=0`, `w<0 → φ=π` |
| Fan-out via copper traces | Fan-out via Dense WDM (100 GHz, one λ per neuron) |
| Sequential clock cycles | All neurons computed simultaneously, at light speed |
| Weights in flash/DRAM | Weights encoded in amplitude + phase of optical carrier |

**The key insight**: coherent IQ detection gives an *exact* dot product in a single optical pass. There is no approximation. The theory–hardware gap is zero.

This opens a path to **inference at the speed of light** — and this repo is the software twin you can run today.

---

## Quick start

```bash
git clone https://github.com/infinitule/fiber-agi
cd fiber-agi
pip install numpy

python3 main.py        # recursive self-improving engine
python3 use_cases.py   # 4 production telecom classifiers
```

No GPU. No CUDA. No PyTorch. No TensorFlow. Runs on any machine made in the last decade.

---

## The architecture

### Layer 0 — Physical substrate

```
                    ┌─────────────────────────────────────────────┐
                    │         COHERENT OPTICAL NEURON             │
                    │                                             │
  E₀ (1550 nm) ──► IQ mod ──► IQ mod ──► ... ──► DCF sum ──► PD │
                    │  w₁        x₁                 E=Σwᵢxᵢ      │
                    │  w₂        x₂                              │
                    │  ...       ...       homodyne → I∝Re(E·E*) │
                    └─────────────────────────────────────────────┘
                              one λ-channel per output neuron
                              Dense WDM — all computed in parallel
```

Physical parameters (ITU-T G.652):

| Component | Spec |
|---|---|
| Laser | 1550 nm C-band, CW |
| Fiber | SMF G.652 — NA=0.1235, V=2.252 (single-mode) |
| Attenuation | 0.2 dB/km @ 1550 nm |
| EDFA | G=20 dB, NF=4 dB |
| DCF | β₂=−100 ps²/nm·km, 28.5 km |
| Detector | R=0.9 A/W (homodyne) |
| WDM spacing | 0.8 nm — 100 GHz DWDM grid |

### Layer 1 — Recursive meta-prompt

Weights are not initialised randomly. They are *generated* by a 3-level seed hierarchy:

```
Level 0 │  P₀ ∈ ℝ⁶⁴                  Fibonacci seed
        │         │
        ▼         ▼
Level 1 │  Ψ(P₀) → (M_W₁, M_W₂)      fixed projection, rank 63/64
        │              │
        ▼              ▼
Level 2 │  MetaGen(Φ, P) → W           outer-product weight matrix
        │
        └─► φ₁ (optical memory)        accumulates across generations
```

The seed generates the generator that generates the weights. The memory vector `φ₁` means the network carries context from every previous generation — it never forgets.

### Layer 2 — Recursive development engine

```
Gen 0  [8 → 16 → 5]       init          98% fiber accuracy
Gen 1  [8 → 24 → 5]       ⊞ widen       98% fiber accuracy
Gen 2  [8 → 24 → 24 → 5]  ⊕ deepen     100% fiber accuracy  ✓ target
```

Each generation either **deepens** (inserts a new fiber ring loop, identity-initialized so accuracy never regresses) or **widens** (expands the narrowest hidden layer). Weights are warm-started — nothing is discarded.

---

## Four production applications

All four run on the **same physical optical hardware**. Only the trained weights change.

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Use Case                          Classes   Network    Acc
  ──────────────────────────────────────────────────────────
  UC1  Multi-Impairment Classifier     5    [8,20,10,5]  100%
  UC2  Adaptive Modulation Control     6    [8,24,12,6]  100%
  UC3  OTDR Fault Diagnosis            5    [8,20,10,5]  100%
  UC4  DCI Link State Classifier       5    [8,20,10,5]  100%
  ──────────────────────────────────────────────────────────
  Mean fiber accuracy                                   100%
  WDM λ-channels deployed                           35–42 λ
  Total parameters                               ~3 500 wts
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

<details>
<summary><b>UC1 — Multi-Impairment Classifier</b> · live fiber link health</summary>

Classifies the dominant impairment on a live coherent link — **without interrupting traffic**.

**Input** (8 receiver-side metrics): EVM, Q-factor, eye opening, BER, spectral skew, RMS jitter, residual CD, DGD  
**Output**: Normal / CD-dominant / PMD-dominant / OSNR-limited / Nonlinear

```
PMD event detected:
  Normal        ░░░░░░░░░░░░░░░░░░░░░░░░░░░░  0.0000
  CD-dominant   ░░░░░░░░░░░░░░░░░░░░░░░░░░░░  0.0000
  PMD-dominant  ████████████████████████████  1.0000  ◄
  OSNR-limited  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░  0.0000
  Nonlinear     ░░░░░░░░░░░░░░░░░░░░░░░░░░░░  0.0000
```
</details>

<details>
<summary><b>UC2 — Adaptive Modulation Controller</b> · maximise spectral efficiency</summary>

Selects the highest-throughput modulation format a given channel can support.

**Input** (8 channel-state parameters): OSNR, path loss, span count, noise figure, residual CD, PMD, nonlinear coefficient, margin  
**Output**: OOK / PAM-4 / PAM-8 / QPSK / QAM-16 / QAM-64

```
Route optimisation — 3 candidate links:
  Short metro          →  QAM-64   6 b/sym   p=1.000
  Regional backbone    →  QAM-16   4 b/sym   p=1.000
  Trans-oceanic        →  PAM-4    2 b/sym   p=0.992
```
</details>

<details>
<summary><b>UC3 — OTDR Fault Diagnosis Engine</b> · sub-second fault ID</summary>

Identifies fault type from OTDR trace features in real time.

**Input** (8 OTDR-derived features): backscatter slope, event power, event distance, pulse width, dead zone, SNR, reflectance, loss rate  
**Output**: Normal / ConnectorLoss / SpliceLoss / FiberBreak / MacroBend

```
NOC alert stream:
  14:02:31  →  FiberBreak      ✗ CRIT   confidence=100.0%
  14:02:45  →  ConnectorLoss   ⚠ WARN   confidence=100.0%
  14:03:02  →  Normal          ✓ OK     confidence=100.0%
  14:03:19  →  MacroBend       ⚠ WARN   confidence=89.9%
```
</details>

<details>
<summary><b>UC4 — DCI Link State Classifier</b> · 400ZR/ZR+ fleet health</summary>

Classifies 400ZR/ZR+ transceiver health from live PM counters for cloud operator fleets.

**Input** (8 PM counters): TX power, RX power, pre-FEC BER, post-FEC BER, CD penalty, PMD penalty, temperature, wavelength drift  
**Output**: Optimal / Degraded / Marginal / Critical / Failed

```
Cloud operator fleet sweep:
  AMS-LON spine  #1   →  Optimal    ✓ No action           p=1.000
  AMS-LON spine  #2   →  Degraded   ⚠ Schedule maint.     p=1.000
  FRA-PAR metro  #1   →  Marginal   ⚠ Re-route traffic    p=1.000
  LHR-DUB access #1   →  Critical   ✗ Immediate failover  p=1.000
  NYC-BOS backup #1   →  Failed     ✗ Dispatch engineer   p=1.000
```
</details>

---

## Tests

```bash
pip install pytest
python3 -m pytest tests/ -v
```

```
50 tests  ·  physics · coherent NN · meta-prompt · trainers · datasets · end-to-end
======================== 50 passed in 115s ========================
```

Test coverage:
- **Physics**: SMF attenuation, EDFA gain (20 dB ±1 dB), DCF linearity, photodetector
- **Coherent NN**: IQ phase encoding, homodyne extraction, softmax normalisation, λ-channel count
- **Meta-prompt**: weight matrix shape/bounds, phi₁ memory accumulation, history trace
- **Trainers**: CE loss descent, convergence on separable data
- **Datasets**: shape, one-hot encoding, all classes represented
- **End-to-end**: all 4 use cases ≥90% fiber accuracy

---

## Repository layout

```
fiber-agi/
├── fiber_physics.py      ITU-T G.652 physics — SMF, EDFA, DCF, PD
├── coherent_nn.py        IQ modulator + homodyne = optical neuron
├── recursive_prompt.py   3-level MetaMetaPrompt weight generator
├── agi_core.py           Dataset generators + ADAM base trainer
├── recursive_dev.py      CrossEntropyTrainer (CE + cosine LR + L2)
├── recursive_dev_v2.py   Recursive self-improving engine V2
├── main.py               → python3 main.py
├── use_cases.py          → python3 use_cases.py
├── tests/
│   └── test_fiber_agi.py 50 tests
├── requirements.txt      numpy>=1.24
└── pyproject.toml        pip installable
```

---

## Extending it

**Larger network:**
```python
# main.py
initial_layer_sizes = [8, 64, 32, 5]
```

**Force more recursive generations:**
```python
# main.py
engine.run(max_gen=10, target_acc=0.999)
```

**Add a new classifier:**
```python
# use_cases.py — copy any run_ucN block
MY_CLASSES = ['ClassA', 'ClassB', 'ClassC']

def _myuc_sample(rng, label):
    # Generate features conditioned on label (label-first strategy)
    if label == 0:
        return np.array([...])   # physically motivated ranges
    ...

def run_uc5(mp, verbose=True):
    X, Y = generate_dataset(_myuc_sample, MY_CLASSES, n=300)
    fiber_nn, acc, _, _ = _train_and_transfer(
        [8, 20, 10, len(MY_CLASSES)], X_tr, Y_tr, X_te, Y_te,
        epochs=700, mp=mp)
    ...
```

**Switch to intensity-mode (V1, 4-pass):**
```python
from fiber_nn import FiberNeuralNetwork   # instead of CoherentFiberNetwork
```

---

## Citation

If you use this in research, please cite the underlying paper:

```bibtex
@article{zang2025fiber,
  title   = {Fiber neural networks for intelligent optical fiber
             communication signal processing},
  author  = {Zang et al.},
  journal = {iOptics},
  year    = {2025}
}
```

---

## License

MIT — use it, fork it, build on it.

---

<div align="center">

**One photon. Four minds. All at light speed.**

*The seed generates its own generator. Recursion is not a technique — it is the substrate.*

[⭐ Star this repo](https://github.com/infinitule/fiber-agi) · [🐛 Open an issue](https://github.com/infinitule/fiber-agi/issues) · [🔀 Fork and extend](https://github.com/infinitule/fiber-agi/fork)

</div>
