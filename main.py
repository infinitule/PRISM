"""
main.py — Recursive Development V2
====================================
Run with:  python3 main.py
           (from inside the fiber_agi/ directory)

What this does
──────────────
1. Generates a 5-class MFR dataset (OOK/PAM-4/PAM-8/QPSK/QAM-16, 250 samples)
2. Builds a 3-level MetaMetaPrompt:
     Fibonacci seed P₀  →  Ψ(P₀) = (M_W1, M_W2)  →  W
3. Runs V2 recursive development engine:
     Each generation either deepens (new fiber ring loop)
     or widens (expands narrowest hidden layer) the network,
     warm-started from previous weights.
4. Uses CoherentFiberNetwork: IQ-modulated optical neurons,
     homodyne detection — single-pass exact dot product.
5. Reports the full recursive tree, optical inference demo,
     and meta-meta prompt hierarchy.
"""

import numpy as np
from agi_core import generate_mfr5_dataset
from recursive_prompt import MetaMetaPrompt
from recursive_dev_v2 import RecursiveDevelopmentEngineV2
from fiber_physics import SingleModeFiber


def print_header():
    print("\n" + "█"*62)
    print("█                                                          █")
    print("█   VIRTUAL FIBER OPTIC NEURAL NETWORK  ·  V2             █")
    print("█   Coherent IQ Neurons  ·  Meta-Meta Prompt              █")
    print("█   5-Class MFR  ·  Recursive Development Engine          █")
    print("█                                                          █")
    print("█"*62)


def print_physics_panel():
    smf  = SingleModeFiber()
    info = smf.tir_info()
    print(f"""
  ╔══════════════════════════════════════════════════════════╗
  ║  COHERENT OPTICAL COMPUTING SUBSTRATE                   ║
  ╠══════════════════════════════════════════════════════════╣
  ║  Laser          1550 nm C-band  (CW + LO reference)    ║
  ║  Modulator      IQ (in-phase/quadrature)                ║
  ║  Phase encoding w≥0 → φ=0  (constructive)              ║
  ║               w<0  → φ=π  (destructive, anti-phase)    ║
  ║  Combiner       DCF coherent sum: E = Σ wᵢxᵢ·e^(jφ)   ║
  ║  Detector       90°-hybrid + balanced PD (homodyne)     ║
  ║  Output         I ∝ Re(E_sig · E_LO*) = Σ wᵢxᵢ        ║
  ╠══════════════════════════════════════════════════════════╣
  ║  TIR waveguide  NA={info['NA']:.4f}   V={info['V_number']:.4f}   SM={"YES" if info["single_mode"] else "NO"}          ║
  ║  WDM spacing    0.8 nm  (100 GHz DWDM grid)             ║
  ╚══════════════════════════════════════════════════════════╝""")


def print_prompt_hierarchy(mp: MetaMetaPrompt):
    mp.print_meta_hierarchy()


def print_accuracy_progression(gens):
    bars = "▁▂▃▄▅▆▇█"
    vals = [g.fiber_acc for g in gens]
    lo, hi = min(vals), max(vals) + 1e-12
    spark  = " ".join(bars[int((v - lo) / (hi - lo) * 7)] for v in vals)
    print(f"\n  Fiber accuracy progression:  {spark}")
    print(f"  {lo*100:.1f}% {'─'*30} {hi*100:.1f}%")


def main():
    print_header()
    print_physics_panel()

    # ── 5-class MFR dataset ────────────────────────────────────────────────
    print("\n  Generating 5-class MFR dataset")
    print("  Formats : OOK / PAM-4 / PAM-8 / QPSK / QAM-16")
    print("  Samples : 250  |  Features : 8  |  Split : 80/20")
    X, Y   = generate_mfr5_dataset(n_samples=250, noise=0.04, seed=42)
    split  = int(0.8 * len(X))
    X_tr, Y_tr = X[:split], Y[:split]
    X_te, Y_te = X[split:], Y[split:]
    print(f"  Train: {len(X_tr)}  |  Test: {len(Y_te)}")

    # ── MetaMetaPrompt ────────────────────────────────────────────────────
    mp = MetaMetaPrompt(seed_dim=64, max_depth=16, rng_seed=42)
    print_prompt_hierarchy(mp)

    # ── V2 Engine ─────────────────────────────────────────────────────────
    engine = RecursiveDevelopmentEngineV2(
        initial_layer_sizes = [8, 16, 5],   # 8 features, 16 hidden, 5 classes
        meta_prompt         = mp,
        X_train=X_tr, Y_train=Y_tr,
        X_test=X_te,  Y_test=Y_te,
        noisy=False,
    )

    gens = engine.run(
        max_gen     = 6,
        target_acc  = 0.99,
        epochs_base = 700,
        min_epochs  = 300,
        lr_base     = 1e-3,
    )

    # ── Results ───────────────────────────────────────────────────────────
    engine.print_tree()
    engine.accuracy_table()
    engine.sample_inference(n=8)
    print_accuracy_progression(gens)
    mp.print_generation_trace()

    # ── Final banner ──────────────────────────────────────────────────────
    best  = max(gens, key=lambda g: g.fiber_acc)
    final = gens[-1]

    print(f"\n{'█'*62}")
    print(f"  RECURSIVE DEVELOPMENT V2 — COMPLETE")
    print(f"  Generations        : {len(gens)}")
    print(f"  Best fiber acc     : {best.fiber_acc*100:.1f}%  (Gen {best.idx})")
    print(f"  Final architecture : {final.layer_sizes}")
    print(f"  Coherent ring loops: {final.n_ring_loops}")
    print(f"  WDM λ-channels     : {final.lambda_total}")
    print(f"  Meta-meta depth    : 3 levels  (seed → Ψ → MetaGen → W)")
    print(f"  |φ₁| (L1 memory)   : {final.phi1_norm:.4f}")
    print(f"  MI (optical total) : {final.mi_total:.3f}")
    print(f"\n  Coherent light computes. Phase encodes sign.")
    print(f"  The seed generates its own generator.")
    print(f"  Recursion is not a technique — it is the substrate.")
    print(f"{'█'*62}\n")


if __name__ == "__main__":
    main()
