"""
prism_agent.py — T7: Agentic Self-Improving Orchestrator
=========================================================
ReAct-style agentic loop: PRISM observes its own accuracy, decides its
next architectural move, executes it, evaluates, and logs the trace —
without human intervention.

Decision policy (rule-based, deterministic):
  acc < 0.70        → AUGMENT_DATA  (expand dataset 2×)
  acc < 0.90        → DEEPEN        (insert new fiber ring loop)
  acc < 0.99        → WIDEN         (expand narrowest hidden layer ×1.5)
  delta < 0.005 ×3  → LOWER_LR     (halve learning rate, reset ADAM)
  gap > 0.05        → RECALIBRATE  (redo per-layer calibration)
  acc >= target     → HALT

Each step is logged as: THOUGHT / ACTION / OBSERVATION

Usage:
    from prism_agent import PRISMAgent, AgentEngine
    engine = AgentEngine(initial_sizes=[8,16,5], X_tr, Y_tr, X_te, Y_te, mp)
    agent  = PRISMAgent(engine, target_acc=0.99, max_steps=20)
    trace  = agent.run()
    agent.print_trace()
"""

import numpy as np
from dataclasses import dataclass, field
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from recursive_dev import CrossEntropyTrainer
from coherent_nn  import CoherentFiberNetwork, minmax_norm
from recursive_prompt import MetaMetaPrompt


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
@dataclass
class AgentStep:
    step:        int
    thought:     str
    action:      str
    observation: str
    fiber_acc:   float
    theory_acc:  float
    arch:        list[int]
    lr:          float
    gen:         int


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class AgentEngine:
    """
    Stateful engine that the PRISMAgent controls.
    Wraps a CrossEntropyTrainer + CoherentFiberNetwork + calibration.
    Exposes the action primitives the agent can invoke.
    """

    def __init__(self, initial_sizes: list[int],
                 X_tr: np.ndarray, Y_tr: np.ndarray,
                 X_te: np.ndarray, Y_te: np.ndarray,
                 mp: MetaMetaPrompt | None = None,
                 rng_seed: int = 42):
        self.sizes   = list(initial_sizes)
        self.X_tr    = X_tr
        self.Y_tr    = Y_tr
        self.X_te    = X_te
        self.Y_te    = Y_te
        self.mp      = mp or MetaMetaPrompt(seed_dim=64, max_depth=16, rng_seed=rng_seed)
        self.rng     = np.random.default_rng(rng_seed)
        self.lr      = 1e-3
        self.gen     = 0
        self.trainer: CrossEntropyTrainer | None = None
        self.fiber_nn: CoherentFiberNetwork | None = None
        self._fiber_acc  = 0.0
        self._theory_acc = 0.0
        self._train(epochs=500)

    # ── Training & transfer ────────────────────────────────────────────────
    def _train(self, epochs: int = 300):
        """Train theoretic model and transfer to fiber NN."""
        trainer = CrossEntropyTrainer(self.sizes, lr=self.lr, l2=1e-4)
        trainer.rng = np.random.default_rng(self.gen)

        # Cosine LR decay
        for ep in range(epochs):
            lr_ep = self.lr * (0.01 + 0.99 * 0.5 * (1 + np.cos(np.pi * ep / epochs)))
            trainer.lr = lr_ep
            idx = np.random.default_rng(ep + self.gen * 1000).permutation(len(self.X_tr))
            for i in idx:
                x_n = minmax_norm(self.X_tr[i])
                trainer.forward(x_n)
                gW, gb = trainer.backward_ce(self.Y_tr[i])
                trainer._adam(gW, gb)

        self._theory_acc = trainer.evaluate(self.X_te, self.Y_te)

        # Build fiber NN
        shapes = [(self.sizes[i+1], self.sizes[i]) for i in range(len(self.sizes)-1)]
        _, encs = self.mp.recursive_unfold(shapes)
        fiber_nn = CoherentFiberNetwork(self.sizes, noisy=False)
        fiber_nn.load_trained_weights(trainer.W, trainer.b)
        fiber_nn.optical_encodings = encs

        # Per-layer calibration
        self._calibrate(fiber_nn, trainer)

        self._fiber_acc = fiber_nn.evaluate(self.X_te, self.Y_te)
        self.trainer    = trainer
        self.fiber_nn   = fiber_nn
        self.gen       += 1

    def _calibrate(self, fiber_nn: CoherentFiberNetwork,
                   trainer: CrossEntropyTrainer):
        """Per-layer output scale calibration."""
        n_hidden = len(fiber_nn.layers) - 1
        for i in range(n_hidden):
            th_vals, fi_vals = [], []
            for x in self.X_tr:
                x_n = minmax_norm(x)
                trainer.forward(x_n)
                th_vals.append(float(np.mean(np.abs(trainer._z[i]))))
                fiber_nn.forward(x)
                fi_vals.append(float(np.mean(np.abs(
                    fiber_nn._layer_outputs[i + 1]))) + 1e-8)
            sc = float(np.clip(np.mean(th_vals) / (np.mean(fi_vals) + 1e-8), 0.1, 8.0))
            fiber_nn.layers[i].W *= sc

    # ── Action primitives ──────────────────────────────────────────────────
    def expand_depth(self):
        """Insert new identity-initialized layer before output."""
        if len(self.sizes) < 3:
            return
        new_width = self.sizes[-2]
        self.sizes = self.sizes[:-1] + [new_width, self.sizes[-1]]
        self.lr   *= 0.7
        self._train(epochs=300)

    def expand_width(self):
        """Widen the narrowest hidden layer by 1.5×."""
        if len(self.sizes) < 3:
            return
        hidden  = self.sizes[1:-1]
        idx     = int(np.argmin(hidden))
        new_w   = max(int(hidden[idx] * 1.5), hidden[idx] + 4)
        self.sizes[1 + idx] = new_w
        self._train(epochs=300)

    def retrain(self, epochs: int = 300):
        """Continue training with current architecture."""
        self._train(epochs=epochs)

    def augment_dataset(self, factor: int = 2):
        """Duplicate training set with small noise."""
        X_aug = np.vstack([self.X_tr] * factor +
                          [self.X_tr + self.rng.normal(0, 0.03, self.X_tr.shape)
                           for _ in range(factor - 1)])
        Y_aug = np.vstack([self.Y_tr] * factor)
        idx   = self.rng.permutation(len(X_aug))
        self.X_tr = X_aug[idx]
        self.Y_tr = Y_aug[idx]

    def recalibrate(self):
        """Re-run calibration without retraining."""
        if self.trainer and self.fiber_nn:
            self._calibrate(self.fiber_nn, self.trainer)
            self._fiber_acc = self.fiber_nn.evaluate(self.X_te, self.Y_te)

    def lower_lr(self):
        """Halve lr and reset ADAM state, then retrain."""
        self.lr *= 0.5
        if self.trainer:
            n = len(self.trainer.W)
            self.trainer.mW = [np.zeros_like(w) for w in self.trainer.W]
            self.trainer.vW = [np.zeros_like(w) for w in self.trainer.W]
            self.trainer.mb = [np.zeros_like(b) for b in self.trainer.b]
            self.trainer.vb = [np.zeros_like(b) for b in self.trainer.b]
            self.trainer.t  = 0
        self._train(epochs=300)

    # ── State accessors ────────────────────────────────────────────────────
    def current_fiber_acc(self) -> float:
        return self._fiber_acc

    def current_theory_acc(self) -> float:
        return self._theory_acc

    def theory_fiber_gap(self) -> float:
        return self._theory_acc - self._fiber_acc

    @property
    def current_arch(self) -> list[int]:
        return list(self.sizes)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class PRISMAgent:
    """
    ReAct-style agentic orchestrator for recursive architecture self-improvement.

    Loop:
      THINK → ACT → OBSERVE → THINK → ...
    Terminates when target_acc is reached or max_steps exhausted.
    """

    ACTION_SPACE = ['DEEPEN', 'WIDEN', 'RETRAIN', 'LOWER_LR',
                    'AUGMENT_DATA', 'RECALIBRATE', 'HALT']

    def __init__(self, engine: AgentEngine,
                 target_acc: float = 0.99,
                 max_steps: int = 20):
        self.engine      = engine
        self.target_acc  = target_acc
        self.max_steps   = max_steps
        self.trace: list[AgentStep] = []
        self.stall_count = 0

    # ── Policy ─────────────────────────────────────────────────────────────
    def _think(self, acc: float, arch: list[int],
               prev_acc: float, gap: float) -> tuple[str, str]:
        """Rule-based policy → (action, rationale)."""
        delta = acc - prev_acc
        n_hidden = len(arch) - 2

        if acc >= self.target_acc:
            return 'HALT', f"Target {self.target_acc:.0%} reached at {acc:.1%} ✓"

        if gap > 0.05:
            return ('RECALIBRATE',
                    f"Theory-fiber gap {gap*100:.1f}% > 5% — redo calibration")

        if delta < 0.002 and self.stall_count >= 2:
            return ('LOWER_LR',
                    f"Stalled {self.stall_count+1} steps (Δ={delta:+.4f}) — halve LR")

        if acc < 0.70:
            return ('AUGMENT_DATA',
                    f"Accuracy {acc:.1%} < 70% — expand dataset to improve coverage")

        if n_hidden < 1 or (acc < 0.90 and n_hidden <= 1):
            return ('DEEPEN',
                    f"Acc {acc:.1%} < 90% and arch {arch} is shallow — insert ring loop")

        if delta < 0.01 and n_hidden >= 1:
            return ('WIDEN',
                    f"Marginal gain Δ={delta:+.4f} — widen narrowest hidden layer")

        return ('RETRAIN',
                f"Moderate gain Δ={delta:+.4f} — continue training current arch")

    # ── Execution ──────────────────────────────────────────────────────────
    def _act(self, action: str):
        """Execute action on the engine."""
        if action == 'DEEPEN':
            self.engine.expand_depth()
        elif action == 'WIDEN':
            self.engine.expand_width()
        elif action == 'RETRAIN':
            self.engine.retrain(epochs=300)
        elif action == 'LOWER_LR':
            self.engine.lower_lr()
        elif action == 'AUGMENT_DATA':
            self.engine.augment_dataset(factor=2)
            self.engine.retrain(epochs=300)
        elif action == 'RECALIBRATE':
            self.engine.recalibrate()

    def run(self) -> list[AgentStep]:
        """Execute the agentic loop. Returns decision trace."""
        prev_acc = 0.0
        for step in range(self.max_steps):
            acc    = self.engine.current_fiber_acc()
            th_acc = self.engine.current_theory_acc()
            gap    = self.engine.theory_fiber_gap()
            arch   = self.engine.current_arch
            lr     = self.engine.lr

            action, thought = self._think(acc, arch, prev_acc, gap)

            if action == 'HALT':
                obs = f"Final: fiber={acc:.1%}  theory={th_acc:.1%}  arch={arch}"
                self.trace.append(AgentStep(
                    step=step, thought=thought, action='HALT',
                    observation=obs, fiber_acc=acc, theory_acc=th_acc,
                    arch=arch, lr=lr, gen=self.engine.gen))
                break

            # Execute
            self._act(action)
            new_acc    = self.engine.current_fiber_acc()
            new_th_acc = self.engine.current_theory_acc()
            new_arch   = self.engine.current_arch
            delta      = new_acc - acc

            obs = (f"fiber: {acc:.1%} → {new_acc:.1%} ({delta:+.4f})  "
                   f"theory: {new_th_acc:.1%}  arch: {new_arch}")

            if abs(delta) < 0.002:
                self.stall_count += 1
            else:
                self.stall_count = 0

            self.trace.append(AgentStep(
                step=step, thought=thought, action=action,
                observation=obs, fiber_acc=new_acc, theory_acc=new_th_acc,
                arch=new_arch, lr=self.engine.lr, gen=self.engine.gen))

            prev_acc = acc

        return self.trace

    def print_trace(self):
        """Pretty-print the decision trace."""
        print(f"\n{'═'*64}")
        print(f"  PRISM AGENT — DECISION TRACE  ({len(self.trace)} steps)")
        print(f"  Target: {self.target_acc:.0%}   Max steps: {self.max_steps}")
        print(f"{'═'*64}")
        for s in self.trace:
            status = '✓ HALT' if s.action == 'HALT' else f'→ {s.action}'
            print(f"\n  Step {s.step:>2}  [{status}]")
            print(f"  THINK  : {s.thought}")
            print(f"  OBSERVE: {s.observation}")
            print(f"  State  : fiber={s.fiber_acc*100:.1f}%  "
                  f"theory={s.theory_acc*100:.1f}%  "
                  f"arch={s.arch}  lr={s.lr:.1e}  gen={s.gen}")

        final = self.trace[-1]
        print(f"\n{'─'*64}")
        print(f"  Final fiber accuracy : {final.fiber_acc*100:.1f}%")
        print(f"  Final architecture   : {final.arch}")
        print(f"  Actions taken        : "
              + ", ".join(dict.fromkeys(s.action for s in self.trace)))
        print(f"{'═'*64}")

    def best_accuracy(self) -> float:
        if not self.trace:
            return 0.0
        return max(s.fiber_acc for s in self.trace)

    def action_counts(self) -> dict:
        from collections import Counter
        return dict(Counter(s.action for s in self.trace))
