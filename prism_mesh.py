"""
prism_mesh.py — T5: Multi-Node WDM Optical Ring Mesh
=====================================================
Simulates a WDM ring topology with N PRISM nodes.
Each node is a trained CoherentFiberNetwork. Activations propagate
over optical fiber spans with realistic delay and amplitude loss.

Physical model:
  Propagation delay:  t = L / (c / n_eff)     [seconds]
  Amplitude loss:     A = 10^(-α·L / 20)       [linear amplitude factor]
  n_eff = 1.4676 at 1550 nm (group effective index, G.652)

Consensus inference:
  Each node infers on its local input. The output (probability vector)
  is attenuated over the link and delivered as partial input to the
  next node. After N rounds, all node outputs are averaged.

Use cases:
  - Distributed impairment monitoring across a metro ring
  - Consensus-based fault localisation in ROADM networks
  - Simulation of propagation delay in coherent OXC rings
"""

import numpy as np
from dataclasses import dataclass, field
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from fiber_physics import SMF_ALPHA_DB_KM, C_LIGHT


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
@dataclass
class FiberSpan:
    """
    One WDM optical link between two PRISM nodes.

    Physical parameters:
      length_km   : fiber span length (km)
      alpha_db_km : attenuation coefficient (dB/km) — G.652 default 0.20
      n_eff       : group effective refractive index — 1.4676 @ 1550 nm
    """
    length_km:   float
    alpha_db_km: float = SMF_ALPHA_DB_KM
    n_eff:       float = 1.4676

    @property
    def propagation_delay_ns(self) -> float:
        """Round-trip propagation delay in nanoseconds."""
        return self.length_km * 1e3 / (C_LIGHT / self.n_eff) * 1e9

    @property
    def loss_db(self) -> float:
        """Total power loss in dB."""
        return self.alpha_db_km * self.length_km

    @property
    def field_loss(self) -> float:
        """Amplitude loss factor (linear, < 1.0)."""
        return 10 ** (-self.loss_db / 20)

    def propagate(self, activation: np.ndarray) -> np.ndarray:
        """
        Attenuate activation vector over fiber span.
        Each element (WDM channel) experiences the same loss.
        """
        return activation * self.field_loss


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
@dataclass
class PRISMNode:
    """
    One node in the optical mesh — a trained CoherentFiberNetwork.

    Attributes:
      node_id     : unique string identifier
      fiber_nn    : trained CoherentFiberNetwork
      class_names : list of class names for this node's task
      neighbors   : dict {neighbor_id: FiberSpan}
    """
    node_id:     str
    fiber_nn:    object          # CoherentFiberNetwork
    class_names: list[str]
    neighbors:   dict = field(default_factory=dict)

    def infer(self, x: np.ndarray) -> np.ndarray:
        """Run coherent NN inference, return probability vector."""
        return self.fiber_nn.forward(x)

    def predict(self, x: np.ndarray) -> tuple[str, float]:
        """Return (class_name, confidence)."""
        probs = self.infer(x)
        idx   = int(np.argmax(probs))
        return self.class_names[idx], float(probs[idx])

    def add_link(self, neighbor_id: str, span: 'FiberSpan'):
        self.neighbors[neighbor_id] = span


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class WDMRingMesh:
    """
    WDM ring of N PRISM nodes connected by fiber spans.

    Topology: node[0] → span[0] → node[1] → span[1] → ... → node[N-1] → span[N-1] → node[0]

    Consensus inference algorithm:
      Round 1: each node infers on the input x. Output is attenuated and
               passed (dimension-projected) to the next node's input.
      Round 2: each node infers on the received activation. Outputs averaged.
      Final:   weighted consensus (weight ∝ 1/span_loss).
    """

    def __init__(self, nodes: list[PRISMNode], ring_spans: list[FiberSpan]):
        if len(nodes) != len(ring_spans):
            raise ValueError(
                f"Expected {len(nodes)} spans (one per inter-node link), "
                f"got {len(ring_spans)}")
        self.nodes   = {n.node_id: n for n in nodes}
        self.ordered = [n.node_id for n in nodes]
        self._spans  = ring_spans

        # Wire ring: node[i] → span[i] → node[(i+1) % N]
        for i, node in enumerate(nodes):
            next_id = nodes[(i + 1) % len(nodes)].node_id
            node.add_link(next_id, ring_spans[i])

        self.total_delay_ns = sum(s.propagation_delay_ns for s in ring_spans)
        self.total_loss_db  = sum(s.loss_db for s in ring_spans)

    # ── Internal utilities ─────────────────────────────────────────────────
    @staticmethod
    def _project(v: np.ndarray, n_target: int) -> np.ndarray:
        """Pad or truncate v to match target input dimension."""
        if len(v) == n_target:
            return v
        if len(v) < n_target:
            return np.pad(v, (0, n_target - len(v)))
        return v[:n_target]

    def _next_node_id(self, current_id: str) -> str:
        neighbors = self.nodes[current_id].neighbors
        return list(neighbors.keys())[0] if neighbors else current_id

    # ── Consensus inference ────────────────────────────────────────────────
    def consensus_infer(self, x: np.ndarray,
                        rounds: int = 2) -> dict:
        """
        Consensus inference around the WDM ring.

        Args:
            x      : input feature vector (applied to first node; propagated)
            rounds : number of ring traversal rounds

        Returns:
            consensus       : average probability vector across all nodes
            node_outputs    : {node_id: probability_vector}
            node_predictions: {node_id: (class_name, confidence)}
            delay_ns        : total ring propagation delay
            loss_db         : total ring power loss
        """
        node_outputs: dict[str, np.ndarray] = {}
        activation   = x.copy()

        for _ in range(rounds):
            for nid in self.ordered:
                node = self.nodes[nid]
                n_in = node.fiber_nn.layers[0].n_in
                x_in = self._project(activation, n_in)
                out  = node.infer(x_in)
                node_outputs[nid] = out

                # Propagate attenuated output to next node
                next_id = self._next_node_id(nid)
                if next_id and next_id in self.nodes:
                    span      = node.neighbors[next_id]
                    activation = span.propagate(out)

        # Weighted consensus: weight = 1 / exp(span_loss_dB / 10)
        weights  = np.array([10 ** (-s.loss_db / 20) for s in self._spans])
        weights /= weights.sum()

        all_probs = np.stack([node_outputs[nid] for nid in self.ordered])
        # Pad/truncate to same shape for averaging (nodes may have different n_classes)
        max_cls  = max(p.shape[0] for p in all_probs)
        padded   = np.array([
            np.pad(p, (0, max_cls - len(p))) if len(p) < max_cls else p
            for p in all_probs
        ])
        consensus = (padded * weights[:, None]).sum(axis=0)

        node_preds = {
            nid: (node.class_names[int(np.argmax(node_outputs[nid]))],
                  float(np.max(node_outputs[nid])))
            for nid, node in self.nodes.items()
        }

        return {
            'consensus':        consensus,
            'node_outputs':     node_outputs,
            'node_predictions': node_preds,
            'delay_ns':         self.total_delay_ns,
            'loss_db':          self.total_loss_db,
        }

    def evaluate_consensus(self, X: np.ndarray, Y: np.ndarray,
                            rounds: int = 2) -> float:
        """
        Evaluate consensus accuracy on a dataset.
        Y is one-hot encoded. Consensus is compared to the majority class.
        """
        n_correct = 0
        max_cls = Y.shape[1]
        for x, y in zip(X, Y):
            result  = self.consensus_infer(x, rounds=rounds)
            cons    = result['consensus'][:max_cls]
            pred    = int(np.argmax(cons))
            true_c  = int(np.argmax(y))
            if pred == true_c:
                n_correct += 1
        return n_correct / len(X)

    # ── Reporting ──────────────────────────────────────────────────────────
    def topology_report(self):
        """Print ring mesh topology summary."""
        n = len(self.ordered)
        print(f"\n  WDM Ring Mesh — {n} nodes")
        print(f"  Total propagation delay : {self.total_delay_ns:.2f} ns")
        print(f"  Total ring loss         : {self.total_loss_db:.2f} dB")
        print(f"\n  {'Node':>14} → {'Next':>14}  {'Length':>8}  "
              f"{'Delay':>8}  {'Loss':>7}  {'λ-ch':>5}")
        print(f"  {'─'*70}")
        for i, nid in enumerate(self.ordered):
            span    = self._spans[i]
            next_id = self.ordered[(i + 1) % n]
            n_lam   = self.nodes[nid].fiber_nn.total_lambda_channels()
            print(f"  {nid:>14} → {next_id:<14}  "
                  f"{span.length_km:>6.0f} km  "
                  f"{span.propagation_delay_ns:>6.2f} ns  "
                  f"{span.loss_db:>5.2f} dB  "
                  f"{n_lam:>5}")

    def node_report(self, x: np.ndarray, rounds: int = 2):
        """Run consensus inference and print per-node decisions."""
        result = self.consensus_infer(x, rounds=rounds)
        print(f"\n  Consensus Inference  (delay={result['delay_ns']:.1f} ns  "
              f"loss={result['loss_db']:.1f} dB)")
        print(f"  {'Node':<14}  {'Class':<14}  {'Confidence':>10}")
        print(f"  {'─'*44}")
        for nid, (cls, conf) in result['node_predictions'].items():
            print(f"  {nid:<14}  {cls:<14}  {conf*100:>9.1f}%")
        cons = result['consensus']
        pred_idx = int(np.argmax(cons))
        # consensus class name from first node's class list
        first_nn  = self.nodes[self.ordered[0]]
        pred_cls  = (first_nn.class_names[pred_idx]
                     if pred_idx < len(first_nn.class_names)
                     else f'class_{pred_idx}')
        print(f"\n  Consensus: {pred_cls}  (p={float(cons[pred_idx]):.3f})")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def build_demo_ring(verbose: bool = True) -> WDMRingMesh:
    """
    Build a 4-node AMS-LON-FRA-PAR ring with MIC classifiers at each node.
    Each node is trained on the same MIC dataset (proof-of-concept).
    """
    from use_cases import generate_mic_dataset, _train_and_transfer, MIC_CLASSES
    from recursive_prompt import MetaMetaPrompt

    mp    = MetaMetaPrompt(seed_dim=64, max_depth=16, rng_seed=42)
    X, Y  = generate_mic_dataset(n=300)
    split = int(0.8 * len(X))

    if verbose:
        print("  Building 4-node WDM ring (training MIC classifier at each node)...")

    # Train one shared model, clone weights to each node
    nn_ref, acc, th_acc, _ = _train_and_transfer(
        [8, 20, 10, 5], X[:split], Y[:split], X[split:], Y[split:],
        epochs=600, mp=mp, verbose=False)

    if verbose:
        print(f"  MIC fiber accuracy at each node: {acc*100:.1f}%")

    # 4-node ring: AMS → LON → FRA → PAR → AMS
    node_ids = ['AMS', 'LON', 'FRA', 'PAR']
    span_lengths = [550, 680, 560, 640]   # km (realistic metro-core distances)

    nodes = [PRISMNode(node_id=nid, fiber_nn=nn_ref, class_names=MIC_CLASSES)
             for nid in node_ids]
    spans = [FiberSpan(length_km=L) for L in span_lengths]
    mesh  = WDMRingMesh(nodes, spans)

    return mesh


if __name__ == "__main__":
    mesh = build_demo_ring(verbose=True)
    mesh.topology_report()

    from use_cases import generate_mic_dataset, MIC_CLASSES, _mic_sample
    import numpy as np
    rng = np.random.default_rng(77)
    x_probe = _mic_sample(rng, label=2)   # PMD event
    mesh.node_report(x_probe)
