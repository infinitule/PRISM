"""
tests/test_fiber_agi.py
=======================
Full test suite for the Virtual Fiber Optic Neural Network.

Run with:   python3 -m pytest tests/ -v
  or:       python3 -m unittest discover tests/
"""

import sys
import os
import unittest
import numpy as np

# Make sure we can import from the package root
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from fiber_physics import (
    SingleModeFiber, EDFA, AmplitudeModulator,
    DispersionCompensatingFiber, Photodetector,
)
from coherent_nn import (
    IQModulator, HomodyneReceiver, CoherentFiberNeuron,
    CoherentFiberNetwork, minmax_norm,
)
from recursive_prompt import MetaMetaPrompt
from agi_core import generate_mfr5_dataset, TheoreticalTrainer
from recursive_dev import CrossEntropyTrainer
from use_cases import (
    generate_mic_dataset, generate_amc_dataset,
    generate_fde_dataset, generate_dci_dataset,
    MIC_CLASSES, AMC_CLASSES, FDE_CLASSES, DCI_CLASSES,
)


# ═══════════════════════════════════════════════════════════════
# PHYSICS LAYER
# ═══════════════════════════════════════════════════════════════

class TestSingleModeFiber(unittest.TestCase):

    def setUp(self):
        self.smf = SingleModeFiber()

    def test_tir_single_mode(self):
        info = self.smf.tir_info()
        self.assertTrue(info['single_mode'],
                        "G.652 SMF must be single-mode at 1550 nm")

    def test_tir_na(self):
        info = self.smf.tir_info()
        self.assertAlmostEqual(info['NA'], 0.1235, places=3)

    def test_tir_v_number(self):
        info = self.smf.tir_info()
        self.assertLess(info['V_number'], 2.405,
                        "V < 2.405 required for single-mode")

    def test_attenuate_reduces_amplitude(self):
        E_in = 1.0
        E_out = self.smf.attenuate(E_in)
        self.assertLess(abs(E_out), abs(E_in),
                        "170 km SMF must attenuate signal")

    def test_attenuate_positive(self):
        E_out = self.smf.attenuate(1.0)
        self.assertGreater(E_out, 0.0)


class TestEDFA(unittest.TestCase):

    def setUp(self):
        self.edfa = EDFA()

    def test_amplifies(self):
        E_in = 0.1
        E_out = self.edfa.amplify(E_in)
        self.assertGreater(abs(E_out), abs(E_in))

    def test_gain_20dB(self):
        """EDFA should provide ~20 dB gain (factor 10 in amplitude)."""
        E_in = 0.01
        E_out = self.edfa.amplify(E_in)
        gain_dB = 20 * np.log10(abs(E_out) / abs(E_in))
        self.assertAlmostEqual(gain_dB, 20.0, delta=1.0)


class TestDCF(unittest.TestCase):

    def setUp(self):
        self.dcf = DispersionCompensatingFiber()

    def test_dot_product_proportional(self):
        """DCF dot product must scale linearly with input magnitude."""
        w = [0.5, 0.3, 0.8]
        x = [0.4, 0.9, 0.2]
        r1 = self.dcf.dot_product(w, x)
        r2 = self.dcf.dot_product([2*v for v in w], x)
        self.assertAlmostEqual(r2, 2 * r1, places=6)

    def test_dot_product_zero(self):
        r = self.dcf.dot_product([0.0, 0.0], [1.0, 1.0])
        self.assertAlmostEqual(r, 0.0, places=8)

    def test_dot_product_sign(self):
        r_pos = self.dcf.dot_product([0.5], [0.5])
        r_neg = self.dcf.dot_product([-0.5], [0.5])
        self.assertGreater(r_pos, 0)
        self.assertLess(r_neg, 0)


class TestPhotodetector(unittest.TestCase):

    def test_detect_proportional(self):
        pd = Photodetector()
        E1 = 0.5
        E2 = 1.0
        p1 = pd.detect(E1)
        p2 = pd.detect(E2)
        self.assertGreater(p2, p1)


# ═══════════════════════════════════════════════════════════════
# COHERENT NEURAL LAYER
# ═══════════════════════════════════════════════════════════════

class TestIQModulator(unittest.TestCase):

    def setUp(self):
        self.iq = IQModulator()

    def test_positive_weight_zero_phase(self):
        """Positive weight → phase=0 → real output positive."""
        E = self.iq.modulate(complex(1.0, 0.0), 0.5)
        self.assertGreater(E.real, 0)
        self.assertAlmostEqual(E.imag, 0.0, places=8)

    def test_negative_weight_pi_phase(self):
        """Negative weight → phase=π → real output negative."""
        E = self.iq.modulate(complex(1.0, 0.0), -0.5)
        self.assertLess(E.real, 0)

    def test_zero_weight_zero_output(self):
        E = self.iq.modulate(complex(1.0, 0.0), 0.0)
        self.assertAlmostEqual(abs(E), 0.0, places=8)

    def test_amplitude_proportional(self):
        E1 = self.iq.modulate(complex(1.0), 0.3)
        E2 = self.iq.modulate(complex(1.0), 0.6)
        self.assertAlmostEqual(abs(E2), 2 * abs(E1), places=6)


class TestHomodyneReceiver(unittest.TestCase):

    def setUp(self):
        self.rx = HomodyneReceiver()

    def test_real_part_extracted(self):
        """Homodyne must return proportional to Re(E_signal)."""
        E_sig = complex(0.8, 0.4)
        result = self.rx.detect(E_sig, E_LO_amplitude=1.0)
        self.assertAlmostEqual(result, self.rx.R * 1.0 * E_sig.real, places=6)

    def test_zero_signal(self):
        result = self.rx.detect(complex(0.0), E_LO_amplitude=1.0)
        self.assertAlmostEqual(result, 0.0, places=8)


class TestCoherentFiberNeuron(unittest.TestCase):

    def setUp(self):
        self.neuron = CoherentFiberNeuron(n_inputs=4)

    def test_output_shape(self):
        w = np.array([0.5, -0.3, 0.8, -0.2])
        x = np.array([1.0, 0.5, -0.4, 0.9])
        result = self.neuron.compute(w, x)
        self.assertIsInstance(result, float)

    def test_sign_preserved(self):
        """Large positive dot product → positive neuron output."""
        w = np.ones(4)
        x = np.ones(4)
        result = self.neuron.compute(w, x)
        self.assertGreater(result, 0.0)

    def test_linearity(self):
        """compute(2w, x) ≈ 2 * compute(w, x) for small weights."""
        w = np.array([0.1, 0.2, 0.15, 0.05])
        x = np.array([0.3, 0.4, 0.5, 0.2])
        r1 = self.neuron.compute(w, x)
        r2 = self.neuron.compute(2 * w, x)
        self.assertAlmostEqual(r2 / r1, 2.0, delta=0.15)


class TestCoherentFiberNetwork(unittest.TestCase):

    def setUp(self):
        self.net = CoherentFiberNetwork([8, 16, 5], noisy=False)

    def test_forward_shape(self):
        x = np.random.default_rng(0).uniform(-1, 1, 8)
        out = self.net.forward(x)
        self.assertEqual(out.shape, (5,))

    def test_forward_softmax(self):
        """Output must be a valid probability distribution."""
        x = np.random.default_rng(1).uniform(-1, 1, 8)
        out = self.net.forward(x)
        self.assertAlmostEqual(float(np.sum(out)), 1.0, places=5)
        self.assertTrue(np.all(out >= 0))

    def test_predict_integer(self):
        x = np.random.default_rng(2).uniform(-1, 1, 8)
        pred = self.net.predict(x)
        self.assertIsInstance(pred, int)
        self.assertIn(pred, list(range(5)))

    def test_evaluate_range(self):
        rng = np.random.default_rng(3)
        X = rng.uniform(-1, 1, (20, 8))
        Y = np.eye(5)[rng.integers(0, 5, 20)]
        acc = self.net.evaluate(X, Y)
        self.assertGreaterEqual(acc, 0.0)
        self.assertLessEqual(acc, 1.0)

    def test_lambda_channels(self):
        """Total λ-channels = sum of output neurons per layer."""
        lam = self.net.total_lambda_channels()
        # [8,16,5]: layer 0 outputs 16, layer 1 outputs 5 → 21
        self.assertEqual(lam, 21)

    def test_load_trained_weights(self):
        """load_trained_weights must not throw and must change outputs."""
        rng = np.random.default_rng(4)
        x = rng.uniform(-1, 1, 8)
        out_before = self.net.forward(x).copy()

        W_new = [rng.uniform(-0.5, 0.5, (16, 8)),
                 rng.uniform(-0.5, 0.5, (5, 16))]
        b_new = [rng.uniform(-0.1, 0.1, 16),
                 rng.uniform(-0.1, 0.1, 5)]
        self.net.load_trained_weights(W_new, b_new)
        out_after = self.net.forward(x)
        self.assertFalse(np.allclose(out_before, out_after))

    def test_minmax_norm(self):
        x = np.array([0.0, 5.0, 10.0])
        n = minmax_norm(x)
        self.assertAlmostEqual(float(n.min()), -1.0, places=6)
        self.assertAlmostEqual(float(n.max()),  1.0, places=6)


# ═══════════════════════════════════════════════════════════════
# RECURSIVE PROMPT
# ═══════════════════════════════════════════════════════════════

class TestMetaMetaPrompt(unittest.TestCase):

    def setUp(self):
        self.mp = MetaMetaPrompt(seed_dim=64, max_depth=4, rng_seed=0)

    def test_seed_norm(self):
        """Fibonacci seed must have non-zero L2 norm."""
        self.assertGreater(np.linalg.norm(self.mp.P0), 0.0)

    def test_generate_weight_matrix_shape(self):
        p = self.mp.P0.copy()
        W = self.mp.generate_weight_matrix(p, n_out=8, n_in=4)
        self.assertEqual(W.shape, (8, 4))

    def test_generate_weight_matrix_bounded(self):
        p = self.mp.P0.copy()
        W = self.mp.generate_weight_matrix(p, n_out=8, n_in=4)
        self.assertLessEqual(np.abs(W).max(), 1.01)

    def test_recursive_unfold_shapes(self):
        shapes = [(16, 8), (5, 16)]
        Ws, encs = self.mp.recursive_unfold(shapes)
        self.assertEqual(len(Ws), 2)
        self.assertEqual(Ws[0].shape, (16, 8))
        self.assertEqual(Ws[1].shape, (5, 16))

    def test_phi1_accumulates(self):
        """phi_1 memory must grow from zero after weight generation."""
        before = np.linalg.norm(self.mp.phi_1)
        self.mp.generate_weight_matrix(self.mp.P0, 8, 4)
        after = np.linalg.norm(self.mp.phi_1)
        self.assertGreater(after, before)

    def test_history_records(self):
        """recursive_unfold must populate history with one entry per layer."""
        shapes = [(8, 4), (3, 8)]
        n_before = len(self.mp.history)
        self.mp.recursive_unfold(shapes)
        self.assertEqual(len(self.mp.history), n_before + len(shapes))


# ═══════════════════════════════════════════════════════════════
# DATASET GENERATORS
# ═══════════════════════════════════════════════════════════════

class TestDatasets(unittest.TestCase):

    def test_mfr5_shape(self):
        X, Y = generate_mfr5_dataset(n_samples=50, seed=0)
        self.assertEqual(X.shape, (50, 8))
        self.assertEqual(Y.shape, (50, 5))

    def test_mfr5_one_hot(self):
        X, Y = generate_mfr5_dataset(n_samples=50, seed=0)
        self.assertTrue(np.all(Y.sum(axis=1) == 1))

    def test_mfr5_all_classes(self):
        X, Y = generate_mfr5_dataset(n_samples=100, seed=0)
        labels = np.argmax(Y, axis=1)
        self.assertEqual(len(np.unique(labels)), 5)

    def test_mic_shape(self):
        X, Y = generate_mic_dataset(n=100)
        self.assertEqual(X.shape[1], 8)
        self.assertEqual(Y.shape[1], len(MIC_CLASSES))

    def test_amc_shape(self):
        X, Y = generate_amc_dataset(n=120)
        self.assertEqual(X.shape[1], 8)
        self.assertEqual(Y.shape[1], len(AMC_CLASSES))

    def test_fde_shape(self):
        X, Y = generate_fde_dataset(n=100)
        self.assertEqual(X.shape[1], 8)
        self.assertEqual(Y.shape[1], len(FDE_CLASSES))

    def test_dci_shape(self):
        X, Y = generate_dci_dataset(n=100)
        self.assertEqual(X.shape[1], 8)
        self.assertEqual(Y.shape[1], len(DCI_CLASSES))

    def test_all_classes_represented(self):
        for gen, n_cls in [
            (lambda: generate_mic_dataset(n=100), len(MIC_CLASSES)),
            (lambda: generate_amc_dataset(n=120), len(AMC_CLASSES)),
            (lambda: generate_fde_dataset(n=100), len(FDE_CLASSES)),
            (lambda: generate_dci_dataset(n=100), len(DCI_CLASSES)),
        ]:
            X, Y = gen()
            labels = np.argmax(Y, axis=1)
            self.assertEqual(len(np.unique(labels)), n_cls,
                             f"Expected {n_cls} classes, got {len(np.unique(labels))}")


# ═══════════════════════════════════════════════════════════════
# TRAINERS
# ═══════════════════════════════════════════════════════════════

class TestTheoreticalTrainer(unittest.TestCase):

    def setUp(self):
        self.trainer = TheoreticalTrainer([4, 8, 3])

    def test_forward_shape(self):
        x = np.array([0.1, 0.5, -0.3, 0.8])
        out = self.trainer.forward(x)
        self.assertEqual(out.shape, (3,))

    def test_forward_softmax(self):
        x = np.array([0.1, 0.5, -0.3, 0.8])
        out = self.trainer.forward(x)
        self.assertAlmostEqual(float(np.sum(out)), 1.0, places=5)

    def test_evaluate_range(self):
        rng = np.random.default_rng(5)
        X = rng.uniform(-1, 1, (20, 4))
        Y = np.eye(3)[rng.integers(0, 3, 20)]
        acc = self.trainer.evaluate(X, Y)
        self.assertGreaterEqual(acc, 0.0)
        self.assertLessEqual(acc, 1.0)


class TestCrossEntropyTrainer(unittest.TestCase):

    def test_loss_decreases(self):
        """CE loss should decrease over training on a simple 2-class task."""
        rng = np.random.default_rng(42)
        trainer = CrossEntropyTrainer([4, 8, 2], lr=1e-2)
        # Class 0: x ~ N([1,1,1,1], 0.1)   Class 1: x ~ N([-1,-1,-1,-1], 0.1)
        X0 = rng.normal( 1.0, 0.1, (20, 4))
        X1 = rng.normal(-1.0, 0.1, (20, 4))
        X  = np.vstack([X0, X1])
        Y  = np.vstack([np.tile([1,0], (20,1)), np.tile([0,1], (20,1))])

        losses = []
        for ep in range(80):
            ep_loss = 0.0
            for i in rng.permutation(len(X)):
                y_hat = trainer.forward(X[i])
                ep_loss += trainer._ce(y_hat, Y[i])
                gW, gb = trainer.backward_ce(Y[i])
                trainer._adam(gW, gb)
            losses.append(ep_loss / len(X))

        self.assertLess(losses[-1], losses[0],
                        "CE loss must decrease during training")

    def test_convergence_on_separable(self):
        """Trainer must reach >90% on a clearly separable 2D problem."""
        rng = np.random.default_rng(7)
        trainer = CrossEntropyTrainer([2, 8, 2], lr=5e-3)
        X0 = rng.normal([2.0, 0.0], 0.2, (40, 2))
        X1 = rng.normal([-2.0, 0.0], 0.2, (40, 2))
        X  = np.vstack([X0, X1])
        Y  = np.vstack([np.tile([1,0], (40,1)), np.tile([0,1], (40,1))])

        for ep in range(200):
            for i in rng.permutation(len(X)):
                trainer.forward(X[i])
                gW, gb = trainer.backward_ce(Y[i])
                trainer._adam(gW, gb)

        acc = trainer.evaluate(X, Y)
        self.assertGreater(acc, 0.90)


# ═══════════════════════════════════════════════════════════════
# END-TO-END: TRAIN → TRANSFER → FIBER ACCURACY
# ═══════════════════════════════════════════════════════════════

class TestEndToEnd(unittest.TestCase):

    def test_mic_fiber_accuracy(self):
        """UC1 coherent fiber NN must reach ≥90% on MIC test set."""
        from use_cases import run_uc1
        mp = MetaMetaPrompt(seed_dim=64, max_depth=4, rng_seed=0)
        _, acc = run_uc1(mp, verbose=False)
        self.assertGreaterEqual(acc, 0.90,
            f"UC1 MIC fiber accuracy {acc:.1%} < 90% threshold")

    def test_amc_fiber_accuracy(self):
        """UC2 coherent fiber NN must reach ≥90% on AMC test set."""
        from use_cases import run_uc2
        mp = MetaMetaPrompt(seed_dim=64, max_depth=4, rng_seed=0)
        _, acc = run_uc2(mp, verbose=False)
        self.assertGreaterEqual(acc, 0.90,
            f"UC2 AMC fiber accuracy {acc:.1%} < 90% threshold")

    def test_fde_fiber_accuracy(self):
        """UC3 coherent fiber NN must reach ≥90% on FDE test set."""
        from use_cases import run_uc3
        mp = MetaMetaPrompt(seed_dim=64, max_depth=4, rng_seed=0)
        _, acc = run_uc3(mp, verbose=False)
        self.assertGreaterEqual(acc, 0.90,
            f"UC3 FDE fiber accuracy {acc:.1%} < 90% threshold")

    def test_dci_fiber_accuracy(self):
        """UC4 coherent fiber NN must reach ≥90% on DCI test set."""
        from use_cases import run_uc4
        mp = MetaMetaPrompt(seed_dim=64, max_depth=4, rng_seed=0)
        _, acc = run_uc4(mp, verbose=False)
        self.assertGreaterEqual(acc, 0.90,
            f"UC4 DCI fiber accuracy {acc:.1%} < 90% threshold")


# ═══════════════════════════════════════════════════════════════
# T8 — QUANTUM HOMODYNE RECEIVER
# ═══════════════════════════════════════════════════════════════

class TestQuantumHomodyneReceiver(unittest.TestCase):

    def setUp(self):
        from fiber_physics import QuantumHomodyneReceiver
        self.QHR = QuantumHomodyneReceiver

    def test_coherent_state_no_advantage(self):
        """At r=0, quantum advantage must equal 1 (no squeezing)."""
        qr = self.QHR(squeezing_r=0.0)
        adv = qr.quantum_advantage(n_photons=100)
        self.assertAlmostEqual(adv, 1.0, places=4)

    def test_squeezing_reduces_variance(self):
        """Squeezing r>0 must reduce x-quadrature variance below 1/4."""
        qr_sq   = self.QHR(squeezing_r=1.0, eta=1.0)
        qr_coh  = self.QHR(squeezing_r=0.0, eta=1.0)
        v_sq    = qr_sq.shot_noise_variance(n_photons=100)
        v_coh   = qr_coh.shot_noise_variance(n_photons=100)
        self.assertLess(v_sq, v_coh)

    def test_snr_improvement_positive(self):
        """SNR improvement must be positive for r>0."""
        qr = self.QHR(squeezing_r=0.5)
        self.assertGreater(qr.snr_improvement_db(), 0.0)

    def test_snr_improvement_zero_at_r0(self):
        """SNR improvement must be 0 dB at r=0."""
        qr = self.QHR(squeezing_r=0.0)
        self.assertAlmostEqual(qr.snr_improvement_db(), 0.0, places=6)

    def test_qfi_exceeds_cfi(self):
        """QFI must exceed CFI for r>0, N>1."""
        qr  = self.QHR(squeezing_r=1.0)
        qfi = qr.quantum_fisher_information(50)
        cfi = qr.classical_fisher_information(50)
        self.assertGreater(qfi, cfi)

    def test_qfi_equals_cfi_at_r0(self):
        """At r=0 and N>>1: QFI ≈ N (cosh(0)=1, sinh(0)=0)."""
        qr  = self.QHR(squeezing_r=0.0)
        N   = 100.0
        qfi = qr.quantum_fisher_information(N)
        self.assertAlmostEqual(qfi, N, places=4)

    def test_snr_8686_per_r(self):
        """ΔSNRdB ≈ 8.686 per unit r (known analytical result)."""
        qr    = self.QHR(squeezing_r=1.0)
        delta = qr.snr_improvement_db()
        self.assertAlmostEqual(delta, 8.686, places=2)

    def test_invalid_r_raises(self):
        """Negative squeezing must raise ValueError."""
        with self.assertRaises(ValueError):
            self.QHR(squeezing_r=-0.1)


# ═══════════════════════════════════════════════════════════════
# T3 — ONLINE FIBER TRAINER (EWC)
# ═══════════════════════════════════════════════════════════════

class TestOnlineFiberTrainer(unittest.TestCase):

    def _make_separable(self):
        rng = np.random.default_rng(11)
        X0  = rng.normal([3, 0], 0.3, (20, 2))
        X1  = rng.normal([-3, 0], 0.3, (20, 2))
        X   = np.vstack([X0, X1])
        Y   = np.vstack([np.tile([1, 0], (20, 1)), np.tile([0, 1], (20, 1))])
        return X, Y

    def test_import(self):
        from recursive_dev import OnlineFiberTrainer
        t = OnlineFiberTrainer([4, 8, 3], lr=1e-3)
        self.assertEqual(t.sizes, [4, 8, 3])

    def test_consolidate_sets_anchor(self):
        from recursive_dev import OnlineFiberTrainer
        X, Y = self._make_separable()
        t = OnlineFiberTrainer([2, 8, 2], lr=1e-3)
        t.consolidate(X, Y, n_samples=10)
        self.assertTrue(t._consolidated)
        self.assertEqual(len(t._anchor_W), len(t.W))
        self.assertEqual(len(t._fisher_W), len(t.W))

    def test_ewc_penalty_zero_before_consolidate(self):
        from recursive_dev import OnlineFiberTrainer
        t = OnlineFiberTrainer([2, 4, 2], lr=1e-3)
        self.assertAlmostEqual(t.ewc_penalty(), 0.0)

    def test_online_update_returns_float(self):
        from recursive_dev import OnlineFiberTrainer
        X, Y = self._make_separable()
        t = OnlineFiberTrainer([2, 4, 2], lr=1e-3)
        loss = t.online_update(X[0], Y[0])
        self.assertIsInstance(loss, float)
        self.assertGreater(loss, 0.0)

    def test_stream_reduces_loss(self):
        from recursive_dev import OnlineFiberTrainer
        X, Y = self._make_separable()
        t    = OnlineFiberTrainer([2, 16, 2], lr=5e-3)
        losses = t.stream_train(X, Y, shuffle=False, verbose_every=0)
        self.assertLess(losses[-1], losses[0])

    def test_ewc_penalty_increases_after_divergence(self):
        """After consolidation, diverging weights → larger EWC penalty."""
        from recursive_dev import OnlineFiberTrainer
        X, Y = self._make_separable()
        t = OnlineFiberTrainer([2, 4, 2], lr=1e-3, lambda_ewc=1.0)
        t.consolidate(X, Y, n_samples=20)
        pen_before = t.ewc_penalty()
        # Force-shift weights
        for w in t.W:
            w += 5.0
        pen_after = t.ewc_penalty()
        self.assertGreater(pen_after, pen_before)


# ═══════════════════════════════════════════════════════════════
# T1 — REAL DATA BRIDGE (CSV / JSON / domain shift)
# ═══════════════════════════════════════════════════════════════

class TestRealDataBridge(unittest.TestCase):

    def setUp(self):
        import tempfile
        self.tmp = tempfile.mkdtemp()

    def _write_csv(self, filename, rows, header):
        import csv
        path = os.path.join(self.tmp, filename)
        with open(path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=header)
            w.writeheader()
            w.writerows(rows)
        return path

    def test_load_csv_basic(self):
        from use_cases import load_csv_dataset
        header = ['OSNR', 'BER', 'label']
        rows   = [{'OSNR': 28.0, 'BER': -7.0, 'label': 'Normal'},
                  {'OSNR': 15.0, 'BER': -4.0, 'label': 'Impaired'},
                  {'OSNR': 10.0, 'BER': -2.0, 'label': 'Critical'},]
        path   = self._write_csv('test.csv', rows, header)
        X, Y, classes = load_csv_dataset(path)
        self.assertEqual(X.shape, (3, 2))
        self.assertEqual(Y.shape, (3, 3))
        self.assertEqual(len(classes), 3)

    def test_load_csv_feature_cols(self):
        from use_cases import load_csv_dataset
        header = ['OSNR', 'BER', 'EVM', 'label']
        rows   = [{'OSNR': 28.0, 'BER': -7.0, 'EVM': 2.0, 'label': 'OK'},
                  {'OSNR': 15.0, 'BER': -4.0, 'EVM': 8.0, 'label': 'Bad'}]
        path   = self._write_csv('test2.csv', rows, header)
        X, Y, classes = load_csv_dataset(path, feature_cols=['OSNR', 'BER'])
        self.assertEqual(X.shape, (2, 2))

    def test_load_json_stream(self):
        from use_cases import load_json_stream
        import json
        path = os.path.join(self.tmp, 'stream.ndjson')
        with open(path, 'w') as f:
            for i in range(5):
                f.write(json.dumps({'osnr': 28.0, 'ber': -7.0,
                                    'fault': 'normal'}) + '\n')
            for i in range(5):
                f.write(json.dumps({'osnr': 10.0, 'ber': -2.0,
                                    'fault': 'break'}) + '\n')
        X, Y, classes = load_json_stream(path, ['osnr', 'ber'], 'fault')
        self.assertEqual(X.shape, (10, 2))
        self.assertEqual(len(classes), 2)

    def test_detect_domain_shift_true(self):
        from use_cases import detect_domain_shift
        X_ref = np.random.default_rng(0).normal(0, 1, (50, 8))
        X_new = X_ref + 10.0   # massive shift
        result = detect_domain_shift(X_ref, X_new)
        self.assertTrue(result['shifted'])
        self.assertGreater(result['max_z'], 2.0)

    def test_detect_domain_shift_false(self):
        from use_cases import detect_domain_shift
        rng   = np.random.default_rng(5)
        X_ref = rng.normal(0, 1, (50, 8))
        X_new = rng.normal(0, 1, (50, 8))   # same distribution
        result = detect_domain_shift(X_ref, X_new, threshold=5.0)
        self.assertFalse(result['shifted'])


# ═══════════════════════════════════════════════════════════════
# T2 — MIC-64 SCALE BREAKOUT
# ═══════════════════════════════════════════════════════════════

class TestMIC64(unittest.TestCase):

    def test_feature_count(self):
        from use_cases import MIC64_FEATURES
        self.assertEqual(len(MIC64_FEATURES), 64)

    def test_sample_shape(self):
        from use_cases import _mic64_sample
        rng = np.random.default_rng(0)
        x   = _mic64_sample(rng, 0)
        self.assertEqual(x.shape, (64,))

    def test_all_labels(self):
        from use_cases import _mic64_sample, MIC_CLASSES
        rng = np.random.default_rng(0)
        for lbl in range(len(MIC_CLASSES)):
            x = _mic64_sample(rng, lbl)
            self.assertEqual(x.shape, (64,))

    def test_dataset_shape(self):
        from use_cases import generate_mic64_dataset
        X, Y = generate_mic64_dataset(n=50, seed=0)
        self.assertEqual(X.shape, (50, 64))
        self.assertEqual(Y.shape, (50, 5))

    def test_one_hot_labels(self):
        from use_cases import generate_mic64_dataset
        _, Y = generate_mic64_dataset(n=50, seed=0)
        np.testing.assert_array_equal(Y.sum(axis=1), np.ones(50))


# ═══════════════════════════════════════════════════════════════
# T4 — PRISM API
# ═══════════════════════════════════════════════════════════════

class TestPRISMAPI(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from prism_api import PRISMApp
        cls.app = PRISMApp(rng_seed=0)

    def test_health_ok(self):
        h = self.app.health()
        self.assertEqual(h['status'], 'ok')
        self.assertIn('mic', h['available_models'])

    def test_models_list(self):
        models = self.app.models()
        ids = [m['id'] for m in models]
        for uc in ['mic', 'amc', 'fde', 'dci']:
            self.assertIn(uc, ids)

    def test_infer_mic_returns_class(self):
        x = [2.0, 14.0, 0.85, -7.0, 0.1, 3.0, 10.0, 1.0]
        result = self.app.infer('mic', x)
        self.assertIn(result['predicted_class'], ['Normal', 'CD-dominant',
            'PMD-dominant', 'OSNR-limited', 'Nonlinear'])
        self.assertAlmostEqual(
            sum(result['probabilities'].values()), 1.0, places=4)

    def test_infer_confidence_in_range(self):
        x = [2.0, 14.0, 0.85, -7.0, 0.1, 3.0, 10.0, 1.0]
        result = self.app.infer('mic', x)
        self.assertGreaterEqual(result['confidence'], 0.0)
        self.assertLessEqual(result['confidence'], 1.0)

    def test_infer_invalid_model_raises(self):
        with self.assertRaises(ValueError):
            self.app.infer('nonexistent_uc', [1.0, 2.0])

    def test_domain_shift_returns_dict(self):
        X_batch = [[2.0, 14.0, 0.85, -7.0, 0.1, 3.0, 10.0, 1.0]] * 5
        result  = self.app.check_domain_shift('mic', X_batch)
        self.assertIn('shifted', result)
        self.assertIn('z_scores', result)


if __name__ == "__main__":
    unittest.main(verbosity=2)
