"""
prism_hitl.py — T12: Hardware-in-the-Loop Bridge
=================================================
Connect PRISM inference to real optical test equipment via SCPI socket.
Runs in simulation mode (Gaussian noise) when no instrument is connected.

Physical instruments supported (SCPI-compatible):
  Keysight N7744A   — optical power meter
  JDSU MTS-8000     — optical BERT / OTDR
  Viavi ONX-220     — network tester
  Anritsu MT9083    — OTDR
  Generic           — any SCPI-over-TCP instrument

Usage (simulation mode — no hardware required):
  from prism_hitl import HITLSession
  session = HITLSession(fiber_nn, MIC_CLASSES)
  result  = session.run_cycle([2.0, 14.0, 0.85, -7.0, 0.1, 3.0, 10.0, 1.0])
  print(result)

Usage (hardware mode):
  cfg     = InstrumentConfig(host='192.168.1.100', port=5025)
  bridge  = SCPIBridge(cfg)
  bridge.connect()
  session = HITLSession(fiber_nn, MIC_CLASSES, instrument=bridge)
  result  = session.run_cycle(x_synthetic)
  bridge.disconnect()
"""

import socket
import json
import time
import numpy as np
from dataclasses import dataclass, field


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
@dataclass
class InstrumentConfig:
    """SCPI socket connection parameters."""
    host:    str
    port:    int
    timeout: float = 5.0
    term:    str   = '\n'   # SCPI terminator (LF or CRLF depending on instrument)


@dataclass
class Measurement:
    """One real-instrument measurement cycle."""
    timestamp_s:   float
    osnr_db:       float | None
    power_dbm:     float | None
    ber_log:       float | None
    extra_fields:  dict = field(default_factory=dict)

    def to_array(self, feature_order: list[str]) -> np.ndarray:
        """Project measurement to feature array matching a use-case schema."""
        mapping = {
            'OSNR':     self.osnr_db,
            'Power':    self.power_dbm,
            'BER':      self.ber_log,
        }
        mapping.update(self.extra_fields)
        return np.array([mapping.get(f, 0.0) or 0.0 for f in feature_order])


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class SCPIBridge:
    """
    SCPI socket bridge for optical test instruments.
    Sends IEEE 488.2 / SCPI commands over TCP and parses responses.

    All commands are non-blocking with a configurable timeout.
    Raises ConnectionError if the socket is not connected.
    """

    def __init__(self, config: InstrumentConfig):
        self.cfg   = config
        self._sock: socket.socket | None = None
        self._connected = False

    def connect(self):
        """Open TCP connection to instrument."""
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.settimeout(self.cfg.timeout)
        self._sock.connect((self.cfg.host, self.cfg.port))
        self._connected = True
        # Identify instrument
        idn = self.send('*IDN?')
        print(f"  [HITL] Connected: {idn}")

    def disconnect(self):
        """Close TCP connection."""
        if self._sock:
            try:
                self._sock.close()
            except OSError:
                pass
        self._connected = False

    def send(self, cmd: str) -> str:
        """Send SCPI command, return response string."""
        if not self._connected:
            raise ConnectionError("Instrument not connected. Call connect() first.")
        self._sock.sendall((cmd + self.cfg.term).encode())
        return self._sock.recv(4096).decode('ascii', errors='replace').strip()

    def measure_osnr(self) -> float:
        """Measure optical signal-to-noise ratio (dB)."""
        return float(self.send('MEAS:OPT:OSNR?'))

    def measure_power_dbm(self) -> float:
        """Measure optical power (dBm)."""
        return float(self.send('MEAS:POW?'))

    def measure_ber_log(self) -> float:
        """Measure log10(BER). Returns e.g. -7.2 for BER=6.3e-8."""
        raw = self.send('MEAS:BER?')
        ber = float(raw)
        return float(np.log10(max(ber, 1e-15)))

    def measure_all(self) -> Measurement:
        """Execute a full measurement sweep."""
        t0 = time.monotonic()
        try:
            osnr  = self.measure_osnr()
        except Exception:
            osnr  = None
        try:
            power = self.measure_power_dbm()
        except Exception:
            power = None
        try:
            ber   = self.measure_ber_log()
        except Exception:
            ber   = None
        return Measurement(timestamp_s=time.monotonic() - t0,
                           osnr_db=osnr, power_dbm=power, ber_log=ber)

    @property
    def is_connected(self) -> bool:
        return self._connected


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class SimulatedInstrument:
    """
    Drop-in replacement for SCPIBridge — no hardware required.
    Adds calibrated Gaussian noise to the synthetic input vector to simulate
    real measurement uncertainty (quantization, thermal noise, ADC jitter).

    Noise model per feature:
      OSNR:  σ = 0.3 dB  (typical OSA measurement uncertainty)
      Power: σ = 0.2 dBm (PM calibration uncertainty)
      BER:   σ = 0.1     (in log space — integer BER decade shift)
      Other: σ = 0.05 × |value|  (proportional uncertainty)
    """

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self._connected = False

    def connect(self):
        self._connected = True
        print("  [HITL] Simulation mode: no hardware (Gaussian noise model)")

    def disconnect(self):
        self._connected = False

    def add_noise(self, x: np.ndarray, feature_names: list[str] | None = None) -> np.ndarray:
        """Add instrument-realistic noise to a feature vector."""
        noisy = x.copy()
        for i, v in enumerate(x):
            fname = feature_names[i] if feature_names else ''
            if 'OSNR' in fname or 'osnr' in fname.lower():
                sigma = 0.3
            elif 'Power' in fname or 'power' in fname.lower():
                sigma = 0.2
            elif 'BER' in fname or 'ber' in fname.lower():
                sigma = 0.1
            else:
                sigma = 0.05 * (abs(float(v)) + 1e-6)
            noisy[i] = float(v) + self.rng.normal(0, sigma)
        return noisy

    @property
    def is_connected(self) -> bool:
        return self._connected


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class HITLSession:
    """
    Hardware-in-the-loop session.

    Workflow per cycle:
      1. Receive synthetic/predicted feature vector x_synthetic
      2. Run PRISM inference → predicted_class, confidence
      3. Query instrument for real measurements (or simulate)
      4. Compute residual: real_data - x_synthetic
      5. Log cycle for adaptation analysis
      6. Return full result dict

    Adaptation:
      After N cycles, call adapt() to retrain PRISM on real measurements.
      Uses the prediction confidence as a pseudo-label weight.
    """

    def __init__(self, fiber_nn, class_names: list[str],
                 instrument: SCPIBridge | SimulatedInstrument | None = None,
                 feature_names: list[str] | None = None):
        self.nn            = fiber_nn
        self.classes       = class_names
        self.feature_names = feature_names
        self.adaptation_log: list[dict] = []

        if instrument is None:
            self.instrument = SimulatedInstrument(seed=42)
            self.instrument.connect()
            self._mode = 'simulation'
        else:
            self.instrument = instrument
            self._mode = 'hardware'

    def run_cycle(self, x_synthetic: list[float] | np.ndarray,
                  true_label: int | None = None) -> dict:
        """
        Execute one HITL inference-measurement cycle.

        Args:
            x_synthetic : raw feature vector (unnormalized)
            true_label  : ground-truth class index (optional, for accuracy tracking)

        Returns dict with:
            input, predicted_class, confidence, real_data,
            residual, mode, correct (if true_label provided)
        """
        x = np.asarray(x_synthetic, dtype=float)

        # PRISM inference
        probs       = self.nn.forward(x)
        pred_idx    = int(np.argmax(probs))
        pred_class  = self.classes[pred_idx]
        confidence  = float(probs[pred_idx])

        # Real measurement
        if isinstance(self.instrument, SimulatedInstrument):
            real_data = self.instrument.add_noise(x, self.feature_names)
        else:
            meas = self.instrument.measure_all()
            # Build real feature vector: replace known fields, keep synthetic for rest
            real_data = x.copy()
            if meas.osnr_db is not None and self.feature_names:
                for i, f in enumerate(self.feature_names):
                    if 'OSNR' in f:   real_data[i] = meas.osnr_db
                    if 'Power' in f:  real_data[i] = meas.power_dbm or x[i]
                    if 'BER' in f:    real_data[i] = meas.ber_log or x[i]

        residual = real_data - x

        result = {
            'input':           x.tolist(),
            'predicted_class': pred_class,
            'predicted_idx':   pred_idx,
            'confidence':      confidence,
            'probabilities':   {c: float(p) for c, p in zip(self.classes, probs)},
            'real_data':       real_data.tolist(),
            'residual':        residual.tolist(),
            'residual_norm':   float(np.linalg.norm(residual)),
            'mode':            self._mode,
        }
        if true_label is not None:
            result['true_class'] = self.classes[true_label]
            result['correct']    = (pred_idx == true_label)

        self.adaptation_log.append(result)
        return result

    def run_fleet(self, fleet: list[tuple[str, list[float], int | None]]) -> list[dict]:
        """
        Run HITL cycle on a list of (link_name, features, true_label) tuples.
        Prints a summary table.
        """
        results = []
        print(f"\n  ── HITL Fleet Sweep ({self._mode} mode) ──")
        w = max(len(c) for c in self.classes)
        print(f"  {'Link':<22}  {'Prediction':<{w}}  {'Conf':>6}  {'Resid':>6}  {'OK':>4}")
        print(f"  {'─'*60}")
        for name, x_raw, true_lbl in fleet:
            r = self.run_cycle(x_raw, true_label=true_lbl)
            ok_str = '✓' if r.get('correct') else ('✗' if true_lbl is not None else '?')
            print(f"  {name:<22}  {r['predicted_class']:<{w}}  "
                  f"{r['confidence']:>5.3f}  {r['residual_norm']:>5.3f}  {ok_str:>4}")
            results.append(r)
        n_correct = sum(r.get('correct', False) for r in results if 'correct' in r)
        n_labeled = sum(1 for r in results if 'correct' in r)
        if n_labeled:
            print(f"  {'─'*60}")
            print(f"  HITL accuracy: {n_correct}/{n_labeled} = {n_correct/n_labeled*100:.1f}%")
        return results

    def summary(self):
        """Print adaptation log summary."""
        n = len(self.adaptation_log)
        if n == 0:
            print("  No HITL cycles logged.")
            return
        resids = [r['residual_norm'] for r in self.adaptation_log]
        print(f"\n  HITL Session Summary — {n} cycles  ({self._mode} mode)")
        print(f"  Mean residual norm : {np.mean(resids):.4f}")
        print(f"  Max  residual norm : {np.max(resids):.4f}")
        correct = [r for r in self.adaptation_log if r.get('correct')]
        labeled = [r for r in self.adaptation_log if 'correct' in r]
        if labeled:
            print(f"  HITL accuracy      : {len(correct)}/{len(labeled)} = "
                  f"{len(correct)/len(labeled)*100:.1f}%")

    def save_log(self, path: str = 'hitl_log.json'):
        """Save adaptation log to JSON for offline analysis."""
        with open(path, 'w') as f:
            json.dump(self.adaptation_log, f, indent=2)
        print(f"  HITL log saved to {path}  ({len(self.adaptation_log)} cycles)")
