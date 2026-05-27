"""
prism_spiking.py — T11: Neuromorphic Spiking Optical Neurons
=============================================================
Leaky integrate-and-fire (LIF) spiking optical neurons.
Information encoded as time-to-first-spike (TTFS): stronger input → earlier spike.

Physical interpretation:
  The LIF threshold corresponds to saturation power of the balanced photodetector.
  The membrane time constant τ_m maps to the carrier lifetime in the active region.
  A spike = homodyne detector crossing saturation → optical limiter fires.

Comparison with continuous IQ neurons:
  IQ neuron  : outputs real-valued dot product (rate coding, synchronous)
  LIF neuron : outputs spike time (temporal coding, asynchronous, no clock needed)

TTFS encoding advantage:
  Latency scales with signal strength — strong signals answered instantly.
  First spike carries maximum information; subsequent spikes are redundant.
  Energy proportional to latency → weak signals consume more energy (realistic).
"""

import numpy as np
from dataclasses import dataclass, field


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class SpikingOpticalNeuron:
    """
    Leaky integrate-and-fire optical neuron.

    Dynamics (Euler integration):
        dV/dt = -V/τ_m + I(t)
        V → 0 after spike (hard reset)

    Refractory period = 1 time unit (no spike allowed immediately after firing).

    Physical mapping:
        V_m       → accumulated photocurrent in detector integrator
        threshold → detector saturation power (PD_RESP × P_sat)
        τ_m       → detector RC time constant (~10 ns for PIN PD)
        dt        → time resolution (0.1 ns default)
    """

    def __init__(self, threshold: float = 1.0, tau_m: float = 10.0,
                 dt: float = 0.1, refractory: float = 1.0):
        self.threshold  = threshold
        self.tau_m      = tau_m       # membrane time constant (time units)
        self.dt         = dt          # integration step (time units)
        self.refractory = refractory  # absolute refractory period (time units)
        self.reset()

    def reset(self):
        self.V_m          = 0.0
        self.t_last_spike = -1e9
        self.spike_times: list[float] = []

    def step(self, I_input: float, t: float) -> bool:
        """Euler step. Returns True if neuron fires at time t."""
        if t - self.t_last_spike < self.refractory:
            return False
        self.V_m += self.dt * (-self.V_m / self.tau_m + I_input)
        if self.V_m >= self.threshold:
            self.spike_times.append(t)
            self.t_last_spike = t
            self.V_m = 0.0
            return True
        return False

    def time_to_first_spike(self, I_input: float, max_t: float = 100.0) -> float:
        """
        Encode input current as latency to first spike.
        Returns max_t if no spike occurs (sub-threshold input).
        Monotonically decreasing with I_input — stronger signal → shorter latency.
        """
        self.reset()
        n_steps = int(max_t / self.dt)
        for k in range(n_steps):
            t = k * self.dt
            if self.step(I_input, t):
                return t
        return max_t

    def spike_count(self, I_input: float, duration: float) -> int:
        """Rate coding: count spikes in a fixed time window."""
        self.reset()
        n_steps = int(duration / self.dt)
        for k in range(n_steps):
            self.step(I_input, k * self.dt)
        return len(self.spike_times)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class SpikingOpticalLayer:
    """
    One layer of spiking optical neurons.
    Each neuron receives a dot-product input (from IQ coherent sum)
    and fires based on TTFS encoding.

    Forward pass:
      dot products → I_inputs → TTFS → output vector
      Normalise TTFS: 0 (earliest = strongest) → 1 (latest = weakest)
    """

    def __init__(self, n_neurons: int, threshold: float = 1.0,
                 tau_m: float = 10.0, dt: float = 0.1, max_t: float = 50.0):
        self.neurons = [SpikingOpticalNeuron(threshold, tau_m, dt)
                        for _ in range(n_neurons)]
        self.max_t   = max_t

    def forward_ttfs(self, dot_products: np.ndarray) -> np.ndarray:
        """
        Convert dot-product activations to normalised TTFS codes.
        Positive inputs are excitatory (fire early), zero/negative → max_t.
        """
        latencies = np.array([
            self.neurons[i].time_to_first_spike(
                max(float(dot_products[i]), 0.0),
                max_t=self.max_t
            )
            for i in range(len(self.neurons))
        ])
        # Normalise: 0 = fired immediately, 1 = never fired
        return 1.0 - latencies / self.max_t   # higher = fired earlier = stronger

    def spike_rates(self, dot_products: np.ndarray,
                    duration: float = 50.0) -> np.ndarray:
        """Rate coding: spikes/duration for each neuron."""
        return np.array([
            self.neurons[i].spike_count(
                max(float(dot_products[i]), 0.0), duration)
            for i in range(len(self.neurons))
        ]) / duration


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
@dataclass
class TTFSProfile:
    """Analysis result for one neuron across a range of input currents."""
    currents:  np.ndarray
    latencies: np.ndarray
    fired:     np.ndarray   # bool — whether neuron fired at all
    threshold: float
    tau_m:     float


def analyze_ttfs_curve(currents: list[float] | np.ndarray,
                       threshold: float = 1.0,
                       tau_m: float = 10.0,
                       dt: float = 0.1,
                       max_t: float = 100.0) -> TTFSProfile:
    """
    Compute TTFS latency curve across a range of input currents.
    Used to verify monotonic decrease (stronger I → shorter latency).
    """
    currents = np.asarray(currents, dtype=float)
    latencies = np.zeros(len(currents))
    fired     = np.zeros(len(currents), dtype=bool)

    neuron = SpikingOpticalNeuron(threshold=threshold, tau_m=tau_m, dt=dt)
    for k, I in enumerate(currents):
        lat = neuron.time_to_first_spike(I, max_t=max_t)
        latencies[k] = lat
        fired[k]     = lat < max_t

    return TTFSProfile(currents=currents, latencies=latencies,
                       fired=fired, threshold=threshold, tau_m=tau_m)


def print_ttfs_table(profile: TTFSProfile):
    """Print TTFS latency vs input current table."""
    print(f"\n  Spiking Optical Neuron — TTFS Profile")
    print(f"  threshold={profile.threshold:.2f}  τ_m={profile.tau_m:.1f}")
    print(f"  {'I_input':>10}  {'Latency':>10}  {'Fired':>6}  {'Bar'}")
    print(f"  {'─'*55}")
    max_lat = profile.latencies.max()
    for I, lat, f in zip(profile.currents, profile.latencies, profile.fired):
        bar = '█' * int((1 - lat / max_lat) * 20) if f else '·' * 3
        fired_str = 'YES' if f else ' NO'
        lat_str   = f'{lat:.2f}' if f else '  ---'
        print(f"  {I:>10.3f}  {lat_str:>10}  {fired_str:>6}  {bar}")


def verify_monotonic_ttfs(currents: list[float] | None = None) -> bool:
    """
    Verify that TTFS is monotonically decreasing with input current.
    Returns True if physically correct (stronger signal → shorter latency).
    """
    if currents is None:
        currents = [0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]
    profile = analyze_ttfs_curve(currents)
    fired_lats = profile.latencies[profile.fired]
    # Among neurons that fired, latency should be monotonically non-increasing
    if len(fired_lats) < 2:
        return False
    return bool(np.all(np.diff(fired_lats) <= 0))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def demo():
    """Demonstration: print TTFS table for standard current range."""
    currents = [0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0]
    profile  = analyze_ttfs_curve(currents)
    print_ttfs_table(profile)
    mono = verify_monotonic_ttfs(currents)
    print(f"\n  Monotonically decreasing TTFS: {'YES ✓' if mono else 'NO ✗'}")
    print(f"  Physical interpretation: TTFS ↓ as I ↑ (stronger signal → earlier spike)")


if __name__ == "__main__":
    demo()
