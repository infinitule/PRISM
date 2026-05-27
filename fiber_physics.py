"""
fiber_physics.py — Virtual Optical Physics Engine
==================================================
Simulates the complete photonic signal pipeline from Zang et al. (iOptics 2025)
and grounded in Curran/Shirk fiber optics fundamentals.

Physical chain:
  Laser → SMF (stretch) → EDFA (amplify+noise) → AM1/AM2 (modulate x, W)
       → DCF (compress = dot product) → PD (detect)

Key equations implemented:
  Eq.6 (modulation):  E_s(T) = E · Σᵢ wᵢxᵢ · Rect_{Ωβ₂l/N}(T - offset_i)
  Eq.7 (compression): E = γ · Σᵢ wᵢxᵢ          ← THIS IS THE DOT PRODUCT
"""

import numpy as np

# ── Physical constants ──────────────────────────────────────────────────────
C_LIGHT   = 2.998e8        # m/s
H_PLANCK  = 6.626e-34      # J·s
LAMBDA_C  = 1550e-9        # m  (C-band, from paper Table 1)
NU_C      = C_LIGHT / LAMBDA_C

# ── SMF G.652 (from paper Table 1) ─────────────────────────────────────────
SMF_ALPHA_DB_KM = 0.20     # dB/km attenuation @ 1550 nm
SMF_BETA2       = -21.7e-27  # s²/m  (typical G.652 @ 1550 nm, ~-21.7 ps²/km)
SMF_LENGTH      = 170e3    # 170 km

# ── DCF (from paper: -100 ps²/(nm·km), 28.5 km) ───────────────────────────
# D_DCF = -100 ps/(nm·km) → β₂_DCF = -D·λ²/(2πc) ≈ +127e-27 s²/m  (opposite to SMF)
DCF_BETA2  = 127e-27       # s²/m (compensates SMF over 170 km)
DCF_LENGTH = 28.5e3        # 28.5 km

# ── EDFA ───────────────────────────────────────────────────────────────────
EDFA_GAIN_DB = 20          # dB
EDFA_NF_DB   = 4           # dB noise figure (from paper)

# ── Photodetector (from paper) ─────────────────────────────────────────────
PD_RESPONSIVITY   = 0.9    # A/W
PD_NOISE_DB       = 4      # dB

# ── Laser / pulse ──────────────────────────────────────────────────────────
PULSE_FREQ   = 50e6        # 50 MHz (from paper)
PULSE_PERIOD = 1.0 / PULSE_FREQ  # 20 ns


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class LaserPulse:
    """
    Coherent CW laser → periodic Gaussian pulses at 1550 nm.
    Pulses are time-stretched by SMF dispersion before modulation.
    """

    def __init__(self, wavelength=LAMBDA_C, rep_rate=PULSE_FREQ,
                 pulse_width_ns=0.5, power_mW=1.0):
        self.lam    = wavelength
        self.nu     = C_LIGHT / wavelength
        self.T_rep  = 1.0 / rep_rate
        self.tau0   = pulse_width_ns * 1e-9   # initial FWHM in seconds
        self.P0     = power_mW * 1e-3         # W
        self.E0     = np.sqrt(self.P0)        # field amplitude

    def field(self, t_array):
        """Complex envelope of one pulse centred at t=0."""
        return self.E0 * np.exp(-t_array**2 / (2 * self.tau0**2))

    def stretched_width(self, beta2, fiber_length, omega_bw):
        """
        After SMF time-stretch:  τ_s = |β₂| · L · Ω  (Eq. 6 notation)
        """
        return abs(beta2) * fiber_length * omega_bw


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class SingleModeFiber:
    """
    SMF G.652 propagation:
      • Total Internal Reflection waveguiding (n_core > n_clad)
      • Attenuation: P_out = P_in · exp(-α·L)
      • Group velocity dispersion (GVD): H(ω) = exp(jβ₂Lω²/2)
      • Time-stretch factor for analog computing ring
    """

    def __init__(self, alpha_db_km=SMF_ALPHA_DB_KM,
                 beta2=SMF_BETA2, length=SMF_LENGTH,
                 n_core=1.4681, n_clad=1.4629):
        # Convert attenuation to Np/m
        self.alpha = alpha_db_km * 1e-3 / (10.0 / np.log(10))
        self.beta2  = beta2
        self.length = length
        # TIR geometry (Curran/Shirk §II.2)
        self.n1 = n_core
        self.n2 = n_clad
        self.NA = np.sqrt(n_core**2 - n_clad**2)
        self.acceptance_angle_deg = np.degrees(np.arcsin(self.NA))
        # Single-mode condition: V = π·d·NA/λ < 2.405 for d = 9 µm
        self.core_diameter = 9e-6  # m
        self.V_number = np.pi * self.core_diameter * self.NA / LAMBDA_C

    # ── Amplitude transfer ────────────────────────────────────────────────
    def attenuate(self, E_field):
        """Power attenuation along fiber length."""
        field_loss = np.exp(-self.alpha * self.length)
        return E_field * field_loss

    def dispersion_transfer(self, omega_array):
        """
        Frequency-domain GVD phase shift: H(ω) = exp(jβ₂Lω²/2)
        Used for analytic pulse shaping.
        """
        return np.exp(1j * self.beta2 * self.length * omega_array**2 / 2)

    def stretch_pulse(self, E_in_t, t_array):
        """
        Time-domain pulse stretching via FFT → GVD phase → IFFT.
        Returns stretched pulse envelope.
        """
        dt = t_array[1] - t_array[0]
        N  = len(t_array)
        omega = 2 * np.pi * np.fft.fftfreq(N, dt)
        E_f   = np.fft.fft(E_in_t)
        E_f  *= self.dispersion_transfer(omega)
        E_f  *= np.exp(-self.alpha * self.length)
        return np.fft.ifft(E_f)

    def tir_info(self):
        return {
            'NA':               self.NA,
            'acceptance_deg':   self.acceptance_angle_deg,
            'V_number':         self.V_number,
            'single_mode':      self.V_number < 2.405,
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class EDFA:
    """
    Erbium-Doped Fiber Amplifier.
    Gain: E_out = √G · E_in
    ASE noise: S_ase = (G-1)·NF·h·ν/2  [W/Hz per polarisation]
    """

    def __init__(self, gain_db=EDFA_GAIN_DB, nf_db=EDFA_NF_DB):
        self.G    = 10 ** (gain_db / 10)
        self.NF   = 10 ** (nf_db  / 10)
        self.sqG  = np.sqrt(self.G)

    def amplify(self, E_field, bandwidth=50e9, rng=None, noisy=True):
        """Amplify field + inject ASE noise."""
        E_out = self.sqG * E_field
        if noisy and rng is not None:
            S_ase    = (self.G - 1) * self.NF * H_PLANCK * NU_C / 2
            noise_pw = S_ase * bandwidth
            sigma    = np.sqrt(noise_pw / 2)
            shape    = E_field.shape if hasattr(E_field, 'shape') else ()
            noise    = rng.normal(0, sigma, shape) + 1j * rng.normal(0, sigma, shape)
            E_out   += noise
        return E_out

    def amplify_noiseless(self, E_field):
        return self.sqG * E_field


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class AmplitudeModulator:
    """
    Intensity (amplitude) modulator — maps synaptic weight to optical amplitude.
    Physical constraint: intensity modulators only encode [0, 1].
    Negative weights handled by positive/negative decomposition (§3, paper).

    E_out = |s| · E_in        s ∈ [0, 1]
    """

    def modulate(self, E_in, signal_value):
        s = float(np.clip(np.abs(signal_value), 0.0, 1.0))
        return s * E_in


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class DispersionCompensatingFiber:
    """
    Dispersion Compensating Fiber — the COMPUTATION ENGINE.

    Physical principle (Eq. 7 from paper):
      After SMF stretches the pulse and AM1/AM2 modulate segments with w and x,
      DCF applies opposite GVD → separated frequency components OVERLAP →
      pulses compress and the compressed energy is:

          E_compressed = γ · Σᵢ wᵢ · xᵢ    ← dot product

    DCF also acts as a low-pass filter, suppressing EDFA noise accumulation.
    """

    def __init__(self, beta2=DCF_BETA2, length=DCF_LENGTH, gamma=1.0):
        self.beta2  = beta2
        self.length = length
        self.gamma  = gamma     # conversion coefficient (Eq. 7)

    def dot_product(self, w_segments, x_segments):
        """
        Core optical computation:
          E = γ · Σᵢ wᵢxᵢ
        Simulates DCF pulse compression → intensity integration.
        """
        return self.gamma * float(np.dot(w_segments, x_segments))

    def dispersion_transfer(self, omega_array):
        return np.exp(1j * self.beta2 * self.length * omega_array**2 / 2)

    def compress_pulse(self, E_stretched_f, omega_array):
        """Frequency-domain DCF compression."""
        return E_stretched_f * self.dispersion_transfer(omega_array)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class Photodetector:
    """
    PIN photodiode: converts optical intensity to photocurrent.
      I = R · |E|²  +  noise

    Responsivity R = 0.9 A/W at 1550 nm (from paper Table 1).
    """

    def __init__(self, responsivity=PD_RESPONSIVITY, nf_db=PD_NOISE_DB):
        self.R      = responsivity
        self.sigma0 = 10 ** (nf_db / 20) * 1e-4  # baseline noise floor (A)

    def detect(self, E_field, rng=None, noisy=True):
        P = np.abs(E_field) ** 2
        I = self.R * P
        if noisy and rng is not None:
            # Shot + thermal noise, scaled by signal
            sigma = self.sigma0 * np.sqrt(np.maximum(P, 1e-15) / 1e-3)
            I    += rng.normal(0, sigma)
        return float(I)

    def detect_noiseless(self, E_field):
        return self.R * float(np.abs(E_field) ** 2)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class QuantumHomodyneReceiver:
    """
    T8 — Quantum extension: squeezed-light homodyne receiver.

    Classical homodyne variance (shot-noise limit):
        V_x^{coh} = 1 / (4N)        N = mean photon number

    Squeezed-light variance (squeezing parameter r ≥ 0):
        V_x^{sq}  = e^{-2r} / 4     (x-quadrature squeezed)
        V_p^{sq}  = e^{+2r} / 4     (p-quadrature anti-squeezed)

    Quantum Fisher Information (phase estimation):
        QFI = N² · sinh²(r) + N · cosh(2r)

    Classical Fisher Information (coherent state):
        CFI = N

    SNR improvement (dB):
        ΔSNRdb = 10·log10(e^{2r}) = 20r/ln(10) ≈ 8.686 · r   [dB]

    Physical reference: Caves 1981; Demkowicz-Dobrzański 2015.
    """

    def __init__(self, squeezing_r: float = 0.0,
                 responsivity: float = PD_RESPONSIVITY,
                 eta: float = 0.95):
        """
        Args:
            squeezing_r  : squeezing parameter r ≥ 0  (r=0 = coherent state)
            responsivity : detector quantum efficiency × responsivity
            eta          : homodyne efficiency (losses in optical path, 0–1)
        """
        if squeezing_r < 0:
            raise ValueError("squeezing_r must be ≥ 0")
        self.r    = float(squeezing_r)
        self.R    = responsivity
        self.eta  = float(np.clip(eta, 1e-6, 1.0))

    # ── Noise variances ────────────────────────────────────────────────────
    def shot_noise_variance(self, n_photons: float) -> float:
        """
        Homodyne x-quadrature shot noise variance.
        Coherent: 1/(4N). Squeezed: e^{-2r}/4. Includes detection loss η.
        """
        v_sq  = np.exp(-2 * self.r) / 4.0
        v_vac = 1.0 / 4.0                   # vacuum variance
        # Measured variance with efficiency loss:  η·V_sq + (1-η)·V_vac
        return float(self.eta * v_sq + (1.0 - self.eta) * v_vac)

    def antisqueezed_variance(self) -> float:
        """p-quadrature anti-squeezed variance: e^{+2r}/4."""
        return float(np.exp(2 * self.r) / 4.0)

    # ── SNR improvement ────────────────────────────────────────────────────
    def snr_improvement_db(self) -> float:
        """
        SNR gain from squeezing vs coherent shot noise (dB).
        ΔSNRdB = 20·r / ln(10) ≈ 8.686 · r
        """
        return float(20.0 * self.r / np.log(10))

    # ── Fisher information ─────────────────────────────────────────────────
    def quantum_fisher_information(self, n_photons: float) -> float:
        """
        QFI for phase estimation with squeezed light.
        QFI = N² sinh²(r) + N cosh(2r)
        At r=0: QFI = N  (shot-noise limited).
        """
        N = float(n_photons)
        return float(N**2 * np.sinh(self.r)**2 + N * np.cosh(2 * self.r))

    def classical_fisher_information(self, n_photons: float) -> float:
        """CFI for coherent state (shot-noise limit): QFI = N."""
        return float(n_photons)

    def quantum_advantage(self, n_photons: float) -> float:
        """
        Ratio QFI / CFI. Values > 1 indicate quantum advantage.
        At r=0: ratio = 1 (no advantage). At r=1, N=100: ~76.
        """
        cfi = self.classical_fisher_information(n_photons)
        if cfi < 1e-15:
            return 1.0
        return float(self.quantum_fisher_information(n_photons) / cfi)

    # ── Effective detection ────────────────────────────────────────────────
    def homodyne_snr(self, signal_amplitude: float, n_photons: float) -> float:
        """
        SNR of homodyne x-quadrature measurement.
        SNR = (R·|E|)² / V_noise  (linear, not dB)
        """
        v_noise = self.shot_noise_variance(n_photons)
        return float((self.R * abs(signal_amplitude))**2 / (v_noise + 1e-30))

    # ── Summary ────────────────────────────────────────────────────────────
    def report(self, n_photons: float = 100.0):
        """Print quantum receiver parameter summary."""
        print(f"\n  Quantum Homodyne Receiver  (r={self.r:.3f}, η={self.eta:.2f})")
        print(f"  {'─'*52}")
        print(f"  Squeezing parameter r    : {self.r:.4f}")
        print(f"  x-variance (shot-noise)  : {self.shot_noise_variance(n_photons):.6f}")
        print(f"  p-variance (anti-sq.)    : {self.antisqueezed_variance():.6f}")
        print(f"  SNR improvement          : {self.snr_improvement_db():.2f} dB")
        print(f"  QFI  (N={n_photons:.0f})           : "
              f"{self.quantum_fisher_information(n_photons):.2f}")
        print(f"  CFI  (N={n_photons:.0f})           : "
              f"{self.classical_fisher_information(n_photons):.2f}")
        print(f"  Quantum advantage        : "
              f"{self.quantum_advantage(n_photons):.2f}×")
        print(f"  Heisenberg limit (N²)    : {n_photons**2:.0f}")
        print(f"  {'─'*52}")
