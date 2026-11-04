"""Channel simulation and 15-second slot synthesis, calibrated like WSJT-X ft8sim.

SNR convention (WSJT-X): signal power / noise power in a 2500 Hz reference bandwidth.
Noise is real white Gaussian, sigma=1 per sample at fs=12 kHz (i.e. spread over 6 kHz).
"""
from dataclasses import dataclass, field

import numpy as np

from . import ft8

BW_RATIO = 2500.0 / (ft8.FS / 2)


def snr_to_amplitude(snr_db):
    """Peak amplitude of a real sinusoid giving snr_db in 2500 Hz with unit-variance noise."""
    return np.sqrt(2 * BW_RATIO) * 10 ** (snr_db / 20)


def gaussian_doppler_process(n, fs, fspread, rng):
    """Complex Gaussian process, unit mean power, Gaussian Doppler spectrum of ~fspread Hz width."""
    if fspread <= 0:
        return np.ones(n, complex)
    nfft = 1 << int(np.ceil(np.log2(n)))
    f = np.fft.fftfreq(nfft, 1 / fs)
    shape = np.exp(-((f / (0.5 * fspread)) ** 2))
    x = (rng.standard_normal(nfft) + 1j * rng.standard_normal(nfft)) * np.sqrt(shape)
    y = np.fft.ifft(x)[:n]
    return y / np.sqrt(np.mean(np.abs(y) ** 2))


def watterson(cwave, fs, fspread, delay_ms, rng):
    """Two-path Watterson HF channel: equal-power paths, independent Gaussian Doppler fading."""
    if fspread <= 0 and delay_ms <= 0:
        return cwave
    n = len(cwave)
    # Fading processes are slow; generate at a low rate and interpolate for speed.
    dec = 50
    m = n // dec + 2
    t_lo = np.arange(m) * dec
    g = []
    for _ in range(2):
        lo = gaussian_doppler_process(m, fs / dec, fspread, rng)
        g.append(np.interp(np.arange(n), t_lo, lo.real) + 1j * np.interp(np.arange(n), t_lo, lo.imag))
    d = int(round(delay_ms * 1e-3 * fs))
    p2 = np.zeros_like(cwave)
    p2[d:] = cwave[: n - d] if d else cwave
    return (g[0] * cwave + g[1] * p2) / np.sqrt(2)


@dataclass
class Signal:
    tones: np.ndarray
    f0: float  # Hz, frequency of tone 0
    dt: float  # s, offset relative to nominal 0.5 s start
    snr_db: float
    fspread: float = 0.0  # Hz Doppler spread (0 = no fading)
    delay_ms: float = 0.0  # second-path delay
    drift_hz_per_s: float = 0.0  # linear frequency drift
    meta: dict = field(default_factory=dict)


def synth_slot(signals, rng, n=ft8.SLOT_SAMPLES, noise=True):
    """Sum of signals + unit-variance white noise, float64, length n (default 15 s @ 12 kHz)."""
    out = rng.standard_normal(n) if noise else np.zeros(n)
    for s in signals:
        w = ft8.complex_wave(s.tones, s.f0)
        if s.drift_hz_per_s:
            t = np.arange(len(w)) / ft8.FS
            w = w * np.exp(1j * np.pi * s.drift_hz_per_s * t**2)
        w = w * np.exp(2j * np.pi * rng.random())
        w = watterson(w, ft8.FS, s.fspread, s.delay_ms, rng)
        k0 = int(round((ft8.NOMINAL_START + s.dt) * ft8.FS))
        a, b = max(k0, 0), min(k0 + len(w), n)
        if b > a:
            out[a:b] += snr_to_amplitude(s.snr_db) * w[a - k0 : b - k0].real
    return out


def write_wav(path, x, scale=100.0):
    """16-bit mono 12 kHz WAV, scaled like ft8sim (rms noise = 100 counts)."""
    from scipy.io import wavfile

    wavfile.write(path, ft8.FS, np.clip(np.round(x * scale), -32767, 32767).astype(np.int16))
