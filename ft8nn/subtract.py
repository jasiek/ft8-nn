"""Subtract decoded FT8 signals from the 12 kHz slot audio (after WSJT-X subtractft8).

The decoded codeword gives the exact tone sequence; a unit-amplitude GFSK reference is built at
the refined (f0, dt), the complex envelope a(t) = LPF{x(t) conj(ref(t))} is tracked with a
~0.3 s window (follows fading and residual phase drift), and 2 Re{a(t) ref(t)} is removed.
"""
import numpy as np
from scipy.signal import fftconvolve

from . import ft8

NFILT = 4001  # ~0.33 s smoothing window at 12 kHz
_WIN = np.sin(np.pi * np.arange(NFILT) / (NFILT - 1)) ** 2
_WIN /= _WIN.sum()


def _envelope_power(x, ref, k0):
    n = len(ref)
    seg = np.zeros(n)  # zero-pad wherever the signal runs past either end of the slot
    lo, hi = max(k0, 0), min(k0 + n, len(x))
    if hi > lo:
        seg[lo - k0 : hi - k0] = x[lo:hi]
    # normalise by the filtered reference power: exact at the signal edges and through the ramps
    a = fftconvolve(seg * np.conj(ref), _WIN, mode="same") / np.maximum(fftconvolve(np.abs(ref) ** 2, _WIN, mode="same"), 1e-3)
    return a, seg


def _block_energy(x, ref, k0, blk=960):
    """Captured energy with ref placed at k0: sum over 80 ms blocks of |sum x conj(ref)|^2 (cheap)."""
    n = len(ref) // blk * blk
    lo, hi = max(k0, 0), min(k0 + n, len(x))
    if hi - lo < blk:
        return 0.0
    p = np.zeros(n, complex)
    p[lo - k0 : hi - k0] = x[lo:hi] * np.conj(ref[lo - k0 : hi - k0])
    return float(np.sum(np.abs(p.reshape(-1, blk).sum(1)) ** 2))


def subtract(x, tones, f0, dt, refine=True, span=48):
    """Return x with the signal (tones, f0, dt) removed. x is modified in place and returned.

    refine: search +-span samples (12 kHz) around dt for the best alignment before subtracting.
    """
    ref = ft8.complex_wave(tones, f0)
    k0 = int(round((ft8.NOMINAL_START + dt) * ft8.FS))
    if refine:
        coarse = max(range(-span, span + 1, 4), key=lambda d: _block_energy(x, ref, k0 + d))
        k0 += max(range(coarse - 3, coarse + 4), key=lambda d: _block_energy(x, ref, k0 + d))
    a, _ = _envelope_power(x, ref, k0)
    model = 2 * np.real(a * ref)
    lo, hi = max(k0, 0), min(k0 + len(ref), len(x))
    x[lo:hi] -= model[lo - k0 : hi - k0]
    return x
