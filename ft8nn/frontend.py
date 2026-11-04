"""FT8 receive front end: coarse candidate search, downsampling, fine sync, symbol spectra.

Pipeline per 15 s slot (12 kHz real audio):
  1. spectrogram (1920-sample rectangular window, hop 480, 3.125 Hz bins)
  2. Costas sync score over (lag, freq) -> candidate list
  3. per candidate: FFT-based band extraction to 200 Hz complex baseband (tone 0 at 0 Hz)
  4. fine sync: dt in 5 ms steps, df in 0.25 Hz steps, maximising Costas energy
  5. symbol spectra cs[79, 8] (complex) and wider-band features for the NN
"""
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import maximum_filter

from . import ft8

FS_BB = 200.0  # baseband sample rate
SPS_BB = 32  # samples per symbol at baseband
NFFT_FULL = 192000  # 16 s at 12 kHz -> 0.0625 Hz bins
NBB = 3200  # 16 s at 200 Hz
DF_FULL = ft8.FS / NFFT_FULL

HOP = 480
NFFT_SPEC = 3840
FPS = ft8.FS / HOP  # spectrogram frames per second (25)
SYM_FRAMES = 4  # frames per symbol


@dataclass
class Candidate:
    f0: float  # Hz, tone-0 frequency
    dt: float  # s, start relative to nominal 0.5 s
    score: float


def spectrogram(x):
    n = (len(x) - ft8.NSPS) // HOP + 1
    idx = np.arange(ft8.NSPS)[None, :] + HOP * np.arange(n)[:, None]
    X = np.fft.rfft(x[idx], n=NFFT_SPEC, axis=1)
    return (X.real**2 + X.imag**2).astype(np.float32)  # (frames, bins), bin = 3.125 Hz


def find_candidates(x, fmin=150.0, fmax=3000.0, dtmin=-1.5, dtmax=2.5, max_cand=250, min_score=1.3):
    S = spectrogram(x)
    T, F = S.shape
    b0, b1 = int(fmin / 3.125), int(fmax / 3.125)
    nf = b1 - b0
    # tone powers P[t, f, tone] with tone spacing = 2 bins
    P = np.stack([S[:, b0 + 2 * k : b0 + 2 * k + nf] for k in range(8)], axis=-1)  # (T, nf, 8)
    lag0 = int(round(ft8.NOMINAL_START * FPS))
    lags = np.arange(int(np.floor(dtmin * FPS)), int(np.ceil(dtmax * FPS)) + 1)
    num = np.zeros((len(lags), nf))
    den = np.zeros((len(lags), nf))
    for off in (0, 36, 72):
        for k, c in enumerate(ft8.COSTAS):
            fr = lag0 + lags + SYM_FRAMES * (off + k)
            valid = (fr >= 0) & (fr < T)
            frc = np.clip(fr, 0, T - 1)
            pk = P[frc]  # (nlag, nf, 8)
            sig = pk[:, :, c]
            tot = pk.sum(-1)
            num += np.where(valid[:, None], sig, 0)
            den += np.where(valid[:, None], (tot - sig) / 7.0, 0)
    score = num / np.maximum(den, 1e-12)
    # normalise by a robust per-slot baseline so the threshold is noise-level independent
    base = np.percentile(score.max(0), 40)
    score = score / max(base, 1e-9)
    peaks = (score == maximum_filter(score, size=(5, 3))) & (score >= min_score)
    li, fi = np.nonzero(peaks)
    order = np.argsort(-score[li, fi])[:max_cand]
    return [
        Candidate(f0=(b0 + fi[o]) * 3.125, dt=lags[li[o]] / FPS, score=float(score[li[o], fi[o]]))
        for o in order
    ]


class Baseband:
    """Holds the full-slot FFT so many candidates can be downsampled cheaply."""

    def __init__(self, x):
        self.X = np.fft.rfft(x, n=NFFT_FULL)
        # noise PSD estimate (median over the 200-3000 Hz band), for normalisation
        band = np.abs(self.X[int(200 / DF_FULL) : int(3000 / DF_FULL)]) ** 2
        self.noise_psd = float(np.median(band)) / np.log(2)  # median of exp dist -> mean

    def downsample(self, f0):
        """Complex baseband (NBB samples @200 Hz) with tone 0 of a signal at f0 mapped to 0 Hz.

        Normalised so white noise has unit variance per complex sample.
        """
        i0 = int(round(f0 / DF_FULL))
        lo, hi = -int(round(1.5 * ft8.TONE_SPACING / DF_FULL)), int(round(8.5 * ft8.TONE_SPACING / DF_FULL))
        k = np.arange(lo, hi)
        src = i0 + k
        ok = (src >= 0) & (src < len(self.X))
        Y = np.zeros(NBB, complex)
        taper_n = 101
        w = np.ones(len(k))
        t = 0.5 * (1 - np.cos(np.pi * np.arange(taper_n) / (taper_n - 1)))
        w[:taper_n] = t
        w[-taper_n:] = t[::-1]
        Y[k[ok] % NBB] = self.X[src[ok]] * w[ok]
        y = np.fft.ifft(Y)
        # Scale: noise var per baseband sample = noise_psd * Nbins / NBB^2
        return y * NBB / np.sqrt(self.noise_psd * (hi - lo))


def _rotate(y, df):
    return y * np.exp(-2j * np.pi * df * np.arange(len(y)) / FS_BB)


def symbol_spectra(y, start, nbins=8, osr=1):
    """Complex DFT of each of the 79 symbols. start = sample index of symbol 0 at 200 Hz.

    Returns (79, nbins*osr...) with bin spacing 6.25/osr Hz starting at 0 Hz.
    """
    idx = start + np.arange(ft8.NN)[:, None] * SPS_BB + np.arange(SPS_BB)[None, :]
    seg = np.where((idx >= 0) & (idx < len(y)), y[np.clip(idx, 0, len(y) - 1)], 0)
    Z = np.fft.fft(seg, n=SPS_BB * osr, axis=1) / np.sqrt(SPS_BB)
    return Z


def _costas_power(y, start):
    Z = symbol_spectra(y, start)
    return sum(np.abs(Z[off + np.arange(7), ft8.COSTAS]) ** 2 for off in ft8.SYNC_POS).sum()


def fine_sync(bb, cand, dt_span=0.05, df_span=2.5, df_step=0.25):
    """Refine (f0, dt). Returns (f0, dt, y_baseband_corrected, start_sample, sync_power)."""
    y0 = bb.downsample(cand.f0)
    start0 = int(round((ft8.NOMINAL_START + cand.dt) * FS_BB))
    best = (-1.0, 0.0, start0)
    # coarse df grid at the coarse start, then dt, then a final df pass
    for df in np.arange(-df_span, df_span + 1e-9, 0.5):
        p = _costas_power(_rotate(y0, df), start0)
        if p > best[0]:
            best = (p, df, start0)
    dts = np.arange(-int(dt_span * FS_BB), int(dt_span * FS_BB) + 1)
    yr = _rotate(y0, best[1])
    for d in dts:
        p = _costas_power(yr, start0 + d)
        if p > best[0]:
            best = (p, best[1], start0 + d)
    for df in best[1] + np.arange(-0.5, 0.5 + 1e-9, df_step):
        p = _costas_power(_rotate(y0, df), best[2])
        if p > best[0]:
            best = (p, df, best[2])
    p, df, start = best
    y = _rotate(y0, df)
    return cand.f0 + df, start / FS_BB - ft8.NOMINAL_START, y, start, p


_SYNC_TONES = np.concatenate([ft8.COSTAS] * 3)  # tone for each of the 21 sync symbols
_SYNC_REF = np.exp(-2j * np.pi * _SYNC_TONES[:, None] * np.arange(SPS_BB)[None, :] / SPS_BB)  # (21,32)
_T_BB = np.arange(NBB) / FS_BB


def costas_grid(y0, start0, dfs, dts):
    """Semi-coherent Costas metric on a (df, dt) grid.

    FT8 is continuous-phase FSK with h=1: every symbol advances an integer number of cycles, so the
    7 symbols of each Costas array add coherently. Coherent 1.12 s integration gives ~7x sharper
    frequency selectivity than per-symbol energy. Arrays are combined non-coherently (fading-robust).
    """
    Y = y0[None, :] * np.exp(-2j * np.pi * dfs[:, None] * _T_BB[None, :])  # (nf, NBB)
    idx = start0 + dts[:, None, None] + SPS_BB * ft8.SYNC_SYMS[None, :, None] + np.arange(SPS_BB)[None, None, :]
    valid = (idx >= 0) & (idx < NBB)
    seg = Y[:, np.clip(idx, 0, NBB - 1)] * valid[None]  # (nf, nt, 21, 32)
    z = np.einsum("ftkn,kn->ftk", seg, _SYNC_REF)
    return (np.abs(z.reshape(len(dfs), len(dts), 3, 7).sum(-1)) ** 2).sum(-1)  # (nf, nt)


def _parabolic(m, i):
    """Sub-grid offset of the peak at index i of 1-D array m (in grid units, |offset|<=0.5)."""
    if i <= 0 or i >= len(m) - 1:
        return 0.0
    a, b, c = np.log(np.maximum(m[i - 1 : i + 2], 1e-30))
    den = a - 2 * b + c
    return float(np.clip(0.5 * (a - c) / den, -0.5, 0.5)) if den < 0 else 0.0


def fractional_shift(y, tau):
    """Advance y by tau samples (y'(n) = y(n + tau)) via an FFT phase ramp."""
    if abs(tau) < 1e-6:
        return y
    k = np.fft.fftfreq(len(y)) * len(y)
    return np.fft.ifft(np.fft.fft(y) * np.exp(2j * np.pi * k * tau / len(y)))


@dataclass
class SyncHyp:
    f0: float
    dt: float
    y: np.ndarray  # baseband, frequency-corrected, fractional delay removed
    start: int  # integer sample index of symbol 0 in y
    metric: float


def fine_sync2(bb, cand, n_hyp=2, df_span=2.5, df_step=0.1, dt_span=12):
    """Refine (f0, dt) with the semi-coherent Costas metric; return up to n_hyp distinct hypotheses."""
    y0 = bb.downsample(cand.f0)
    start0 = int(round((ft8.NOMINAL_START + cand.dt) * FS_BB))
    dfs = np.arange(-df_span, df_span + 1e-9, df_step)
    dts = np.arange(-dt_span, dt_span + 1)
    M = costas_grid(y0, start0, dfs, dts)
    peaks = (M == maximum_filter(M, size=(7, 5)))
    fi, ti = np.nonzero(peaks)
    order = np.argsort(-M[fi, ti])[:n_hyp]
    hyps = []
    for o in order:
        i, j = fi[o], ti[o]
        df = dfs[i] + _parabolic(M[:, j], i) * df_step
        dtf = dts[j] + _parabolic(M[i, :], j)
        y = fractional_shift(_rotate(y0, df), dtf - round(dtf))
        start = start0 + int(round(dtf))
        hyps.append(SyncHyp(cand.f0 + df, (start0 + dtf) / FS_BB - ft8.NOMINAL_START, y, start, float(M[i, j])))
    return hyps


def classical_llr(cs):
    """WSJT-X style single-symbol non-coherent metric (nsym=1), max-log over Gray-mapped tones.

    cs: (79, 8) complex symbol spectra. Returns 174 LLRs, log P(1)/P(0) convention, scaled like WSJT-X.
    """
    s = np.abs(cs[ft8.DATA_POS])  # (58, 8)
    vals = ft8.GRAY_INV  # tone -> 3-bit value
    llr = np.zeros((58, 3))
    for b in range(3):
        bit = (vals >> (2 - b)) & 1  # bit b (MSB first) for each tone
        llr[:, b] = s[:, bit == 1].max(1) - s[:, bit == 0].max(1)
    llr = llr.reshape(-1)
    return 2.83 * llr / max(np.std(llr), 1e-9)


def coherent_llr(cs, nsym):
    """WSJT-X nsym=2/3 metric: coherent sum over groups of nsym data symbols (exploits phase continuity)."""
    if nsym == 1:
        return classical_llr(cs)
    d = cs[ft8.DATA_POS]  # (58, 8)
    tone_val = ft8.GRAY_INV
    nbits = 3 * nsym
    llr = np.zeros(174)
    combos = np.array(np.meshgrid(*[np.arange(8)] * nsym, indexing="ij")).reshape(nsym, -1).T  # (8^n, n) tones
    bits_of = np.zeros((len(combos), nbits), np.uint8)
    for j in range(nsym):
        v = tone_val[combos[:, j]]
        for b in range(3):
            bits_of[:, 3 * j + b] = (v >> (2 - b)) & 1
    for g0 in range(0, 58, nsym):
        g = min(nsym, 58 - g0)
        if g < nsym:  # tail: fall back to smaller groups
            sub = coherent_llr_group(d[g0 : g0 + g], g, tone_val)
            llr[3 * g0 : 3 * (g0 + g)] = sub
            continue
        s = np.abs(sum(d[g0 + j, combos[:, j]] for j in range(nsym)))
        for b in range(nbits):
            llr[3 * g0 + b] = s[bits_of[:, b] == 1].max() - s[bits_of[:, b] == 0].max()
    return 2.83 * llr / max(np.std(llr), 1e-9)


def coherent_llr_group(d, g, tone_val):
    out = np.zeros(3 * g)
    for j in range(g):
        s = np.abs(d[j])
        for b in range(3):
            bit = (tone_val >> (2 - b)) & 1
            out[3 * j + b] = s[bit == 1].max() - s[bit == 0].max()
    return out
