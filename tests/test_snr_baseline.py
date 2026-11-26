"""Reported SNR against jt9's noise floor (frontend.spectrum_baseline, a port of get_spectrum_baseline.f90 +
baseline.f90): the floor is the noise at the signal's own frequency, so a receiver passband that is not flat and a
busy band do not shift the SNR. C == Python on the SNR column is tools/golden_c.py's job."""
import numpy as np
import pytest

from ft8nn import channel, decoder, ft8
from ft8nn import frontend as fe


def _signals(rng, f0s, snr):
    out = []
    for f0 in f0s:
        bits = rng.integers(0, 2, 77).astype(np.uint8)
        bits[74:77] = (0, 0, 1)  # a standard message, so the decoder's plausibility check keeps it
        out.append(channel.Signal(ft8.encode77(bits)[1], f0, rng.uniform(-0.3, 0.5), snr))
    return out


def _shape(x, gain_db):
    """x through a receiver passband: gain_db(f) in dB."""
    X = np.fft.rfft(x)
    return np.fft.irfft(X * 10 ** (gain_db(np.fft.rfftfreq(len(x), 1 / ft8.FS)) / 20), len(x))


def _snrs(x, f0s):
    """{true f0: reported SNR} of the decodes near each f0 (classical metrics, 2 passes)."""
    ds = decoder.decode_slot(x / np.std(x), passes=2, cochannel=None, repeat=None)
    got = {}
    for f0 in f0s:
        near = [d for d in ds if abs(d.f0 - f0) < 3]
        if near:
            got[f0] = near[0].snr
    return got


def test_white_noise_floor_is_the_median():
    """On white noise jt9's floor and the per-slot median agree: noise_ratio ~ 1 everywhere in the fit range."""
    x = np.random.default_rng(1).standard_normal(ft8.SLOT_SAMPLES)
    bb = fe.Baseband(x)
    r = [10 * np.log10(bb.noise_ratio(f)) for f in np.arange(250, 3950, 100)]
    assert max(abs(v) for v in r) < 0.6


F0S = [400.0, 900.0, 1400.0, 1900.0, 2400.0, 2900.0]


def test_flat_noise_reads_the_true_snr():
    rng = np.random.default_rng(2)
    x = channel.synth_slot(_signals(rng, F0S, -10.0), rng)
    got = _snrs(x, F0S)
    assert len(got) == len(F0S)
    assert all(abs(v + 10.0) < 1.0 for v in got.values()), got


def test_passband_slope_does_not_shift_the_snr(monkeypatch):
    """Signal and noise through the same smooth passband (+4 dB at 1.65 kHz, -8 dB at the edges): the SNR at each
    frequency is unchanged, and the local floor reads it. The per-slot median read the edges ~5.5 dB low and the
    middle ~2.5 dB high."""
    rng = np.random.default_rng(3)
    x = _shape(channel.synth_slot(_signals(rng, F0S, -8.0), rng), lambda f: 4 - 12 * ((f - 1650) / 1500) ** 2)
    got = _snrs(x, F0S)
    assert len(got) >= 5
    assert all(abs(v + 8.0) < 1.5 for v in got.values()), got
    orig = decoder.estimate_snr
    monkeypatch.setattr(decoder, "estimate_snr", lambda h, b, bb=None: orig(h, b))  # the per-slot median
    old = _snrs(x, F0S)
    assert old[400.0] < -12 and old[2900.0] < -12 and old[1400.0] > -6


def test_busy_band_does_not_lower_the_snr():
    """25 strong signals (+5 dB) around a -12 dB one: the 10th-percentile floor passes between them."""
    rng = np.random.default_rng(4)
    crowd = [f for f in np.arange(250, 2900, 100.0) if abs(f - 1500) > 120]
    x = channel.synth_slot(_signals(rng, crowd, 5.0) + _signals(rng, [1500.0], -12.0), rng)
    got = _snrs(x, [1500.0])
    assert abs(got[1500.0] + 12.0) < 1.5, got


@pytest.mark.parametrize("nfa,nfb", [(200, 4000), (200, 3000), (100, 4910)])
def test_fit_range_is_jt9s(nfa, nfb):
    """The polynomial is fitted over [nfa, nfb] in 3.125 Hz bins and held constant outside it."""
    x = np.random.default_rng(5).standard_normal(ft8.SLOT_SAMPLES)
    sbase, _ = fe.spectrum_baseline(x, nfa, nfb)
    ia, ib = fe.base_bins(nfa, nfb)
    assert (ia, ib) == (round(nfa / 3.125), round(nfb / 3.125))
    assert np.all(sbase[:ia] == sbase[ia]) and np.all(sbase[ib + 1 :] == sbase[ib])
