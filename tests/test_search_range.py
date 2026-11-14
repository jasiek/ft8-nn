"""Candidate search range: WSJT-X's [nfa, nfb] clamped as jt9 clamps it (frontend.search_range), the C library's
ft8nn_set_range the same, and the search really covering 3.1..4.9 kHz (it stopped at 3100 Hz before 2026-11-14)."""
import ctypes
import pathlib

import numpy as np
import pytest

from ft8nn import frontend as fe
from ft8nn import ft8

ROOT = pathlib.Path(__file__).resolve().parents[1]
LIB = next((p for p in (ROOT / "c/build/libft8nn.dylib", ROOT / "c/build/libft8nn.so") if p.exists()), None)
MODEL = ROOT / "models/v3/ft8nn.onnx"

CASES = [
    ((None, None), (150.0, 4000.0)),     # no request: the default
    ((0, 0), (150.0, 4000.0)),           # nfb <= nfa: unset
    ((-1, 3000), (150.0, 4000.0)),
    ((200, 4000), (200.0, 4000.0)),      # jt9 file mode's default, as is
    ((200, 3000), (200.0, 3000.0)),
    ((0, 5000), (100.0, 4910.0)),        # both ends clamped
    ((50, 3000), (100.0, 3000.0)),
    ((0, 50), (100.0, 150.0)),           # narrow window keeps its 50 Hz width at the bottom
    ((4950, 5000), (4860.0, 4910.0)),    # ... and at the top
    ((1480, 1520), (1480.0, 1520.0)),    # a re-decode around the QSO frequency
    ((4920, 5200), (4920.0, 4923.125)),  # wide window entirely above the cap: never empty
]


@pytest.mark.parametrize("req,want", CASES)
def test_search_range(req, want):
    assert fe.search_range(*req) == pytest.approx(want)


@pytest.mark.skipif(LIB is None or not MODEL.exists(), reason="needs c/build/libft8nn and models/v3/ft8nn.onnx")
def test_c_set_range_equals_python():
    lib = ctypes.CDLL(str(LIB))
    lib.ft8nn_open.restype = ctypes.c_void_p
    lib.ft8nn_open.argtypes = [ctypes.c_char_p, ctypes.c_int]
    lib.ft8nn_set_range.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.c_double]
    lib.ft8nn_get_range.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
    lib.ft8nn_close.argtypes = [ctypes.c_void_p]
    ctx = lib.ft8nn_open(str(MODEL).encode(), 1)
    assert ctx
    try:
        lo, hi = ctypes.c_double(), ctypes.c_double()
        lib.ft8nn_get_range(ctx, ctypes.byref(lo), ctypes.byref(hi))
        assert (lo.value, hi.value) == (fe.FMIN_DEFAULT, fe.FMAX_DEFAULT)
        for (nfa, nfb), want in CASES:
            lib.ft8nn_set_range(ctx, -1 if nfa is None else nfa, -1 if nfb is None else nfb)
            lib.ft8nn_get_range(ctx, ctypes.byref(lo), ctypes.byref(hi))
            assert (lo.value, hi.value) == pytest.approx(want), (nfa, nfb)
    finally:
        lib.ft8nn_close(ctx)


def _tone_slot(f0, rng):
    """15 s of noise with one strong FT8 signal (random tones) at f0, starting at the nominal 0.5 s."""
    tones = ft8.encode77(rng.integers(0, 2, 77).astype(np.uint8))[1]
    x = np.zeros(15 * ft8.FS)
    ph, n0 = 0.0, int(ft8.NOMINAL_START * ft8.FS)
    for k, t in enumerate(tones):
        f = f0 + t * ft8.TONE_SPACING
        n = np.arange(ft8.NSPS)
        x[n0 + k * ft8.NSPS + n] = np.sin(ph + 2 * np.pi * f * n / ft8.FS)
        ph += 2 * np.pi * f * ft8.NSPS / ft8.FS
    return x + 0.3 * rng.standard_normal(len(x))


@pytest.mark.parametrize("f0", [3500.0, 4500.0])
def test_candidates_above_3100(f0):
    x = _tone_slot(f0, np.random.default_rng(int(f0)))
    near = lambda cs: [c for c in cs if abs(c.f0 - f0) < 3.2 and abs(c.dt) < 0.1]
    lo, hi = fe.search_range(200, 4910)
    assert near(fe.find_candidates(x, fmin=lo, fmax=hi))
    assert not near(fe.find_candidates(x, fmin=200, fmax=3100))  # the old top
    if f0 < fe.FMAX_DEFAULT:
        assert near(fe.find_candidates(x))
