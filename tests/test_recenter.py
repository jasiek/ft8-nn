"""DT re-centring: a receiver whose clock or audio stream is off puts every signal at the same wrong DT,
which the 4 s search window then only partly covers (decoder.decode_slot dt_recenter)."""
import numpy as np

from ft8nn import decoder, frontend as fe, ft8
from ft8nn.channel import Signal, synth_slot


def test_dt_shift_median_gate():
    assert decoder._dt_shift([-2.0] * (decoder.DT_RECENTER_MIN - 1), 0.8) == 0.0  # too few decodes
    assert decoder._dt_shift([0.3, 0.1, 0.2, 0.4, 0.2, 0.3], 0.8) == 0.0  # normal receiver: no shift
    assert decoder._dt_shift([-1.63] * 6, 0.8) == -1.64  # snapped to the 40 ms lag grid
    assert decoder._dt_shift([-9.0] * 6, 0.8) == -2.48  # clamped to DT_RECENTER_MAX, then snapped


def test_dt_shift_makes_an_early_candidate_an_ordinary_one():
    """At DT -2.1 only the early track (plain BP/OSD, no erasure or a-priori stages) sees the signal."""
    rng = np.random.default_rng(7)
    tones = ft8.encode77(rng.integers(0, 2, 77).astype(np.uint8))[1]
    x = synth_slot([Signal(tones, 1500.0, -2.1, 10.0)], rng)
    at = lambda cands: [c for c in cands if abs(c.f0 - 1500) < 6 and abs(c.dt + 2.1) < 0.1]
    plain = at(fe.find_candidates(x))
    shifted = at(fe.find_candidates(x, dt_shift=-2.0))
    assert len(plain) == 1 and plain[0].early
    assert len(shifted) == 1 and not shifted[0].early


def test_recentring_decodes_a_slot_the_window_does_not_reach():
    """DT -2.9 s is outside the search window altogether, early track included."""
    rng = np.random.default_rng(11)
    sigs, keys = [], []
    for i in range(4):
        bits = rng.integers(0, 2, 77).astype(np.uint8)
        sigs.append(Signal(ft8.encode77(bits)[1], 1000.0 + 97 * i, -2.9, 10.0))
        keys.append("".join(map(str, bits)))
    x = synth_slot(sigs, rng)
    # context 30 s earlier carries the receiver's offset; far enough in frequency to rule out context a-priori
    ctx = [(300.0 + 20 * i, -2.9, np.zeros(77, dtype=np.uint8)) for i in range(decoder.DT_RECENTER_MIN)]
    off = {d.key for d in decoder.decode_slot(x, context=ctx, dt_recenter=0)} & set(keys)
    on = {d.key for d in decoder.decode_slot(x, context=ctx)} & set(keys)
    print(f"DT -2.9 s: {len(off)} of 4 decoded without re-centring, {len(on)} with")
    assert not off and len(on) >= 3
