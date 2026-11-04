import numpy as np

from ft8nn import decoder, ft8
from ft8nn.channel import Signal, synth_slot
from ft8nn.subtract import subtract


def _tones(rng):
    return ft8.encode77(rng.integers(0, 2, 77).astype(np.uint8))


def test_subtraction_removes_strong_signal():
    rng = np.random.default_rng(3)
    _, t = _tones(rng)
    x = synth_slot([Signal(t, 1000.3, 0.137, 10.0)], rng)
    noise_only = np.var(synth_slot([], np.random.default_rng(99)))
    before = np.var(x)
    subtract(x, t, 1000.3, 0.133)  # 4 ms dt error (~2.7x our sync rms): refinement must fix it
    after = np.var(x)
    print(f"var before {before:.2f} after {after:.3f} noise {noise_only:.3f}")
    assert after < noise_only * 1.02


def test_multipass_recovers_overlapped_weak_signal():
    rng = np.random.default_rng(4)
    (cw1, t1), (cw2, t2) = _tones(rng), _tones(rng)
    x = synth_slot([Signal(t1, 1200.0, 0.1, 0.0), Signal(t2, 1225.0, 0.25, -16.0)], rng)
    one = {d.key for d in decoder.decode_slot(x, passes=1)}
    multi = {d.key for d in decoder.decode_slot(x, passes=3)}
    key = lambda cw: "".join(map(str, cw[:77]))
    assert key(cw1) in one and key(cw1) in multi
    assert key(cw2) in multi
