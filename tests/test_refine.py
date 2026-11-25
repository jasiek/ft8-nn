import json
from pathlib import Path

import numpy as np

from ft8nn import decoder, ft8, message
from ft8nn.channel import Signal, synth_slot


def _std_payload():
    pool = json.loads((Path(ft8.__file__).parent / "data/messages.json").read_text())
    for m in pool:
        b = np.array([int(c) for c in m["bits"]], dtype=np.uint8)
        if b[74:77].tolist() == [0, 0, 1] and message.plausible(b, deep=True):
            return b
    raise RuntimeError("no suitable message")


def test_refine_stage_rescues_offset_signal_and_skips_noise():
    rng = np.random.default_rng(19)
    b77 = _std_payload()
    # Signal sits at 1500.4 Hz, dt=0.205 s (+0.4 Hz, +1 baseband sample from seed at 1500.0 Hz, dt=0.200 s)
    x = synth_slot([Signal(ft8.encode77(b77)[1], 1500.4, 0.205, -14.0)], rng)
    seed = [(1500.0, 0.200, 35.0, np.zeros(174, np.float32))]
    got = decoder._refine_stage(x, decoder.classical_metrics, seed, {}, *decoder.REFINE, 2, 40, 40.0, 2)
    assert len(got) == 1
    assert got[0].bits77.tobytes() == b77.tobytes()
    assert got[0].method in ("bp_rf", "osd_rf", "osd_bp_rf")

    noise = synth_slot([], rng)
    assert decoder._refine_stage(noise, decoder.classical_metrics, seed, {}, *decoder.REFINE, 2, 40, 40.0, 2) == []
