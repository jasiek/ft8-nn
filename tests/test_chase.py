import json
from pathlib import Path

import numpy as np

from ft8nn import decoder, ft8, ldpc, message
from ft8nn.channel import Signal, synth_slot


def _std_payload():
    """Pick a real plausible standard QSO message (i3=1) from the message pool."""
    pool = json.loads((Path(ft8.__file__).parent / "data/messages.json").read_text())
    for m in pool:
        b = np.array([int(c) for c in m["bits"]], dtype=np.uint8)
        if (
            b[74:77].tolist() == [0, 0, 1]
            and message._int(b[0:28]) >= message.NTOKENS + message.MAX22
            and message._int(b[29:58]) >= message.NTOKENS + message.MAX22
            and message._int(b[59:74]) < message.MAXGRID4
            and message.plausible(b, deep=True)
        ):
            return b
    raise RuntimeError("no suitable message")


QSO_GRID = _std_payload()
QSO_RR73 = QSO_GRID.copy()
QSO_RR73[58] = 0
QSO_RR73[59:74] = [((message.MAXGRID4 + 3) >> (14 - i)) & 1 for i in range(15)]


def _noisy_llr(bits77, sigma, rng):
    """BPSK-equivalent soft LLRs for bits77 with additive Gaussian noise of std sigma."""
    s = 2.0 * ft8.encode77(bits77)[0] - 1.0
    return (s + rng.normal(0.0, sigma, size=174)).astype(np.float32)


def test_chase_combines_two_failed_slots():
    """Two slots that each fail single-slot BP/OSD combine and decode via _chase_stage."""
    rng = np.random.default_rng(42)
    x = synth_slot([Signal(ft8.encode77(QSO_GRID)[1], 1500.0, 0.2, -23.0)], rng)
    l_prev = _noisy_llr(QSO_GRID, 1.35, rng)
    l_cur = _noisy_llr(QSO_GRID, 1.35, rng)
    # Verify neither slot decodes alone via BP
    assert not ldpc.bp_decode(l_prev.astype(np.float64)[None])[1][0]
    assert not ldpc.bp_decode(l_cur.astype(np.float64)[None])[1][0]
    cur_failed = [(1500.0, 0.20, 48.0, l_cur)]
    prev_failed = [(1500.4, 0.22, 48.0, l_prev)]
    got = decoder._chase_stage(x, cur_failed, [prev_failed], {}, *decoder.CHASE, 2, 40, 40.0)
    assert len(got) == 1
    assert got[0].bits77.tobytes() == QSO_GRID.tobytes()
    assert got[0].method in ("bp_ch", "osd_ch")


def test_chase_rejects_silent_current_slot():
    """A strong previous-slot LLR combined with pure noise in the current slot must not leak."""
    rng = np.random.default_rng(7)
    noise = synth_slot([], rng)
    l_prev = (3.5 * (2.0 * ft8.encode77(QSO_GRID)[0] - 1.0)).astype(np.float32)
    l_cur = rng.normal(0.0, 1.0, size=174).astype(np.float32)
    got = decoder._chase_stage(noise, [(1500.0, 0.20, 55.0, l_cur)], [[(1500.0, 0.20, 10.0, l_prev)]], {}, *decoder.CHASE, 2, 40, 40.0)
    assert got == []


def test_chase_rejects_stale_message_when_sender_moved_on():
    """If t-30 had QSO_GRID (strong) and slot t switched to QSO_RR73 (weak), successor margin rejects QSO_GRID."""
    rng = np.random.default_rng(11)
    x = synth_slot([Signal(ft8.encode77(QSO_RR73)[1], 1500.0, 0.2, -22.0)], rng)
    l_prev = (3.5 * (2.0 * ft8.encode77(QSO_GRID)[0] - 1.0)).astype(np.float32)
    l_cur = (1.2 * (2.0 * ft8.encode77(QSO_RR73)[0] - 1.0) + rng.normal(0.0, 0.4, size=174)).astype(np.float32)
    got = decoder._chase_stage(x, [(1500.0, 0.20, 42.0, l_cur)], [[(1500.0, 0.20, 10.0, l_prev)]], {}, *decoder.CHASE, 2, 40, 40.0)
    assert all(d.bits77.tobytes() != QSO_GRID.tobytes() for d in got)


def test_chase_skips_sender_already_decoded_in_slot():
    """If the sender already has a decode in `found`, Chase must not emit a second message for that sender."""
    rng = np.random.default_rng(42)
    x = synth_slot([Signal(ft8.encode77(QSO_GRID)[1], 1500.0, 0.2, -23.0)], rng)
    l_prev = _noisy_llr(QSO_GRID, 1.35, rng)
    l_cur = _noisy_llr(QSO_GRID, 1.35, rng)
    existing = decoder.Decode(QSO_RR73.copy(), 1500.0, 0.20, 5.0, "bp")
    got = decoder._chase_stage(
        x,
        [(1500.0, 0.20, 48.0, l_cur)],
        [[(1500.0, 0.20, 48.0, l_prev)]],
        {existing.key: existing},
        *decoder.CHASE,
        2,
        40,
        40.0,
    )
    assert got == []
