import numpy as np

from ft8nn import decoder, ft8, message
from ft8nn.channel import Signal, synth_slot


def _payload(n28a, n28b, r, g15, i3=1):
    bits = lambda v, n: [(v >> (n - 1 - i)) & 1 for i in range(n)]
    return np.array(bits(n28a, 28) + [0] + bits(n28b, 28) + [0, r] + bits(g15, 15) + bits(i3, 3), dtype=np.uint8)


CALL = message.NTOKENS + message.MAX22 + 123456  # any standard-call value: the stage never unpacks it
CQ = _payload(2, CALL, 0, 1234)  # CQ <call> <grid>
QSO = _payload(CALL + 1, CALL, 0, 1234)  # <call> <call> <grid>


def _cq_nonsense(c):
    return 2 <= message._int(c[0:28]) < message.NTOKENS and (c[58] or message._int(c[59:74]) > message.MAXGRID4 + 1)


def test_cq_prior_has_no_unsendable_competitors():
    old = decoder._successors(CQ, [1234], sendable_only=False)
    new = decoder._successors(CQ, [1234])
    assert any(_cq_nonsense(c) for c in old)  # the plan-12 set: CQ + R, CQ + report/RRR/RR73/73
    assert not any(_cq_nonsense(c) for c in new)
    assert all(message.plausible(c) for c in new)


def test_qso_prior_keeps_every_competitor():
    old = decoder._successors(QSO, [1234], sendable_only=False)
    new = decoder._successors(QSO, [1234])
    assert [c.tobytes() for c in old] == [c.tobytes() for c in new]  # report -> R-report -> RR73/RRR/73 all kept


def test_repeat_stage_accepts_a_cq_with_no_competitor_left():
    rng = np.random.default_rng(5)
    x = synth_slot([Signal(ft8.encode77(CQ)[1], 1500.0, 0.2, -12.0)], rng)
    got = decoder._repeat_stage(x, decoder.classical_metrics, {}, [(1500.0, 0.2, CQ)], 2, *decoder.REPEAT)
    assert [d.bits77.tobytes() for d in got] == [CQ.tobytes()] and got[0].method == "rp"
    noise = synth_slot([], rng)
    assert decoder._repeat_stage(noise, decoder.classical_metrics, {}, [(1500.0, 0.2, CQ)], 2, *decoder.REPEAT) == []


def test_repeat_stage_accepts_clear_winning_successor():
    rng = np.random.default_rng(6)
    rr73 = _payload(CALL + 1, CALL, 0, message.MAXGRID4 + 3)
    x = synth_slot([Signal(ft8.encode77(rr73)[1], 1500.0, 0.2, -10.0)], rng)
    got = decoder._repeat_stage(x, decoder.classical_metrics, {}, [(1500.0, 0.2, QSO)], 2, *decoder.REPEAT)
    assert [d.bits77.tobytes() for d in got] == [rr73.tobytes()] and got[0].method == "rp"


def test_i4_context_and_repeat():
    # Build a plausible i3=4 CQ message: CQ W1AW/4 (n12=0, c58="W1AW/4", iflip=0, nrpt=0, icq=1, i3=4)
    c58 = 0
    for ch in "W1AW/4".rjust(11):
        c58 = 38 * c58 + message.C38.index(ch)
    bits = lambda v, n: [(v >> (n - 1 - i)) & 1 for i in range(n)]
    i4_cq = np.array(bits(0, 12) + bits(c58, 58) + [0, 0, 0, 1, 1, 0, 0], dtype=np.uint8)
    assert message.unpack77(i4_cq) == "CQ W1AW/4"
    pats = decoder.context_patterns([(1500.0, 0.2, i4_cq)], 1501.0, 0.22)
    assert len(pats) == 1 and pats[0][0] == "i4_call"
    rng = np.random.default_rng(7)
    x = synth_slot([Signal(ft8.encode77(i4_cq)[1], 1500.0, 0.2, -12.0)], rng)
    got = decoder._repeat_stage(x, decoder.classical_metrics, {}, [(1500.0, 0.2, i4_cq)], 2, *decoder.REPEAT)
    assert [d.bits77.tobytes() for d in got] == [i4_cq.tobytes()] and got[0].method == "rp"
    noise = synth_slot([], rng)
    assert decoder._repeat_stage(noise, decoder.classical_metrics, {}, [(1500.0, 0.2, i4_cq)], 2, *decoder.REPEAT) == []
