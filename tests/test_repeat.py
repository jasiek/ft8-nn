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
