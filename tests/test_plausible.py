"""message.plausible: protocol-impossible messages and (deep) unallocated callsign blocks are rejected."""
import json
from pathlib import Path

from ft8nn import message

POOL = [m for m in json.loads(Path("ft8nn/data/messages.json").read_text())]


def bits(text):
    return [int(c) for c in next(m["bits"] for m in POOL if m["text"] == text)]


def setfield(b, lo, hi, v):
    b = list(b)
    b[lo:hi] = [int(c) for c in format(v, f"0{hi - lo}b")]
    return b


def first(pred):
    return next(m["text"] for m in POOL if pred(m["text"]))


def test_real_messages_pass():
    assert all(message.plausible([int(c) for c in m["bits"]]) for m in POOL)


def test_cq_with_r_flag_rejected():
    cq = bits(first(lambda t: t.startswith("CQ ") and len(t.split()) == 3))
    assert message.plausible(cq)
    r = list(cq)
    r[58] = 1  # unpacks as "CQ <call> R <grid>", which WSJT-X never sends
    assert message.unpack77(r).split()[-2] == "R"
    assert not message.plausible(r)


def test_report_range():
    b = bits(first(lambda t: t.endswith(("-10", "-11", "-12", "+05"))))
    for irpt, ok in [(5, True), (84, True), (86, True), (105, True), (126, True), (135, True), (136, False), (300, False)]:
        assert message.plausible(setfield(b, 59, 74, message.MAXGRID4 + irpt)) == ok, irpt
    assert not message.plausible(setfield(b, 59, 74, message.MAXGRID4))  # "SA00" is not a locator


def test_deep_call_blocks():
    good = bits(first(lambda t: t.split()[0].startswith(("K", "W", "DL", "JA"))))
    bad = bits(first(lambda t: t.split()[0][:2] in ("QT", "QU", "HM", "ZH")))
    assert message.plausible(good, deep=True)
    assert message.plausible(bad) and not message.plausible(bad, deep=True)


def i3_4(call, icq=0):
    """An i3=4 payload: h12 c58 h1 r2 c1 (c58 = the call right-aligned in 11 base-38 characters)."""
    n58 = 0
    for ch in call.rjust(11):
        n58 = 38 * n58 + message.C38.index(ch)
    return [0] * 12 + [int(c) for c in format(n58, "058b")] + [0, 0, 0, icq] + [1, 0, 0]


def test_deep_nonstandard_call():
    for call in ("PJ4/K1ABC", "HB9HRR/QRP", "YO1BRANCUSI", "PC40EUDXF", "A60FD/14", "SX200H"):  # all seen on the air
        assert message.plausible(i3_4(call), deep=True), call
        assert message.unpack77(i3_4(call, icq=1)) == "CQ " + call
    for call in ("K1A BC", "K1ABC//P", "/K1ABC", "ABCDEFG", "1234567", "QT1ABC", "A/B/C/D"):
        assert message.plausible(i3_4(call)) and not message.plausible(i3_4(call), deep=True), call
    b = i3_4("K1ABC")
    b[12:70] = [1] * 58  # 2^58 - 1 > 38^11: no 11-character call packs to it
    assert not message.plausible(b, deep=True)


def test_deep_dxpedition_and_rtty_calls():
    good, bad = bits(first(lambda t: t.split()[0].startswith(("K", "W")))), bits(first(lambda t: t.split()[0][:2] in ("QT", "QU", "HM", "ZH")))
    dx = good[0:28] + bad[0:28] + [0] * 15 + [0, 0, 1] + [0, 0, 0]  # i3=0 n3=1: c28 c28 h10 r5
    assert message.plausible(dx) and not message.plausible(dx, deep=True)
    dx[28:56] = good[0:28]
    assert message.plausible(dx, deep=True)
    rtty = [1] + good[0:28] + bad[0:28] + [0] * 17 + [0, 1, 1]  # i3=3: t1 c28 c28 R1 r3 s13
    assert message.plausible(rtty) and not message.plausible(rtty, deep=True)


def test_eu_vhf_locator_range():
    b = [0] * 74 + [1, 0, 1]  # i3=5: h12 h22 R1 r3 s11 g25
    assert message.plausible(b)
    b = setfield(b, 49, 74, message.MAXGRID6)
    assert not message.plausible(b)
