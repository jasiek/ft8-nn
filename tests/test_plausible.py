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
