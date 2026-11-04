"""FT8 77-bit message unpacking (port of WSJT-X packjt77 unpack77, common message types).

Supported: i3=1/2 standard (incl. /R, /P, hashed calls, CQ variants), i3=4 nonstandard calls,
i3=0 n3=0 free text, n3=1 DXpedition, n3=5 telemetry. Others are rendered as '<i3.n3:hex>'.
Hashed callsigns resolve through a HashTable that learns from every decoded full callsign.
"""
import numpy as np

NTOKENS = 2063592
MAX22 = 4194304
MAXGRID4 = 32400
C1 = " 0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
C2 = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
C3 = "0123456789"
C4 = " ABCDEFGHIJKLMNOPQRSTUVWXYZ"
C38 = " 0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ/"
C42 = " 0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ+-./?"


def _int(bits):
    v = 0
    for b in bits:
        v = (v << 1) | int(b)
    return v


def ihashcall(call, m):
    """WSJT-X callsign hash (10, 12 or 22 bits)."""
    c = (call.strip().upper() + " " * 11)[:11]
    n = 0
    for ch in c:
        n = 38 * n + max(C38.find(ch), 0)
    return ((47055833459 * n) & 0xFFFFFFFFFFFFFFFF) >> (64 - m)


class HashTable:
    def __init__(self):
        self.h10, self.h12, self.h22 = {}, {}, {}

    def add(self, call):
        call = call.strip()
        if len(call) < 3 or call.startswith("<") or call in ("CQ", "DE", "QRZ"):
            return
        self.h10[ihashcall(call, 10)] = call
        self.h12[ihashcall(call, 12)] = call
        self.h22[ihashcall(call, 22)] = call

    def lookup(self, table, h):
        c = getattr(self, table).get(h)
        return f"<{c}>" if c else "<...>"


def unpack28(n28, ht):
    if n28 < NTOKENS:
        if n28 <= 2:
            return ["DE", "QRZ", "CQ"][n28]
        if n28 <= 1002:
            return f"CQ {n28 - 3:03d}"
        n = n28 - 1003
        s = ""
        for _ in range(4):
            s = C4[n % 27] + s
            n //= 27
        return "CQ " + s.strip()
    n28 -= NTOKENS
    if n28 < MAX22:
        return ht.lookup("h22", n28) if ht else "<...>"
    n = n28 - MAX22
    i1, n = divmod(n, 36 * 10 * 27 * 27 * 27)
    i2, n = divmod(n, 10 * 27 * 27 * 27)
    i3, n = divmod(n, 27 * 27 * 27)
    i4, n = divmod(n, 27 * 27)
    i5, i6 = divmod(n, 27)
    call = (C1[i1] + C2[i2] + C3[i3] + C4[i4] + C4[i5] + C4[i6]).strip()
    if ht:
        ht.add(call)
    return call


def _report(ir, igrid4):
    if igrid4 <= MAXGRID4:
        j1, r = divmod(igrid4, 18 * 10 * 10)
        j2, r = divmod(r, 10 * 10)
        j3, j4 = divmod(r, 10)
        g = chr(65 + j1) + chr(65 + j2) + str(j3) + str(j4)
        return ("R " + g) if ir else g
    irpt = igrid4 - MAXGRID4
    if irpt == 1:
        return ""
    if irpt in (2, 3, 4):
        return ["RRR", "RR73", "73"][irpt - 2]
    isnr = irpt - 35
    if isnr > 50:
        isnr -= 101
    s = f"{isnr:+03d}"
    return ("R" + s) if ir else s


def unpack77(bits, ht=None):
    """77-bit array -> message text (WSJT-X formatting)."""
    b = np.asarray(bits).astype(int)
    i3 = _int(b[74:77])
    if i3 in (1, 2):
        n28a, ipa, n28b, ipb, ir, igrid4 = _int(b[0:28]), b[28], _int(b[29:57]), b[57], b[58], _int(b[59:74])
        sfx = "/R" if i3 == 1 else "/P"
        c1 = unpack28(n28a, ht)
        c2 = unpack28(n28b, ht)
        if ipa and not c1.startswith("<"):
            c1 += sfx
        if ipb and not c2.startswith("<"):
            c2 += sfx
        rpt = _report(ir, igrid4)
        return " ".join(p for p in (c1, c2, rpt) if p)
    if i3 == 4:
        n12, n58, iflip, nrpt, icq = _int(b[0:12]), _int(b[12:70]), b[70], _int(b[71:73]), b[73]
        s = ""
        for _ in range(11):
            s = C38[n58 % 38] + s
            n58 //= 38
        c11 = s.strip()
        if ht:
            ht.add(c11)
        h = ht.lookup("h12", n12) if ht else "<...>"
        if icq:
            return "CQ " + c11
        calls = f"{c11} {h}" if iflip else f"{h} {c11}"
        return calls + ("", " RRR", " RR73", " 73")[nrpt]
    if i3 == 0:
        n3 = _int(b[71:74])
        if n3 == 0:
            n = _int(b[0:71])
            s = ""
            for _ in range(13):
                s = C42[n % 42] + s
                n //= 42
            return s.strip()
        if n3 == 1:  # DXpedition mode: "K1ABC RR73; W9XYZ <KH1/KH7Z> -08"
            c1, c2 = unpack28(_int(b[0:28]), ht), unpack28(_int(b[28:56]), ht)
            h = ht.lookup("h10", _int(b[56:66])) if ht else "<...>"
            return f"{c1} RR73; {c2} {h} {2 * _int(b[66:71]) - 30:+03d}"
        if n3 == 5:
            return f"{_int(b[0:71]):018X}".lstrip("0") or "0"
        return f"<0.{n3}:{_int(b[0:71]):018x}>"
    return f"<{i3}:{_int(b[0:74]):019x}>"
