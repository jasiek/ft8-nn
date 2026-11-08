"""Bit-exact check of our encoder against WSJT-X `ft8code` output."""
import os
import re
import shutil
import subprocess

import numpy as np
import pytest

from ft8nn import ft8

# WSJT-X's ft8code: $FT8CODE, else on PATH (Linux package wsjtx), else the macOS app bundle; tests skip without it
FT8CODE = os.environ.get("FT8CODE") or shutil.which("ft8code") or "/Applications/wsjtx.app/Contents/MacOS/ft8code"
needs_ft8code = pytest.mark.skipif(not os.path.exists(FT8CODE), reason="WSJT-X ft8code not found")
MESSAGES = ["CQ K1ABC FN42", "K1ABC W9XYZ -11", "W9XYZ K1ABC R-09", "K1ABC W9XYZ RR73", "SP5XYZ G4ABC JO01"]


def ft8code(msg):
    out = subprocess.run([FT8CODE, msg], capture_output=True, text=True).stdout
    lines = out.splitlines()
    get = lambda hdr: lines[next(i for i, l in enumerate(lines) if l.startswith(hdr)) + 1].strip()
    bits = lambda s: np.array([int(c) for c in s], np.uint8)
    tones = np.array([int(c) for c in re.sub(r"\s", "", get("  Sync"))])
    return bits(get("Source-encoded")), bits(get("14-bit CRC")), bits(get("83 Parity")), tones


@needs_ft8code
@pytest.mark.parametrize("msg", MESSAGES)
def test_matches_ft8code(msg):
    p77, crc, parity, tones = ft8code(msg)
    a91 = ft8.add_crc(p77)
    assert (a91[77:] == crc).all()
    assert ft8.crc_ok(a91)
    cw, t = ft8.encode77(p77)
    assert (cw[91:] == parity).all()
    assert (t == tones).all()
    assert (ft8.H.astype(int) @ cw % 2 == 0).all()


def test_waveform_tone_frequencies():
    tones = ft8.codeword_to_tones(ft8.ldpc_encode(ft8.add_crc(np.zeros(77, np.uint8))))
    w = ft8.complex_wave(tones, 1000.0)
    inst_f = np.diff(np.unwrap(np.angle(w))) * ft8.FS / (2 * np.pi)
    mid = inst_f[np.arange(ft8.NN) * ft8.NSPS + ft8.NSPS // 2]  # mid-symbol frequency
    assert np.allclose(mid, 1000.0 + tones * ft8.TONE_SPACING, atol=0.05)
