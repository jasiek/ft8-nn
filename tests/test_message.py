import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from ft8nn import message

# WSJT-X's ft8code: $FT8CODE, else on PATH (Linux package wsjtx), else the macOS app bundle; tests skip without it
FT8CODE = os.environ.get("FT8CODE") or shutil.which("ft8code") or "/Applications/wsjtx.app/Contents/MacOS/ft8code"
needs_ft8code = pytest.mark.skipif(not os.path.exists(FT8CODE), reason="WSJT-X ft8code not found")
POOL = Path("ft8nn/data/messages.json")  # made by tools/make_message_pool.py (needs ft8code)


def bits_of(msg):
    lines = subprocess.run([FT8CODE, msg], capture_output=True, text=True).stdout.splitlines()
    s = lines[next(i for i, l in enumerate(lines) if l.startswith("Source-encoded")) + 1].strip()
    return np.array([int(c) for c in s])


@pytest.mark.skipif(not POOL.exists(), reason="ft8nn/data/messages.json not present")
def test_message_pool_roundtrip():
    pool = json.loads(POOL.read_text())
    bad = [(m["text"], message.unpack77([int(c) for c in m["bits"]])) for m in pool]
    bad = [b for b in bad if b[0] != b[1]]
    assert not bad, bad[:5]


@needs_ft8code
@pytest.mark.parametrize(
    "msg,expect",
    [
        ("CQ DX K1ABC FN42", "CQ DX K1ABC FN42"),
        ("CQ 290 K1ABC FN42", "CQ 290 K1ABC FN42"),
        ("K1ABC/R W9XYZ/R R FN42", "K1ABC/R W9XYZ/R R FN42"),
        ("G4ABC/P PA9XYZ JO22", "G4ABC/P PA9XYZ JO22"),
        ("K1ABC W9XYZ R-07", "K1ABC W9XYZ R-07"),
        ("TNX BOB 73 GL", "TNX BOB 73 GL"),
        ("CQ PJ4/K1ABC", "CQ PJ4/K1ABC"),
        ("<W9XYZ> PJ4/K1ABC RR73", "<...> PJ4/K1ABC RR73"),
        ("QRZ W9XYZ EN37", "QRZ W9XYZ EN37"),
        ("K1ABC RR73; W9XYZ <KH1/KH7Z> -08", "K1ABC RR73; W9XYZ <...> -08"),
    ],
)
def test_ft8code_roundtrip(msg, expect):
    assert message.unpack77(bits_of(msg)) == expect


@needs_ft8code
def test_hash_resolution():
    ht = message.HashTable()
    message.unpack77(bits_of("CQ W9XYZ EN37"), ht)  # learn W9XYZ
    assert message.unpack77(bits_of("<W9XYZ> PJ4/K1ABC RR73"), ht) == "<W9XYZ> PJ4/K1ABC RR73"
