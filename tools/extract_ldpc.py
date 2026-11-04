"""Extract FT8 LDPC(174,91) tables from ft8_lib's constants.c (parsed as text, never compiled).

Writes ft8nn/data/ldpc.npz with:
  gen  : (83, 91) uint8  generator rows: parity[i] = gen[i] . a91 mod 2
  Nm   : (83, 7)  int16  bit indices (0-based) per check, -1 = padding
  Mn   : (174, 3) int16  check indices (0-based) per bit
"""
import re
import sys
from pathlib import Path

import numpy as np

src = Path(sys.argv[1] if len(sys.argv) > 1 else "third_party/ft8_lib/ft8/constants.c").read_text()


def table(name):
    m = re.search(name + r"\[[^\]]*\]\[[^\]]*\]\s*=\s*\{(.*?)\};", src, re.S)
    rows = re.findall(r"\{([^{}]*)\}", m.group(1))
    return [[int(v, 0) for v in r.replace(" ", "").split(",") if v] for r in rows]


gen_bytes = np.array(table("kFTX_LDPC_generator"), dtype=np.uint8)
gen = np.unpackbits(gen_bytes, axis=1)[:, :91]
Nm = np.array(table("kFTX_LDPC_Nm"), dtype=np.int16) - 1
Mn = np.array(table("kFTX_LDPC_Mn"), dtype=np.int16) - 1
assert gen.shape == (83, 91) and Nm.shape == (83, 7) and Mn.shape == (174, 3)

# Consistency: Nm and Mn must describe the same bipartite graph.
H = np.zeros((83, 174), np.uint8)
for m, row in enumerate(Nm):
    H[m, row[row >= 0]] = 1
H2 = np.zeros_like(H)
for n, row in enumerate(Mn):
    H2[row, n] = 1
assert (H == H2).all(), "Nm/Mn mismatch"
# Generator must be orthogonal to H: H . [I | gen^T]^T = 0
G = np.concatenate([np.eye(91, dtype=np.uint8), gen.T], axis=1)  # (91,174) systematic
assert ((G.astype(int) @ H.T.astype(int)) % 2 == 0).all(), "G not orthogonal to H"

out = Path("ft8nn/data/ldpc.npz")
np.savez_compressed(out, gen=gen, Nm=Nm, Mn=Mn)
print("wrote", out, "edges:", int(H.sum()))
