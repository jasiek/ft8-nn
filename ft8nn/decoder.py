"""End-to-end FT8 slot decoder with a pluggable soft-metric stage.

decode_slot(x) -> list of Decode(bits77, f0, dt, ...)

Stages: candidate search -> semi-coherent fine sync (top-k hypotheses) -> soft metrics
(classical nsym=1/2/3 and/or neural, batched across candidates) -> BP -> OSD fallback -> CRC.
"""
from dataclasses import dataclass, field

import numpy as np

from . import frontend as fe
from . import ldpc


@dataclass
class Decode:
    bits77: np.ndarray
    f0: float
    dt: float
    sync: float
    method: str
    info: dict = field(default_factory=dict)

    @property
    def key(self):
        return "".join(map(str, self.bits77))


def classical_metrics(hyps):
    """For each hypothesis: list of 174-LLR vectors (nsym=1,2,3)."""
    out = []
    for h in hyps:
        cs = fe.symbol_spectra(h.y, h.start)[:, :8]
        out.append([fe.coherent_llr(cs, n) for n in (1, 2, 3)])
    return out


def decode_slot(x, llr_fn=classical_metrics, max_cand=250, n_hyp=2, osd_order=2, osd_max_hard=40):
    """Decode one 15 s slot (real audio @12 kHz, any scale).

    llr_fn(hyps) -> per-hypothesis list of LLR vectors to try in order (first success wins).
    """
    cands = fe.find_candidates(x, max_cand=max_cand)
    bb = fe.Baseband(x)
    hyps, owner = [], []
    for ci, c in enumerate(cands):
        for h in fe.fine_sync2(bb, c, n_hyp=n_hyp):
            hyps.append(h)
            owner.append(ci)
    if not hyps:
        return []
    metrics = llr_fn(hyps)
    found = {}
    done_cand = set()
    for h, ci, llrs in zip(hyps, owner, metrics):
        if ci in done_cand:
            continue
        if any(abs(h.f0 - d.f0) < 3 and abs(h.dt - d.dt) < 0.05 for d in found.values()):
            done_cand.add(ci)
            continue
        for k, llr in enumerate(llrs):
            (a91, info), = ldpc.decode(llr[None, :], osd_order=osd_order, osd_max_hard=osd_max_hard)
            if a91 is None:
                continue
            if not a91[:77].any():  # all-zero payload: classic false decode
                break
            d = Decode(a91[:77].copy(), h.f0, h.dt, h.metric, info["method"], {**info, "metric": k})
            found.setdefault(d.key, d)
            done_cand.add(ci)
            break
    return list(found.values())
