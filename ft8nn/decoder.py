"""End-to-end FT8 slot decoder with a pluggable soft-metric stage.

decode_slot(x) -> list of Decode(bits77, f0, dt, ...)

Stages: candidate search -> semi-coherent fine sync (top-k hypotheses) -> soft metrics
(classical nsym=1/2/3 and/or neural, batched across candidates) -> BP -> OSD fallback -> CRC.
"""
from dataclasses import dataclass, field

import numpy as np

from . import frontend as fe
from . import ft8, ldpc
from .subtract import subtract


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


def decode_slot(x, llr_fn=classical_metrics, max_cand=600, n_hyp=2, osd_order=2, osd_max_hard=40, passes=3, osd_gate=40.0, osd_metrics=1):
    """Decode one 15 s slot (real audio @12 kHz, any scale), with multi-pass subtraction.

    llr_fn(hyps) -> per-hypothesis list of LLR vectors to try in order (first success wins).
    Each pass decodes, then subtracts every new decode from the audio so that weaker signals
    overlapping stronger ones become visible to the next pass's candidate search.
    osd_metrics: OSD is tried only for the first `osd_metrics` metrics of each hypothesis (the
    calibrated NN); later fallback metrics get BP only, which keeps false decodes rare.
    """
    x = np.array(x, dtype=np.float64, copy=True)
    found = {}
    for p in range(passes):
        new = _decode_pass(x, llr_fn, max_cand, n_hyp, osd_order, osd_max_hard, found, osd_gate, osd_metrics)
        for d in new:
            d.info["pass"] = p
            found[d.key] = d
            subtract(x, ft8.encode77(d.bits77)[1], d.f0, d.dt)
        if not new:
            break
    return list(found.values())


def _decode_pass(x, llr_fn, max_cand, n_hyp, osd_order, osd_max_hard, known, osd_gate, osd_metrics):
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
    # batched BP over every (hypothesis, metric) pair
    flat = [(hi, k) for hi, llrs in enumerate(metrics) for k in range(len(llrs))]
    L = np.stack([metrics[hi][k] for hi, k in flat])
    bp_bits, bp_ok, bp_nhard = ldpc.bp_decode(L)
    # expected bit errors under the LLRs: ~70 for noise, <40 for anything OSD can rescue
    exp_err = (1.0 / (1.0 + np.exp(np.abs(L)))).sum(1)
    row = {key: r for r, key in enumerate(flat)}
    found = {}
    done_cand = set()
    for hi, (h, ci) in enumerate(zip(hyps, owner)):
        if ci in done_cand:
            continue
        if any(abs(h.f0 - d.f0) < 3 and abs(h.dt - d.dt) < 0.05 for d in found.values()):
            done_cand.add(ci)
            continue
        for k in range(len(metrics[hi])):
            r = row[(hi, k)]
            a91 = None
            if bp_ok[r] and ft8.crc_ok(bp_bits[r, :91]):
                a91, info = bp_bits[r, :91], {"method": "bp", "nhard": int(bp_nhard[r])}
            elif osd_order >= 0 and k < osd_metrics and exp_err[r] < osd_gate:
                cw, crc, dist, nh = ldpc.osd_decode(L[r], osd_order)
                if crc and nh <= osd_max_hard:
                    a91, info = cw[:91], {"method": "osd", "nhard": nh, "dist": dist}
            if a91 is None:
                continue
            if not a91[:77].any():  # all-zero payload: classic false decode
                break
            d = Decode(a91[:77].copy(), h.f0, h.dt, h.metric, info["method"], {**info, "metric": k, "exp_err": float(exp_err[r])})
            if d.key not in known:
                found.setdefault(d.key, d)
            done_cand.add(ci)
            break
    return list(found.values())
