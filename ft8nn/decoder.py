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
    snr: float = float("nan")  # dB in 2500 Hz (WSJT-X convention)

    @property
    def key(self):
        return "".join(map(str, self.bits77))


def _ap_cq():
    """A-priori pattern for 'CQ <call> <grid>' (WSJT-X AP type 1): n28=2 ('CQ'), ipa=0, i3=1.
    Returns (bit indices, bit values) within the 77-bit payload (= first 77 codeword bits)."""
    idx = np.r_[0:29, 74:77]
    val = np.r_[[(2 >> (27 - i)) & 1 for i in range(28)], 0, 0, 0, 1].astype(np.uint8)
    return idx, val


AP_CQ = _ap_cq()


NOISE_BIN = fe.NBB / (int(round(8.5 * ft8.TONE_SPACING / fe.DF_FULL)) + int(round(1.5 * ft8.TONE_SPACING / fe.DF_FULL)))
SNR_OFFSET = 0.23  # calibration vs true SNR on synthetic AWGN (tools/calib_snr.py)


def estimate_snr(h, bits77):
    """SNR (dB, 2500 Hz ref) from the decoded tone sequence: mean power on the transmitted tones vs
    the front end's normalised per-bin noise power. 10*log10(2500/6.25) = 26.0 dB bandwidth term."""
    tones = ft8.encode77(bits77)[1]
    Z = fe.symbol_spectra(h.y, h.start)
    ps = float(np.mean(np.abs(Z[np.arange(ft8.NN), tones]) ** 2))
    return 10 * np.log10(max(ps - NOISE_BIN, 1e-3 * NOISE_BIN) / NOISE_BIN) - 26.0 + SNR_OFFSET


def classical_metrics(hyps):
    """For each hypothesis: list of 174-LLR vectors (nsym=1,2,3)."""
    out = []
    for h in hyps:
        cs = fe.symbol_spectra(h.y, h.start)[:, :8]
        out.append([fe.coherent_llr(cs, n) for n in (1, 2, 3)])
    return out


def decode_slot(x, llr_fn=classical_metrics, max_cand=600, n_hyp=2, osd_order=2, osd_max_hard=40, passes=3, osd_gate=40.0, osd_metrics=1, ap=(AP_CQ,), ap_gate=44.0, erase_gate=15.0, cand_kw=None):
    """Decode one 15 s slot (real audio @12 kHz, any scale), with multi-pass subtraction.

    llr_fn(hyps) -> per-hypothesis list of LLR vectors to try in order (first success wins).
    Each pass decodes, then subtracts every new decode from the audio so that weaker signals
    overlapping stronger ones become visible to the next pass's candidate search.
    osd_metrics: OSD is tried only for the first `osd_metrics` metrics of each hypothesis (the
    calibrated NN); later fallback metrics get BP only, which keeps false decodes rare.
    ap: a-priori bit patterns (indices, values) tried on the NN metric when normal decoding fails,
    for hypotheses whose expected error count is below ap_gate. Default: CQ.
    erase_gate: hypotheses whose NN metric is this confident (expected bit errors below the gate) but
    still fail get a burst-erasure retry (_try_erase); 0 disables.
    cand_kw: extra keyword arguments for frontend.find_candidates (e.g. per_symbol, fmax).
    """
    x = np.array(x, dtype=np.float64, copy=True)
    found = {}
    for p in range(passes):
        new = _decode_pass(x, llr_fn, max_cand, n_hyp, osd_order, osd_max_hard, found, osd_gate, osd_metrics, ap, ap_gate, erase_gate, cand_kw or {})
        for d in new:
            d.info["pass"] = p
            found[d.key] = d
            subtract(x, ft8.encode77(d.bits77)[1], d.f0, d.dt)
        if not new:
            break
    return list(found.values())


def _decode_pass(x, llr_fn, max_cand, n_hyp, osd_order, osd_max_hard, known, osd_gate, osd_metrics, ap, ap_gate, erase_gate, cand_kw):
    cands = fe.find_candidates(x, max_cand=max_cand, **cand_kw)
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
            if a91 is None and k == 0 and ap and exp_err[r] < ap_gate:
                a91, info = _try_ap(L[r], ap, osd_order, osd_max_hard)
            if a91 is None and k == 0 and exp_err[r] < erase_gate:
                a91, info = _try_erase(L[r])
            if a91 is None:
                continue
            if not a91[:77].any():  # all-zero payload: classic false decode
                break
            d = Decode(a91[:77].copy(), h.f0, h.dt, h.metric, info["method"], {**info, "metric": k, "exp_err": float(exp_err[r])})
            d.snr = estimate_snr(h, d.bits77)
            if d.key not in known:
                found.setdefault(d.key, d)
            done_cand.add(ci)
            break
    return list(found.values())


# Burst-erasure windows over the 58 data symbols: (first, end) data-symbol index pairs.
ERASE_WINDOWS = sorted({(a, min(58, a + w)) for w in (6, 10, 14, 18) for a in range(0, 58 - w + 2, 2)})


def _try_erase(llr):
    """Confident LLRs that still fail usually carry a burst of confidently wrong symbols: the sender
    changed the message mid-transmission (WSJT-X switches tones at once), so part of the frame belongs
    to the old message. Zero the LLRs of each window of data symbols in turn and run BP; first CRC
    pass wins."""
    batch = np.repeat(llr[None], len(ERASE_WINDOWS), 0)
    for k, (a, b) in enumerate(ERASE_WINDOWS):
        batch[k, 3 * a : 3 * b] = 0.0
    bits, ok, nh = ldpc.bp_decode(batch)
    for k in np.nonzero(ok)[0]:
        if ft8.crc_ok(bits[k, :91]):
            return bits[k, :91], {"method": "bp_er", "nhard": int(nh[k]), "window": ERASE_WINDOWS[k]}
    return None, None


def _try_ap(llr, patterns, osd_order, osd_max_hard):
    """Pin known payload bits to strong LLRs (WSJT-X: apmag = 1.01 * max|llr|), then BP, then OSD."""
    mag = 1.01 * float(np.abs(llr).max())
    for pi, (idx, val) in enumerate(patterns):
        l2 = llr.copy()
        l2[idx] = np.where(val == 1, mag, -mag)
        bits, ok, nh = ldpc.bp_decode(l2[None])
        cand = None
        if ok[0] and ft8.crc_ok(bits[0, :91]):
            cand = (bits[0, :91], {"method": f"bp_ap{pi}", "nhard": int(nh[0])})
        elif osd_order >= 0:
            cw, crc, dist, nhh = ldpc.osd_decode(l2, osd_order)
            if crc and nhh <= osd_max_hard:
                cand = (cw[:91], {"method": f"osd_ap{pi}", "nhard": nhh, "dist": dist})
        if cand is not None and (cand[0][idx] == val).all():
            return cand
    return None, None
