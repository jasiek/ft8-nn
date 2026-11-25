"""LDPC(174,91) decoding: batched sum-product BP and ordered-statistics decoding (OSD).

LLR convention throughout: llr = log P(bit=1) / P(bit=0)  (positive -> 1), same as ft8_lib and
as a sigmoid-output NN (logit of P(bit=1)).
"""
import numpy as np

from . import ft8

_NM = ft8.LDPC_NM.astype(np.int64)
_PAD = _NM < 0
_NM_IDX = np.where(_PAD, 0, _NM)  # (83,7)

# For each bit n, the flat indices (into 83*7 edge array) of its 3 edges.
_EDGE_OF_BIT = np.zeros((174, 3), np.int64)
for _n in range(174):
    for _k, _m in enumerate(ft8.LDPC_MN[_n]):
        _j = int(np.where(_NM[_m] == _n)[0][0])
        _EDGE_OF_BIT[_n, _k] = _m * 7 + _j

_H = ft8.H.astype(np.uint8)
_G = np.concatenate([np.eye(91, dtype=np.uint8), ft8.LDPC_GEN.T], axis=1)  # (91,174) systematic


def syndrome_ok(bits):
    """bits (B,174) uint8 -> (B,) bool, all parity checks satisfied."""
    return n_unsatisfied(bits) == 0


def n_unsatisfied(bits):
    """bits (B,174) uint8 -> (B,) int, parity checks not satisfied."""
    return ((bits.astype(np.int32) @ _H.T.astype(np.int32)) & 1).sum(-1)


# BP stops a run that is going nowhere (as WSJT-X's bpdecode174_91, with its own numbers): after `patience` iterations
# in a row without fewer unsatisfied parity checks, once at least `min_iter` iterations are done and more than `floor`
# checks are still unsatisfied. Nearly every BP run is a noise hypothesis that never converges and used to run all 30
# iterations. Chosen on validation run 1 (10.2M BP runs, 39,561 converge): WSJT-X's (5, 10, 15) would stop 55 runs
# that converge later; (8, 5, 15) stops 2, whose decodes OSD then finds (same 34,727 decodes, 2 labelled osd0 instead
# of bp0), and keeps 40% of the iterations; (8, 15, 15) stops none and keeps 54%. None: always `iters` iterations.
BP_STOP = (8, 5, 15)


def bp_decode(llr, iters=30, stop=BP_STOP):
    """Batched sum-product BP.

    llr: (B,174). Returns (bits (B,174) uint8, ok (B,) bool, n_hard_errors (B,) int).
    n_hard_errors = number of channel hard decisions that disagree with the decoded codeword.
    stop: (patience, min_iter, floor), see BP_STOP; None runs every row to `iters` or convergence.
    """
    llr = np.atleast_2d(np.asarray(llr, np.float64))
    B = llr.shape[0]
    Lp = -llr  # internal: log P0/P1
    R = np.zeros((B, 83, 7))
    bits = (llr > 0).astype(np.uint8)
    done = syndrome_ok(bits)
    out = bits.copy()
    live = ~done  # rows still iterating: not converged, not stopped
    last = np.zeros(B, int)
    cnt = np.zeros(B, int)
    for it in range(iters):
        act = live.copy()
        if not act.any():
            break
        Ra = R[act]
        tot = Lp[act] + Ra.reshape(Ra.shape[0], -1)[:, _EDGE_OF_BIT].sum(-1)
        Q = tot[:, _NM_IDX] - Ra
        T = np.tanh(np.clip(Q, -40, 40) / 2)
        T[:, _PAD] = 1.0
        T = np.where(np.abs(T) < 1e-12, 1e-12, T)
        ext = np.prod(T, axis=-1, keepdims=True) / T
        Ra = 2 * np.arctanh(np.clip(ext, -0.9999999, 0.9999999))
        Ra[:, _PAD] = 0.0
        R[act] = Ra
        post = Lp[act] + Ra.reshape(Ra.shape[0], -1)[:, _EDGE_OF_BIT].sum(-1)
        b = (post < 0).astype(np.uint8)
        nc = n_unsatisfied(b)
        ok = nc == 0
        ia = np.where(act)[0]
        out[ia] = b
        done[ia[ok]] = True
        live[ia[ok]] = False
        if stop:
            patience, min_iter, floor = stop
            cnt[ia] = np.where(nc < last[ia], 0, cnt[ia] + 1) if it > 0 else 0
            last[ia] = nc
            live[ia[~ok & (cnt[ia] >= patience) & (it + 1 >= min_iter) & (nc > floor)]] = False
    nhard = ((llr > 0).astype(np.uint8) != out).sum(-1)
    return out, done, nhard


def bp_posterior_sums(llr, n=2):
    """Running sums of the BP posteriors (llr sign) after iterations 1..n, as WSJT-X's decode174_91 saves them
    (zsave) for OSD: BP's first iterations pull the bits of a weak codeword towards consistency even when BP itself
    never converges. No early stop. llr (174,) -> (n,174)."""
    Lp = -np.asarray(llr, np.float64)[None]
    R = np.zeros((1, 83, 7))
    zsum = np.zeros(174)
    out = np.empty((n, 174))
    for it in range(n):
        tot = Lp + R.reshape(1, -1)[:, _EDGE_OF_BIT].sum(-1)
        Q = tot[:, _NM_IDX] - R
        T = np.tanh(np.clip(Q, -40, 40) / 2)
        T[:, _PAD] = 1.0
        T = np.where(np.abs(T) < 1e-12, 1e-12, T)
        ext = np.prod(T, axis=-1, keepdims=True) / T
        R = 2 * np.arctanh(np.clip(ext, -0.9999999, 0.9999999))
        R[:, _PAD] = 0.0
        zsum -= (Lp + R.reshape(1, -1)[:, _EDGE_OF_BIT].sum(-1))[0]
        out[it] = zsum
    return out


def _gf2_reduce(Gp):
    """Row-reduce Gp (91,174) over GF(2) choosing pivots left-to-right. Returns (Gr, pivot_cols)."""
    G = Gp.copy().astype(bool)
    k = G.shape[0]
    pivots = []
    r = 0
    for c in range(G.shape[1]):
        rows = np.nonzero(G[r:, c])[0]
        if rows.size == 0:
            continue
        p = r + rows[0]
        if p != r:
            G[[r, p]] = G[[p, r]]
        hit = G[:, c].copy()
        hit[r] = False
        G[hit] ^= G[r]
        pivots.append(c)
        r += 1
        if r == k:
            break
    return G.astype(np.uint8), np.array(pivots)


_OSD_PAIRS = {}
_OSD_TRIPS = {}


def _osd_pairs(k2):
    if k2 not in _OSD_PAIRS:
        lo = 91 - k2
        i, j = np.triu_indices(k2, 1)
        _OSD_PAIRS[k2] = (lo + i, lo + j)
    return _OSD_PAIRS[k2]


def _osd_trips(k3):
    if k3 not in _OSD_TRIPS:
        lo = 91 - k3
        a, b, c = [], [], []
        for i in range(k3):
            for j in range(i + 1, k3):
                for k in range(j + 1, k3):
                    a.append(lo + i)
                    b.append(lo + j)
                    c.append(lo + k)
        _OSD_TRIPS[k3] = (np.array(a, np.intp), np.array(b, np.intp), np.array(c, np.intp))
    return _OSD_TRIPS[k3]


def osd_decode(llr, order=2, k2=None, k3=28):
    """Ordered-statistics decoding of one 174-LLR vector (exact single-argmin ML search).

    order 0: hard decision on the most reliable basis; 1: + all single flips; 2: + pairs among the
    k2 least reliable basis positions (default 40 for order 2, 60 for order 3); 3: + triples among
    the k3 least reliable basis positions (default 28). Uses exact lower-bound pruning on the
    flipped pivot reliabilities (bit-identical to exhaustive evaluation). Returns (bits174, ok_crc,
    dist, n_hard_errors), where dist is the reliability-weighted distance (sum |llr| over
    disagreeing positions).
    """
    if k2 is None:
        k2 = 60 if order >= 3 else 40
    llr = np.asarray(llr, np.float64)
    rel = np.abs(llr)
    hard = (llr > 0).astype(np.uint8)
    perm = np.argsort(-rel, kind="stable")
    Gr, piv = _gf2_reduce(_G[:, perm])
    hp, rp = hard[perm], rel[perm]
    info = hp[piv]  # hard decisions on the most reliable basis
    c0 = (info.astype(np.int32) @ Gr.astype(np.int32)) & 1

    cands = [c0[None, :]]
    if order >= 1:
        cands.append(c0[None, :] ^ Gr)  # flip each info bit
    C01 = np.concatenate(cands).astype(np.uint8)
    d01 = ((C01 != hp[None, :]) * rp[None, :]).sum(-1)
    best = int(np.argmin(d01))
    bd = float(d01[best])
    bw = C01[best]

    p_rel = rp[piv]
    if order >= 2 and k2 >= 2:
        gi, gj = _osd_pairs(k2)
        keep = (p_rel[gi] + p_rel[gj]) < bd + 1e-12
        if keep.any():
            C2 = (c0[None, :] ^ Gr[gi[keep]] ^ Gr[gj[keep]]).astype(np.uint8)
            d2 = ((C2 != hp[None, :]) * rp[None, :]).sum(-1)
            m2 = int(np.argmin(d2))
            if float(d2[m2]) < bd:
                bd = float(d2[m2])
                bw = C2[m2]
    if order >= 3 and k3 >= 3:
        ga, gb, gc = _osd_trips(k3)
        keep = (p_rel[ga] + p_rel[gb] + p_rel[gc]) < bd + 1e-12
        if keep.any():
            C3 = (c0[None, :] ^ Gr[ga[keep]] ^ Gr[gb[keep]] ^ Gr[gc[keep]]).astype(np.uint8)
            d3 = ((C3 != hp[None, :]) * rp[None, :]).sum(-1)
            m3 = int(np.argmin(d3))
            if float(d3[m3]) < bd:
                bd = float(d3[m3])
                bw = C3[m3]

    cw = np.empty(174, np.uint8)
    cw[perm] = bw
    return cw, ft8.crc_ok(cw[:91]), bd, int((cw != hard).sum())


def decode(llr, bp_iters=30, osd_order=2, osd_max_hard=40):
    """BP first; OSD fallback when BP fails. Returns list of (a91 or None, info dict) per row."""
    llr = np.atleast_2d(llr)
    bits, ok, nhard = bp_decode(llr, bp_iters)
    res = []
    for i in range(len(llr)):
        if ok[i] and ft8.crc_ok(bits[i, :91]):
            res.append((bits[i, :91], {"method": "bp", "nhard": int(nhard[i])}))
            continue
        if osd_order >= 0:
            cw, crc, dist, nh = osd_decode(llr[i], osd_order)
            if crc and nh <= osd_max_hard:
                res.append((cw[:91], {"method": "osd", "nhard": nh, "dist": dist}))
                continue
        res.append((None, {"method": "fail"}))
    return res
