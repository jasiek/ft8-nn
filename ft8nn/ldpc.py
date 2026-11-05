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
    return ((bits.astype(np.int32) @ _H.T.astype(np.int32)) & 1).sum(-1) == 0


def bp_decode(llr, iters=30):
    """Batched sum-product BP.

    llr: (B,174). Returns (bits (B,174) uint8, ok (B,) bool, n_hard_errors (B,) int).
    n_hard_errors = number of channel hard decisions that disagree with the decoded codeword.
    """
    llr = np.atleast_2d(np.asarray(llr, np.float64))
    B = llr.shape[0]
    Lp = -llr  # internal: log P0/P1
    R = np.zeros((B, 83, 7))
    bits = (llr > 0).astype(np.uint8)
    done = syndrome_ok(bits)
    out = bits.copy()
    for _ in range(iters):
        act = ~done
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
        ok = syndrome_ok(b)
        ia = np.where(act)[0]
        out[ia] = b
        done[ia[ok]] = True
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


def osd_decode(llr, order=2, k2=40):
    """Ordered-statistics decoding of one 174-LLR vector.

    order 0: hard decision on the most reliable basis; 1: + all single flips; 2: + pairs among the
    k2 least reliable basis positions. Returns (bits174, ok_crc, dist, n_hard_errors), where dist is
    the reliability-weighted distance (sum |llr| over disagreeing positions).
    """
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
    if order >= 2:
        # least reliable basis positions are the last pivots (pivots are in reliability order)
        low = np.arange(91)[-k2:]
        i, j = np.triu_indices(len(low), 1)
        cands.append(c0[None, :] ^ Gr[low[i]] ^ Gr[low[j]])
    C = np.concatenate(cands).astype(np.uint8)
    dist = ((C != hp[None, :]) * rp[None, :]).sum(-1)
    best = int(np.argmin(dist))
    cw = np.empty(174, np.uint8)
    cw[perm] = C[best]
    return cw, ft8.crc_ok(cw[:91]), float(dist[best]), int((cw != hard).sum())


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
