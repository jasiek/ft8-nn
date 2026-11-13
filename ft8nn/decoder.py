"""End-to-end FT8 slot decoder with a pluggable soft-metric stage.

decode_slot(x) -> list of Decode(bits77, f0, dt, ...)

Stages: candidate search -> semi-coherent fine sync (top-k hypotheses) -> soft metrics
(classical nsym=1/2/3 and/or neural, batched across candidates) -> BP -> OSD fallback -> CRC.
"""
from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import maximum_filter

from . import frontend as fe
from . import ft8, ldpc, message
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

# Context a-priori: a station keeps its audio frequency and usually continues the same exchange, so a standard
# message decoded 30 or 60 s earlier (same period) near f predicts the next one there. Patterns, strongest first:
# pair = both call fields + i3 (61 bits: the QSO continues), sender = second call + i3 (32 bits: new partner, after CQ).
# Validation run 2 prototype: +1117 new decodes; with every receiver given another receiver's context instead
# (all decodes would be false) 0 of 211,315 hypotheses decoded.
CTX_PAIR = np.r_[0:58, 74:77]
CTX_SENDER = np.r_[29:58, 74:77]
CTX_DF = 10.0  # Hz
# A sender's clock is stable: run 2, same sender within CTX_DF 30/60 s later, |dDT| p90 0.05 s, p99 0.49 s. Without this
# limit a late copy (DT 2.36) of a strong neighbour decoded as `9R2NKT NI8T R IC50` and then fed itself as context.
CTX_DDT = 0.5  # s
# Repeat stage (_repeat_stage, plan 12): (rho_min, margin, LLR clip), chosen on validation run 1.
REPEAT = (0.40, 0.15, 2.0)
# Runner-up rho for a sender with no sendable competitor (its earlier messages are all CQs, and _successors no longer
# offers a CQ with a report or the R flag), so such a sender needs rho >= 0.50. Run-1 shuffled-context null
# (tools/rp_competitors.py, the same 223,643 tests as above): with runner-up 0, 6 false accepts; 0.30, 1 (the
# highest uncontested wrong CQ scores 0.459); 0.35, 0 - and 1 at margin 0.10, as for the plan-12 competitor set.
REPEAT_SOLO = 0.35
# Competitor set of the repeat stage: True = only messages a station could send (_sendable); False = the plan-12 set
# (bench --repeat-competitors shipped), which reproduces the decodes made before this change.
REPEAT_SENDABLE = True
# Co-channel retry (_cochannel_stage, plan 7.1 item 1c): (peaks per cell within the near dt span, peaks in the far dt
# band, df span Hz, near and far dt spans in baseband samples, minimum data-sync score), chosen on validation run 1.
COCHANNEL = (2, 1, 8.0, 24, 50, 3.0)


def context_patterns(context, f0, dt):
    """[(kind, idx, val)] for the context messages [(f0, dt, bits77)] within CTX_DF and CTX_DDT of (f0, dt); standard
    messages (i3 1/2) only."""
    out, seen = [], set()
    for fc, dc, b in context:
        if abs(fc - f0) > CTX_DF or abs(dc - dt) > CTX_DDT or b[74:77].tolist() not in ([0, 0, 1], [0, 1, 0]):
            continue
        for kind, idx in (("pair", CTX_PAIR), ("sender", CTX_SENDER)):
            key = (kind, b[idx].tobytes())
            if key not in seen:
                seen.add(key)
                out.append((kind, idx, b[idx]))
    return out


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


def decode_slot(x, llr_fn=classical_metrics, max_cand=600, n_hyp=2, osd_order=2, osd_max_hard=40, passes=4, osd_gate=40.0, osd_metrics=1, ap=(AP_CQ,), ap_gate=44.0, erase_gate=15.0, erase_osd_dmax=3.0, osd_snaps=2, context=(), ctx_gate=48.0, cand_kw=None, reuse_df=62.5, dt_recenter=0.8, repeat=REPEAT, cochannel=COCHANNEL, prior=()):
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
    erase_osd_dmax: wide-window (20-24 symbol) OSD-0 erasure stage of _try_erase, accepted only below this
    reliability-weighted distance on the kept bits; 0 disables.
    osd_snaps: when OSD on the NN LLRs fails, OSD again on the running sums of BP posteriors after iterations
    1..osd_snaps (_try_osd_bp, WSJT-X's zsave); 0 disables.
    context: [(f0, dt, bits77)] decoded 30 and 60 s earlier on this receiver; context a-priori (_try_ctx) for
    hypotheses within CTX_DF and CTX_DDT of one of them whose expected error count is below ctx_gate. Empty: off.
    cand_kw: extra keyword arguments for frontend.find_candidates (e.g. per_symbol, fmax).
    reuse_df: in passes after the first, skip a candidate (same frequency bin, lag and track) already tried in an
    earlier pass if no decode subtracted since then lies within reuse_df Hz of it: subtraction only changes the
    audio near the subtracted signal, so its hypotheses would get (nearly) the same LLRs and fail again. 62.5 Hz
    covers the candidate's baseband window (-1.5..8.5 tones), fine-sync range and the subtracted signal's width.
    A/B nn_v2_rs: run 2 -1 decode of 27,930, 1/3 less time; 30 Hz lost 2 of 330 on 8 busy slots. 0: off.
    dt_recenter: when the median DT of the decodes already in hand (this slot's, plus the context's from 30/60 s
    earlier) is at least this many seconds from 0, the receiver's own clock or audio stream is off by that much
    and every signal sits at the same wrong DT; later passes then search a window shifted by that median
    (find_candidates(dt_shift=)). 0 disables.
    repeat: (rho_min, margin, clip) for the repeat stage (_repeat_stage) after the last pass, which tests each context
    message whose sender has no decode in the slot as a whole known message (needs context); None: off.
    cochannel: (near peaks, far peaks, df span, near dt span, far dt span, min score) for the co-channel retry
    (_cochannel_stage) after the last pass, which searches every decode's own cell in the residual for a second
    signal hidden under it; None: off.
    prior: decodes this slot already has, from an early pass on the first part of the audio (WSJT-X asks at ~11.8 s).
    They are subtracted first and count as found (never reported twice, and the co-channel and repeat stages see
    them), are returned with info["prior"] = True, and are never dropped: they have been sent already.
    """
    x = np.array(x, dtype=np.float64, copy=True)
    found = {}
    for d in prior:
        d.info["prior"] = True
        found[d.key] = d
        subtract(x, ft8.encode77(d.bits77)[1], d.f0, d.dt)
    tried, subs = {}, []  # candidate key -> pass last tried; (f0, pass) of every subtracted decode
    for p in range(passes):
        skip = None
        if reuse_df and p:
            skip = lambda c: (q := tried.get(_cand_key(c))) is not None and not any(
                ps >= q and abs(fs - c.f0) < reuse_df for fs, ps in subs)
        ckw = dict(cand_kw or {})
        if dt_recenter and "dt_shift" not in ckw:
            sh = _dt_shift([d.dt for d in found.values()] + [c[1] for c in context], dt_recenter)
            if sh:
                ckw["dt_shift"] = sh
        new = _decode_pass(x, llr_fn, max_cand, n_hyp, osd_order, osd_max_hard, found, osd_gate, osd_metrics, ap, ap_gate, erase_gate, erase_osd_dmax, osd_snaps, context, ctx_gate, ckw, skip, tried, p)
        for d in new:
            d.info["pass"] = p
            found[d.key] = d
            subs.append((d.f0, p))
            subtract(x, ft8.encode77(d.bits77)[1], d.f0, d.dt)
        if not new:
            break
    if cochannel and found:
        for d in _cochannel_stage(x, llr_fn, found, *cochannel, osd_order, osd_max_hard, osd_gate, osd_snaps):
            d.info["pass"] = passes
            found[d.key] = d
            subtract(x, ft8.encode77(d.bits77)[1], d.f0, d.dt)
    if repeat and context:
        for d in _repeat_stage(x, llr_fn, found, context, n_hyp, *repeat):
            found[d.key] = d
    _drop_ctx_dupes(found)
    return list(found.values())


# Repeat stage (plan 12): a station that sent a standard message 30 or 60 s ago often sends exactly the same one
# again (CQ, grid). With all 77 bits known the whole codeword is known, so instead of decoding, test it: rho =
# sum(l*s)/sum(|l|) between the NN LLRs and the codeword s = +-1 (1 = perfect agreement, ~0 = noise). The test runs
# last, on the residual after every decode of the slot is subtracted, at the earlier decode's (f0, dt) - no
# candidate search needed. Competitors: the sender's other earlier messages and every message the same two calls
# could send next (same R flag both ways x grid/RRR/RR73/73/reports -30..+30, and its own CQ with any grid it sent),
# so a sender who moved on to the next over is not read as repeating the last one; a payload 1 bit away still
# scores up to rho 0.54 on a perfect signal, which is why the margin is needed. Only the earlier messages themselves
# are emitted. LLRs are clipped to +-clip first: interference on a few symbols can give |l| ~ 9 on a handful of bits
# (run 2, 07_12m 22:29:30: bits 0-14 only, which every CQ shares), and unclipped they dominate rho.
# Thresholds (rho 0.40, margin 0.15, clip 2) chosen on validation run 1: 0 of 223,643 shuffled-context tests (each
# receiver given another band's context: every acceptance false) pass; at margin 0.10, 1.
REPEAT_REPORTS = [message.MAXGRID4 + v for v in (2, 3, 4)] + [message.MAXGRID4 + 35 + n for n in range(-30, 31)]
_CW_CACHE = {}


def _codeword(bits77):
    k = bits77.tobytes()
    if k not in _CW_CACHE:
        if len(_CW_CACHE) > 200000:
            _CW_CACHE.clear()
        _CW_CACHE[k] = 2.0 * ft8.encode77(bits77)[0] - 1.0
    return _CW_CACHE[k]


def _sendable(c):
    """False for a competitor no station transmits: anything message.plausible rejects (a CQ with the R flag), and a
    CQ carrying RRR/RR73/73 or a report in place of its locator. For a CQ prior these sit one 15-bit field from the
    truth, so on a weak signal they score close behind it and the margin test failed real repeats: on runs 2-4 they
    were the runner-up in 226 of the 320 repeats lost on the margin (docs/miss-profile-dtcrp.md)."""
    if not message.plausible(c):
        return False
    return not (2 <= message._int(c[0:28]) < message.NTOKENS and message._int(c[59:74]) > message.MAXGRID4 + 1)


def _successors(b, grids=(), sendable_only=True):
    """Payloads with b's two call fields and i3, every R flag and g15 a standard sender could use next, plus the
    sender's own CQ with each grid in `grids` (a sender who went back to calling CQ: run 3, `R3RZ BD1AUJ ON80`
    read again while jt9 and JTDX both decoded `CQ BD1AUJ ON80`). sendable_only: leave out what no station sends
    (_sendable); False gives the competitor set shipped with plan 12."""
    out = []
    for gv in grids:
        c = b.copy()
        c[0:28] = [(2 >> (27 - i)) & 1 for i in range(28)]
        c[28], c[58] = 0, 0
        c[59:74] = [(gv >> (14 - i)) & 1 for i in range(15)]
        out.append(c)
    g = message._int(b[59:74])
    for gv in ([g] if g < message.MAXGRID4 else []) + REPEAT_REPORTS:
        gb = np.array([(gv >> (14 - i)) & 1 for i in range(15)], dtype=np.uint8)
        for r in (0, 1):
            c = b.copy()
            c[58], c[59:74] = r, gb
            if not sendable_only or _sendable(c):
                out.append(c)
    return out


def _repeat_stage(x, llr_fn, found, context, n_hyp, rho_min, margin, clip, log=None):
    """Decodes for context messages whose sender has no decode in this slot (see the comment above). log: list to
    append (best rho, runner-up rho, best is an earlier message, best payload) to for every tested sender."""
    have_s = {_sender_key(d) for d in found.values()}
    have_b = set(found)
    groups = {}  # sender -> [(f0, dt, bits77)]
    for fc, dc, b in context:
        if b[74:77].tolist() not in ([0, 0, 1], [0, 1, 0]) or "".join(map(str, b)) in have_b:
            continue
        sk = b[29:58].tobytes()
        if sk not in have_s:
            groups.setdefault(sk, []).append((fc, dc, b))
    if not groups:
        return []
    bb = fe.Baseband(x)
    out = []
    for msgs in groups.values():
        prior = {m[2].tobytes(): m[2] for m in msgs}
        alts = dict(prior)
        grids = sorted({message._int(b[59:74]) for b in prior.values() if message._int(b[59:74]) < message.MAXGRID4})
        for b in prior.values():
            for c in _successors(b, grids, sendable_only=REPEAT_SENDABLE):
                alts.setdefault(c.tobytes(), c)
        keys = list(alts)
        S = np.stack([_codeword(alts[k]) for k in keys])  # (n_alt, 174)
        best = None
        for fc, dc in dict.fromkeys((m[0], m[1]) for m in msgs):  # each distinct location, in context order
            hs = fe.fine_sync2(bb, _Loc(fc, dc), n_hyp=n_hyp)
            if not hs:
                continue
            for h, llrs in zip(hs, llr_fn(hs)):
                l = np.clip(llrs[0], -clip, clip)
                rho = S @ l / (np.abs(l).sum() + 1e-9)
                order = np.argsort(-rho, kind="stable")  # ties: lowest index, as the C port
                i1 = order[0]
                # a sender whose only earlier message is a CQ has no sendable competitor: the runner-up is then the
                # noise ceiling REPEAT_SOLO, so the test asks rho >= REPEAT_SOLO + margin instead of rho >= rho_min
                r2 = float(rho[order[1]]) if len(order) > 1 else REPEAT_SOLO
                if best is None or rho[i1] > best[0]:
                    best = (float(rho[i1]), r2, keys[i1], h)
        if best is None:
            continue
        r1, r2, k1, h = best
        if log is not None:
            log.append((r1, r2, k1 in prior, alts[k1]))
        if k1 in prior and r1 >= rho_min and r1 - r2 >= margin:
            d = Decode(prior[k1].copy(), h.f0, h.dt, h.metric, "rp", {"rho": r1, "rho2": r2, "metric": 0})
            d.snr = estimate_snr(h, d.bits77)
            out.append(d)
    return out


# Co-channel retry (plan 7.1 item 1c). Two stations within a few Hz and tens of ms of each other share one candidate
# cell, because every FT8 signal carries the same three Costas arrays. We decode the stronger and subtract it, and as
# the amplitude was fitted to the sum, the subtraction takes most of the weaker one's sync with it; its data symbols
# carry another message and survive. No candidate search or Costas fine sync finds it afterwards. So after the last
# pass, at every decode's own cell the residual is searched with a data-symbol metric (frontend.data_sync) over df +-8
# Hz x dt +-50 baseband samples (+-0.25 s; 0.5 Hz x 2-sample grid). The best local maxima scoring at least min_score
# go to the NN as 3x3 hypotheses (COCHANNEL_REFINE): `near` of them within +-24 samples, `far` beyond. Two zones,
# because one +-50 window with 2 peaks lost 6 near recoveries to far peaks while gaining 7 far ones (run 1). NN
# metric, BP, OSD and OSD on BP sums as in a pass; plausibility rules; a result that is a decode we already have, or
# a ghost of any decode in the slot, is skipped.
# Validation run 1, offline on the nn_v3_dtcrp residuals (34,295 cells, 14.7 NN hypotheses per cell): +253 decodes =
# 174 slot-confirmed, 74 active, 0 unconfirmed, 5 unscorable; 72 of them are among the 146 jt9-only decodes within
# 10 Hz / 0.4 s of one of ours (near zone only: 67; a brute 561-point grid over df +-8 Hz x dt +-80 ms, no data-sync
# search: 69). min_score 3.0: with the cells of 1,000 busy real slots forced onto noise-only audio (43,474 cells) the
# best score of a cell is at most 2.77, so no hypothesis reaches the NN; with no score limit those cells gave
# 1,173,798 NN hypotheses and 0 decodes.
COCHANNEL_REFINE = (0.4, 1)  # Hz, baseband samples around each data-sync peak (3 x 3 hypotheses)
COCHANNEL_GRID = (0.5, 2)  # data-sync grid steps: Hz, baseband samples


def _nn_only(llr_fn):
    """The NN metric alone if llr_fn offers it (nn_metric.make_llr_fn), else llr_fn."""
    return getattr(llr_fn, "nn", llr_fn)


def _cochannel_stage(x, llr_fn, found, near, far, span, tnear, tfar, min_score, osd_order, osd_max_hard, osd_gate, osd_snaps, log=None):
    """Decodes hidden under the cells of the decodes in `found` (see the comment above). log: list to append
    (cell index, best data-sync score, outcome) to for every cell."""
    bb = fe.Baseband(x)
    gf, gt = COCHANNEL_GRID
    rdf, rdt = COCHANNEL_REFINE
    dfs = np.arange(-span - 2 * gf, span + 2 * gf + 1e-9, gf)  # one step beyond the window each way, so a peak on its
    dts = np.arange(-tfar - gt, tfar + gt + 1, gt)  # edge must be a true local maximum
    infreq = np.abs(dfs)[:, None] <= span + 1e-9
    zones = ((near, infreq & (np.abs(dts)[None, :] <= tnear)), (far, infreq & (np.abs(dts)[None, :] > tnear) & (np.abs(dts)[None, :] <= tfar)))
    offs = [(0.0, 0)] + [(a, b) for a in (-rdf, 0.0, rdf) for b in (-rdt, 0, rdt) if (a, b) != (0.0, 0)]
    cells = list(found.values())
    hyps, group, best = [], [], []
    for ci, d in enumerate(cells):
        y0 = bb.downsample(d.f0)
        start0 = int(round((ft8.NOMINAL_START + d.dt) * fe.FS_BB))
        M = fe.data_sync(y0, start0, dfs, dts)
        lm = (M == maximum_filter(M, size=3)) & (M >= min_score)
        best.append(float(M[zones[0][1] | zones[1][1]].max()))
        for z, (k, zone) in enumerate(zones):
            fi, ti = np.nonzero(lm & zone)
            for rank, o in enumerate(np.argsort(-M[fi, ti], kind="stable")[:k]):
                for a, b in offs:
                    df, s = dfs[fi[o]] + a, start0 + int(dts[ti[o]]) + b
                    hyps.append(fe.SyncHyp(d.f0 + df, s / fe.FS_BB - ft8.NOMINAL_START, fe._rotate(y0, df), s, float(M[fi[o], ti[o]])))
                    group.append((ci, z, rank))
    out, outcome = {}, {}
    if hyps:
        L = np.stack([l[0] for l in _nn_only(llr_fn)(hyps)])
        bits, ok, nh = ldpc.bp_decode(L)
        exp_err = (1.0 / (1.0 + np.exp(np.abs(L)))).sum(1)
        done = set()
        for i, (h, g) in enumerate(zip(hyps, group)):
            if g in done:
                continue
            a91 = None
            if ok[i] and ft8.crc_ok(bits[i, :91]):
                a91, info = _plausible(bits[i, :91], {"method": "bp_cc", "nhard": int(nh[i])})
            if a91 is None and osd_order >= 0 and exp_err[i] < osd_gate:
                cw, crc, dist, n = ldpc.osd_decode(L[i], osd_order)
                if crc and n <= osd_max_hard:
                    a91, info = _plausible(cw[:91], {"method": "osd_cc", "nhard": n, "dist": dist})
                if a91 is None and osd_snaps:
                    a91, info = _try_osd_bp(L[i], osd_snaps, osd_order, osd_max_hard)
                    if a91 is not None:
                        a91, info = _plausible(a91, {**info, "method": "osd_bp_cc"})
            if a91 is None or not a91[:77].any():
                continue
            d = Decode(a91[:77].copy(), h.f0, h.dt, h.metric, info["method"], {**info, "metric": 0, "exp_err": float(exp_err[i])})
            if d.key in found or d.key in out:
                outcome.setdefault(g[0], "known")
                continue
            if any(_is_ghost(d, e) for e in cells + list(out.values())):
                outcome.setdefault(g[0], "ghost")
                continue
            d.snr = estimate_snr(h, d.bits77)
            out[d.key] = d
            outcome[g[0]] = "new"
            done.add(g)
    if log is not None:
        log += [(ci, best[ci], outcome.get(ci, "")) for ci in range(len(cells))]
    return list(out.values())


@dataclass
class _Loc:
    """A fine_sync2 starting point that is not a search candidate."""
    f0: float
    dt: float


def _cand_key(c):
    return round(c.f0 / 3.125), round(c.dt * fe.FPS), c.early


DT_RECENTER_MIN = 6  # decodes needed before their median DT is trusted as the receiver's offset
DT_RECENTER_MAX = 2.5  # s, largest shift applied (jt9's own search floor), so the window cannot walk away


def _dt_shift(dts, gate):
    """Receiver DT offset from decoded (CRC-checked) DTs: their median, snapped to the lag grid.

    Senders' clocks scatter by a few tenths of a second, so a median this far from 0 over this many decodes is
    the receiver's own offset (a KiwiSDR stream that slips, or a sound card / clock that is off in an app)."""
    if len(dts) < DT_RECENTER_MIN:
        return 0.0
    med = float(np.median(dts))
    med = min(max(med, -DT_RECENTER_MAX), DT_RECENTER_MAX)
    return 0.0 if abs(med) < gate else round(med * fe.FPS) / fe.FPS


def _decode_pass(x, llr_fn, max_cand, n_hyp, osd_order, osd_max_hard, known, osd_gate, osd_metrics, ap, ap_gate, erase_gate, erase_osd_dmax, osd_snaps, context, ctx_gate, cand_kw, skip=None, tried=None, p=0):
    cands = fe.find_candidates(x, max_cand=max_cand, **cand_kw)
    bb = fe.Baseband(x)
    hyps, owner = [], []
    for ci, c in enumerate(cands):
        if skip is not None and skip(c):
            continue
        if tried is not None:
            tried[_cand_key(c)] = p
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
        # early-DT candidates have weaker sync evidence: no OSD on BP sums or under a-priori bits, no erasure stages
        early = cands[ci].early
        for k in range(len(metrics[hi])):
            r = row[(hi, k)]
            # every stage's result must be a message that could really be sent (_plausible); if not, the next stage
            # gets its turn
            a91 = None
            if bp_ok[r] and ft8.crc_ok(bp_bits[r, :91]):
                a91, info = _plausible(bp_bits[r, :91], {"method": "bp", "nhard": int(bp_nhard[r])})
            if a91 is None and osd_order >= 0 and k < osd_metrics and exp_err[r] < osd_gate:
                cw, crc, dist, nh = ldpc.osd_decode(L[r], osd_order)
                if crc and nh <= osd_max_hard:
                    a91, info = _plausible(cw[:91], {"method": "osd", "nhard": nh, "dist": dist})
                if a91 is None and osd_snaps and not early:
                    a91, info = _plausible(*_try_osd_bp(L[r], osd_snaps, osd_order, osd_max_hard))
            if a91 is None and k == 0 and context and not early and exp_err[r] < ctx_gate:
                pats = context_patterns(context, h.f0, h.dt)
                if pats:
                    a91, info = _plausible(*_try_ctx(L[r], pats))
            if a91 is None and k == 0 and ap and exp_err[r] < ap_gate:
                a91, info = _plausible(*_try_ap(L[r], ap, -1 if early else osd_order, osd_max_hard))
            if a91 is None and k == 0 and exp_err[r] < erase_gate and not early:
                a91, info = _plausible(*_try_erase(L[r], erase_osd_dmax))
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
    # Wide-window OSD erasure and OSD on BP sums can turn a ghost of a strong signal into a CRC-valid random message:
    # at a wrong lag and/or tone bin the NN reads the real signal's tones confidently in the wrong places. Drop such a
    # decode if its data tones mostly equal a neighbour's tones shifted by whole symbols and whole tones (_is_ghost).
    # Checked after the whole pass (before subtraction), so candidate order doesn't matter.
    others = [d for d in found.values() if d.method not in GHOST_PRONE] + list(known.values())
    for key in [k for k, d in found.items() if d.method in GHOST_PRONE]:
        if any(_is_ghost(found[key], e) for e in others):
            del found[key]
    _drop_ctx_dupes(found, known.values())
    return list(found.values())


# Ghost test: on validation runs 1+2 the 6 garbage osd_er decodes matched a neighbour's shifted tones on 53-62% of
# 43-56 comparable data symbols; real osd_er decodes at most 23% (random: 1/8).
GHOST_MAX_DF, GHOST_MIN_SYMS, GHOST_MATCH = 50.0, 20, 0.4
# Methods checked for ghosts. osd_bp: on run 2 (nn_v2_ob) the test flagged 1 of 512 osd_bp decodes, the only
# unconfirmed one (`0J4SMG/R TS6KAQ/R R AO20`, 12 symbols after a -1.5 dB signal 1.6 Hz away).
GHOST_PRONE = ("osd_er", "osd_bp")
CTX_METHODS = ("bp_cx", "osd_cx")


def _sender_key(d):
    """Second call field (+ ipb) of a standard message, else None."""
    return d.bits77[29:58].tobytes() if d.bits77[74:77].tolist() in ([0, 0, 1], [0, 1, 0]) else None


def _drop_ctx_dupes(found, known=()):
    """A station sends one message per slot. A context decode whose sender another decode of this slot already has
    is a second hypothesis on that signal (other DT) read with the pinned calls: drop it (in place)."""
    have = {_sender_key(d) for d in list(found.values()) + list(known) if d.method not in CTX_METHODS}
    for key in [k for k, d in found.items() if d.method in CTX_METHODS]:
        sk = _sender_key(found[key])
        if sk in have and not found[key].info.get("prior"):
            del found[key]
        else:
            have.add(sk)


def _is_ghost(d, e):
    """True if d's data tones equal e's tones shifted by k symbols and m tones on >= GHOST_MATCH of the
    >= GHOST_MIN_SYMS data symbols where e's shifted tone falls inside d's 8-tone band."""
    df = d.f0 - e.f0
    if abs(df) >= GHOST_MAX_DF:
        return False
    td, te = ft8.encode77(d.bits77)[1], ft8.encode77(e.bits77)[1]
    x, y = df / ft8.TONE_SPACING, (d.dt - e.dt) * ft8.FS / ft8.NSPS
    for m in range(int(np.floor(x)), int(np.ceil(x)) + 1):
        for k in range(int(np.floor(y)) - 1, int(np.ceil(y)) + 2):
            j = ft8.DATA_POS + k
            ok = (j >= 0) & (j < ft8.NN)
            t = te[j[ok]] - m
            band = (t >= 0) & (t < 8)
            n = int(band.sum())
            if n >= GHOST_MIN_SYMS and (td[ft8.DATA_POS[ok]][band] == t[band]).sum() >= GHOST_MATCH * n:
                return True
    return False


# Burst-erasure windows over the 58 data symbols: (first, end) data-symbol index pairs.
ERASE_WINDOWS = sorted({(a, min(58, a + w)) for w in (6, 10, 14, 18) for a in range(0, 58 - w + 2, 2)})
# Wider bursts (message changed early or late in the frame): too many erasures for BP, so OSD-0. At most 24
# symbols (72 bits) erased leaves >= 102 kept bits: 91 form the OSD basis, the other >= 11 plus CRC-14 check it.
ERASE_OSD_WINDOWS = sorted({(a, min(58, a + w)) for w in (20, 22, 24) for a in range(0, 58 - w + 2, 2)})


def _plausible(a91, info):
    """(a91, info) if the payload could be a real transmission (message.plausible; the callsign-block test only for
    OSD-family methods, which can make CRC-valid words out of noise), else (None, None)."""
    if a91 is not None and not message.plausible(a91[:77], deep=info["method"].startswith("osd")):
        return None, None
    return a91, info


def _try_erase(llr, osd_dmax=3.0):
    """Confident LLRs that still fail usually carry a burst of confidently wrong symbols: the sender
    changed the message mid-transmission (WSJT-X switches tones at once), so part of the frame belongs
    to the old message. Zero the LLRs of each window of data symbols in turn and run BP; first CRC
    pass wins. Then, if osd_dmax > 0, OSD-0 on each wide window: the CRC-passing result with the smallest
    distance (sum |llr| over kept bits that disagree) wins if that distance is below osd_dmax."""
    batch = np.repeat(llr[None], len(ERASE_WINDOWS), 0)
    for k, (a, b) in enumerate(ERASE_WINDOWS):
        batch[k, 3 * a : 3 * b] = 0.0
    bits, ok, nh = ldpc.bp_decode(batch)
    for k in np.nonzero(ok)[0]:
        if ft8.crc_ok(bits[k, :91]):
            return bits[k, :91], {"method": "bp_er", "nhard": int(nh[k]), "window": ERASE_WINDOWS[k]}
    if osd_dmax > 0:
        best = None
        for a, b in ERASE_OSD_WINDOWS:
            e = llr.copy()
            e[3 * a : 3 * b] = 0.0
            cw, crc, dist, n = ldpc.osd_decode(e, 0)
            if crc and dist < osd_dmax and (best is None or dist < best[1]):
                best = (cw, dist, n, (a, b))
        if best is not None:
            cw, dist, n, win = best
            return cw[:91], {"method": "osd_er", "nhard": n, "dist": dist, "window": win}
    return None, None


def _try_osd_bp(llr, n, osd_order, osd_max_hard):
    """OSD on the running sums of BP posteriors after iterations 1..n (ldpc.bp_posterior_sums). The hard-error
    count is taken against the channel LLRs, as for plain OSD. Prototype on validation run 2 (decodes the default
    decoder missed): snapshots 1-2 add 243 checkable decodes, 154 slot-confirmed, 0 unconfirmed; snapshot 3 adds
    one unconfirmed garbage decode, so n = 2 (WSJT-X's maxosd at its deepest setting is also 2)."""
    hard = (llr > 0).astype(np.uint8)
    for s, z in enumerate(ldpc.bp_posterior_sums(llr, n)):
        cw, crc, dist, _ = ldpc.osd_decode(z, osd_order)
        nh = int((cw != hard).sum())
        if crc and nh <= osd_max_hard:
            return cw[:91], {"method": "osd_bp", "nhard": nh, "dist": dist, "snap": s + 1}
    return None, None


def _try_ctx(llr, patterns):
    """Context a-priori: pin each pattern's bits (as _try_ap), BP then OSD-2; first CRC pass that keeps the pinned
    bits wins. No hard-error limit: the wrong-context test (0 false of 211,315) was run without one."""
    mag = 1.01 * float(np.abs(llr).max())
    hard = (llr > 0).astype(np.uint8)
    for kind, idx, val in patterns:
        l2 = llr.copy()
        l2[idx] = np.where(val == 1, mag, -mag)
        bits, ok, _ = ldpc.bp_decode(l2[None])
        if ok[0] and ft8.crc_ok(bits[0, :91]):
            cw, info = bits[0], {"method": "bp_cx"}
        else:
            cw, crc, dist, _ = ldpc.osd_decode(l2, 2)
            if not crc:
                continue
            info = {"method": "osd_cx", "dist": dist}
        if (cw[idx] == val).all():
            return cw[:91], {**info, "ctx": kind, "nhard": int((cw != hard).sum())}
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
