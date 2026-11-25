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

# Context a-priori: a station keeps its audio frequency and usually continues the same exchange, so a message decoded
# 30..180 s earlier (same period) near f predicts the next one there. Patterns, strongest first:
# standard (i3=1/2): pair = both call fields + i3 (61 bits), sender = second call + i3 (32 bits).
# non-standard (i3=4): i4_pair = h12 + c58 + iflip + i3 (74 bits, QSO only), i4_call = c58 + i3 (61 bits).
CTX_PAIR = np.r_[0:58, 74:77]
CTX_SENDER = np.r_[29:58, 74:77]
CTX_I4_PAIR = np.r_[0:71, 74:77]
CTX_I4_CALL = np.r_[12:70, 74:77]
CTX_DF = 10.0  # Hz
# A sender's clock is stable: run 2, same sender within CTX_DF 30/60 s later, |dDT| p90 0.05 s, p99 0.49 s. Without this
# limit a late copy (DT 2.36) of a strong neighbour decoded as `9R2NKT NI8T R IC50` and then fed itself as context.
CTX_DDT = 0.5  # s
CTX_MAX_HARD = 52  # hard-error cap for osd_cx (run 1 wrong-band 30..180 s null: rejects the sole false hit at nhard=57)
CTX_AGES = (30, 60, 90, 120, 150, 180)  # same-parity context lookback, s (6 periods = 3 min)
# Repeat stage (_repeat_stage, plan 12): (rho_min, margin, LLR clip), chosen on validation run 1.
REPEAT = (0.40, 0.15, 2.0)
# Clear-winning successor thresholds in _repeat_stage (k1 not in prior): tighter than exact repeats because ~130
# successors compete per sender. Tuned on run 1: +220 new decodes (114 slot, 106 active, 0 unconf, 42/42 jt9 exact
# match, 0 stale), 0 / 140,539 wrong-band shuffle null.
REPEAT_SUCC = (0.55, 0.20)
# Runner-up rho for a sender with no sendable competitor (its earlier messages are all CQs, and _successors no longer
# offers a CQ with a report or the R flag), so such a sender needs rho >= 0.50. Run-1 shuffled-context null
# (tools/rp_competitors.py, the same 223,643 tests as above): with runner-up 0, 6 false accepts; 0.30, 1 (the
# highest uncontested wrong CQ scores 0.459); 0.35, 0 - and 1 at margin 0.10, as for the plan-12 competitor set.
REPEAT_SOLO = 0.35
# Competitor set of the repeat stage: True = only messages a station could send (_sendable); False = the plan-12 set
# (bench --repeat-competitors shipped), which reproduces the decodes made before this change.
REPEAT_SENDABLE = True
# Early pass (WSJT-X asks at ~11.8 s; C ft8nn_decode_early): subtraction passes, so it is done before the final request
# at 14.4 s. On 40 busy run-1 slots (M4, all performance cores) passes 0-1 end ~1.55 s after the request, 2-3 at
# ~2.1 / 2.3 s and the co-channel and repeat stages at ~2.7 s; the final pass (prior=) finds what the early one left.
EARLY_PASSES = 2
# Co-channel retry (_cochannel_stage, plan 7.1 item 1c): (peaks per cell within the near dt span, peaks in the far dt
# band, df span Hz, near and far dt spans in baseband samples, minimum data-sync score, neighbour gate), chosen on
# validation run 1. Neighbour gate: each peak's own hypothesis goes to the NN first and its 8 neighbours only if that
# one's expected errors (sum of 1/(1+e^|LLR|)) are below it. On run 1 the stage's 322 decodes came 206 from the peak
# itself (expected errors <= 39.3) and 116 from a neighbour of a peak at <= 61.8, while 65% of the 53,221 peaks were
# at >= 66; so 66 skips 58% of the stage's NN hypotheses and loses none of them.
COCHANNEL = (2, 1, 8.0, 24, 50, 3.0, 66.0)
# Cross-slot soft LLR Chase combining (_chase_stage): combine failed NN LLRs of this slot with failed NN LLRs from
# 30 and/or 60 s earlier (same parity) at the same (f0, dt), with zero extra NN calls.
# (df_max Hz, dt_max s, combine LLR clip, min rho on slot t, min rho on previous slot, min successor margin on slot t),
# tuned on validation run 1 (433 strict new decodes = 335 slot-confirmed, 92 active, 0 unconfirmed; 0 false on 8x
# cross-band shuffle null of 106k pairs; 0 jt9 disagreements).
CHASE = (2.5, 0.12, 4.0, 0.30, 0.20, 0.05)
CHASE_EE_MAX = 68.0  # keep failed non-early NN hypotheses below this expected-error ceiling (noise ~70)
CHASE_CAP = 256      # max failed hypotheses kept per slot (lowest exp_err first)


def context_patterns(context, f0, dt):
    """[(kind, idx, val)] for context messages [(f0, dt, bits77)] within CTX_DF and CTX_DDT of (f0, dt):
    standard (i3 1/2: pair, sender) and non-standard (i3 4: i4_pair for QSO, i4_call)."""
    out, seen = [], set()
    for fc, dc, b in context:
        if abs(fc - f0) > CTX_DF or abs(dc - dt) > CTX_DDT:
            continue
        i3 = b[74:77].tolist()
        if i3 in ([0, 0, 1], [0, 1, 0]):
            pats = (("pair", CTX_PAIR), ("sender", CTX_SENDER))
        elif i3 == [1, 0, 0]:
            pats = (("i4_call", CTX_I4_CALL),) if b[73] else (("i4_pair", CTX_I4_PAIR), ("i4_call", CTX_I4_CALL))
        else:
            continue
        for kind, idx in pats:
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


def _scaled_max_cand(max_cand, cand_kw):
    """Scale candidate cap with searched bandwidth (900 at default 150..4000 Hz, 1101 at 200..4910 Hz,
    floored at max_cand and capped at 1200) so wideband search does not starve in-band candidates."""
    fmin = (cand_kw or {}).get("fmin", fe.FMIN_DEFAULT)
    fmax = (cand_kw or {}).get("fmax", fe.FMAX_DEFAULT)
    w = max(0.0, float(fmax) - float(fmin))
    return max(max_cand, min(1200, int(round(max_cand * 1.5 * w / (fe.FMAX_DEFAULT - fe.FMIN_DEFAULT)))))


def decode_slot(x, llr_fn=classical_metrics, max_cand=600, n_hyp=2, osd_order=2, osd_max_hard=40, passes=4, osd_gate=40.0, osd_metrics=1, ap=(AP_CQ,), ap_gate=44.0, erase_gate=15.0, erase_bp_gate=30.0, erase_osd_dmax=3.0, osd_snaps=2, context=(), ctx_gate=48.0, cand_kw=None, reuse_df=62.5, dt_recenter=0.8, repeat=REPEAT, cochannel=COCHANNEL, prior=(), scale_cand=True, post_pass=True, chase=CHASE, chase_hist=(), failed_out=None):
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
    cochannel: (near peaks, far peaks, df span, near dt span, far dt span, min score, neighbour gate) for the co-channel retry
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
    mc = _scaled_max_cand(max_cand, cand_kw) if scale_cand else max_cand
    tried, subs = {}, []  # candidate key -> pass last tried; (f0, pass) of every subtracted decode
    failed_map = {} if (chase or failed_out is not None) else None
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
        new = _decode_pass(x, llr_fn, mc, n_hyp, osd_order, osd_max_hard, found, osd_gate, osd_metrics, ap, ap_gate, erase_gate, erase_bp_gate, erase_osd_dmax, osd_snaps, context, ctx_gate, ckw, skip, tried, p, failed_map)
        for d in new:
            d.info["pass"] = p
            found[d.key] = d
            subs.append((d.f0, p))
            subtract(x, ft8.encode77(d.bits77)[1], d.f0, d.dt)
        if not new:
            break
    post_f0s = []
    if cochannel and found:
        for d in _cochannel_stage(x, llr_fn, found, *cochannel, osd_order, osd_max_hard, osd_gate, osd_snaps):
            d.info["pass"] = passes
            found[d.key] = d
            subtract(x, ft8.encode77(d.bits77)[1], d.f0, d.dt)
            post_f0s.append(d.f0)
    if chase and chase_hist and failed_map:
        cur_failed = _finalize_failed(failed_map, found)
        for d in _chase_stage(x, cur_failed, chase_hist, found, *chase, osd_order, osd_max_hard, osd_gate):
            d.info["pass"] = passes
            found[d.key] = d
            subtract(x, ft8.encode77(d.bits77)[1], d.f0, d.dt)
            post_f0s.append(d.f0)
    if repeat and context:
        for d in _repeat_stage(x, llr_fn, found, context, n_hyp, *repeat):
            d.info["pass"] = passes
            found[d.key] = d
            if post_pass:
                subtract(x, ft8.encode77(d.bits77)[1], d.f0, d.dt)
                post_f0s.append(d.f0)
    if post_pass and post_f0s:
        df_win = reuse_df or 62.5
        skip = lambda c: not any(abs(fs - c.f0) < df_win for fs in post_f0s)
        ckw = dict(cand_kw or {})
        if dt_recenter and "dt_shift" not in ckw:
            sh = _dt_shift([d.dt for d in found.values()] + [c[1] for c in context], dt_recenter)
            if sh:
                ckw["dt_shift"] = sh
        new = _decode_pass(x, llr_fn, mc, n_hyp, osd_order, osd_max_hard, found, osd_gate, osd_metrics, ap, ap_gate, erase_gate, erase_bp_gate, erase_osd_dmax, osd_snaps, context, ctx_gate, ckw, skip, None, passes + 1, failed_map)
        for d in new:
            d.info["pass"] = passes + 1
            found[d.key] = d
    _drop_ctx_dupes(found)
    if failed_out is not None and failed_map:
        failed_out.extend(_finalize_failed(failed_map, found))
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


def _i4_successors(b):
    """Sendable i3=4 competitors: for CQ c58 (icq=1) only itself; for QSO (icq=0), all 4 nrpt values (0..3) with
    same h12/c58/iflip, plus CQ c58 when iflip==0 (c58 is the sender)."""
    if b[73]:
        return [b.copy()]
    out = []
    for nrpt in range(4):
        c = b.copy()
        c[71], c[72], c[73] = (nrpt >> 1) & 1, nrpt & 1, 0
        out.append(c)
    if not b[70]:
        c = b.copy()
        c[0:12] = 0
        c[70] = c[71] = c[72] = 0
        c[73] = 1
        out.append(c)
    return out


def _repeat_stage(x, llr_fn, found, context, n_hyp, rho_min, margin, clip, log=None):
    """Decodes for context messages whose sender has no decode in this slot (standard i3=1/2 and non-standard i3=4).
    Accepts exact repeats at (rho_min, margin) and clear-winning sendable successors at REPEAT_SUCC. log: list to
    append (best rho, runner-up rho, best is an earlier message, best payload) to for every tested sender."""
    have_s = {_sender_key(d) for d in found.values()}
    have_s.discard(None)
    have_b = set(found)
    groups = {}  # sender_key -> [(f0, dt, bits77)]
    for fc, dc, b in context:
        if "".join(map(str, b)) in have_b:
            continue
        sk = _bits_sender_key(b)
        if sk is not None and sk not in have_s:
            groups.setdefault(sk, []).append((fc, dc, b))
    if not groups:
        return []
    bb = fe.Baseband(x)
    out = []
    succ_rho, succ_margin = REPEAT_SUCC
    for sk, msgs in groups.items():
        is_i4 = sk[0] != "std"
        prior = {m[2].tobytes(): m[2] for m in msgs}
        alts = dict(prior)
        if is_i4:
            for b in prior.values():
                for c in _i4_successors(b):
                    alts.setdefault(c.tobytes(), c)
        else:
            grids = sorted({message._int(b[59:74]) for b in prior.values() if message._int(b[59:74]) < message.MAXGRID4})
            for b in prior.values():
                for c in _successors(b, grids, sendable_only=REPEAT_SENDABLE):
                    alts.setdefault(c.tobytes(), c)
        keys = list(alts)
        S = np.stack([_codeword(alts[k]) for k in keys])  # (n_alt, 174)
        # Keep at most 2 distinct locations (newest first, separated by >= 1 Hz or >= 0.05 s) so 6-slot history
        # does not redundantly re-evaluate the same fine_sync2 window.
        locs = []
        for m in msgs:
            fc, dc = m[0], m[1]
            if any(abs(fc - uf) < 1.0 and abs(dc - ud) < 0.05 for uf, ud in locs):
                continue
            locs.append((fc, dc))
            if len(locs) == 2:
                break
        best = None
        for fc, dc in locs:
            hs = fe.fine_sync2(bb, _Loc(fc, dc), n_hyp=n_hyp)
            if not hs:
                continue
            for h, llrs in zip(hs, llr_fn(hs)):
                l = np.clip(llrs[0], -clip, clip)
                rho = S @ l / (np.abs(l).sum() + 1e-9)
                order = np.argsort(-rho, kind="stable")  # ties: lowest index, as the C port
                i1 = order[0]
                # a sender whose only earlier message is a CQ has no sendable competitor: the runner-up is then the
                # noise ceiling REPEAT_SOLO; i3=4 groups have at most 5 competitors, so floor their runner-up too
                r2 = float(rho[order[1]]) if len(order) > 1 else REPEAT_SOLO
                if is_i4 and r2 < REPEAT_SOLO:
                    r2 = REPEAT_SOLO
                if best is None or rho[i1] > best[0]:
                    best = (float(rho[i1]), r2, keys[i1], h)
        if best is None:
            continue
        r1, r2, k1, h = best
        in_pri = k1 in prior
        if log is not None:
            log.append((r1, r2, in_pri, alts[k1]))
        ok_acc = (r1 >= rho_min and r1 - r2 >= margin) if in_pri else (r1 >= succ_rho and r1 - r2 >= succ_margin)
        if ok_acc:
            d = Decode(alts[k1].copy(), h.f0, h.dt, h.metric, "rp", {"rho": r1, "rho2": r2, "metric": 0, "succ": not in_pri})
            d.snr = estimate_snr(h, d.bits77)
            out.append(d)
    return out


# Co-channel retry (plan 7.1 item 1c). Two stations within a few Hz and tens of ms of each other share one candidate
# cell, because every FT8 signal carries the same three Costas arrays. We decode the stronger and subtract it, and as
# the amplitude was fitted to the sum, the subtraction takes most of the weaker one's sync with it; its data symbols
# carry another message and survive. No candidate search or Costas fine sync finds it afterwards. So after the last
# pass, at every decode's own cell the residual is searched with a data-symbol metric (frontend.data_sync) over df +-8
# Hz x dt +-50 baseband samples (+-0.25 s; 0.5 Hz x 2-sample grid). The best local maxima scoring at least min_score
# go to the NN as 3x3 hypotheses (COCHANNEL_REFINE; the 8 around the peak only when the peak's own NN expected errors
# are below nb_gate, see COCHANNEL): `near` of them within +-24 samples, `far` beyond. Two zones,
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


def _cochannel_stage(x, llr_fn, found, near, far, span, tnear, tfar, min_score, nb_gate, osd_order, osd_max_hard, osd_gate, osd_snaps, log=None):
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
                i, j = int(fi[o]), int(ti[o])
                # Parabolic sub-grid centring on M (as fine_sync2); data_sync's 32-sample window peaks ~1 sample late
                # relative to symbol-aligned NN features, so subtracting 1.0 centres the 3x3 refinement symmetrically.
                df0 = dfs[i] + fe._parabolic(M[:, j], i) * gf
                s0 = start0 + int(round(dts[j] + fe._parabolic(M[i, :], j) * gt - 1.0))
                for a, b in offs:
                    df, s = df0 + a, s0 + b
                    hyps.append(fe.SyncHyp(d.f0 + df, s / fe.FS_BB - ft8.NOMINAL_START, fe._rotate(y0, df), s, float(M[i, j])))
                    group.append((ci, z, rank))
    out, outcome = {}, {}
    if hyps:
        # the peak itself first (offs[0]); its 8 neighbours only where the NN finds the peak's own bits clean enough
        nn = _nn_only(llr_fn)
        L = np.zeros((len(hyps), 174))
        L[::9] = np.stack([l[0] for l in nn(hyps[::9])])
        exp_err = (1.0 / (1.0 + np.exp(np.abs(L)))).sum(1)
        run = np.zeros(len(hyps), bool)
        run[::9] = True
        for k in np.nonzero(exp_err[::9] < nb_gate)[0]:
            run[9 * k + 1 : 9 * k + 9] = True
        nb = np.nonzero(run & (np.arange(len(hyps)) % 9 != 0))[0]
        if len(nb):
            L[nb] = np.stack([l[0] for l in nn([hyps[i] for i in nb])])
            exp_err[nb] = (1.0 / (1.0 + np.exp(np.abs(L[nb])))).sum(1)
        bits, ok, nh = (np.zeros((len(hyps), 174), np.uint8), np.zeros(len(hyps), bool), np.zeros(len(hyps), int))
        bits[run], ok[run], nh[run] = ldpc.bp_decode(L[run])
        done = set()
        for i, (h, g) in enumerate(zip(hyps, group)):
            if g in done or not run[i]:
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


def _finalize_failed(failed_map, found):
    """Filter failed NN hypotheses near any decoded signal, sort by (exp_err, f0, dt), cap at CHASE_CAP."""
    decs = list(found.values())
    kept = [v for v in failed_map.values() if not any(abs(v[0] - d.f0) < 3.0 and abs(v[1] - d.dt) < 0.05 for d in decs)]
    kept.sort(key=lambda v: (v[2], v[0], v[1]))
    return kept[:CHASE_CAP]


def _chase_margin(lt, b77, clip=REPEAT[2]):
    """(rho(lt, b77), margin over best sendable successor on lt). For CQs and non-standard messages with no sendable
    successor, the runner-up ceiling is REPEAT_SOLO (as _repeat_stage)."""
    l = np.clip(lt, -clip, clip)
    denom = float(np.abs(l).sum()) + 1e-9
    s0 = _codeword(b77)
    r0 = float((l @ s0) / denom)
    if b77[74:77].tolist() not in ([0, 0, 1], [0, 1, 0]):
        return r0, r0 - REPEAT_SOLO
    g = message._int(b77[59:74])
    grids = [g] if g < message.MAXGRID4 else []
    k0 = b77.tobytes()
    alts = [c for c in _successors(b77, grids, sendable_only=REPEAT_SENDABLE) if c.tobytes() != k0]
    if not alts:
        return r0, r0 - REPEAT_SOLO
    S = np.stack([_codeword(c) for c in alts])
    r2 = float((S @ l).max() / denom)
    return r0, r0 - r2


def _chase_stage(x, cur_failed, chase_hist, found, df_max, dt_max, clip, rho_t_min, rho_p_min, margin_t, osd_order, osd_max_hard, osd_gate):
    """Cross-slot soft LLR Chase combining: combine failed NN LLRs of this slot with failed NN LLRs from 30 and/or
    60 s earlier within (df_max, dt_max), decode with BP then gated OSD, and gate on uncombined slot agreement
    (rho_t, rho_p) and successor margin (margin_t) so stale messages from t-30/t-60 cannot leak."""
    prevs = [p for p in chase_hist if p]
    if not cur_failed or not prevs:
        return []
    prev_arr = [(np.array([e[0] for e in p]), np.array([e[1] for e in p]), p) for p in prevs]
    comb_L, comb_meta = [], []
    for hi, (f0_i, dt_i, _, lt_f32) in enumerate(cur_failed):
        lt_i = lt_f32.astype(np.float64)
        lt_c = np.clip(lt_i, -clip, clip)
        matches = []
        for f0_p, dt_p, p in prev_arr:
            df = np.abs(f0_p - f0_i)
            dd = np.abs(dt_p - dt_i)
            ok = (df <= df_max) & (dd <= dt_max)
            if not ok.any():
                continue
            cand_idx = np.nonzero(ok)[0]
            dist = (df[cand_idx] / df_max) ** 2 + (dd[cand_idx] / dt_max) ** 2
            best_j = int(cand_idx[int(np.argmin(dist))])
            lp_raw = p[best_j][3].astype(np.float64)
            matches.append((lp_raw, np.clip(lp_raw, -clip, clip)))
        if not matches:
            continue
        for lp_raw, lp_c in matches:
            comb_L.append(lt_c + lp_c)
            comb_meta.append((hi, f0_i, dt_i, lt_i, (lp_raw,)))
        if len(matches) == 2:
            comb_L.append(lt_c + matches[0][1] + matches[1][1])
            comb_meta.append((hi, f0_i, dt_i, lt_i, (matches[0][0], matches[1][0])))
    if not comb_L:
        return []
    Lc = np.stack(comb_L)
    bits, ok, nh = ldpc.bp_decode(Lc)
    ee_c = (1.0 / (1.0 + np.exp(np.abs(Lc)))).sum(1)
    have_s = {_sender_key(d) for d in found.values()}
    have_s.discard(None)
    cells = list(found.values())
    out, done_hi, bb = {}, set(), None
    rho_clip = REPEAT[2]
    for r in range(len(Lc)):
        hi, f0_i, dt_i, lt_i, lps = comb_meta[r]
        if hi in done_hi:
            continue
        a91, info = None, None
        if ok[r] and ft8.crc_ok(bits[r, :91]):
            a91, info = _plausible(bits[r, :91], {"method": "bp_ch", "nhard": int(nh[r])})
        if a91 is None and osd_order >= 0 and ee_c[r] < osd_gate:
            cw, crc, dist, nho = ldpc.osd_decode(Lc[r], osd_order)
            if crc and nho <= osd_max_hard:
                a91, info = _plausible(cw[:91], {"method": "osd_ch", "nhard": nho, "dist": dist})
            if a91 is None and osd_order == 2:
                cw, crc, dist, nho = ldpc.osd_decode(Lc[r], 3)
                if crc and nho <= osd_max_hard:
                    a91, info = _plausible(cw[:91], {"method": "osd_ch", "nhard": nho, "dist": dist})
        if a91 is None or not a91[:77].any():
            continue
        b77 = a91[:77].copy()
        d = Decode(b77, f0_i, dt_i, 0.0, info["method"], {**info, "metric": 0, "exp_err": float(ee_c[r])})
        if d.key in found or d.key in out:
            continue
        sk = _sender_key(d)
        if sk is not None and sk in have_s:
            continue
        if any(_is_ghost(d, e) for e in cells + list(out.values())):
            continue
        rt, mt = _chase_margin(lt_i, b77, rho_clip)
        if rt < rho_t_min or mt < margin_t:
            continue
        s0 = _codeword(b77)
        rp = max(
            float((np.clip(lp, -rho_clip, rho_clip) @ s0) / (np.abs(np.clip(lp, -rho_clip, rho_clip)).sum() + 1e-9))
            for lp in lps
        )
        if rp < rho_p_min:
            continue
        if bb is None:
            bb = fe.Baseband(x)
        y = bb.downsample(f0_i)
        start = int(round((ft8.NOMINAL_START + dt_i) * fe.FS_BB))
        d.snr = estimate_snr(fe.SyncHyp(f0_i, dt_i, y, start, 0.0), b77)
        d.info.update({"rho_t": rt, "rho_p": rp, "margin_t": mt, "n_slots": 1 + len(lps)})
        out[d.key] = d
        if sk is not None:
            have_s.add(sk)
        done_hi.add(hi)
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


def _decode_pass(x, llr_fn, max_cand, n_hyp, osd_order, osd_max_hard, known, osd_gate, osd_metrics, ap, ap_gate, erase_gate, erase_bp_gate, erase_osd_dmax, osd_snaps, context, ctx_gate, cand_kw, skip=None, tried=None, p=0, failed_map=None):
    cands = fe.find_candidates(x, max_cand=max_cand, **cand_kw)
    bb = fe.Baseband(x)
    hyps, owner, h_idx = [], [], []
    for ci, c in enumerate(cands):
        if skip is not None and skip(c):
            continue
        ck = _cand_key(c)
        if tried is not None:
            tried[ck] = p
        if failed_map is not None and not c.early:
            for q in range(n_hyp):
                failed_map.pop((ck, q), None)
        for q, h in enumerate(fe.fine_sync2(bb, c, n_hyp=n_hyp)):
            hyps.append(h)
            owner.append(ci)
            h_idx.append(q)
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
        decoded = False
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
            if a91 is None and k == 0 and erase_gate > 0 and exp_err[r] < max(erase_gate, erase_bp_gate) and not early:
                a91, info = _plausible(*_try_erase(L[r], erase_osd_dmax if exp_err[r] < erase_gate else 0.0))
            if a91 is None:
                continue
            if not a91[:77].any():  # all-zero payload: classic false decode
                break
            d = Decode(a91[:77].copy(), h.f0, h.dt, h.metric, info["method"], {**info, "metric": k, "exp_err": float(exp_err[r])})
            d.snr = estimate_snr(h, d.bits77)
            if d.key not in known:
                found.setdefault(d.key, d)
            done_cand.add(ci)
            decoded = True
            break
        if not decoded and failed_map is not None and not early:
            r0 = row[(hi, 0)]
            ee0 = float(exp_err[r0])
            if ee0 < CHASE_EE_MAX:
                failed_map[(_cand_key(cands[ci]), h_idx[hi])] = (float(h.f0), float(h.dt), ee0, L[r0].astype(np.float32))
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


def _bits_sender_key(b):
    """Sender identity key for standard (i3 1/2) and non-standard (i3 4) messages, else None."""
    i3 = b[74:77].tolist()
    if i3 in ([0, 0, 1], [0, 1, 0]):
        return ("std", b[29:58].tobytes())
    if i3 == [1, 0, 0]:
        if b[73] or not b[70]:  # CQ c58 or <h12> c58 -> sender is c58
            return ("i4_c58", b[12:70].tobytes())
        return ("i4_h12", b[0:12].tobytes(), b[12:70].tobytes())
    return None


def _sender_key(d):
    """Sender identity key of a Decode, else None."""
    return _bits_sender_key(d.bits77)


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
    """Context a-priori: pin each pattern's bits (as _try_ap), BP then OSD-2, then fallback OSD-3; first CRC pass
    that keeps the pinned bits wins (osd_cx capped at CTX_MAX_HARD hard errors against the channel LLRs)."""
    mag = 1.01 * float(np.abs(llr).max())
    hard = (llr > 0).astype(np.uint8)
    for kind, idx, val in patterns:
        l2 = llr.copy()
        l2[idx] = np.where(val == 1, mag, -mag)
        bits, ok, _ = ldpc.bp_decode(l2[None])
        if ok[0] and ft8.crc_ok(bits[0, :91]) and (bits[0, idx] == val).all():
            return bits[0, :91], {"method": "bp_cx", "ctx": kind, "nhard": int((bits[0] != hard).sum())}
        cw, crc, dist, _ = ldpc.osd_decode(l2, 2)
        nh = int((cw != hard).sum()) if crc else 174
        if not (crc and nh <= CTX_MAX_HARD and (cw[idx] == val).all()):
            cw, crc, dist, _ = ldpc.osd_decode(l2, 3)
            nh = int((cw != hard).sum()) if crc else 174
            if not (crc and nh <= CTX_MAX_HARD and (cw[idx] == val).all()):
                continue
        return cw[:91], {"method": "osd_cx", "dist": dist, "ctx": kind, "nhard": nh}
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
