"""Decode real 15 s slot recordings and compare with jt9 decodes (no ground truth).

  uv run python bench/run_real.py datasets/ft8-training-data/recordings --model runs/v1/model.pt --tag nn_v1
  uv run python bench/run_real.py datasets/ft8-training-data/recordings --compare-only --tag nn_v1

Recordings layout: <root>/<station>/slots/*.wav (+ optional decodes.txt from jt9:
"<slot> <hhmmss> <snr> <dt> <freq> ~ <text>"). The recordings folder is only read; our
decodes go to data/real/<station>/decodes_<tag>.txt.

Comparison: per slot, normalised message text (hashed calls -> <...>, jt9 '?'/'aN' marks removed).
Our-only decodes are checked for corroboration: does any callsign in it appear in jt9's decodes
anywhere in that station's recording? Random false decodes produce random callsigns, so a high
corroboration rate means our extra decodes are real.
"""
import argparse
import os
import re
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.io import wavfile

_LLR = None
_NHYP = 2
_CKW = {}
_EG = 15.0
_EOD = 3.0
_OSN = 2
_NP = 4
_CTX = True
_RDF = 62.5
_DTR = 0.8
_TRUNC = None
_CTXF = None
_PRIOR = None
_AP = True
_EF = None
_LLR_E = None


def _init(model, also_classical, n_hyp, early_nn="", ckw=None, erase_gate=15.0, erase_osd_dmax=3.0, osd_snaps=2, passes=4, context=True, reuse_df=62.5, dt_recenter=0.8, repeat=None, cochannel=None, sendable=True, truncate=None, context_from=None, prior=None, ap=True, early_final=None):
    global _EF, _LLR_E, _AP, _LLR, _NHYP, _CKW, _EG, _EOD, _OSN, _NP, _CTX, _RDF, _DTR, _RP, _CC, _TRUNC, _CTXF, _PRIOR
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import torch

    torch.set_num_threads(1)
    from ft8nn import decoder, nn_metric

    _LLR = nn_metric.make_llr_fn(model, also_classical=also_classical, n_valid=truncate, erase="erase" in early_nn,
                                 mask_median="median" in early_nn) if model else decoder.classical_metrics
    _NHYP = n_hyp
    _CKW = ckw or {}
    _EG = erase_gate
    _EOD = erase_osd_dmax
    _OSN = osd_snaps
    _NP = passes
    _CTX = context
    _RDF = reuse_df
    _DTR = dt_recenter
    _RP = repeat
    _CC = cochannel
    _TRUNC = truncate
    _CTXF = context_from
    _PRIOR = prior
    _AP = ap
    _EF = early_final
    if early_final:
        _LLR_E = nn_metric.make_llr_fn(model, also_classical=also_classical, n_valid=early_final, mask_median=True)
    decoder.REPEAT_SENDABLE = sendable  # repeat-stage competitor set (--repeat-competitors)


def _cc_arg(v):
    """--cochannel NEAR_PEAKS,FAR_PEAKS,DF_SPAN,NEAR_DT,FAR_DT,MIN_SCORE or 'off' -> decode_slot(cochannel=)."""
    if v == "off":
        return None
    kn, kf, span, tn, tf, thr = v.split(",")
    return int(kn), int(kf), float(span), int(tn), int(tf), float(thr)


def _slot_time(stem):
    try:
        return datetime.strptime(stem, "%y%m%d_%H%M%S")
    except ValueError:
        return None


def _load_decodes(path):
    """{slot: [(f0, dt, bits77, method, snr)]} from a decodes_<tag>.txt file."""
    out = defaultdict(list)
    if path.exists():
        for line in path.read_text().splitlines():
            slot, f0, dt, meth, _, hexbits, snr = line.split("\t")[:7]
            bits = np.array([int(c) for c in f"{int(hexbits, 16):077b}"], np.uint8)
            out[slot].append((float(f0), float(dt), bits, re.sub(r"\d+$", "", meth), float(snr)))
    return out


def _load_context(path):
    """{slot time: [(f0, dt, bits77)]} from a decodes_<tag>.txt file (the --context-from decodes)."""
    return {_slot_time(s): [d[:3] for d in v] for s, v in _load_decodes(path).items()}


def _decode_chain(paths):
    """Decode slots in time order; each gets the decodes of the slots 30 and 60 s earlier as context
    (a receiver running continuously). Chains are one period (even/odd slots) of one station.
    With --context-from the context is that file's decodes instead of this run's own (an early pass
    has the final decodes of the earlier slots, not their early ones)."""
    done, out = {}, []
    station = Path(paths[0]).parent.parent.name
    if _CTXF:
        done = _load_context(Path(_CTXF[0]) / station / f"decodes_{_CTXF[1]}.txt")
    prior = _load_decodes(Path(_PRIOR[0]) / station / f"decodes_{_PRIOR[1]}.txt") if _PRIOR else {}
    for path in paths:
        t = _slot_time(Path(path).stem)
        ctx = [c for b in (30, 60) for c in done.get(t - timedelta(seconds=b), [])] if t and _CTX else []
        stem, res = _decode_early_final(path, ctx) if _EF else _decode(path, ctx, prior.get(Path(path).stem, ()))
        if t and not _CTXF:
            done[t] = [(r[1], r[2], r[0]) for r in res]
        out.append((stem, res))
    return out


def _decode(path, context=(), prior=()):
    from ft8nn import decoder

    prior = [decoder.Decode(b, f0, dt, 0.0, m, {"metric": 0}, snr) for f0, dt, b, m, snr in prior]
    sr, x = wavfile.read(path)
    assert sr == 12000, f"{path}: expected 12 kHz, got {sr}"
    x = x.astype(np.float64)
    if x.ndim > 1:
        x = x.mean(1)
    if len(x) < 180000:
        x = np.pad(x, (0, 180000 - len(x)))
    x = x[: _TRUNC or 180000]  # an early pass: only the first _TRUNC samples have arrived (the C library gets just those)
    ds = decoder.decode_slot(x / (np.std(x) + 1e-9), llr_fn=_LLR, n_hyp=_NHYP, erase_gate=_EG, erase_osd_dmax=_EOD, osd_snaps=_OSN, passes=_NP, context=context, cand_kw=_CKW, reuse_df=_RDF, dt_recenter=_DTR, repeat=_RP, cochannel=_CC, prior=prior, ap=(decoder.AP_CQ,) if _AP else ())
    return Path(path).stem, [(d.bits77, d.f0, d.dt, d.method, d.info.get("metric"), d.snr) for d in ds]


def _decode_early_final(path, context=()):
    """--early-final N: as ft8nn_cli --early N - an early pass on the first N samples (ft8nn_decode_early: masked NN
    median, EARLY_PASSES subtraction passes, no co-channel or repeat stage, no CQ AP, no OSD on BP sums, no wide-window
    erasure OSD), then the whole slot with its decodes as prior (ft8nn_decode_final). Returns the union, early first."""
    from ft8nn import decoder

    sr, x = wavfile.read(path)
    x = x.astype(np.float64)
    if x.ndim > 1:
        x = x.mean(1)
    x = np.pad(x, (0, max(0, 180000 - len(x))))[:180000]
    kw = dict(n_hyp=_NHYP, erase_gate=_EG, context=context, cand_kw=_CKW, reuse_df=_RDF, dt_recenter=_DTR)
    xe = x[:_EF]
    early = decoder.decode_slot(xe / (np.std(xe) + 1e-9), llr_fn=_LLR_E, erase_osd_dmax=0, osd_snaps=0, ap=(),
                                passes=decoder.EARLY_PASSES, repeat=None, cochannel=None, **kw)
    ds = decoder.decode_slot(x / (np.std(x) + 1e-9), llr_fn=_LLR, erase_osd_dmax=_EOD, osd_snaps=_OSN, prior=early,
                             passes=_NP, repeat=_RP, cochannel=_CC, **kw)
    return Path(path).stem, [(d.bits77, d.f0, d.dt, d.method, d.info.get("metric"), d.snr) for d in ds]


def norm(t):
    t = re.sub(r"(\s+(\?|a\d))+\s*$", "", t.strip()).strip()  # jt9 may append both: "... ? a1"
    return re.sub(r"<[^>]*>", "<...>", t)


CALL = re.compile(r"^(?=.*\d)(?=.*[A-Z])[A-Z0-9/]{3,11}$")
NOT_CALL = re.compile(r"^(R?[+-]\d\d|RR73|RRR|73|[A-R]{2}\d\d|CQ|DE|QRZ|R|DX|\d+)$")


def calls_in(text):
    return {w for w in text.split() if CALL.match(w) and not NOT_CALL.match(w)}


def load_jt9(path):
    out = defaultdict(list)
    for line in Path(path).read_text().splitlines():
        m = re.match(r"^(\S+)\s+\d{6}\s+(-?\d+)\s+(-?[\d.]+)\s+(\d+)\s+~\s+(.*)$", line)
        if m:
            out[m[1]].append((int(m[2]), float(m[3]), int(m[4]), norm(m[5])))
    return out


def load_ours(path):
    out = defaultdict(list)
    for line in Path(path).read_text().splitlines():
        parts = line.split("\t")  # slot f0 dt method text [hexbits] [snr]
        slot, f0, dt, meth, text = parts[:5]
        snr = float(parts[6]) if len(parts) > 6 else float("nan")
        out[slot].append((float(f0), float(dt), meth, norm(text), snr))
    return out


def compare(station, jt9, ours):
    slots = sorted(set(jt9) | set(ours))
    jt9_calls = set().union(*[calls_in(d[3]) for v in jt9.values() for d in v]) if jt9 else set()
    tot = Counter()
    only_ours, only_jt9 = [], []
    for s in slots:
        a = {d[3] for d in jt9.get(s, [])}
        b = {d[3] for d in ours.get(s, [])}
        osnr = {d[3]: d[4] for d in ours.get(s, [])}
        tot["jt9"] += len(a)
        tot["ours"] += len(b)
        tot["both"] += len(a & b)
        for t in b - a:
            only_ours.append((s, t, osnr[t]))
        snr = {d[3]: d[0] for d in jt9.get(s, [])}
        for t in a - b:
            only_jt9.append((s, t, snr[t]))
    corro = [t for _, t, _ in only_ours if calls_in(t) & jt9_calls]
    nocall = [t for _, t, _ in only_ours if not calls_in(t)]
    return tot, only_ours, only_jt9, corro, nocall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--model", default=None)
    ap.add_argument("--tag", default="nn_v1")
    ap.add_argument("--also-classical", action="store_true")
    ap.add_argument("--n-hyp", type=int, default=2)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--stations", default=None, help="comma-separated prefixes, e.g. 00,01")
    ap.add_argument("--per-symbol", action="store_true", help="per-symbol Costas sync ratio")
    ap.add_argument("--fmax", type=float, default=3100.0)
    ap.add_argument("--base", default="median", choices=["max40", "median"], help="candidate score baseline")
    ap.add_argument("--erase-gate", type=float, default=15.0, help="burst-erasure retry below this expected error count (0 = off)")
    ap.add_argument("--erase-osd-dmax", type=float, default=3.0, help="wide-window OSD erasure stage distance limit (0 = off)")
    ap.add_argument("--osd-snaps", type=int, default=2, help="OSD on BP posterior sums after iterations 1..N (0 = off)")
    ap.add_argument("--passes", type=int, default=4, help="decode-subtract passes per slot")
    ap.add_argument("--reuse-df", type=float, default=62.5, help="skip candidates tried before with no subtraction within this many Hz (0 = off)")
    ap.add_argument("--dt-recenter", type=float, default=0.8, help="shift the DT search window by the median DT of the decodes so far (own + context) when it is at least this far from 0 (0 = off)")
    ap.add_argument("--no-ap", action="store_true", help="no CQ a-priori decoding")
    ap.add_argument("--no-context", action="store_true", help="no context a-priori (slots decoded independently)")
    ap.add_argument("--repeat", default="0.4,0.15,2", help="repeat stage RHO,MARGIN,CLIP; 'off' to disable")
    ap.add_argument("--repeat-competitors", default="sendable", choices=["sendable", "shipped"],
                    help="repeat-stage competitors: only messages a station could send, or the plan-12 set")
    ap.add_argument("--cochannel", default="2,1,8,24,50,3", help="co-channel retry NEAR_PEAKS,FAR_PEAKS,DF_SPAN,NEAR_DT,FAR_DT,MIN_SCORE (dt in 5 ms baseband samples); 'off' to disable")
    ap.add_argument("--truncate", type=int, default=None, help="zero the audio from this sample on (an early pass: "
                    "WSJT-X asks at 41 or 47 half-symbols, 141696 or 162432 samples)")
    ap.add_argument("--early-final", type=int, default=None, help="N: early pass on the first N samples, then the final "
                    "pass with its decodes as prior, context from the union (as c/build/ft8nn_cli --early N)")
    ap.add_argument("--early-nn", default="", help="with --truncate: 'erase' (LLRs of data symbols past the cut set to "
                    "0) and/or 'median' (NN feature normalisation over the symbols before the cut), comma-separated")
    ap.add_argument("--context-from", default=None, help="TAG: context a-priori from <out>/<station>/decodes_TAG.txt "
                    "(the final decodes 30/60 s earlier) instead of this run's own decodes")
    ap.add_argument("--prior", default=None, help="TAG: each slot's decodes in <out>/<station>/decodes_TAG.txt (an early "
                    "pass) are subtracted first and kept as found (decoder.decode_slot(prior=)); the output is the union")
    ap.add_argument("--out", default="data/real", help="output root: <out>/<station>/decodes_<tag>.txt")
    ap.add_argument("--compare-only", action="store_true")
    ap.add_argument("--show", type=int, default=8, help="examples to print per category")
    a = ap.parse_args()

    from ft8nn import message

    root = Path(a.root)
    stations = sorted(p for p in root.iterdir() if (p / "slots").is_dir())
    if a.stations:
        pre = tuple(a.stations.split(","))
        stations = [s for s in stations if s.name.startswith(pre)]
    outroot = Path(a.out)

    if not a.compare_only:
        with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(a.model, a.also_classical, a.n_hyp, a.early_nn, dict(per_symbol=a.per_symbol, fmax=a.fmax, base=a.base), a.erase_gate, a.erase_osd_dmax, a.osd_snaps, a.passes, not a.no_context, a.reuse_df, a.dt_recenter, None if a.repeat == "off" else tuple(map(float, a.repeat.split(","))), _cc_arg(a.cochannel), a.repeat_competitors == "sendable", a.truncate, (a.out, a.context_from) if a.context_from else None, (a.out, a.prior) if a.prior else None, not a.no_ap, a.early_final)) as ex:
            jobs = []  # (station index, chain): one chain per station and period, or one per file without context
            for si, st in enumerate(stations):
                files = sorted(str(f) for f in (st / "slots").glob("*.wav"))
                if a.no_context:
                    jobs += [(si, [f]) for f in files]
                else:
                    per = defaultdict(list)
                    for f in files:
                        t = _slot_time(Path(f).stem)
                        per[t.second // 15 % 2 if t else f].append(f)
                    jobs += [(si, c) for c in per.values()]
            results = defaultdict(dict)
            for (si, _), out in zip(jobs, ex.map(_decode_chain, [c for _, c in jobs])):
                results[si].update(out)
            for si, st in enumerate(stations):
                files = sorted((st / "slots").glob("*.wav"))
                res = results[si]
                ht = message.HashTable()  # learn hashes in time order, like a running WSJT-X session
                lines = []
                for slot in sorted(res):
                    for bits, f0, dt, meth, k, snr in res[slot]:
                        hexbits = f"{int(''.join(map(str, bits)), 2):020x}"
                        lines.append(f"{slot}\t{f0:.1f}\t{dt:.2f}\t{meth}{k}\t{message.unpack77(bits, ht)}\t{hexbits}\t{snr:.1f}")
                od = outroot / st.name
                od.mkdir(parents=True, exist_ok=True)
                (od / f"decodes_{a.tag}.txt").write_text("\n".join(lines) + "\n")
                print(f"{st.name}: {len(files)} slots, {len(lines)} decodes", flush=True)

    print(f"\n{'station':40s} {'jt9':>6s} {'ours':>6s} {'both':>6s} {'ours-only':>10s} {'corrob.':>8s} {'jt9-only':>9s}")
    grand = Counter()
    examples_o, examples_j, all_jt9_only, all_ours_only = [], [], [], []
    for st in stations:
        jf, of = st / "decodes.txt", outroot / st.name / f"decodes_{a.tag}.txt"
        if not (jf.exists() and of.exists()):
            print(f"{st.name:40s} {'(no jt9 decodes yet)' if of.exists() else '(not decoded)'}")
            continue
        tot, oo, oj, corro, nocall = compare(st.name, load_jt9(jf), load_ours(of))
        grand.update(tot)
        grand["only_ours"] += len(oo)
        grand["corro"] += len(corro)
        grand["only_jt9"] += len(oj)
        all_jt9_only += [x[2] for x in oj]
        examples_o += [(st.name[:6], t, t in corro) for _, t, _ in oo[: a.show]]
        all_ours_only += [x[2] for x in oo if x[2] == x[2]]
        examples_j += [(st.name[:6], t, snr) for _, t, snr in oj[: a.show]]
        print(f"{st.name:40s} {tot['jt9']:6d} {tot['ours']:6d} {tot['both']:6d} {len(oo):10d} {len(corro):8d} {len(oj):9d}")
    if grand["jt9"]:
        print(f"{'TOTAL':40s} {grand['jt9']:6d} {grand['ours']:6d} {grand['both']:6d} {grand['only_ours']:10d} {grand['corro']:8d} {grand['only_jt9']:9d}")
        print(f"\nours / jt9 = {grand['ours'] / grand['jt9']:.3f};  ours-only corroborated: {grand['corro'] / max(grand['only_ours'], 1):.0%}")
        if all_jt9_only:
            h = Counter(min(max(s // 3 * 3, -27), 0) for s in all_jt9_only)
            print("jt9-only by jt9 SNR: " + "  ".join(f"{k:+d}:{v}" for k, v in sorted(h.items())))
        if all_ours_only:
            h = Counter(min(max(int(s // 3 * 3), -27), 0) for s in all_ours_only)
            print("ours-only by our SNR: " + "  ".join(f"{k:+d}:{v}" for k, v in sorted(h.items())))
        print("\nexamples ours-only (✓ = callsign also heard by jt9):")
        for st, t, c in examples_o[: 3 * a.show]:
            print(f"  {st} {'✓' if c else ' '} {t}")
        print("examples jt9-only (jt9 SNR):")
        for st, t, snr in examples_j[: 3 * a.show]:
            print(f"  {st} {snr:+3d} {t}")


if __name__ == "__main__":
    main()
