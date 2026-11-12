"""Replay the slots of tools/miss_profile.py's misses: why the repeat stage, the candidate search and the NN miss them.

  uv run python tools/miss_replay.py MISSES.tsv --out-tsv REPLAY.tsv [--slots N] [--seed 1] [--workers 4]

Each reference text is packed with WSJT-X's ft8code (exact 77 bits, so the codeword is known). Per slot, in the
order the decoder works: subtract our saved non-`rp` decodes (data/val/run<r>/<st>/decodes_<ours>.txt), then
  rp_*     emulate the repeat stage (decoder._repeat_stage) for the missed sender alone, with our decodes of 30 and
           60 s earlier as context: rp_state = noctx (no standard message of that sender in the context), have
           (the stage sees the sender as decoded already), accept, successor (best match is one of the competitor
           messages, not an earlier one), lowrho (best < 0.40), margin (runner-up within 0.15); rp_r1/rp_r2 best
           and runner-up rho; rp_ref = rho of the reference's own message at the same hypotheses (also when it is
           not among the stage's messages); rp_in_alts = the reference's message is one the stage scores
then subtract our `rp` decodes too (the residual a further pass would see) and
  cand     NOCAND (no search candidate within 5 Hz / 0.2 s of the reference's frequency and DT) / FAIL (candidate,
           BP + OSD-2 find no codeword) / OK (decodes here) / OTHER (decodes to a different message), with E =
           expected bit errors of the candidate's first hypothesis (sum of 1 / (1 + e^|LLR|))
  forced   the same decode straight at the reference's (frequency, DT), no candidate search (fine sync +-
           its usual range): OK / FAIL / OTHER, its E, and forced_rho = rho of the reference's message there
rho = sum(l * s) / sum(|l|) over the 174 codeword bits, LLRs clipped to +-2 as in the repeat stage.

--lookback: the repeat stage run backwards - context from our decodes 30 and 60 s LATER, tested on the slot's
residual after all its decodes (repeat stage included) are subtracted, i.e. what re-testing a slot once the next
slots are decoded would find. --rp-only skips the candidate/forced tests and the slots with nothing to test.
"""
import argparse, csv, os, random, re, subprocess, sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
V = os.environ.get("V") or "datasets/ft8-validation-set"
_FN = None


def _init(model):
    global _FN
    import torch
    torch.set_num_threads(1)
    from ft8nn import nn_metric
    _FN = nn_metric.make_llr_fn(model)


def _ft8code(text):
    out = subprocess.run(["ft8code", text], capture_output=True, text=True).stdout.splitlines()
    try:
        i = next(i for i, l in enumerate(out) if l.startswith("Source-encoded"))
    except StopIteration:
        return None, None
    row = next(l for l in out if re.match(r"^\s*1\.", l))
    return out[i + 1].strip(), row[42:78].strip()


def ft8code(text):
    """(77 bits, ft8code's decoded text) for the transmission behind a reference decoder's text. Decoders print a
    resolved hashed call without brackets (JTDX: `YO4SX 4L/US8ICM +01`, sent as `YO4SX <4L/US8ICM> +01`), so when
    the text does not pack back to itself, try bracketing each call-like word and keep the first variant that does."""
    plain = lambda t: re.sub(r"[<>]", "", t).split()
    w = text.split()
    for i in [None] + [i for i, x in enumerate(w) if re.search(r"\d", x) and re.search(r"[A-Z]", x) and len(x) >= 3]:
        t = text if i is None else " ".join(f"<{x}>" if j == i else x for j, x in enumerate(w))
        bits, dec_text = _ft8code(t)
        if bits and plain(dec_text) == plain(text):
            return bits, dec_text
    return _ft8code(text)


def load_ours(path):
    out = defaultdict(list)
    for line in Path(path).read_text().splitlines():
        p = line.split("\t")
        b = np.array(list(bin(int(p[5], 16))[2:].zfill(77)), dtype=np.uint8)
        out[p[0]].append((b, float(p[1]), float(p[2]), p[3]))
    return out


def audio(path):
    from scipy.io import wavfile
    _, x = wavfile.read(path)
    x = x.astype(np.float64)
    if x.ndim > 1:
        x = x.mean(1)
    x = np.pad(x, (0, max(0, 180000 - len(x))))[:180000]
    return x / (np.std(x) + 1e-9)


def _group(ctx, rb):
    """The context messages the repeat stage would test for the sender of payload rb (standard messages only)."""
    std = lambda b: b[74:77].tolist() in ([0, 0, 1], [0, 1, 0])
    if rb is None or not std(rb):
        return []
    return [(fc, dc, b) for fc, dc, b in ctx if std(b) and b[29:58].tobytes() == rb[29:58].tobytes()]


def job(args):
    from ft8nn import decoder as dec, frontend as fe, ft8, ldpc
    from ft8nn.subtract import subtract
    wav, ours, ctx, misses, lookback, rp_only = args
    x = audio(wav)
    found = {}
    for b, f0, dt, meth in ours:
        if lookback or not meth.startswith("rp"):  # look-back: the slot is fully decoded (repeat stage included) first
            subtract(x, ft8.encode77(b)[1], f0, dt)
            d = dec.Decode(b, f0, dt, 0.0, meth)
            found[d.key] = d
    bb = fe.Baseband(x)
    have_s = {dec._sender_key(d) for d in found.values()}
    clipl = lambda l: np.clip(l, -2.0, 2.0)
    rho = lambda S, l: S @ clipl(l) / (np.abs(clipl(l)).sum() + 1e-9)
    res = []
    for m in misses:
        r = dict(key=m["key"])
        rb = m["bits"]
        cw_ref = dec._codeword(rb) if rb is not None else None
        grp = _group(ctx, rb)
        if not grp:
            r.update(rp_state="noctx")
        elif rb[29:58].tobytes() in have_s:
            r.update(rp_state="have")
        else:
            prior = {g[2].tobytes(): g[2] for g in grp}
            alts = dict(prior)
            grids = sorted({dec.message._int(b[59:74]) for b in prior.values() if dec.message._int(b[59:74]) < dec.message.MAXGRID4})
            for b in prior.values():
                for c in dec._successors(b, grids):
                    alts.setdefault(c.tobytes(), c)
            keys = list(alts)
            S = np.stack([dec._codeword(alts[k]) for k in keys] + [cw_ref])
            # competitors no station sends: a CQ with the R flag or with a report/RRR/RR73/73 in place of a grid
            M = dec.message
            sane = np.array([not (2 <= M._int(alts[k][0:28]) < M.NTOKENS and (alts[k][58] or M._int(alts[k][59:74]) > M.MAXGRID4))
                             for k in keys])
            best, ref_best = None, -9.0
            for fc, dc in dict.fromkeys((g[0], g[1]) for g in grp):
                hs = fe.fine_sync2(bb, dec._Loc(fc, dc), n_hyp=2)
                if not hs:
                    continue
                for h, llrs in zip(hs, _FN(hs)):
                    rr = rho(S, llrs[0])
                    ref_best = max(ref_best, float(rr[-1]))
                    rr = rr[:-1]
                    o = np.argsort(-rr, kind="stable")
                    if best is None or rr[o[0]] > best[0]:
                        s2 = [i for i in o[1:] if sane[i]]
                        best = (float(rr[o[0]]), float(rr[o[1]]), keys[o[0]], keys[o[1]],
                                float(rr[s2[0]]) if s2 else -1.0)
            if best is None:
                r.update(rp_state="nosync")
            else:
                r1, r2, k1, k2, r2s = best
                r.update(rp_r2_sane=f"{r2s:.3f}")
                state = ("successor" if k1 not in prior else "lowrho" if r1 < 0.40 else "margin" if r1 - r2 < 0.15
                         else "accept")
                txt = lambda k: dec.message.unpack77(alts[k])
                plaus = lambda k: int(bool(dec.message.plausible(alts[k])))
                r.update(rp_best=txt(k1), rp_second=txt(k2), rp_second_plausible=plaus(k2),
                         rp_second_prior=int(k2 in prior))
                r.update(rp_state=state, rp_r1=f"{r1:.3f}", rp_r2=f"{r2:.3f}", rp_ref=f"{ref_best:.3f}",
                         rp_best_is_ref=int(k1 == rb.tobytes()), rp_in_alts=int(rb.tobytes() in alts),
                         rp_n_alts=len(keys), rp_min_df=f"{min(abs(g[0] - m['freq']) for g in grp):.1f}")
        res.append(r)
    if rp_only:
        return res
    for b, f0, dt, meth in ours:
        if meth.startswith("rp") and not lookback:
            subtract(x, ft8.encode77(b)[1], f0, dt)
    cands = fe.find_candidates(x)
    bb = fe.Baseband(x)

    def try_decode(hs):
        E0, outcome = None, "FAIL"
        for h, (l,) in zip(hs, _FN(hs)):
            E = float((1 / (1 + np.exp(np.abs(l)))).sum())
            E0 = E if E0 is None else E0
            a, _ = ldpc.decode(l[None])[0]
            if a is not None:
                return ("OK" if rb is not None and (a[:77] == rb).all() else "OTHER"), E0, l
        return outcome, E0, None

    for m, r in zip(misses, res):
        rb = m["bits"]
        near = [c for c in cands if abs(c.f0 - m["freq"]) < 5 and abs(c.dt - m["dt"]) < 0.2]
        if not near:
            r.update(cand="NOCAND")
        else:
            c = min(near, key=lambda c: abs(c.f0 - m["freq"]) + 10 * abs(c.dt - m["dt"]))
            o, E, _ = try_decode(fe.fine_sync2(bb, c, 2))
            r.update(cand=o, cand_E=f"{E:.1f}", cand_score=f"{c.score:.2f}")
        hs = fe.fine_sync2(bb, dec._Loc(float(m["freq"]), float(m["dt"])), n_hyp=2)
        if hs:
            o, E, _ = try_decode(hs)
            fr = max(float(rho(dec._codeword(rb)[None], l[0])[0]) for l in _FN(hs)) if rb is not None else float("nan")
            r.update(forced=o, forced_E=f"{E:.1f}", forced_rho=f"{fr:.3f}")
        else:
            r.update(forced="NOSYNC")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tsv")
    ap.add_argument("--out-tsv", required=True)
    ap.add_argument("--ours", default="nn_v3_dtcrp")
    ap.add_argument("--val", default="data/val")
    ap.add_argument("--model", default="models/v3/model.pt")
    ap.add_argument("--slots", type=int, default=0, help="random sample of this many slots (0 = all)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--lookback", action="store_true",
                    help="repeat-stage context from our decodes 30/60 s LATER, tested on the fully decoded slot's residual")
    ap.add_argument("--rp-only", action="store_true", help="only the repeat-stage emulation (no candidate/forced test)")
    ap.add_argument("--only", help="only misses with this miss_profile.py flag set (e.g. repeat)")
    a = ap.parse_args()

    rows = [r for r in csv.DictReader(open(a.tsv), delimiter="\t") if not a.only or r[a.only] == "1"]
    by_slot = defaultdict(list)
    for r in rows:
        by_slot[(r["run"], r["st"], r["slot"])].append(r)
    keys = sorted(by_slot)
    if a.slots:
        random.Random(a.seed).shuffle(keys)
        keys = sorted(keys[:a.slots])
    print(f"{len(keys)} slots, {sum(len(by_slot[k]) for k in keys)} misses; packing texts with ft8code", flush=True)
    ours = {}
    jobs = []
    for run, st, slot in keys:
        if (run, st) not in ours:
            ours[run, st] = load_ours(Path(a.val) / f"run{run}" / st / f"decodes_{a.ours}.txt")
        o = ours[run, st]
        t = datetime.strptime(slot, "%y%m%d_%H%M%S")
        sgn = 1 if a.lookback else -1
        ctx = [(f0, dt, b) for back in (30, 60) for b, f0, dt, _ in o.get((t + timedelta(seconds=sgn * back)).strftime("%y%m%d_%H%M%S"), [])]
        ms = []
        for r in by_slot[run, st, slot]:
            bits, dec_text = ft8code(r["text"])
            r["packed"] = dec_text or ""
            ms.append(dict(key=(run, st, slot, r["sender"]), freq=int(r["freq"]), dt=float(r["dt"]),
                           bits=np.array([int(c) for c in bits], np.uint8) if bits else None))
        if a.rp_only and not any(_group(ctx, m["bits"]) for m in ms):
            continue  # nothing for the repeat stage to test in this slot
        jobs.append((str(Path(V) / f"run{run}" / "recordings" / st / "slots" / f"{slot}.wav"), o.get(slot, []), ctx, ms,
                     a.lookback, a.rp_only))
    print(f"{len(jobs)} slots to replay", flush=True)
    out = {}
    with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(a.model,)) as ex:
        for i, res in enumerate(ex.map(job, jobs, chunksize=1)):
            for r in res:
                out[r["key"]] = r
            if i % 50 == 0:
                print(f"  {i}/{len(jobs)} slots {datetime.now():%H:%M:%S}", flush=True)
    cols = ["rp_state", "rp_r1", "rp_r2", "rp_ref", "rp_best_is_ref", "rp_in_alts", "rp_n_alts", "rp_min_df", "rp_best",
            "rp_second", "rp_second_plausible", "rp_second_prior", "rp_r2_sane", "cand", "cand_E", "cand_score", "forced", "forced_E",
            "forced_rho"]
    base = [c for c in rows[0].keys() if c != "packed"] + ["packed"]
    with open(a.out_tsv, "w") as fh:
        fh.write("\t".join(base + cols) + "\n")
        for k in keys:
            for r in by_slot[k]:
                x = out.get((r["run"], r["st"], r["slot"], r["sender"]), {})
                fh.write("\t".join([str(r.get(c, "")) for c in base] + [str(x.get(c, "")) for c in cols]) + "\n")
    print(f"wrote {a.out_tsv}")


if __name__ == "__main__":
    main()
