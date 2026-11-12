"""Repeat stage with the shipped competitor set vs sendable competitors only (decoder._successors(sendable_only=)),
emulated from saved decodes as tools/rp_eval.py does, both sets scored from the same rho matrix.

  uv run python tools/rp_competitors.py REC --decodes data/val/run1 --base nn_v3 --shuffle 0 1 2 3 4 5 --dump F
  uv run python tools/rp_competitors.py REC --decodes data/val/runN --base nn_v3_dtcrp --real --tag-old T1 --tag-new T2

Per receiver and slot, in time order: subtract the slot's base decodes that are not `rp` (what the passes produce)
and run the stage on the residual.
--shuffle n...: context from the n-th other-band receiver of the same run (same slot times), so every acceptance is
  false: the false-accept null of plan 12, all offsets sharing one residual per slot. --dump writes one row per tested
  sender and offset: best rho, runner-up rho and whether the best is an earlier message, for both competitor sets.
--real: context from this receiver's own decodes 30/60 s earlier plus the stage's own earlier acceptances, chained
  separately for each set; writes <station>/decodes_<tag-old|tag-new>.txt = base non-rp decodes + that set's
  acceptances (score with collect/verify.py, stale-check with tools/rp_stale.py).
"""
import argparse, os, sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_FN = None


def _init(model):
    global _FN
    import torch
    torch.set_num_threads(1)
    from ft8nn import nn_metric
    _FN = nn_metric.make_llr_fn(model)


def load(path, skip_rp=True):
    out = defaultdict(list)
    for line in Path(path).read_text().splitlines():
        p = line.split("\t")
        if skip_rp and p[3].startswith("rp"):
            continue
        b = np.array(list(bin(int(p[5], 16))[2:].zfill(77)), dtype=np.uint8)
        out[p[0]].append((b, float(p[1]), float(p[2]), p[3], p[4], line))
    return out


def audio(path):
    from scipy.io import wavfile
    _, x = wavfile.read(path)
    x = x.astype(np.float64)
    if x.ndim > 1:
        x = x.mean(1)
    x = np.pad(x, (0, max(0, 180000 - len(x))))[:180000]
    return x / (np.std(x) + 1e-9)


def score_group(bb, msgs, clip):
    """Both competitor sets for one sender: {set: (r1, r2, best is earlier, best payload, hypothesis)}."""
    from ft8nn import decoder as dec, frontend as fe, message
    prior = {m[2].tobytes(): m[2] for m in msgs}
    alts = dict(prior)
    new_keys = set(prior)
    grids = sorted({message._int(b[59:74]) for b in prior.values() if message._int(b[59:74]) < message.MAXGRID4})
    for b in prior.values():
        for c in dec._successors(b, grids, sendable_only=False):
            alts.setdefault(c.tobytes(), c)
        for c in dec._successors(b, grids, sendable_only=True):
            new_keys.add(c.tobytes())
    keys = list(alts)
    S = np.stack([dec._codeword(alts[k]) for k in keys])
    sub = np.array([i for i, k in enumerate(keys) if k in new_keys])
    best = {"old": None, "new": None}
    for fc, dc in dict.fromkeys((m[0], m[1]) for m in msgs):
        hs = fe.fine_sync2(bb, dec._Loc(fc, dc), n_hyp=2)
        if not hs:
            continue
        for h, llrs in zip(hs, _FN(hs)):
            l = np.clip(llrs[0], -clip, clip)
            rho = S @ l / (np.abs(l).sum() + 1e-9)
            for name, idx in (("old", np.arange(len(keys))), ("new", sub)):
                rr = rho[idx]
                o = np.argsort(-rr, kind="stable")
                r1 = float(rr[o[0]])
                r2 = float(rr[o[1]]) if len(o) > 1 else 0.0
                if best[name] is None or r1 > best[name][0]:
                    k1 = keys[idx[o[0]]]
                    best[name] = (r1, r2, k1 in prior, alts[k1], h)
    return best


def groups(context, found):
    """The stage's grouping (decoder._repeat_stage): standard messages of senders with no decode in the slot."""
    from ft8nn import decoder as dec
    have_s = {dec._sender_key(d) for d in found.values()}
    have_b = set(found)
    g = {}
    for fc, dc, b in context:
        if b[74:77].tolist() not in ([0, 0, 1], [0, 1, 0]) or "".join(map(str, b)) in have_b:
            continue
        sk = b[29:58].tobytes()
        if sk not in have_s:
            g.setdefault(sk, []).append((fc, dc, b))
    return g


def run_station(job):
    from ft8nn import decoder as dec, ft8
    from ft8nn.subtract import subtract
    slots_dir, own, shuffle_src, real, rho_min, margin, clip, solo = job
    T = lambda s: datetime.strptime(s, "%y%m%d_%H%M%S")
    acc = {"old": defaultdict(list), "new": defaultdict(list)}
    lines = {"old": [], "new": []}
    dump = []
    for slot in sorted(own.keys() | {p.stem for p in Path(slots_dir).glob("*.wav")}):
        t = T(slot)
        back = [(t - timedelta(seconds=s)).strftime("%y%m%d_%H%M%S") for s in (30, 60)]
        ctxs = {}
        if real:
            for name in ("old", "new"):
                ctx, text_of = [], {}
                for s0 in back:
                    for b, f0, dt, _, text, _ in own.get(s0, []):
                        ctx.append((f0, dt, b))
                        text_of[b.tobytes()] = text
                    for b, f0, dt, text in acc[name].get(s0, []):
                        ctx.append((f0, dt, b))
                        text_of[b.tobytes()] = text
                ctxs[name] = (ctx, text_of)
        for off, src in shuffle_src:
            ctx, text_of = [], {}
            for s0 in back:
                for b, f0, dt, _, text, _ in src.get(s0, []):
                    ctx.append((f0, dt, b))
                    text_of[b.tobytes()] = text
            ctxs[off] = (ctx, text_of)
        if not any(c for c, _ in ctxs.values()):
            continue
        x = audio(Path(slots_dir) / f"{slot}.wav")
        found = {}
        for b, f0, dt, meth, _, _ in own.get(slot, []):
            subtract(x, ft8.encode77(b)[1], f0, dt)
            d = dec.Decode(b, f0, dt, 0.0, meth)
            found[d.key] = d
        from ft8nn import frontend as fe
        bb = fe.Baseband(x)
        cache = {}

        def score(sk, msgs):
            k = (sk, tuple((m[0], m[1], m[2].tobytes()) for m in msgs))
            if k not in cache:
                cache[k] = score_group(bb, msgs, clip)
            return cache[k]

        for name, (ctx, text_of) in ctxs.items():
            for sk, msgs in groups(ctx, found).items():
                res = score(sk, msgs)
                if name in ("old", "new"):
                    r = res[name]
                    if r and name == "new" and r[1] == 0.0:  # no sendable competitor: runner-up = the noise floor
                        r = (r[0], solo) + r[2:]
                    if r and r[2] and r[0] >= rho_min and r[0] - r[1] >= margin:
                        r1, r2, _, b, h = r
                        text = text_of[b.tobytes()]
                        acc[name][slot].append((b, h.f0, h.dt, text))
                        hexb = format(int("".join(map(str, b)), 2), "x").zfill(20)
                        snr = dec.estimate_snr(h, b)
                        lines[name].append(f"{slot}\t{h.f0:.1f}\t{h.dt:.2f}\trp\t{text}\t{hexb}\t{snr:.1f}\t{r1:.3f}\t{r2:.3f}")
                else:
                    o, n = res["old"], res["new"]
                    if o is None:
                        continue
                    dump.append(f"{name}\t{slot}\t{o[0]:.4f}\t{o[1]:.4f}\t{int(o[2])}\t{n[0]:.4f}\t{n[1]:.4f}\t{int(n[2])}"
                                f"\t{text_of.get(n[3].tobytes(), '?')}")
    return slots_dir, lines, dump


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--decodes", required=True)
    ap.add_argument("--base", default="nn_v3_dtcrp")
    ap.add_argument("--model", default="models/v3/model.pt")
    ap.add_argument("--shuffle", type=int, nargs="*", default=[])
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--tag-old")
    ap.add_argument("--tag-new")
    ap.add_argument("--dump")
    ap.add_argument("--rho", type=float, default=0.40)
    ap.add_argument("--margin", type=float, default=0.15)
    ap.add_argument("--clip", type=float, default=2.0)
    ap.add_argument("--solo", type=float, default=None, help="runner-up rho when the sendable set has no competitor (default: decoder.REPEAT_SOLO)")
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    if a.solo is None:
        from ft8nn import decoder
        a.solo = decoder.REPEAT_SOLO

    root, out = Path(a.root), Path(a.decodes)
    stations = sorted(p.name for p in root.iterdir() if (p / "slots").is_dir())
    own = {st: load(out / st / f"decodes_{a.base}.txt") for st in stations}
    band = lambda st: st.split("_")[1]
    jobs = []
    for i, st in enumerate(stations):
        others = [s for s in stations[i + 1:] + stations[:i] if band(s) != band(st)]
        src = [(f"shuffle{k}", own[others[k % len(others)]]) for k in a.shuffle]
        jobs.append((str(root / st / "slots"), own[st], src, a.real, a.rho, a.margin, a.clip, a.solo))
    dump, tot = [], {"old": 0, "new": 0}
    # --dump: one file per station next to it (<dump>.d/<station>.tsv), written as each station finishes, so an
    # interrupted run resumes where it stopped (shuffle mode only: real mode chains need the whole station anyway)
    part = Path(str(a.dump) + ".d") if a.dump else None
    if part:
        part.mkdir(exist_ok=True)
        done = {p.stem for p in part.glob("*.tsv")}
        for st in sorted(done):
            dump += [x for x in (part / f"{st}.tsv").read_text().splitlines() if x]
        if not a.real:
            jobs = [j for j in jobs if Path(j[0]).parent.name not in done]
    from concurrent.futures import as_completed
    with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(a.model,)) as ex:
        for fut in as_completed([ex.submit(run_station, j) for j in jobs]):
            slots_dir, lines, d = fut.result()
            st = Path(slots_dir).parent.name
            dump += [f"{st}\t{x}" for x in d]
            if part and d:
                (part / f"{st}.tsv").write_text("".join(f"{st}\t{x}\n" for x in d))
            msg = f"  {st:48s} {datetime.now():%H:%M:%S}"
            if a.real:
                base = "".join(l[5] + "\n" for s in sorted(own[st]) for l in own[st][s])
                for name, tag in (("old", a.tag_old), ("new", a.tag_new)):
                    tot[name] += len(lines[name])
                    if tag:
                        (out / st / f"decodes_{tag}.txt").write_text(base + "".join(l + "\n" for l in lines[name]))
                msg += f"  rp accepted: shipped set {len(lines['old'])}, sendable set {len(lines['new'])}"
            if a.shuffle:
                msg += f"  {len(d)} shuffled tests"
            print(msg, flush=True)
    if a.real:
        print(f"total rp accepted: shipped competitors {tot['old']}, sendable competitors {tot['new']}")
    if a.dump:
        Path(a.dump).write_text("station\tctx\tslot\tr1_old\tr2_old\tprior_old\tr1_new\tr2_new\tprior_new\ttext\n"
                                + "".join(x + "\n" for x in dump))
    if dump:
        rows = [x.split("\t") for x in dump]
        for name, c1, c2, cp in (("shipped", 3, 4, 5), ("sendable", 6, 7, 8)):
            for rho, mg in ((0.40, 0.15), (0.40, 0.10), (0.35, 0.15), (0.40, 0.20), (0.45, 0.15), (0.50, 0.15)):
                n = sum(r[cp] == "1" and float(r[c1]) >= rho and float(r[c1]) - float(r[c2]) >= mg for r in rows)
                print(f"  null, {name} competitors: {n} of {len(rows)} tests pass at rho {rho:.2f} / margin {mg:.2f}")


if __name__ == "__main__":
    main()
