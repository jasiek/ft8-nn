"""Evaluate the repeat stage (decoder._repeat_stage, plan 12) from saved decodes, without re-running the decoder.

  uv run python tools/rp_eval.py ROOT --decodes OUT [--base nn_v3] [--tag nn_v3_rp] [--rho 0.4 --margin 0.15]
                                 [--shuffle] [--dump scores.tsv] [--workers 8]

The stage runs last in decode_slot, on the slot's residual, so it can be emulated exactly: per receiver and slot,
in time order, subtract the slot's base decodes (OUT/<station>/decodes_<base>.txt) from the audio and run the stage
with the decodes of 30 and 60 s earlier as context - the base decodes plus the stage's own earlier acceptances,
as the C/Python chain would carry them. Writes OUT/<station>/decodes_<tag>.txt = base + accepted lines (score it
with collect/verify.py --tag <tag>).

--shuffle: every receiver gets its context from a receiver on another band in the same run (same slot times), so
every acceptance is false; this is the false-decode bound. No chaining in that mode. --shuffle-offset n picks the
n-th other-band receiver, so several runs give independent nulls.
--dump: thresholds off (accept nothing), write one row per tested sender: station, slot, best rho, runner-up rho,
whether the best is one of the earlier messages, its text. For choosing --rho / --margin offline.
"""
import argparse, os, sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from scipy.io import wavfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_FN = None


def _init(model):
    global _FN
    import torch
    torch.set_num_threads(1)
    from ft8nn import nn_metric
    _FN = nn_metric.make_llr_fn(model)


def load(path):
    """{slot: [(bits77, f0, dt, method, text, line)]} from an ft8nn decodes file."""
    out = defaultdict(list)
    for line in Path(path).read_text().splitlines():
        p = line.split("\t")
        b = np.array(list(bin(int(p[5], 16))[2:].zfill(77)), dtype=np.uint8)
        out[p[0]].append((b, float(p[1]), float(p[2]), p[3], p[4], line))
    return out


def audio(path):
    _, x = wavfile.read(path)
    x = x.astype(np.float64)
    if x.ndim > 1:
        x = x.mean(1)
    if len(x) < 180000:
        x = np.pad(x, (0, 180000 - len(x)))
    return x[:180000] / (np.std(x) + 1e-9)


def run_station(job):
    from ft8nn import decoder as dec, ft8
    from ft8nn.subtract import subtract
    slots_dir, own, ctx_src, rho, margin, clip, chain, dump = job
    T = lambda s: datetime.strptime(s, "%y%m%d_%H%M%S")
    acc = defaultdict(list)  # slot -> accepted [(bits, f0, dt, text)]
    lines, rows = [], []
    for slot in sorted(own.keys() | {p.stem for p in Path(slots_dir).glob("*.wav")}):
        t = T(slot)
        ctx, text_of = [], {}
        for back in (30, 60):
            s0 = (t - timedelta(seconds=back)).strftime("%y%m%d_%H%M%S")
            for b, f0, dt, _, text, _ in ctx_src.get(s0, []):
                ctx.append((f0, dt, b))
                text_of[b.tobytes()] = text
            for b, f0, dt, text in (acc.get(s0, []) if chain else []):
                ctx.append((f0, dt, b))
                text_of[b.tobytes()] = text
        if not ctx:
            continue
        x = audio(Path(slots_dir) / f"{slot}.wav")
        found = {}
        for b, f0, dt, meth, _, _ in own.get(slot, []):
            subtract(x, ft8.encode77(b)[1], f0, dt)
            d = dec.Decode(b, f0, dt, 0.0, meth)
            found[d.key] = d
        log = [] if dump else None
        new = dec._repeat_stage(x, _FN, found, ctx, 2, rho, margin, clip, log=log)
        for d in new:
            text = text_of[d.bits77.tobytes()]
            acc[slot].append((d.bits77, d.f0, d.dt, text))
            hexb = format(int("".join(map(str, d.bits77)), 2), "x").zfill(20)
            lines.append(f"{slot}\t{d.f0:.1f}\t{d.dt:.2f}\trp\t{text}\t{hexb}\t{d.snr:.1f}\t{d.info['rho']:.3f}\t{d.info['rho2']:.3f}")
        for r1, r2, is_prior, b in (log or []):
            rows.append(f"{slot}\t{r1:.4f}\t{r2:.4f}\t{int(is_prior)}\t{text_of.get(b.tobytes(), '?')}")
    return slots_dir, lines, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="recordings root (<station>/slots/*.wav)")
    ap.add_argument("--decodes", required=True, help="root of <station>/decodes_<base>.txt; output goes here too")
    ap.add_argument("--base", default="nn_v3")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--model", default="models/v3/model.pt")
    ap.add_argument("--rho", type=float, default=0.4)
    ap.add_argument("--margin", type=float, default=0.15)
    ap.add_argument("--clip", type=float, default=2.0, help="LLR clip before rho")
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--shuffle-offset", type=int, default=0, help="use the n-th other-band receiver (0 = next)")
    ap.add_argument("--dump", default=None)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    root, out = Path(a.root), Path(a.decodes)
    stations = sorted(p.name for p in root.iterdir() if (p / "slots").is_dir())
    own = {st: load(out / st / f"decodes_{a.base}.txt") for st in stations}
    band = lambda st: st.split("_")[1]
    jobs = []
    for i, st in enumerate(stations):
        src = st
        if a.shuffle:
            others = [s for s in stations[i + 1:] + stations[:i] if band(s) != band(st)]
            src = others[a.shuffle_offset % len(others)]
        rho, margin = (9.0, 0.0) if a.dump else (a.rho, a.margin)  # dump: accept nothing, log every tested sender
        jobs.append((str(root / st / "slots"), own[st], own[src], rho, margin, a.clip, not a.shuffle, bool(a.dump)))
    total, rows = 0, []
    with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(a.model,)) as ex:
        for slots_dir, lines, r in ex.map(run_station, jobs):
            st = Path(slots_dir).parent.name
            total += len(lines)
            rows += [f"{st}\t{x}" for x in r]
            print(f"  {st:48s} {len(lines):5d} accepted", flush=True)
            if a.tag and not a.dump:
                base = (out / st / f"decodes_{a.base}.txt").read_text()
                (out / st / f"decodes_{a.tag}.txt").write_text(base + "".join(l + "\n" for l in lines))
    print(f"total accepted {total}{' (shuffled context: all false)' if a.shuffle else ''}")
    if a.dump:
        Path(a.dump).write_text("station\tslot\trho1\trho2\tprior\ttext\n" + "".join(x + "\n" for x in rows))
        print(f"wrote {len(rows)} rows to {a.dump}")


if __name__ == "__main__":
    main()
