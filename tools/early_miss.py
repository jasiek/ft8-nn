"""Why an early pass (audio cut at N samples) misses decodes the full-length decode makes.

  uv run python tools/early_miss.py RECORDINGS --full nn_v3_ccrpc --early e118_nn --truncate 141696 \
      [--out data/val/run2] [--model models/v3/model.pt] [--workers 6] [--tsv OUT.tsv]

Lost = payloads in <out>/<station>/decodes_<full>.txt but not in decodes_<early>.txt for the same slot. Per lost
decode:
  cut      data symbols whose end lies past the cut, from the full decode's DT (of 58; 3 bits each)
  forced_full / forced_cut   NN LLRs at the full decode's (f0, DT) (fine sync over its usual range), BP then OSD-2:
           OK (that payload), OTHER, FAIL, on the full-length audio with the full run's other decodes subtracted, and
           on the cut audio with the early run's decodes subtracted (the residual each decode ends up with); E =
           expected bit errors of the best hypothesis there (sum of 1 / (1 + e^|LLR|))
  cand     a search candidate within 5 Hz / 0.2 s of the full decode's (f0, DT) in that cut residual: yes / no
Classes: "full-only stage" (forced_full fails: the full decode needed context, the repeat stage, AP or OSD beyond
order 2, so this test says nothing), "cut" (OK at full length, FAIL when cut: the missing tail took too much),
"search" (OK when cut but no candidate), "later stage" (OK when cut, candidate there: subtraction order, gates).
"""
import argparse
import csv
import os
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_FN = None
_N = None


def _init(model, n):
    global _FN, _N
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import torch
    torch.set_num_threads(1)
    from ft8nn import nn_metric
    _FN = nn_metric.make_llr_fn(model)
    _N = n


def _load(path):
    out = defaultdict(dict)
    if path.exists():
        for line in path.read_text().splitlines():
            p = line.split("\t")
            out[p[0]][p[5]] = p  # slot f0 dt method text hexbits snr
    return out


def _force(x, f0, dt, bits):
    from ft8nn import frontend as fe, ldpc

    bb = fe.Baseband(x)
    res, E = "FAIL", float("nan")
    for h in fe.fine_sync2(bb, fe.Candidate(f0, dt, 0.0), n_hyp=2):
        llr = np.asarray(_FN([h])[0][0])
        e = float((1 / (1 + np.exp(np.abs(llr)))).sum())
        E = e if E != E else min(E, e)
        a91, _ = ldpc.decode(llr)[0]
        if a91 is not None:
            if (a91[:77] == bits).all():
                return "OK", round(e, 1)
            res = "OTHER"
    return res, round(E, 1)


def _bits(hexbits):
    return np.array([int(c) for c in f"{int(hexbits, 16):077b}"], np.uint8)


def _slot(job):
    from scipy.io import wavfile
    from ft8nn import frontend as fe, ft8
    from ft8nn.subtract import subtract

    wav, lost, kept_full, kept_early = job
    _, x = wavfile.read(wav)
    x = x.astype(np.float64)
    if x.ndim > 1:
        x = x.mean(1)
    x = np.pad(x, (0, max(0, 180000 - len(x))))[:180000]
    xf = x / (np.std(x) + 1e-9)
    xc = x.copy()
    xc[_N:] = 0
    xc /= np.std(xc[:_N]) + 1e-9
    for y, kept in ((xf, kept_full), (xc, kept_early)):
        for p in kept:
            subtract(y, ft8.encode77(_bits(p[5]))[1], float(p[1]), float(p[2]))
    xc[_N:] = 0  # subtraction leaks a little past the cut (envelope smoothing)
    cands = fe.find_candidates(xc)
    t_cut = _N / ft8.FS
    out = []
    for p in lost:
        f0, dt, bits = float(p[1]), float(p[2]), _bits(p[5])
        ends = ft8.NOMINAL_START + dt + (ft8.DATA_POS + 1) * ft8.NSPS / ft8.FS
        ff, ef = _force(xf, f0, dt, bits)
        fc, ec = _force(xc, f0, dt, bits)
        cand = any(abs(c.f0 - f0) < 5 and abs(c.dt - dt) < 0.2 for c in cands)
        out.append(dict(station=Path(wav).parent.parent.name, slot=p[0], f0=f0, dt=dt, method=p[3].rstrip("0123456789"),
                        snr=float(p[6]), text=p[4], cut=int((ends > t_cut).sum()), forced_full=ff, E_full=ef,
                        forced_cut=fc, E_cut=ec, cand="yes" if cand else "no"))
    return out


def _class(r):
    if r["forced_full"] != "OK":
        return "full-only stage"
    if r["forced_cut"] != "OK":
        return "cut"
    return "search" if r["cand"] == "no" else "later stage"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--full", required=True)
    ap.add_argument("--early", required=True)
    ap.add_argument("--truncate", type=int, required=True)
    ap.add_argument("--out", default="data/val/run2")
    ap.add_argument("--model", default="models/v3/model.pt")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--tsv", default=None)
    a = ap.parse_args()

    jobs, n_full, n_early, n_new = [], 0, 0, 0
    for st in sorted(p for p in Path(a.root).iterdir() if (p / "slots").is_dir()):
        full = _load(Path(a.out) / st.name / f"decodes_{a.full}.txt")
        early = _load(Path(a.out) / st.name / f"decodes_{a.early}.txt")
        for slot in sorted(f.stem for f in (st / "slots").glob("*.wav")):
            n_full += len(full[slot])
            n_early += len(early[slot])
            n_new += len(set(early[slot]) - set(full[slot]))
            lost = [full[slot][k] for k in full[slot] if k not in early[slot]]
            if lost:
                kept = [p for k, p in full[slot].items() if k in early[slot]]
                jobs.append((str(st / "slots" / f"{slot}.wav"), lost, kept, list(early[slot].values())))
    with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(a.model, a.truncate)) as ex:
        rows = [r for rs in ex.map(_slot, jobs) for r in rs]
    if a.tsv:
        with open(a.tsv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t")
            w.writeheader()
            w.writerows(rows)

    print(f"full {a.full}: {n_full} decodes; early {a.early} (cut at {a.truncate} samples, {a.truncate / 12000:.2f} s): "
          f"{n_early}; lost {len(rows)}, new {n_new}")
    print("\nby class:  " + "  ".join(f"{k} {v}" for k, v in Counter(map(_class, rows)).most_common()))
    print("by method in the full decode:  " + "  ".join(f"{k} {v}" for k, v in Counter(r["method"] for r in rows).most_common()))
    classes = ("full-only stage", "cut", "search", "later stage")
    head = "".join(f"{c:>17s}" for c in classes)
    print(f"\ncut data symbols   lost{head}")
    for lo, hi in ((0, 2), (3, 5), (6, 9), (10, 14), (15, 58)):
        sel = Counter(_class(r) for r in rows if lo <= r["cut"] <= hi)
        print(f"  {lo:2d}-{hi:2d}           {sum(sel.values()):5d}" + "".join(f"{sel[c]:17d}" for c in classes))
    print(f"\nSNR (full decode)  lost{head}")
    for lo in range(-27, 3, 3):
        sel = Counter(_class(r) for r in rows if lo <= r["snr"] < lo + 3 or (lo == -27 and r["snr"] < lo) or (lo == 0 and r["snr"] >= 0))
        print(f"  {lo:+3d}               {sum(sel.values()):5d}" + "".join(f"{sel[c]:17d}" for c in classes))

if __name__ == "__main__":
    main()
