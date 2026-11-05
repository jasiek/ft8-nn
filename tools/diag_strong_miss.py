"""Classify jt9-only decodes above an SNR threshold by the stage that loses them.

  uv run python tools/diag_strong_miss.py runs/v2/model.pt nn_v2 -9 [max_slots] [--root REC] [--out OUT] [--max-snr S]

--root: recordings root (default ../ft8-training-data/recordings); --out: our decodes root (default data/real).

For each slot: subtract our saved decodes, then look for a candidate near the jt9 signal in the
residual. Stages: NOCAND (no candidate within 5 Hz / 0.2 s), FAIL (candidate, NN decode fails;
E = expected bit errors), OK (decodes in the residual: lost to pass/candidate budget or dedupe).
"""
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.io import wavfile

sys.path.insert(0, "bench")
import run_real as rr  # noqa: E402
from ft8nn import frontend as fe, ft8, ldpc, nn_metric  # noqa: E402
from ft8nn.subtract import subtract  # noqa: E402

import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument("model"); _ap.add_argument("tag"); _ap.add_argument("thr", type=int)
_ap.add_argument("max_slots", type=int, nargs="?", default=40)
_ap.add_argument("--root", default="/Users/jps/tmp/ft8-training-data/recordings")
_ap.add_argument("--out", default="data/real")
_ap.add_argument("--max-snr", type=int, default=99)
_a = _ap.parse_args()
model, tag, thr, max_slots = _a.model, _a.tag, _a.thr, _a.max_slots
root, outroot = Path(_a.root), Path(_a.out)
fn = nn_metric.make_llr_fn(model)

todo = []
for st in sorted(p for p in root.iterdir() if (p / "slots").is_dir()):
    ours = rr.load_ours(outroot / st.name / f"decodes_{tag}.txt")
    raw = {}
    for line in (st / "decodes.txt").read_text().splitlines():
        p = line.split("~")
        if len(p) == 2:
            raw[(line.split()[0], rr.norm(p[1]))] = p[1].strip()
    miss = defaultdict(list)
    for s, v in rr.load_jt9(st / "decodes.txt").items():
        names = {d[3] for d in ours.get(s, [])}
        for snr, dt, f, t in v:
            if thr <= snr <= _a.max_snr and t not in names and not raw[(s, t)].rstrip().endswith(("a1", "a2", "a3")):
                miss[s].append((snr, dt, f, t))
    todo += [(st, s, m, ours.get(s, [])) for s, m in miss.items()]

stage = Counter()
for st, s, misses, ours in todo[:max_slots]:
    _, x = wavfile.read(st / "slots" / f"{s}.wav")
    x = x.astype(float)
    x = np.pad(x, (0, max(0, 180000 - len(x))))[:180000] / (x.std() + 1e-9)
    # re-synthesise our decodes from the hex payloads for subtraction
    lines = (outroot / st.name / f"decodes_{tag}.txt").read_text().splitlines()
    for ln in lines:
        p = ln.split("\t")
        if p[0] == s:
            bits = np.array(list(bin(int(p[5], 16))[2:].zfill(77)), dtype=np.uint8)
            subtract(x, ft8.encode77(bits)[1], float(p[1]), float(p[2]))
    cands = fe.find_candidates(x)
    bb = fe.Baseband(x)
    for snr, dt, f, t in misses:
        near = [c for c in cands if abs(c.f0 - f) < 5 and abs(c.dt - dt) < 0.2]
        nb = [f"{o[0]:.0f}/{o[1]:+.2f}" for o in ours if abs(o[0] - f) < 50 and abs(o[1] - dt) < 1]
        if not near:
            st_ = "NOCAND"
            info = ""
        else:
            hs = fe.fine_sync2(bb, near[0], 2)
            res = []
            ok = False
            for h, (l,) in zip(hs, fn(hs)):
                a, _ = ldpc.decode(l[None])[0]
                E = float((1 / (1 + np.exp(np.abs(l)))).sum())
                ok |= a is not None
                res.append(f"{h.f0:.1f}/{h.dt:+.2f} E={E:.0f}")
            st_ = "OK" if ok else "FAIL"
            info = f"sc {near[0].score:.1f} " + "; ".join(res)
        stage[st_] += 1
        print(f"{st.name[:6]} {s} {snr:+3d} {dt:+.1f} {f:5d} {t:24s} {st_:6s} {info} | ours<50Hz: {nb}", flush=True)
print(stage, f"({len(todo)} slots with misses, {min(len(todo), max_slots)} examined)")
