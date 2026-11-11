"""Why context a-priori misses repeats: signals the reference decoder gets that we decoded verbatim 30/60 s earlier.

  uv run python tools/diag_ctx_miss.py data/logs/jtdx_cmp/jtdx_only_slot.tsv [--n 300] [--model models/v3/model.pt]

Input: the --tsv of tools/diag_ref_only.py (run, station, slot, snr, dt, freq, marker, text). Keeps the misses whose
exact text we decoded in the same period 30 or 60 s earlier within CTX_DF (the prior decode), samples --n of them,
and replays each slot: subtract all our decodes of the slot, find candidates, take the one nearest the prior decode
within CTX_DF / CTX_DDT, and report the first gate that loses it - NOCAND (no candidate there), EARLY (early-DT
candidate, context skipped), GATED (expected bit errors >= ctx_gate), CTXFAIL (pinned BP/OSD-2 finds no codeword with
the pinned bits), IMPLAUS (message.plausible rejects it), CTXOK (decodes here: lost elsewhere, e.g. dedupe/budget).

Also a full-message test, the JTDX '^' idea: with all 77 payload bits known the whole 174-bit codeword is known, so
score how well the NN's LLRs agree with it, rho = sum(l * s) / sum(|l|), s = +-1 per codeword bit (1 = perfect
agreement, ~0 = noise). Scored at the nearest candidate's hypotheses and, with no candidate search, directly at the
prior decode's (f0, dt). Two null distributions say what threshold would keep false decodes out: the same LLRs
against the codewords of up to 30 other messages decoded on this receiver (wrong message, right audio), and the right
message tested at the same (f0, dt) in the same slot on up to 3 receivers on other bands (station absent: the case
of a repeat hypothesised after the sender stopped, as in the context AP wrong-receiver test).
"""
import argparse, random, re, sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.io import wavfile

sys.path.insert(0, "bench")
from ft8nn import decoder as dec, frontend as fe, ft8, message, nn_metric  # noqa: E402
from ft8nn.subtract import subtract  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("tsv")
ap.add_argument("--n", type=int, default=300)
ap.add_argument("--model", default="models/v3/model.pt")
ap.add_argument("--tag", default="nn_v3")
ap.add_argument("--out", default="data/val")
ap.add_argument("--rec", default="datasets/ft8-validation-set")
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--ctx-gate", type=float, default=48.0)
a = ap.parse_args()

T = lambda s: datetime.strptime(s, "%y%m%d_%H%M%S").timestamp()
strip = lambda t: re.sub(r"[<>]", "", t)
bits_of = lambda h: np.array(list(bin(int(h, 16))[2:].zfill(77)), dtype=np.uint8)

ours = {}
def load(run, st):
    if (run, st) not in ours:
        rows = []
        for ln in (Path(a.out) / f"run{run}" / st / f"decodes_{a.tag}.txt").read_text().splitlines():
            p = ln.split("\t")
            rows.append(dict(slot=p[0], f0=float(p[1]), dt=float(p[2]), text=strip(p[4]), bits=bits_of(p[5])))
        ours[run, st] = rows
    return ours[run, st]

todo = []
for ln in Path(a.tsv).read_text().splitlines():
    run, st, slot, snr, dt, freq, mark, text = ln.split("\t")
    t0 = T(slot)
    prior = [o for o in load(run, st) if o["text"] == text and t0 - T(o["slot"]) in (30, 60)
             and abs(o["f0"] - int(freq)) <= dec.CTX_DF]
    if prior:
        todo.append((run, st, slot, int(snr), mark, text, max(prior, key=lambda o: T(o["slot"]))))
print(f"{len(todo)} misses with the same text decoded by us 30/60 s earlier within {dec.CTX_DF} Hz", flush=True)
random.Random(a.seed).shuffle(todo)
todo = todo[:a.n]

fn = nn_metric.make_llr_fn(a.model)
rho = lambda l, cw: float((l * (2.0 * cw - 1)).sum() / (np.abs(l).sum() + 1e-9))
stage, rows = Counter(), []
true_c, null_c, true_d, null_d, null_rx = [], [], [], [], []
band = lambda st: st.split("_")[1]
by_run = defaultdict(list)
for p in sorted((Path(a.rec)).glob("run*/recordings/*/slots")):
    by_run[p.parts[-4][3:]].append(p.parent.name)
for i, (run, st, slot, snr, mark, text, pr) in enumerate(todo):
    _, x = wavfile.read(Path(a.rec) / f"run{run}" / "recordings" / st / "slots" / f"{slot}.wav")
    x = x.astype(float)
    x = np.pad(x, (0, max(0, 180000 - len(x))))[:180000] / (x.std() + 1e-9)
    for o in load(run, st):
        if o["slot"] == slot:
            subtract(x, ft8.encode77(o["bits"])[1], o["f0"], o["dt"])
    cw = ft8.encode77(pr["bits"])[0]
    others = {o["text"]: o["bits"] for o in load(run, st) if o["text"] != text}
    wrong = [ft8.encode77(b)[0] for b in list(others.values())[:30]]
    bb = fe.Baseband(x)

    # full-message test straight at the prior decode's (f0, dt): no candidate search
    hs = fe.fine_sync2(bb, SimpleNamespace(f0=pr["f0"], dt=pr["dt"], score=0.0, early=False), 2)
    L = [l for (l,) in fn(hs)]
    true_d.append(max(rho(l, cw) for l in L))
    null_d += [max(rho(l, w) for l in L) for w in wrong]

    for st2 in [s2 for s2 in by_run[run] if band(s2) != band(st)][:3]:
        w2 = Path(a.rec) / f"run{run}" / "recordings" / st2 / "slots" / f"{slot}.wav"
        if w2.exists():
            _, y = wavfile.read(w2)
            y = y.astype(float)
            y = np.pad(y, (0, max(0, 180000 - len(y))))[:180000] / (y.std() + 1e-9)
            h2 = fe.fine_sync2(fe.Baseband(y), SimpleNamespace(f0=pr["f0"], dt=pr["dt"], score=0.0, early=False), 2)
            null_rx.append(max(rho(l, cw) for (l,) in fn(h2)))

    cands = fe.find_candidates(x)
    near = [c for c in cands if abs(c.f0 - pr["f0"]) <= dec.CTX_DF and abs(c.dt - pr["dt"]) <= dec.CTX_DDT]
    if not near:
        st_, info = "NOCAND", ""
    else:
        c = min(near, key=lambda c: abs(c.f0 - pr["f0"]))
        hs = fe.fine_sync2(bb, c, 2)
        L = [l for (l,) in fn(hs)]
        E = [float((1 / (1 + np.exp(np.abs(l)))).sum()) for l in L]
        true_c.append(max(rho(l, cw) for l in L))
        null_c += [max(rho(l, w) for l in L) for w in wrong]
        pats = dec.context_patterns([(pr["f0"], pr["dt"], pr["bits"])], hs[0].f0, hs[0].dt)
        if c.early:
            st_ = "EARLY"
        elif E[0] >= a.ctx_gate:
            st_ = "GATED"
        else:
            a91, inf = dec._try_ctx(L[0], pats)
            st_ = "CTXFAIL" if a91 is None else ("IMPLAUS" if dec._plausible(a91, inf)[0] is None else "CTXOK")
        info = f"E={E[0]:.0f} rho={true_c[-1]:.2f}"
    stage[st_] += 1
    print(f"{i:3d} run{run} {st[:8]} {slot} {snr:+3d} {mark} {text:26s} {st_:7s} {info} rho@prior={true_d[-1]:.2f}",
          flush=True)

print("\nfirst gate that loses the repeat:", dict(stage.most_common()))
for name, t, n in (("at the nearest candidate; null: wrong message", true_c, null_c),
                   ("at the prior (f0, dt); null: wrong message", true_d, null_d),
                   ("at the prior (f0, dt); null: right message, receiver on another band", true_d, null_rx)):
    t, n = np.array(t), np.array(n)
    print(f"\nfull-message rho {name}: right message n={len(t)}, wrong messages n={len(n)}")
    print("  percentiles      5%   25%   50%   75%   95%   max")
    print("  right       " + " ".join(f"{v:5.2f}" for v in np.percentile(t, [5, 25, 50, 75, 95, 100])))
    print("  wrong       " + " ".join(f"{v:5.2f}" for v in np.percentile(n, [5, 25, 50, 75, 95, 100])))
    for thr in (0.2, 0.25, 0.3, 0.35, 0.4, 0.5):
        print(f"  rho >= {thr:.2f}: right {100 * (t >= thr).mean():5.1f}%   wrong {int((n >= thr).sum())} of {len(n)}")
