#!/usr/bin/env python3
"""Compare decoders on validation runs from collect/verify.py's per-decode files.

    python3 tools/compare_decoders.py --runs 2 3 4 --tags jt9 jtdx jtdx_deep nn_v3

Reads <out>/run<r>/<station>/verified_<tag>.tsv (jt9: the recordings' own
verified_jt9.tsv), so verify.py must have been run for every tag first.
Prints decodes / slot-confirmed / unconf per run and pooled, then the overlap
of each pair as (slot, sending callsign) - the identity verify.py scores, which
ignores hashed-call and AP-marker differences between decoders - with the tier
of each side's exclusive pairs (slot-confirmed exclusives are real signals the
other decoder missed).
"""
import argparse, os
from collections import Counter

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V = os.environ.get("V") or os.path.join(REPO, "datasets", "ft8-validation-set")


def load(run, tag, out):
    """-> (Counter of tiers over decodes, {(station, slot, sender): best tier})."""
    rec = os.path.join(V, f"run{run}", "recordings")
    root = rec if tag == "jt9" else os.path.join(out, f"run{run}")
    tiers, pairs = Counter(), {}
    rank = {"slot": 0, "active": 1, "repeat": 2, "unconf": 3}
    for st in sorted(os.listdir(rec)):
        p = os.path.join(root, st, f"verified_{tag}.tsv")
        if not os.path.isdir(os.path.join(rec, st, "slots")):
            continue
        if not os.path.exists(p):
            raise SystemExit(f"missing {p}: run collect/verify.py --tag {tag}")
        for line in open(p).read().splitlines()[1:]:
            f = line.split("\t")
            slot, sender, tier = f[0], f[3], f[4]
            tiers[tier] += 1
            if tier in rank:
                k = (st, slot, sender)
                if k not in pairs or rank[tier] < rank[pairs[k]]:
                    pairs[k] = tier
    return tiers, pairs


def row(t):
    return sum(t.values()), t["slot"], t["unconf"]


def pct(a, b):
    return f"{100 * (a - b) / b:+.1f}%" if b else "-"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["2", "3", "4"])
    ap.add_argument("--tags", nargs="+", default=["jt9", "jtdx", "jtdx_deep", "nn_v3"])
    ap.add_argument("--out", default=os.path.join(REPO, "data", "val"))
    a = ap.parse_args()

    data = {(r, t): load(r, t, a.out) for r in a.runs for t in a.tags}
    pooled = {t: (sum((data[r, t][0] for r in a.runs), Counter()),
                  {(r,) + k: v for r in a.runs for k, v in data[r, t][1].items()})
              for t in a.tags}
    base = a.tags[0]

    print(f"decodes / slot-confirmed / unconf; change vs {base} in brackets\n")
    print(f"{'run':8s}" + "".join(f"{t:>38s}" for t in a.tags))
    for r in a.runs + ["pooled"]:
        cells = []
        for t in a.tags:
            d, s, u = row(pooled[t][0] if r == "pooled" else data[r, t][0])
            bd, bs, _ = row(pooled[base][0] if r == "pooled" else data[r, base][0])
            extra = "" if t == base else f" ({pct(d, bd)}/{pct(s, bs)})"
            cells.append(f"{d:,} / {s:,} / {u}{extra}")
        print(f"{r:8s}" + "".join(f"{c:>38s}" for c in cells))

    print("\noverlap, pooled, as (slot, sending callsign) pairs")
    print("  A-only and B-only are split by verify.py tier: slot / active / repeat / unconf")
    for i, x in enumerate(a.tags):
        for y in a.tags[i + 1:]:
            px, py = pooled[x][1], pooled[y][1]
            both = px.keys() & py.keys()
            ox = Counter(px[k] for k in px.keys() - py.keys())
            oy = Counter(py[k] for k in py.keys() - px.keys())
            fmt = lambda c: f"{sum(c.values()):6,} ({c['slot']:,} / {c['active']:,} / {c['repeat']:,} / {c['unconf']:,})"
            print(f"  {x:>9s} vs {y:<9s} both {len(both):7,}   {x}-only {fmt(ox)}   {y}-only {fmt(oy)}")


if __name__ == "__main__":
    main()
