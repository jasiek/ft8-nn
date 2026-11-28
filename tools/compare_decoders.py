#!/usr/bin/env python3
"""Compare decoders on validation sets from collect/verify.py's per-decode TSVs.

Block-set mode (default, new validation set runs/r02, r05..r14):
    python3 tools/compare_decoders.py --set third --tags jt9 jtdx jtdx_deep ft8lib nn_v4
    python3 tools/compare_decoders.py --set small --tags jt9 jtdx jtdx_deep ft8lib nn_v4

Legacy full-run mode (v1 runs 2, 3, 4):
    python3 tools/compare_decoders.py --set none --runs 2 3 4 --tags jt9 jtdx jtdx_deep nn_v3

Reads <out>/<run>/<station>/verified_<tag>_<set>.tsv (or legacy verified_<tag>.tsv),
reporting decodes, slot, xconf, confirmed (slot + xconf), active, repeat, unconf,
strict exclusive confirmation (xconf-strict / unconfirmable when verify.py --others
was run), timing from timing_<tag>_<set>.json when present, and pairwise
(run, station, slot, sending callsign) overlaps.
"""
import argparse
from collections import Counter
import json
import os
import re

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V = os.path.abspath(os.environ.get("V") or os.path.join(REPO, "datasets", "ft8-validation-set"))
RANK = {"slot": 0, "xconf": 1, "active": 2, "repeat": 3, "unconf": 4}


def resolve_set(spec):
    if os.path.exists(spec):
        return os.path.abspath(spec)
    cand = os.path.join(V, "sets", spec if spec.endswith(".txt") else f"{spec}.txt")
    if os.path.exists(cand):
        return cand
    raise SystemExit(f"set file not found: {spec}")


def read_set_runs(path):
    out = []
    for line in open(path):
        w = line.split("#", 1)[0].split()
        if len(w) == 3 and re.match(r"^\d{6}_\d{6}$", w[2]):
            out.append((os.path.normpath(w[0]), w[1], w[2]))
    return out


def parse_tsv(path, st, tiers, excl, pairs):
    with open(path) as fh:
        hdr = fh.readline().rstrip("\n").split("\t")
        has_excl = "base_tier" in hdr
        ti = hdr.index("tier")
        bi = hdr.index("base_tier") if has_excl else ti
        ei = hdr.index("excl") if has_excl else None
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) <= bi:
                continue
            slot, sender, base = f[0], f[3], f[bi]
            tiers[base] += 1
            if has_excl and f[ei] == "1" and base != "n/a":
                st_tier = f[ti]
                excl[st_tier] += 1
                excl[f"{st_tier}:{base}"] += 1
            if base in RANK:
                k = (st, slot, sender)
                if k not in pairs or RANK[base] < RANK[pairs[k]]:
                    pairs[k] = base


def load_set_run(run, stations, set_name, tag, out_root):
    rec = os.path.join(V, run, "recordings")
    droot = os.path.join(out_root, run)
    tiers, excl, pairs = Counter(), Counter(), {}
    for st in stations:
        cands = [
            os.path.join(droot, st, f"verified_{tag}_{set_name}.tsv"),
            os.path.join(rec, st, f"verified_{tag}_{set_name}.tsv"),
        ]
        p = next((c for c in cands if os.path.exists(c)), None)
        if not p:
            raise SystemExit(f"missing verified_{tag}_{set_name}.tsv for {run}/{st}; "
                             f"run tools/run_set.py --set {set_name} --verify-only --tag {tag}")
        parse_tsv(p, st, tiers, excl, pairs)
    return tiers, excl, pairs


def load_legacy(run, tag, out_root):
    rec = os.path.join(V, f"run{run}" if run != "1" else "", "recordings")
    root = rec if tag == "jt9" and not os.path.isdir(os.path.join(out_root, f"run{run}")) \
        else os.path.join(out_root, f"run{run}")
    tiers, excl, pairs = Counter(), Counter(), {}
    for st in sorted(os.listdir(rec)):
        if not os.path.isdir(os.path.join(rec, st, "slots")):
            continue
        cands = [os.path.join(root, st, f"verified_{tag}.tsv"),
                 os.path.join(rec, st, f"verified_{tag}.tsv")]
        p = next((c for c in cands if os.path.exists(c)), None)
        if not p:
            raise SystemExit(f"missing verified_{tag}.tsv for run{run}/{st}: run collect/verify.py --tag {tag}")
        parse_tsv(p, st, tiers, excl, pairs)
    return tiers, excl, pairs


def pct(a, b):
    return f"{100 * (a - b) / b:+.1f}%" if b else "-"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="third",
                    help="small | third | full | path/to/set.txt, or 'none' for legacy --runs")
    ap.add_argument("--runs", nargs="+", default=["2", "3", "4"], help="legacy v1 runs when --set none")
    ap.add_argument("--tags", nargs="+", default=["jt9", "jtdx", "jtdx_deep", "ft8lib", "nn_v4"])
    ap.add_argument("--out", default=None, help="decodes root (default: data/val/<set>, or data/val for --set none)")
    ap.add_argument("--per-run", action="store_true", help="also print per-run breakdown")
    a = ap.parse_args()

    if a.set != "none":
        set_path = resolve_set(a.set)
        set_name = os.path.splitext(os.path.basename(set_path))[0]
        out_root = os.path.abspath(a.out or os.path.join(REPO, "data", "val", set_name))
        blocks = read_set_runs(set_path)
        runs = sorted({r for r, _, _ in blocks})
        rx_by_run = {r: sorted({st for rr, st, _ in blocks if rr == r}) for r in runs}
        data = {(r, t): load_set_run(r, rx_by_run[r], set_name, t, out_root) for r in runs for t in a.tags}
        title = f"set '{set_name}' ({len(blocks)} blocks, {len(blocks) * 8:,} scored slots across {len(runs)} runs)"
    else:
        set_name = None
        out_root = os.path.abspath(a.out or os.path.join(REPO, "data", "val"))
        runs = a.runs
        data = {(r, t): load_legacy(r, t, out_root) for r in runs for t in a.tags}
        title = f"legacy runs {', '.join(runs)}"

    pooled = {
        t: (
            sum((data[r, t][0] for r in runs), Counter()),
            sum((data[r, t][1] for r in runs), Counter()),
            {(r,) + k: v for r in runs for k, v in data[r, t][2].items()},
        )
        for t in a.tags
    }
    base = a.tags[0]
    bt = pooled[base][0]
    b_dec = sum(bt.values())
    b_chk = b_dec - bt["n/a"]
    b_slot = bt["slot"]
    b_conf = bt["slot"] + bt["xconf"]

    print(f"=== Decoder comparison on {title} (vs {base}) ===\n")
    hdr = (f"{'decoder':12s} {'decodes':>9s} {'vs ' + base:>8s} {'checkable':>9s} "
           f"{'slot':>8s} {'xconf':>7s} {'confirmed':>10s} {'vs ' + base:>8s} {'conf%':>6s} "
           f"{'active':>7s} {'repeat':>6s} {'unconf':>6s} {'n/a':>5s}")
    print(hdr)
    print("-" * len(hdr))
    for t in a.tags:
        c = pooled[t][0]
        dec = sum(c.values())
        chk = dec - c["n/a"]
        conf = c["slot"] + c["xconf"]
        vs_d = "–" if t == base else pct(dec, b_dec)
        vs_c = "–" if t == base else pct(conf, b_conf)
        cpct = f"{100 * conf / max(chk, 1):.1f}%"
        print(f"{t:12s} {dec:9,d} {vs_d:>8s} {chk:9,d} "
              f"{c['slot']:8,d} {c['xconf']:7,d} {conf:10,d} {vs_c:>8s} {cpct:>6s} "
              f"{c['active']:7,d} {c['repeat']:6,d} {c['unconf']:6,d} {c['n/a']:5,d}")

    if any(pooled[t][1] for t in a.tags):
        print("\nStrict exclusive confirmation (decodes found by no other compared decoder on that receiver):")
        ehdr = (f"{'decoder':12s} {'exclusives':>10s} {'xconf-strict':>13s} {'strict%':>8s} "
                f"{'unconfirmable':>14s}  unconfirmable base tiers")
        print(ehdr)
        print("-" * len(ehdr))
        for t in a.tags:
            ex = pooled[t][1]
            xs, uc = ex["xconf-strict"], ex["unconfirmable"]
            ne = xs + uc
            sp = f"{100 * xs / ne:.1f}%" if ne else "–"
            ub = ", ".join(f"{b}:{ex[f'unconfirmable:{b}']}"
                           for b in ("slot", "xconf", "active", "repeat", "unconf")
                           if ex[f"unconfirmable:{b}"])
            print(f"{t:12s} {ne:10,d} {xs:13,d} {sp:>8s} {uc:14,d}  {ub or '–'}")

    if set_name:
        timings = {}
        for t in a.tags:
            tp = os.path.join(out_root, f"timing_{t}_{set_name}.json")
            if os.path.exists(tp):
                timings[t] = json.load(open(tp))
        if timings:
            print("\nTiming summary:")
            thdr = f"{'decoder':12s} {'workers':>7s} {'wall (min)':>10s} {'wall/slot':>10s} {'core-s/slot':>12s}  machine"
            print(thdr)
            print("-" * len(thdr))
            for t in a.tags:
                if t not in timings:
                    continue
                tm = timings[t]
                ns = max(tm.get("slots_decoded") or 1, 1)
                ws = tm["wall_s"]
                cs = tm.get("cpu_s") or 0.0
                print(f"{t:12s} {tm['workers']:7d} {ws / 60:10.2f} {ws / ns:9.2f}s "
                      f"{(f'{cs / ns:.2f}s' if cs else '–'):>12s}  {tm.get('machine', '')}")

    if a.per_run:
        print(f"\nPer-run decodes / confirmed (slot+xconf) / unconf:")
        print(f"{'run':10s}" + "".join(f"{t:>28s}" for t in a.tags))
        for r in runs:
            cells = []
            for t in a.tags:
                c = data[r, t][0]
                cells.append(f"{sum(c.values()):,} / {c['slot'] + c['xconf']:,} / {c['unconf']}")
            print(f"{r:10s}" + "".join(f"{cell:>28s}" for cell in cells))

    print("\nPairwise overlap, pooled, as (run, station, slot, sending callsign) pairs:")
    print("  exclusive counts split as: total (confirmed [slot+xconf] / active / repeat / unconf)")
    fmt = lambda c: (f"{sum(c.values()):6,} "
                     f"({c['slot'] + c['xconf']:,} [{c['slot']:,}+{c['xconf']:,}] / "
                     f"{c['active']:,} / {c['repeat']:,} / {c['unconf']:,})")
    for i, x in enumerate(a.tags):
        for y in a.tags[i + 1:]:
            px, py = pooled[x][2], pooled[y][2]
            both = px.keys() & py.keys()
            ox = Counter(px[k] for k in px.keys() - py.keys())
            oy = Counter(py[k] for k in py.keys() - px.keys())
            print(f"  {x:>10s} vs {y:<10s} both {len(both):7,}   {x}-only {fmt(ox)}   {y}-only {fmt(oy)}")


if __name__ == "__main__":
    main()
