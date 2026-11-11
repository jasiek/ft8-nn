#!/usr/bin/env python3
"""Profile the signals a reference decoder (JTDX by default) gets and nn_v3 misses.

    python3 tools/diag_ref_only.py [--ref jtdx] [--ours nn_v3] [--tier slot] [--list N]

A miss is a (station, slot, sending callsign) pair in the reference's
verified_<ref>.tsv, at the given verify.py tier, that is absent from ours at any
tier - the identity tools/compare_decoders.py counts. Each miss is joined to the
reference's decode line (SNR, DT, frequency, message) and, for JTDX, its
service marker; then broken down by SNR, marker, station, DT, frequency, message
type, whether jt9 also decodes it, and whether we decode anything else within
10 Hz in that slot (a decode at the same spot with a different sender).
"""
import argparse, os, re
from collections import Counter, defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V = os.environ.get("V") or os.path.join(REPO, "datasets", "ft8-validation-set")
JT9 = re.compile(r"^(\S+)\s+\d{6}\s+(-?\d+)\s+(-?[\d.]+)\s+(\d+)\s+~\s+(.*)$")


def verified(path):
    """{(slot, sender): tier}, best tier per pair, plus rows by (slot, freq, text)."""
    rank = {"slot": 0, "active": 1, "repeat": 2, "unconf": 3}
    pairs, rows = {}, {}
    for line in open(path).read().splitlines()[1:]:
        slot, freq, text, sender, tier = line.split("\t")[:5]
        rows[(slot, round(float(freq)), text.strip())] = (sender, tier)
        if tier in rank and ((slot, sender) not in pairs or rank[tier] < rank[pairs[slot, sender]]):
            pairs[slot, sender] = tier
    return pairs, rows


def msg_type(text):
    w = text.split()
    if not w:
        return "empty"
    if w[0] in ("CQ", "QRZ"):
        return "CQ"
    last = w[-1]
    if last in ("RR73", "RRR", "73"):
        return "RR73/RRR/73"
    if re.match(r"^R?[+-]\d\d$", last):
        return "report"
    if re.match(r"^[A-R]{2}\d\d$", last):
        return "grid"
    return "other"


def binned(x, edges):
    for e in edges:
        if x < e:
            return f"< {e:+d}"
    return f">= {edges[-1]:+d}"


def show(title, c, total, order=None):
    print(f"\n{title}")
    for k in (order or [k for k, _ in c.most_common()]):
        if c[k]:
            print(f"  {k:24s} {c[k]:6,}  {100 * c[k] / total:5.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["2", "3", "4"])
    ap.add_argument("--ref", default="jtdx")
    ap.add_argument("--ours", default="nn_v3")
    ap.add_argument("--tier", default="slot")
    ap.add_argument("--out", default=os.path.join(REPO, "data", "val"))
    ap.add_argument("--list", type=int, default=0, help="print the N strongest misses")
    ap.add_argument("--tsv", help="write every miss here (run, station, slot, snr, dt, freq, marker, text)")
    a = ap.parse_args()

    misses = []
    for r in a.runs:
        rec = os.path.join(V, f"run{r}", "recordings")
        for st in sorted(d for d in os.listdir(rec) if os.path.isdir(os.path.join(rec, d, "slots"))):
            d = os.path.join(a.out, f"run{r}", st)
            ref_pairs, ref_rows = verified(os.path.join(d, f"verified_{a.ref}.tsv"))
            our_pairs, _ = verified(os.path.join(d, f"verified_{a.ours}.tsv"))
            jt9_pairs, _ = verified(os.path.join(rec, st, "verified_jt9.tsv"))
            ours_f = defaultdict(list)
            for line in open(os.path.join(d, f"decodes_{a.ours}.txt")).read().splitlines():
                p = line.split("\t")
                ours_f[p[0]].append(float(p[1]))
            marks = {}
            mp = os.path.join(d, f"markers_{a.ref}.tsv")
            if os.path.exists(mp):
                for line in open(mp).read().splitlines():
                    slot, freq, mark, text = line.split("\t")
                    marks[(slot, int(freq), text.strip())] = mark
            seen = set()
            for line in open(os.path.join(d, f"decodes_{a.ref}.txt")).read().splitlines():
                m = JT9.match(line)
                if not m:
                    continue
                slot, snr, dt, freq, text = m[1], int(m[2]), float(m[3]), int(m[4]), m[5].strip()
                sender, tier = ref_rows.get((slot, freq, text), (None, None))
                if tier != a.tier or (slot, sender) in our_pairs or (slot, sender) in seen:
                    continue
                if ref_pairs.get((slot, sender)) != a.tier:
                    continue
                seen.add((slot, sender))
                misses.append(dict(run=r, st=st, slot=slot, snr=snr, dt=dt, freq=freq, text=text,
                                   mark=marks.get((slot, freq, text), "?"),
                                   jt9=(slot, sender) in jt9_pairs,
                                   near=any(abs(f - freq) < 10 for f in ours_f.get(slot, []))))
    n = len(misses)
    print(f"{a.ref}-only pairs at tier '{a.tier}', missed by {a.ours} at every tier: {n:,}")
    edges = [-20, -15, -10, -5, 0, 5]
    show("SNR (as the reference reports it)", Counter(binned(m["snr"], edges) for m in misses), n,
         [f"< {e:+d}" for e in edges] + [f">= {edges[-1]:+d}"])
    show("reference stage marker ('-' BP/OSD, '^' tone matching, '*' AP)", Counter(m["mark"] for m in misses), n)
    show("jt9 also decodes it", Counter("yes" if m["jt9"] else "no" for m in misses), n)
    show("we decode something else within 10 Hz in that slot",
         Counter("yes" if m["near"] else "no" for m in misses), n)
    show("message type", Counter(msg_type(m["text"]) for m in misses), n)
    show("DT (s)", Counter(binned(round(m["dt"] * 10), [-10, -5, 0, 5, 10, 15, 20]) for m in misses), n)
    show("frequency (Hz)", Counter(f"{m['freq'] // 500 * 500}-{m['freq'] // 500 * 500 + 499}" for m in misses), n,
         [f"{f}-{f + 499}" for f in range(0, 3500, 500)])
    show("station", Counter(f"run{m['run']} {m['st'][:30]}" for m in misses), n)
    if a.list:
        print(f"\nstrongest {a.list}:")
        for m in sorted(misses, key=lambda m: -m["snr"])[:a.list]:
            print(f"  run{m['run']} {m['st'][:8]} {m['slot']} {m['snr']:+3d} {m['dt']:+.1f} {m['freq']:5d} "
                  f"{m['mark']} jt9={'y' if m['jt9'] else 'n'} near={'y' if m['near'] else 'n'}  {m['text']}")
    if a.tsv:
        with open(a.tsv, "w") as fh:
            for m in misses:
                fh.write("\t".join(str(m[k]) for k in ("run", "st", "slot", "snr", "dt", "freq", "mark", "text")) + "\n")


if __name__ == "__main__":
    main()
