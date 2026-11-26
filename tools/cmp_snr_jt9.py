"""Our reported SNR vs jt9's on the decodes both made: by jt9 SNR, by audio frequency and per receiver.

    python3 tools/cmp_snr_jt9.py --tag nn_v3_f4bp --tag nn_v3_snr \\
        --run datasets/ft8-validation-set/run2/recordings:data/val/run2 [--run REC:DEC ...]

A decode pairs with jt9's (the station's decodes.txt) on the same slot and message text with f0 within 6 Hz. Every
--tag must have decoded the pair, so the columns compare the same decodes. Each cell is the median of ours - jt9
(dB) with the count; the "slope" line fits ours against jt9 for jt9 SNR -20..+20 (1 = the same scale).
"""
import argparse
import re
from collections import defaultdict
from pathlib import Path

JT9 = re.compile(r"(\S+)\s+\d+\s+(-?\d+)\s+(-?[\d.]+)\s+(\d+)\s+~\s+(.*?)(?:\s+(?:\?|a\d))*\s*$")  # without jt9's ? / aN marks


def median(v):
    v = sorted(v)
    n = len(v)
    return (v[n // 2] + v[(n - 1) // 2]) / 2


def slope(pairs):
    """Least-squares slope of ours on jt9."""
    n = len(pairs)
    mx, my = sum(p[0] for p in pairs) / n, sum(p[1] for p in pairs) / n
    return sum((x - mx) * (y - my) for x, y in pairs) / sum((x - mx) ** 2 for x, _ in pairs)


def load(rec, dec, tags):
    """[(station, f0, jt9 snr, {tag: our snr})] for the decodes every tag shares with jt9."""
    out = []
    for st in sorted(p for p in Path(rec).iterdir() if (p / "decodes.txt").exists()):
        jt = defaultdict(list)
        for line in (st / "decodes.txt").read_text().splitlines():
            m = JT9.match(line)
            if m:
                jt[(m[1], m[5])].append((int(m[2]), int(m[4])))
        ours = {}
        for t in tags:
            f = Path(dec) / st.name / f"decodes_{t}.txt"
            if not f.exists():
                break
            for line in f.read_text().splitlines():
                p = line.split("\t")
                for js, jf in jt.get((p[0], p[4].strip()), []):
                    if abs(jf - float(p[1])) < 6:
                        ours.setdefault((p[0], p[4].strip(), jf), {"f0": float(p[1]), "jt9": js})[t] = float(p[6])
        else:
            out += [(st.name, d["f0"], d["jt9"], d) for d in ours.values() if all(t in d for t in tags)]
    return out


def table(title, rows, key, tags, order=None):
    groups = defaultdict(list)
    for r in rows:
        groups[key(r)].append(r)
    print(f"\n{title:22s}" + "".join(f"{t:>22s}" for t in tags))
    for k in order or sorted(groups):
        g = groups.get(k)
        if g:
            print(f"{str(k):22s}" + "".join(f"{median([r[3][t] - r[2] for r in g]):+15.1f} n={len(g):<5d}" for t in tags))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, metavar="REC:DEC", help="recordings root : decodes root")
    ap.add_argument("--tag", action="append", required=True)
    a = ap.parse_args()
    rows = []
    for r in a.run:
        rec, dec = r.split(":")
        rows += load(rec, dec, a.tag)
    print(f"{len(rows)} decodes shared with jt9 by every tag; cells: median ours - jt9 (dB)")
    mid = [r for r in rows if -20 <= r[2] <= 20]
    print("slope (jt9 -20..+20)  " + "".join(f"{slope([(r[2], r[3][t]) for r in mid]):22.2f}" for t in a.tag))
    print("all                   " + "".join(f"{median([r[3][t] - r[2] for r in rows]):+22.1f}" for t in a.tag))
    table("jt9 SNR", rows, lambda r: max(-24, min(20, r[2] // 4 * 4)), a.tag)
    table("f0 (Hz)", rows, lambda r: int(r[1] // 250 * 250), a.tag)
    table("station", rows, lambda r: r[0][:24], a.tag)


if __name__ == "__main__":
    main()
