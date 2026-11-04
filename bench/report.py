"""Compare decoders on a benchmark set: decode rate per SNR, 50% threshold, false decodes.

  uv run python bench/report.py data/bench/awgn [--dt-max 0.5]
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np


def norm(t):
    """Strip jt9 decorations: trailing '?' (low-confidence AP) and 'a1'..'a7' AP tags."""
    return re.sub(r"\s+(\?|a\d)\s*$", "", t.strip()).strip()


def curve(manifest, decodes, text_idx, dt_min, dt_max):
    rows = {}
    for e in manifest:
        got = {norm(d[text_idx]) for d in decodes.get(e["file"], [])}
        sel = [s for s in e["signals"] if dt_min <= s["dt"] <= dt_max]
        truth_all = {s["text"] for s in e["signals"]}
        r = rows.setdefault(e["snr"], [0, 0, 0])
        r[0] += sum(s["text"] in got for s in sel)
        r[1] += len(sel)
        r[2] += len(got - truth_all)
    return dict(sorted(rows.items()))


def threshold(rows, level=0.5):
    """SNR where decode rate crosses `level` (linear interpolation)."""
    pts = [(snr, f / t) for snr, (f, t, _) in rows.items() if t]
    for (s0, r0), (s1, r1) in zip(pts, pts[1:]):
        if r0 < level <= r1:
            return s0 + (level - r0) / (r1 - r0) * (s1 - s0)
    return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("set")
    ap.add_argument("--dt-min", type=float, default=-9)
    ap.add_argument("--dt-max", type=float, default=9)
    a = ap.parse_args()
    root = Path(a.set)
    manifest = json.loads((root / "manifest.json").read_text())
    results = {}
    for p in sorted(root.glob("results_*.json")):
        r = json.loads(p.read_text())
        tag = p.stem.removeprefix("results_")
        text_idx = 3 if tag.startswith("jt9") else 0
        results[tag] = curve(manifest, r["decodes"], text_idx, a.dt_min, a.dt_max)
    tags = list(results)
    snrs = sorted({s for r in results.values() for s in r})
    print(f"{root.name}  (DT {a.dt_min}..{a.dt_max})   decode rate %  [false decodes]")
    print("  SNR " + "".join(f"{t:>22s}" for t in tags))
    for s in snrs:
        cells = []
        for t in tags:
            f, n, fa = results[t].get(s, (0, 0, 0))
            cells.append(f"{100 * f / max(n, 1):6.1f}% [{fa:3d}]".rjust(22))
        print(f"  {s:4d}" + "".join(cells))
    print("  50%  " + "".join(f"{threshold(results[t]):+21.2f} " for t in tags))
    print("  90%  " + "".join(f"{threshold(results[t], 0.9):+21.2f} " for t in tags))


if __name__ == "__main__":
    main()
