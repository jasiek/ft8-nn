"""Run WSJT-X jt9 over a benchmark set and score decodes against the manifest.

  uv run python bench/run_jt9.py data/bench/awgn [--depth 3] [--workers 8]

Writes <set>/results_jt9.json and prints decode rate per SNR.
"""
import argparse
import json
import re
import subprocess
import tempfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

JT9 = "/usr/local/bin/jt9"
LINE = re.compile(r"^(\d{6})\s+(-?\d+)\s+(-?[\d.]+)\s+(\d+)\s+~\s+(.*?)\s*(?:a\d)?\s*$")


def run_chunk(files, depth, extra):
    """jt9 on a batch of files in a private temp dir. Returns {file_tag: [(snr, dt, freq, text)]}."""
    with tempfile.TemporaryDirectory() as tmp:
        cmd = [JT9, "-8", "-d", str(depth), "-a", tmp, "-t", tmp, *extra, *map(str, files)]
        out = subprocess.run(cmd, capture_output=True, text=True, cwd=tmp).stdout
    res = defaultdict(list)
    # jt9 prints decodes per file followed by <DecodeFinished>; map in order.
    blocks = out.split("<DecodeFinished>")
    for f, blk in zip(files, blocks):
        for line in blk.splitlines():
            m = LINE.match(line.strip())
            if m:
                res[f.name].append((int(m[2]), float(m[3]), int(m[4]), m[5].strip()))
    return res


def score(manifest, decodes):
    per_snr = defaultdict(lambda: [0, 0, 0])  # [found, total, false]
    for e in manifest:
        truth = {s["text"] for s in e["signals"]}
        got = {d[3] for d in decodes.get(e["file"], [])}
        k = per_snr[e["snr"]]
        k[0] += len(truth & got)
        k[1] += len(e["signals"])
        k[2] += len(got - truth)
    return dict(sorted(per_snr.items()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("set")
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=10)
    ap.add_argument("--extra", default="", help="extra jt9 args, e.g. '-c K1ABC'")
    a = ap.parse_args()

    root = Path(a.set).resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    files = [root / e["file"] for e in manifest]
    chunks = [files[i : i + a.chunk] for i in range(0, len(files), a.chunk)]
    decodes = {}
    with ThreadPoolExecutor(a.workers) as ex:
        for r in ex.map(lambda c: run_chunk(c, a.depth, a.extra.split()), chunks):
            decodes.update(r)

    table = score(manifest, decodes)
    tag = f"jt9_d{a.depth}" + ("_" + a.extra.replace(" ", "") if a.extra else "")
    (root / f"results_{tag}.json").write_text(json.dumps({"decodes": decodes, "per_snr": table}, indent=1))
    print(f"{tag} on {root.name}:  SNR  decoded/total  rate  false")
    for snr, (f, t, fa) in table.items():
        print(f"  {snr:4d}  {f:4d}/{t:<4d}  {f / t:6.1%}  {fa}")


if __name__ == "__main__":
    main()
