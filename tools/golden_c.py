"""Golden test: C decoder (c/build/ft8nn_cli) vs the Python decoder's saved results.

    uv run python tools/golden_c.py real  <recordings/station_dir> [--n 20]
    uv run python tools/golden_c.py awgn  data/bench/awgn [--n 60] [--workers 6]

real: decodes the first n slots of one station in time order (one context, so hash learning matches)
      and compares every output field with data/real/<station>/decodes_nn_v2.txt.
awgn: compares decoded payload sets per file with data/bench/awgn/results_nn_v2.json.
--threads N (default 4) runs the C decoder on N threads (results must not depend on it); --stream decodes through
ft8nn_decode_stream, so the decodes handed out pass by pass are what is compared.
"""
import argparse
import json
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "c/build/ft8nn_cli"
ONNX = ROOT / "models/v3/ft8nn.onnx"
CLI_OPTS = ["--threads", "4"]


def run_cli(files):
    out = subprocess.run([str(CLI), *CLI_OPTS, str(ONNX), *map(str, files)], capture_output=True, text=True, check=True)
    return [l.split("\t") for l in out.stdout.splitlines() if l]


def real(a):
    st = Path(a.path)
    files = sorted((st / "slots").glob("*.wav"))[: a.n]
    slots = {f.stem for f in files}
    ref = defaultdict(list)
    for l in (ROOT / a.out / st.name / f"decodes_{a.tag}.txt").read_text().splitlines():
        p = l.split("\t")
        if p[0] in slots:
            ref[p[0]].append(p)
    got = defaultdict(list)
    for p in run_cli(files):
        got[p[0]].append(p)
    nref = sum(map(len, ref.values()))
    ngot = sum(map(len, got.values()))
    exact = same_bits = 0
    for s in sorted(slots):
        r = {p[5]: p for p in ref[s]}
        g = {p[5]: p for p in got[s]}
        same_bits += len(r.keys() & g.keys())
        for k in r.keys() & g.keys():
            if r[k] == g[k]:
                exact += 1
            else:
                print(f"  {s} field diff:\n    py {r[k]}\n    c  {g[k]}")
        for k in r.keys() - g.keys():
            print(f"  {s} only py: {r[k][1:5]}")
        for k in g.keys() - r.keys():
            print(f"  {s} only c:  {g[k][1:5]}")
    print(f"{st.name}: {len(files)} slots, py {nref} decodes, c {ngot}, same payload {same_bits}, all fields equal {exact}")
    return nref == ngot == exact


def awgn(a):
    root = Path(a.path)
    man = json.loads((root / "manifest.json").read_text())
    pool = {m["bits"]: m["text"] for m in json.loads((ROOT / "ft8nn/data/messages.json").read_text())}
    ref = json.loads((root / f"results_{a.tag}.json").read_text())["decodes"]
    # spread over SNRs: every k-th file
    files = [root / e["file"] for e in man][:: max(1, len(man) // a.n)][: a.n]
    chunks = [files[i :: a.workers] for i in range(a.workers)]
    got = defaultdict(set)
    with ThreadPoolExecutor(a.workers) as ex:
        for res in ex.map(run_cli, chunks):
            for p in res:
                bits = format(int(p[5], 16), "077b")
                got[p[0] + ".wav"].add(pool.get(bits, "??" + bits))
    nref = ngot = same = 0
    for f in files:
        r = {d[0] for d in ref[f.name]}
        g = got[f.name]
        nref, ngot, same = nref + len(r), ngot + len(g), same + len(r & g)
        if r != g:
            print(f"  {f.name}: only py {sorted(r - g)}  only c {sorted(g - r)}")
    print(f"awgn: {len(files)} files, py {nref} decodes, c {ngot}, same {same}")
    return nref == ngot == same


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["real", "awgn"])
    ap.add_argument("path")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--tag", default="nn_v3_cc", help="Python result tag to compare against")
    ap.add_argument("--onnx", default=None, help="model for the C side (default models/v3/ft8nn.onnx)")
    ap.add_argument("--out", default="data/real", help="root of the Python decodes (real mode), e.g. data/val/run2")
    ap.add_argument("--threads", type=int, default=4, help="C decode threads (0 = one per performance core)")
    ap.add_argument("--stream", action="store_true", help="decode with ft8nn_decode_stream")
    a = ap.parse_args()
    global ONNX, CLI_OPTS
    ONNX = Path(a.onnx) if a.onnx else ONNX
    CLI_OPTS = ["--threads", str(a.threads)] + (["--stream"] if a.stream else [])
    ok = real(a) if a.mode == "real" else awgn(a)
    print("PASS" if ok else "DIFF")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
