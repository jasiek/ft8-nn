"""Run the ft8nn decoder over a benchmark set and score it like bench/run_jt9.py.

  uv run python bench/run_ours.py data/bench/awgn [--tag classical] [--model path.pt]

Decoded payloads are mapped to text through the message pool; payloads not in the pool are
counted as false decodes (every transmitted message comes from the pool).
"""
import argparse
import json
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.io import wavfile

POOL = {m["bits"]: m["text"] for m in json.loads(Path("ft8nn/data/messages.json").read_text())}
_ARGS = None


def _init(args):
    global _ARGS, _LLR
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    _ARGS = args
    from ft8nn import decoder

    _LLR = decoder.classical_metrics
    if args.model:
        import torch

        torch.set_num_threads(1)
        from ft8nn import nn_metric

        _LLR = nn_metric.make_llr_fn(args.model, also_classical=args.also_classical)


def _decode_file(path):
    from ft8nn import decoder

    _, x = wavfile.read(path)
    ds = decoder.decode_slot(x / 100.0, llr_fn=_LLR, n_hyp=_ARGS.n_hyp, osd_order=_ARGS.osd_order, osd_max_hard=_ARGS.osd_max_hard, ap=() if _ARGS.no_ap else (decoder.AP_CQ,), erase_gate=_ARGS.erase_gate, erase_osd_dmax=_ARGS.erase_osd_dmax, osd_snaps=_ARGS.osd_snaps, passes=_ARGS.passes, reuse_df=_ARGS.reuse_df, dt_recenter=_ARGS.dt_recenter, cand_kw=dict(per_symbol=_ARGS.per_symbol, fmax=_ARGS.fmax, base=_ARGS.base))
    return Path(path).name, [
        (POOL.get(d.key, "??" + d.key), round(d.f0, 1), round(d.dt, 2), d.method, d.info.get("metric"), d.info.get("nhard"))
        for d in ds
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("set")
    ap.add_argument("--tag", default="classical")
    ap.add_argument("--model", default=None)
    ap.add_argument("--also-classical", action="store_true")
    ap.add_argument("--n-hyp", type=int, default=2)
    ap.add_argument("--per-symbol", action="store_true", help="per-symbol Costas sync ratio")
    ap.add_argument("--fmax", type=float, default=3100.0)
    ap.add_argument("--base", default="median", choices=["max40", "median"], help="candidate score baseline")
    ap.add_argument("--no-ap", action="store_true", help="disable a-priori (CQ) decoding")
    ap.add_argument("--erase-gate", type=float, default=15.0, help="burst-erasure retry below this expected error count (0 = off)")
    ap.add_argument("--erase-osd-dmax", type=float, default=3.0, help="wide-window OSD erasure stage distance limit (0 = off)")
    ap.add_argument("--osd-snaps", type=int, default=2, help="OSD on BP posterior sums after iterations 1..N (0 = off)")
    ap.add_argument("--passes", type=int, default=4, help="decode-subtract passes per slot")
    ap.add_argument("--reuse-df", type=float, default=62.5, help="skip candidates tried before with no subtraction within this many Hz (0 = off)")
    ap.add_argument("--dt-recenter", type=float, default=0.8, help="shift the DT search window by the median DT of the decodes so far when it is at least this far from 0 (0 = off)")
    ap.add_argument("--repeat", default="", help="accepted for run_real.py parity; inert here (slots have no context)")
    ap.add_argument("--osd-order", type=int, default=2)
    ap.add_argument("--osd-max-hard", type=int, default=40)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--snrs", default=None, help="restrict to SNR range a:b")
    a = ap.parse_args()

    root = Path(a.set)
    manifest = json.loads((root / "manifest.json").read_text())
    if a.snrs:
        lo, hi = (int(v) for v in a.snrs.split(":"))
        manifest = [e for e in manifest if lo <= e["snr"] <= hi]
    files = [str(root / e["file"]) for e in manifest]
    with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(a,)) as ex:
        decodes = dict(ex.map(_decode_file, files, chunksize=2))

    per_snr = defaultdict(lambda: [0, 0, 0])
    for e in manifest:
        truth = {s["text"] for s in e["signals"]}
        got = {d[0] for d in decodes.get(e["file"], [])}
        k = per_snr[e["snr"]]
        k[0] += len(truth & got)
        k[1] += len(truth)
        k[2] += len(got - truth)
    table = dict(sorted(per_snr.items()))
    (root / f"results_{a.tag}.json").write_text(json.dumps({"decodes": decodes, "per_snr": table}, indent=1))
    print(f"{a.tag} on {root.name}:  SNR  decoded/total  rate  false")
    for snr, (f, t, fa) in table.items():
        print(f"  {snr:4d}  {f:4d}/{t:<4d}  {f / t:6.1%}  {fa}")


if __name__ == "__main__":
    main()
