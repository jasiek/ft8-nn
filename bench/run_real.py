"""Decode real 15 s slot recordings and compare with jt9 decodes (no ground truth).

  uv run python bench/run_real.py ../ft8-training-data/recordings --model runs/v1/model.pt --tag nn_v1
  uv run python bench/run_real.py ../ft8-training-data/recordings --compare-only --tag nn_v1

Recordings layout: <root>/<station>/slots/*.wav (+ optional decodes.txt from jt9:
"<slot> <hhmmss> <snr> <dt> <freq> ~ <text>"). The recordings folder is only read; our
decodes go to data/real/<station>/decodes_<tag>.txt.

Comparison: per slot, normalised message text (hashed calls -> <...>, jt9 '?'/'aN' marks removed).
Our-only decodes are checked for corroboration: does any callsign in it appear in jt9's decodes
anywhere in that station's recording? Random false decodes produce random callsigns, so a high
corroboration rate means our extra decodes are real.
"""
import argparse
import os
import re
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.io import wavfile

_LLR = None
_NHYP = 2
_CKW = {}
_EG = 15.0
_EOD = 3.0
_OSN = 2


def _init(model, also_classical, n_hyp, ckw=None, erase_gate=15.0, erase_osd_dmax=3.0, osd_snaps=2):
    global _LLR, _NHYP, _CKW, _EG, _EOD, _OSN
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import torch

    torch.set_num_threads(1)
    from ft8nn import decoder, nn_metric

    _LLR = nn_metric.make_llr_fn(model, also_classical=also_classical) if model else decoder.classical_metrics
    _NHYP = n_hyp
    _CKW = ckw or {}
    _EG = erase_gate
    _EOD = erase_osd_dmax
    _OSN = osd_snaps


def _decode(path):
    from ft8nn import decoder

    sr, x = wavfile.read(path)
    assert sr == 12000, f"{path}: expected 12 kHz, got {sr}"
    x = x.astype(np.float64)
    if x.ndim > 1:
        x = x.mean(1)
    if len(x) < 180000:
        x = np.pad(x, (0, 180000 - len(x)))
    ds = decoder.decode_slot(x[:180000] / (np.std(x) + 1e-9), llr_fn=_LLR, n_hyp=_NHYP, erase_gate=_EG, erase_osd_dmax=_EOD, osd_snaps=_OSN, cand_kw=_CKW)
    return Path(path).stem, [(d.bits77, d.f0, d.dt, d.method, d.info.get("metric"), d.snr) for d in ds]


def norm(t):
    t = re.sub(r"(\s+(\?|a\d))+\s*$", "", t.strip()).strip()  # jt9 may append both: "... ? a1"
    return re.sub(r"<[^>]*>", "<...>", t)


CALL = re.compile(r"^(?=.*\d)(?=.*[A-Z])[A-Z0-9/]{3,11}$")
NOT_CALL = re.compile(r"^(R?[+-]\d\d|RR73|RRR|73|[A-R]{2}\d\d|CQ|DE|QRZ|R|DX|\d+)$")


def calls_in(text):
    return {w for w in text.split() if CALL.match(w) and not NOT_CALL.match(w)}


def load_jt9(path):
    out = defaultdict(list)
    for line in Path(path).read_text().splitlines():
        m = re.match(r"^(\S+)\s+\d{6}\s+(-?\d+)\s+(-?[\d.]+)\s+(\d+)\s+~\s+(.*)$", line)
        if m:
            out[m[1]].append((int(m[2]), float(m[3]), int(m[4]), norm(m[5])))
    return out


def load_ours(path):
    out = defaultdict(list)
    for line in Path(path).read_text().splitlines():
        parts = line.split("\t")  # slot f0 dt method text [hexbits] [snr]
        slot, f0, dt, meth, text = parts[:5]
        snr = float(parts[6]) if len(parts) > 6 else float("nan")
        out[slot].append((float(f0), float(dt), meth, norm(text), snr))
    return out


def compare(station, jt9, ours):
    slots = sorted(set(jt9) | set(ours))
    jt9_calls = set().union(*[calls_in(d[3]) for v in jt9.values() for d in v]) if jt9 else set()
    tot = Counter()
    only_ours, only_jt9 = [], []
    for s in slots:
        a = {d[3] for d in jt9.get(s, [])}
        b = {d[3] for d in ours.get(s, [])}
        osnr = {d[3]: d[4] for d in ours.get(s, [])}
        tot["jt9"] += len(a)
        tot["ours"] += len(b)
        tot["both"] += len(a & b)
        for t in b - a:
            only_ours.append((s, t, osnr[t]))
        snr = {d[3]: d[0] for d in jt9.get(s, [])}
        for t in a - b:
            only_jt9.append((s, t, snr[t]))
    corro = [t for _, t, _ in only_ours if calls_in(t) & jt9_calls]
    nocall = [t for _, t, _ in only_ours if not calls_in(t)]
    return tot, only_ours, only_jt9, corro, nocall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--model", default=None)
    ap.add_argument("--tag", default="nn_v1")
    ap.add_argument("--also-classical", action="store_true")
    ap.add_argument("--n-hyp", type=int, default=2)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--stations", default=None, help="comma-separated prefixes, e.g. 00,01")
    ap.add_argument("--per-symbol", action="store_true", help="per-symbol Costas sync ratio")
    ap.add_argument("--fmax", type=float, default=3100.0)
    ap.add_argument("--base", default="median", choices=["max40", "median"], help="candidate score baseline")
    ap.add_argument("--erase-gate", type=float, default=15.0, help="burst-erasure retry below this expected error count (0 = off)")
    ap.add_argument("--erase-osd-dmax", type=float, default=3.0, help="wide-window OSD erasure stage distance limit (0 = off)")
    ap.add_argument("--osd-snaps", type=int, default=2, help="OSD on BP posterior sums after iterations 1..N (0 = off)")
    ap.add_argument("--out", default="data/real", help="output root: <out>/<station>/decodes_<tag>.txt")
    ap.add_argument("--compare-only", action="store_true")
    ap.add_argument("--show", type=int, default=8, help="examples to print per category")
    a = ap.parse_args()

    from ft8nn import message

    root = Path(a.root)
    stations = sorted(p for p in root.iterdir() if (p / "slots").is_dir())
    if a.stations:
        pre = tuple(a.stations.split(","))
        stations = [s for s in stations if s.name.startswith(pre)]
    outroot = Path(a.out)

    if not a.compare_only:
        with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(a.model, a.also_classical, a.n_hyp, dict(per_symbol=a.per_symbol, fmax=a.fmax, base=a.base), a.erase_gate, a.erase_osd_dmax, a.osd_snaps)) as ex:
            for st in stations:
                files = sorted(str(f) for f in (st / "slots").glob("*.wav"))
                res = dict(ex.map(_decode, files, chunksize=2))
                ht = message.HashTable()  # learn hashes in time order, like a running WSJT-X session
                lines = []
                for slot in sorted(res):
                    for bits, f0, dt, meth, k, snr in res[slot]:
                        hexbits = f"{int(''.join(map(str, bits)), 2):020x}"
                        lines.append(f"{slot}\t{f0:.1f}\t{dt:.2f}\t{meth}{k}\t{message.unpack77(bits, ht)}\t{hexbits}\t{snr:.1f}")
                od = outroot / st.name
                od.mkdir(parents=True, exist_ok=True)
                (od / f"decodes_{a.tag}.txt").write_text("\n".join(lines) + "\n")
                print(f"{st.name}: {len(files)} slots, {len(lines)} decodes", flush=True)

    print(f"\n{'station':40s} {'jt9':>6s} {'ours':>6s} {'both':>6s} {'ours-only':>10s} {'corrob.':>8s} {'jt9-only':>9s}")
    grand = Counter()
    examples_o, examples_j, all_jt9_only, all_ours_only = [], [], [], []
    for st in stations:
        jf, of = st / "decodes.txt", outroot / st.name / f"decodes_{a.tag}.txt"
        if not (jf.exists() and of.exists()):
            print(f"{st.name:40s} {'(no jt9 decodes yet)' if of.exists() else '(not decoded)'}")
            continue
        tot, oo, oj, corro, nocall = compare(st.name, load_jt9(jf), load_ours(of))
        grand.update(tot)
        grand["only_ours"] += len(oo)
        grand["corro"] += len(corro)
        grand["only_jt9"] += len(oj)
        all_jt9_only += [x[2] for x in oj]
        examples_o += [(st.name[:6], t, t in corro) for _, t, _ in oo[: a.show]]
        all_ours_only += [x[2] for x in oo if x[2] == x[2]]
        examples_j += [(st.name[:6], t, snr) for _, t, snr in oj[: a.show]]
        print(f"{st.name:40s} {tot['jt9']:6d} {tot['ours']:6d} {tot['both']:6d} {len(oo):10d} {len(corro):8d} {len(oj):9d}")
    if grand["jt9"]:
        print(f"{'TOTAL':40s} {grand['jt9']:6d} {grand['ours']:6d} {grand['both']:6d} {grand['only_ours']:10d} {grand['corro']:8d} {grand['only_jt9']:9d}")
        print(f"\nours / jt9 = {grand['ours'] / grand['jt9']:.3f};  ours-only corroborated: {grand['corro'] / max(grand['only_ours'], 1):.0%}")
        if all_jt9_only:
            h = Counter(min(max(s // 3 * 3, -27), 0) for s in all_jt9_only)
            print("jt9-only by jt9 SNR: " + "  ".join(f"{k:+d}:{v}" for k, v in sorted(h.items())))
        if all_ours_only:
            h = Counter(min(max(int(s // 3 * 3), -27), 0) for s in all_ours_only)
            print("ours-only by our SNR: " + "  ".join(f"{k:+d}:{v}" for k, v in sorted(h.items())))
        print("\nexamples ours-only (✓ = callsign also heard by jt9):")
        for st, t, c in examples_o[: 3 * a.show]:
            print(f"  {st} {'✓' if c else ' '} {t}")
        print("examples jt9-only (jt9 SNR):")
        for st, t, snr in examples_j[: 3 * a.show]:
            print(f"  {st} {snr:+3d} {t}")


if __name__ == "__main__":
    main()
