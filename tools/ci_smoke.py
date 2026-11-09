"""CI check without the recordings: the C library must decode exactly what the Python decoder does.

Default: the trained model (models/v3), exported to ONNX here, on synthetic slots: random payloads at -24..-8 dB (near the decode limit) in
AWGN, numbered file names (no context a-priori), so BP, OSD, a-priori and subtraction all take part. --model random:
a randomly initialised SoftDemod (fixed seed) instead; its LLRs are meaningless, but the classical fallback metrics
still decode the strong signals. Either way the whole C pipeline (candidates, sync, ONNX Runtime, BP/OSD,
subtraction, unpacking) is compared with Python field by field, as on real data (tools/golden_c.py).

  uv run python tools/ci_smoke.py [--dir data/ci] [--slots 8] [--model models/v3/model.pt|random]

Needs c/build/ft8nn_cli (make -C c). Exit status 0 = C and Python agree and decoded something.
"""
import argparse
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ft8nn import ft8, model  # noqa: E402
from ft8nn.channel import Signal, synth_slot, write_wav  # noqa: E402


def run(*cmd):
    print("+", " ".join(map(str, cmd)), flush=True)
    r = subprocess.run([str(c) for c in cmd], cwd=ROOT, capture_output=True, text=True)
    print(r.stdout[-3000:], r.stderr[-3000:] if r.returncode else "", sep="", flush=True)
    if r.returncode:
        sys.exit(f"failed ({r.returncode}): {cmd[0]} {cmd[1] if len(cmd) > 1 else ''}")
    return r.stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/ci")
    ap.add_argument("--slots", type=int, default=8)
    ap.add_argument("--per-slot", type=int, default=8)
    ap.add_argument("--model", default="models/v3/model.pt", help="checkpoint, or 'random'")
    a = ap.parse_args()
    d = (ROOT / a.dir).resolve()
    slots = d / "rec" / "synthetic" / "slots"
    slots.mkdir(parents=True, exist_ok=True)

    if a.model == "random":
        torch.manual_seed(0)
        ckpt = d / "random.pt"
        torch.save({"state_dict": model.SoftDemod().state_dict(), "arch": {}, "step": 0}, ckpt)
    else:
        ckpt = ROOT / a.model
    onnx = d / "model.onnx"
    run(sys.executable, "tools/export_onnx.py", ckpt, onnx)

    rng = np.random.default_rng(1)
    spacing = 2400 / a.per_slot
    for i in range(a.slots):
        sigs = []
        for k in range(a.per_slot):
            _, tones = ft8.encode77(ft8.random_payloads(1, rng)[0])
            f0 = 300 + spacing * k + rng.uniform(0, spacing - 60)
            sigs.append(Signal(tones, f0, rng.uniform(-0.4, 1.0), rng.uniform(-16, -4) if a.model == "random" else rng.uniform(-24, -8)))
        write_wav(slots / f"000000_{i:06d}.wav", synth_slot(sigs, rng))

    run(sys.executable, "bench/run_real.py", d / "rec", "--model", ckpt, "--also-classical", "--tag", "ci",
        "--workers", "2", "--no-context", "--out", d / "py")
    out = run(sys.executable, "tools/golden_c.py", "real", slots.parent, "--n", a.slots, "--tag", "ci",
              "--onnx", onnx, "--out", d / "py")
    m = re.search(r"py (\d+) decodes, c (\d+), same payload (\d+), all fields equal (\d+)", out)
    if not m or "PASS" not in out:
        sys.exit("C and Python differ")
    if int(m.group(1)) < a.slots * a.per_slot // 3:
        sys.exit(f"only {m.group(1)} decodes of {a.slots * a.per_slot} strong signals: decoder broken?")
    print(f"OK: {m.group(1)} decodes, C == Python in every field")


if __name__ == "__main__":
    main()
