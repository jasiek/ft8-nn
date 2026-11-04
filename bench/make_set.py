"""Generate a benchmark set of 15 s FT8 slots with known content.

Each WAV holds K non-overlapping signals at the same SNR. A manifest.json records ground truth.

  uv run python bench/make_set.py --out data/bench/awgn --channel awgn --snrs -26:-17 --files 30
"""
import argparse
import json
from pathlib import Path

import numpy as np

from ft8nn import ft8
from ft8nn.channel import Signal, synth_slot, write_wav

CHANNELS = {
    # name: (fspread Hz, delay ms)
    "awgn": (0.0, 0.0),
    "moderate": (0.5, 1.0),  # ITU-R F.1487-ish mid-latitude moderate
    "disturbed": (1.0, 2.0),
    "polar": (3.0, 2.0),
}


def parse_range(s):
    a, b = (int(v) for v in s.split(":"))
    return list(range(a, b + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--channel", default="awgn", choices=CHANNELS)
    ap.add_argument("--snrs", default="-26:-17")
    ap.add_argument("--files", type=int, default=30, help="files per SNR")
    ap.add_argument("--per-file", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    pool = json.loads(Path("ft8nn/data/messages.json").read_text())
    rng = np.random.default_rng(a.seed)
    fspread, delay = CHANNELS[a.channel]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = []
    idx = 0
    for snr in parse_range(a.snrs):
        for _ in range(a.files):
            idx += 1
            sigs = []
            spacing = 2400 / a.per_file
            for k in range(a.per_file):
                m = pool[rng.integers(len(pool))]
                bits = np.array([int(c) for c in m["bits"]], np.uint8)
                _, tones = ft8.encode77(bits)
                f0 = 300 + spacing * k + rng.uniform(0, spacing - 60)
                dt = rng.uniform(-0.4, 1.0)
                sigs.append(Signal(tones, f0, dt, snr, fspread, delay, meta={"text": m["text"]}))
            name = f"000000_{idx:06d}.wav"
            write_wav(out / name, synth_slot(sigs, rng))
            manifest.append(
                {
                    "file": name,
                    "snr": snr,
                    "channel": a.channel,
                    "signals": [{"text": s.meta["text"], "f0": s.f0, "dt": s.dt} for s in sigs],
                }
            )
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"wrote {idx} files to {out}")


if __name__ == "__main__":
    main()
