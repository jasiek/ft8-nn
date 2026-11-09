"""Bank of real receiver noise at the decoder's 200 Hz baseband, for training (synth_bb real_noise_p).

  uv run python tools/make_noise_bank.py REC_ROOT DECODES_ROOT TAG OUT.npy [--per-slot 24] [--workers 10]
  e.g. datasets/ft8-training-data/recordings data/real nn_v2_pl data/noise/train_bank.npy

For every slot under REC_ROOT/<station>/slots: normalise as the decoder does, subtract every decode listed in
DECODES_ROOT/<station>/decodes_<TAG>.txt, then cut --per-slot baseband segments (frontend.Baseband.downsample,
the decoder's own conversion and scaling) at random frequencies 200..2900 Hz. What is left is real receiver
noise, AGC behaviour, carriers and interference, plus the weak signals nobody decoded. Use the training
recordings only: validation runs 1 and 2 stay unseen. Writes OUT (segments, complex64, read by train.py
--noise-bank) and OUT_who.npy ("station/slot" of each segment).
"""
import argparse, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.io import wavfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ft8nn import frontend as fe, ft8  # noqa: E402
from ft8nn.subtract import subtract  # noqa: E402


def work(job):
    path, decodes, per_slot, seed = job
    _, x = wavfile.read(path)
    x = x.astype(float)
    x = np.pad(x, (0, max(0, 180000 - len(x))))[:180000]
    x = x / (np.std(x) + 1e-9)
    for f0, dt, hexbits in decodes:
        bits = np.array([int(c) for c in format(int(hexbits, 16), "077b")], np.uint8)
        subtract(x, ft8.encode77(bits)[1], f0, dt)
    bb = fe.Baseband(x)
    rng = np.random.default_rng(seed)
    f0s = rng.uniform(200, 2900, per_slot)
    return np.stack([bb.downsample(f) for f in f0s]).astype(np.complex64), f0s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rec"); ap.add_argument("dec"); ap.add_argument("tag"); ap.add_argument("out")
    ap.add_argument("--per-slot", type=int, default=24); ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()
    jobs, meta = [], []
    for st in sorted(p for p in Path(a.rec).iterdir() if (p / "slots").is_dir()):
        dec = {}
        for line in (Path(a.dec) / st.name / f"decodes_{a.tag}.txt").read_text().splitlines():
            p = line.split("\t")
            dec.setdefault(p[0], []).append((float(p[1]), float(p[2]), p[5]))
        for w in sorted((st / "slots").glob("*.wav")):
            jobs.append((str(w), dec.get(w.stem, []), a.per_slot, len(jobs)))
            meta.append((st.name, w.stem))
    segs, who = [], []
    with ProcessPoolExecutor(a.workers) as ex:
        for (st, slot), (s, f) in zip(meta, ex.map(work, jobs, chunksize=4)):
            segs.append(s); who += [f"{st}/{slot}"] * len(f)
    y = np.concatenate(segs)
    np.save(a.out, y)
    np.save(a.out.removesuffix(".npy") + "_who.npy", np.array(who))
    print(f"{len(jobs)} slots -> {len(y)} segments of {y.shape[1]} samples, {y.nbytes / 2**20:.0f} MB")


if __name__ == "__main__":
    main()
