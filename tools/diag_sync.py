"""Where do weak signals get lost? candidate search -> fine sync -> NN+LDPC, vs oracle sync."""
import json
import sys

import numpy as np
from scipy.io import wavfile

from ft8nn import frontend as fe
from ft8nn import ft8, ldpc, nn_metric
from ft8nn.frontend import Candidate

snr = int(sys.argv[1]) if len(sys.argv) > 1 else -22
model = sys.argv[2] if len(sys.argv) > 2 else "runs/v1/model_mid.pt"
pool = {m["text"]: m["bits"] for m in json.load(open("ft8nn/data/messages.json"))}
man = [e for e in json.load(open("data/bench/awgn/manifest.json")) if e["snr"] == snr][:20]
llr_fn = nn_metric.make_llr_fn(model)

stats = dict(n=0, cand=0, dec_pipe=0, dec_oracle_cand=0, dec_true_sync=0)
df_err, dt_err = [], []
for e in man:
    _, x = wavfile.read("data/bench/awgn/" + e["file"])
    x = x / 100.0
    cands = fe.find_candidates(x)
    bb = fe.Baseband(x)
    for s in e["signals"]:
        stats["n"] += 1
        cw = ft8.encode77(np.array([int(c) for c in pool[s["text"]]], np.uint8))[0]

        def try_dec(f0, dt, fine):
            if fine:
                f, d, y, start, _ = fe.fine_sync(bb, Candidate(f0, dt, 0))
            else:
                f, d = f0, dt
                y = bb.downsample(f0)
                start = int(round((ft8.NOMINAL_START + dt) * fe.FS_BB))
            (a91, _), = ldpc.decode(llr_fn(None, y, start)[0][None])
            return a91 is not None and (a91 == cw[:91]).all(), f, d

        near = [c for c in cands if abs(c.f0 - s["f0"]) < 4 and abs(c.dt - s["dt"]) < 0.1]
        if near:
            stats["cand"] += 1
            ok, f, d = try_dec(near[0].f0, near[0].dt, True)
            stats["dec_pipe"] += ok
            df_err.append(f - s["f0"]); dt_err.append(d - s["dt"])
        ok, f, d = try_dec(s["f0"] + np.random.uniform(-1.5, 1.5), s["dt"] + np.random.uniform(-0.04, 0.04), True)
        stats["dec_oracle_cand"] += ok
        ok, _, _ = try_dec(s["f0"], s["dt"], False)
        stats["dec_true_sync"] += ok
print(f"SNR {snr}: ", {k: f"{v}/{stats['n']}" for k, v in stats.items()})
df_err, dt_err = np.array(df_err), np.array(dt_err)
print(f"bias: df {df_err.mean():+.3f} Hz, dt {dt_err.mean()*1000:+.2f} ms"); print(f"fine-sync error: df rms {np.sqrt(np.mean(df_err**2)):.3f} Hz (|df|>0.5: {np.mean(abs(df_err)>0.5):.0%}), dt rms {np.sqrt(np.mean(dt_err**2))*1000:.1f} ms (|dt|>10ms: {np.mean(abs(dt_err)>0.01):.0%})")
