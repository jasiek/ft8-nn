"""Compare fine-sync accuracy (old vs semi-coherent) and decode rate with NN, vs oracle."""
import json, sys
import numpy as np
from scipy.io import wavfile
from ft8nn import frontend as fe, ft8, ldpc, nn_metric
from ft8nn.frontend import Candidate

snr = int(sys.argv[1]); model = sys.argv[2] if len(sys.argv) > 2 else "runs/v1/model_mid.pt"
pool = {m["text"]: m["bits"] for m in json.load(open("ft8nn/data/messages.json"))}
man = [e for e in json.load(open("data/bench/awgn/manifest.json")) if e["snr"] == snr][:20]
llr_fn = nn_metric.make_llr_fn(model)
rng = np.random.default_rng(0)
err = {"old": [], "new": []}; dec = {"old": 0, "new": 0, "new2": 0, "oracle": 0}; n = 0
def ok(y, start, cw):
    (a91, _), = ldpc.decode(llr_fn(None, y, start)[0][None]); return a91 is not None and (a91 == cw[:91]).all()
for e in man:
    _, x = wavfile.read("data/bench/awgn/" + e["file"]); bb = fe.Baseband(x / 100.0)
    for s in e["signals"]:
        n += 1
        cw = ft8.encode77(np.array([int(c) for c in pool[s["text"]]], np.uint8))[0]
        c = Candidate(s["f0"] + rng.uniform(-1.56, 1.56), s["dt"] + rng.uniform(-0.02, 0.02), 0)  # coarse-grid-like error
        f, d, y, st, _ = fe.fine_sync(bb, c); err["old"].append((f - s["f0"], d - s["dt"])); dec["old"] += ok(y, st, cw)
        hs = fe.fine_sync2(bb, c, n_hyp=2); h = hs[0]
        err["new"].append((h.f0 - s["f0"], h.dt - s["dt"]))
        r = ok(h.y, h.start, cw); dec["new"] += r
        dec["new2"] += r or any(ok(g.y, g.start, cw) for g in hs[1:])
        k0 = (ft8.NOMINAL_START + s["dt"]) * fe.FS_BB
        dec["oracle"] += ok(fe.fractional_shift(bb.downsample(s["f0"]), k0 - round(k0)), int(round(k0)), cw)
print(f"SNR {snr}  decodes/{n}:", dec)
for k, v in err.items():
    v = np.array(v); print(f"  {k}: df rms {np.sqrt((v[:,0]**2).mean()):.3f} Hz  bias {v[:,0].mean():+.3f} | dt rms {np.sqrt((v[:,1]**2).mean())*1e3:.1f} ms  bias {v[:,1].mean()*1e3:+.1f}")
