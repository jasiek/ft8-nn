"""Calibrate decoder.estimate_snr against true SNR on a synthetic benchmark set."""
import json, sys
import numpy as np
from scipy.io import wavfile
from ft8nn import decoder, nn_metric

pool = {m["bits"]: m["text"] for m in json.load(open("ft8nn/data/messages.json"))}
man = json.load(open("data/bench/awgn/manifest.json"))
fn = nn_metric.make_llr_fn(sys.argv[1] if len(sys.argv) > 1 else "runs/v1/model.pt")
rows = []
for snr in (-20, -18, -16):
    for e in [e for e in man if e["snr"] == snr][:3]:
        _, x = wavfile.read("data/bench/awgn/" + e["file"])
        for d in decoder.decode_slot(x / 100.0, llr_fn=fn):
            if pool.get(d.key) in {s["text"] for s in e["signals"]}:
                rows.append((snr, d.snr))
r = np.array(rows)
for snr in np.unique(r[:, 0]):
    v = r[r[:, 0] == snr, 1]
    print(f"true {snr:+.0f}: est mean {v.mean():+.2f} sd {v.std():.2f} (n={len(v)})")
print(f"suggested SNR_OFFSET += {-(r[:, 1] - r[:, 0]).mean():+.2f}")
