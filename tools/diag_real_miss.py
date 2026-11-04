"""For each jt9-only decode in a real slot: after our full multi-pass decode + subtraction, is
there a candidate near it in the residual, and does fine sync + NN decode it there?"""
import sys
sys.path.insert(0, "bench")
import numpy as np
from scipy.io import wavfile
import run_real as rr
from ft8nn import frontend as fe, ldpc, nn_metric, message, decoder, ft8
from ft8nn.subtract import subtract

st = "/Users/jps/tmp/ft8-training-data/recordings/" + sys.argv[1]; slot = sys.argv[2]
_, x = wavfile.read(f"{st}/slots/{slot}.wav"); x = x.astype(float); x /= x.std()
fn = nn_metric.make_llr_fn("runs/v1/model.pt")
ds = decoder.decode_slot(x, llr_fn=fn)
ours = {rr.norm(message.unpack77(d.bits77)): d for d in ds}
res_x = x.copy()
for d in ds:
    subtract(res_x, ft8.encode77(d.bits77)[1], d.f0, d.dt)
cands = fe.find_candidates(res_x); bb = fe.Baseband(res_x)
for snr, dt, f, t in sorted(rr.load_jt9(f"{st}/decodes.txt")[slot]):
    if t in ours: continue
    nb = [f"{dd.f0:.0f}/{dd.dt:+.2f}" for dd in ds if abs(dd.f0 - f) < 50]
    near = [(i, c) for i, c in enumerate(cands) if abs(c.f0 - f) < 5 and abs(c.dt - dt) < 0.2]
    out = []
    for i, c in near[:1]:
        for h, (l,) in zip(*(lambda hs: (hs, fn(hs)))(fe.fine_sync2(bb, c, 2))):
            a, _ = ldpc.decode(l[None])[0]
            E = (1 / (1 + np.exp(np.abs(l)))).sum()
            out.append(f"#{i} sc {c.score:.1f} ->{h.f0:.1f}/{h.dt:+.2f} E={E:.0f} {'OK' if a is not None else 'fail'}")
    print(f"{snr:+3d} {dt:+.1f} {f:5d} {t:26s} our decodes within 50Hz: {nb} | {'; '.join(out) if out else 'NO CAND (residual)'}")
