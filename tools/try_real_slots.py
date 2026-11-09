"""Quick check on a few real slots: our decodes vs jt9 (counts + misses with jt9 SNR)."""
import sys, time
sys.path.insert(0, "bench")
import run_real as rr
from ft8nn import message
st = "datasets/ft8-training-data/recordings/" + sys.argv[1]
slots = sys.argv[2].split(",")
rr._init("runs/v1/model.pt", "--cl" in sys.argv, 2)
jt9 = rr.load_jt9(st + "/decodes.txt")
T = [0, 0, 0]
for s in slots:
    t = time.time(); _, ds = rr._decode(f"{st}/slots/{s}.wav"); el = time.time() - t
    ours = {rr.norm(message.unpack77(d[0])) for d in ds}
    j = {d[3]: d[0] for d in jt9[s]}
    T[0] += len(j); T[1] += len(ours); T[2] += len(ours & set(j))
    print(f"{s}: jt9 {len(j)} ours {len(ours)} both {len(ours & set(j))}  {el:.1f}s | ours-only {sorted(ours - set(j))} | jt9-only SNRs {sorted(j[t] for t in set(j) - ours)}")
print("TOTAL jt9 %d ours %d both %d" % tuple(T))
