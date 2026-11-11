"""Stale-repeat check for the repeat stage: an 'rp' decode is wrong if a reference decoder decoded a different
message from the same sender in that slot (the sender moved on; the stage read the last over again).

  python3 tools/rp_stale.py --decodes data/val/run1 --tag nn_v3_rp --ref REC/<station>/decodes.txt [--ref ...]
  (--ref names a decodes file pattern with {st}: e.g. datasets/.../recordings/{st}/decodes.txt)

Counts rp decodes whose (slot, sender) the reference also decoded: same text (agree), different text (stale),
and those the reference did not decode at all. A hashed call printed as '...' (jt9 has no hash table) matches any
call; a hashed sender is not checkable.
"""
import argparse, os, re
from collections import Counter, defaultdict

JT9 = re.compile(r"^(\S+)\s+\d{6}\s+-?\d+\s+-?[\d.]+\s+\d+\s+~\s+(.*)$")
norm = lambda t: re.sub(r"[<>]", "", re.sub(r"(\s+(\?|a\d))+\s*$", "", t.strip()))


def sender(t):
    w = t.split()
    if len(w) < 2:
        return None
    return w[2] if w[0] == "CQ" and len(w) > 3 else w[1]


ap = argparse.ArgumentParser()
ap.add_argument("--decodes", required=True)
ap.add_argument("--tag", default="nn_v3_rp")
ap.add_argument("--ref", action="append", required=True)
ap.add_argument("--show", type=int, default=10)
a = ap.parse_args()
c, ex = Counter(), []
for st in sorted(os.listdir(a.decodes)):
    p = os.path.join(a.decodes, st, f"decodes_{a.tag}.txt")
    if not os.path.exists(p):
        continue
    ref = defaultdict(set)
    for pat in a.ref:
        rp = pat.format(st=st)
        if os.path.exists(rp):
            for line in open(rp):
                m = JT9.match(line)
                if m:
                    t = norm(m[2]); ref[(m[1], sender(t))].add(t)
    for line in open(p):
        f = line.rstrip("\n").split("\t")
        if not f[3].startswith("rp"):
            continue
        t = norm(f[4]); k = (f[0], sender(t))
        if k[1] in (None, "..."):
            c["sender hashed: not checkable"] += 1
            continue
        same = lambda u: len(u.split()) == len(t.split()) and all(
            x == y or "..." in (x, y) for x, y in zip(u.split(), t.split()))
        if k not in ref:
            c["reference has no decode of this sender in the slot"] += 1
        elif any(same(u) for u in ref[k]):
            c["reference decoded the same message"] += 1
        else:
            c["STALE: reference decoded a different message"] += 1
            ex.append(f"  {st[:10]} {f[0]} {f[1]:>7s}  rp: {t:28s} ref: {' | '.join(sorted(ref[k]))}"
                      + (f"  rho {f[7]} / {f[8]}" if len(f) > 8 else ""))
for k, v in c.most_common():
    print(f"{v:6d}  {k}")
print("\n".join(ex[:a.show]))
