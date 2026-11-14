#!/usr/bin/env python3
"""Profile the slot-confirmed signals other decoders get and ours misses, by mechanism (no audio).

    python3 tools/miss_profile.py [--ours nn_v3_dtcrp] [--refs jt9 jtdx jtdx_deep] [--runs 2 3 4] --tsv OUT.tsv

A miss is a (station, slot, sending callsign) pair that at least one reference decoder reports at verify.py tier
'slot' and that ours does not report at any tier - the identity tools/compare_decoders.py counts. Each miss gets the
references that have it (with SNR, DT, frequency, text and, for JTDX, its stage marker) and these flags, all from
saved decodes:
  scorer   we decoded it, but verify.py cannot see it: our text names the sender as a resolved hashed call
           `<CALL>` (verify.py treats any `<` as unchecked), or our text equals the reference's once `<>` is dropped
  dtwin    reference DT outside our search window [-1.5, 2.5] s, shifted by the receiver's median DT when DT
           re-centring would arm (|median| >= 0.8 s over the slot's decodes)
  edge     frequency below 200 Hz or above 3000 Hz (the recordings pass 200-3000 Hz; our search covered 150-3100 Hz until 2026-11-14, now 150-4000)
  cochan   we decode a different sender within 10 Hz and 0.4 s in the same slot (nearest one is recorded);
           cochan_both: the reference also decoded that neighbour's sender in the slot (it got both signals)
  repeat   we decoded the exact text on this receiver 30 or 60 s earlier within 10 Hz (CTX_DF): repeat-stage input
  rep_any  the exact text decoded by us on this receiver at any frequency 30-300 s earlier
  rep_next the exact text decoded by us 30 or 60 s LATER within 10 Hz (a station's first overs, before we have it)
  pair     the same first two words (e.g. `K1ABC W9XYZ`) in our decodes 30/60 s earlier within 10 Hz and 0.5 s:
           context a-priori (bp_cx/osd_cx) would pin them
  sender   the sender in our decodes 30/60 s earlier within 10 Hz and 0.5 s (context a-priori's sender pattern)
  heard    the sender is in our decodes on this receiver at any time in the recording
"""
import argparse, os, re, sys
from collections import Counter, defaultdict
from datetime import datetime

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "collect"))
from verify import sender as v_sender, norm_text  # noqa: E402

V = os.environ.get("V") or os.path.join(REPO, "datasets", "ft8-validation-set")
JT9 = re.compile(r"^(\S+)\s+\d{6}\s+(-?\d+)\s+(-?[\d.]+)\s+(\d+)\s+~\s+(.*)$")
T = lambda s: datetime.strptime(s, "%y%m%d_%H%M%S").timestamp()
strip = lambda t: re.sub(r"[<>]", "", norm_text(t)).strip()
RANK = {"slot": 0, "active": 1, "repeat": 2, "unconf": 3}


def verified(path):
    """({(slot, sender): best tier}, {(slot, freq int, text): (sender, tier)})"""
    pairs, rows = {}, {}
    for line in open(path).read().splitlines()[1:]:
        slot, freq, text, snd, tier = line.split("\t")[:5]
        rows[(slot, round(float(freq)), text.strip())] = (snd, tier)
        if tier in RANK and ((slot, snd) not in pairs or RANK[tier] < RANK[pairs[slot, snd]]):
            pairs[slot, snd] = tier
    return pairs, rows


def ref_lines(path, rows, marks):
    """[(slot, snr, dt, freq, text, sender, tier, marker)] for a jt9-format decodes file."""
    out = []
    for line in open(path).read().splitlines():
        m = JT9.match(line)
        if not m:
            continue
        slot, snr, dt, freq, raw = m[1], int(m[2]), float(m[3]), int(m[4]), m[5]
        text = norm_text(raw)
        snd, tier = rows.get((slot, freq, text), rows.get((slot, freq, raw.strip()), (None, None)))
        out.append((slot, snr, dt, freq, text, snd, tier, marks.get((slot, freq, raw.strip()), marks.get((slot, freq, text), ""))))
    return out


def ours_lines(path):
    out = defaultdict(list)
    for line in open(path).read().splitlines():
        p = line.split("\t")
        out[p[0]].append(dict(f0=float(p[1]), dt=float(p[2]), method=p[3].rstrip("0123456789") or p[3], text=p[4].strip(),
                              snr=float(p[6]) if len(p) > 6 else float("nan")))
    return out


def words2(text):
    w = strip(text).split()
    return tuple(w[:2]) if len(w) >= 2 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["2", "3", "4"])
    ap.add_argument("--ours", default="nn_v3_dtcrp")
    ap.add_argument("--refs", nargs="+", default=["jt9", "jtdx", "jtdx_deep"])
    ap.add_argument("--out", default=os.path.join(REPO, "data", "val"))
    ap.add_argument("--tsv", required=True)
    a = ap.parse_args()

    misses = []
    dt_off, snr_off = [], defaultdict(list)  # reference vs ours on the same decode, for the conventions
    for r in a.runs:
        rec = os.path.join(V, f"run{r}", "recordings")
        for st in sorted(d for d in os.listdir(rec) if os.path.isdir(os.path.join(rec, d, "slots"))):
            d = os.path.join(a.out, f"run{r}", st)
            our_pairs, _ = verified(os.path.join(d, f"verified_{a.ours}.tsv"))
            ours = ours_lines(os.path.join(d, f"decodes_{a.ours}.txt"))
            by_text = defaultdict(list)  # stripped text -> [(t, f0)]
            heard = set()
            for slot, lst in ours.items():
                for o in lst:
                    by_text[strip(o["text"])].append((T(slot), o["f0"]))
                    s = v_sender(strip(o["text"]))
                    if s:
                        heard.add(s)
            refs = {}
            for ref in a.refs:
                if ref == "jt9":
                    vp, dp, mp = (os.path.join(rec, st, "verified_jt9.tsv"), os.path.join(rec, st, "decodes.txt"), None)
                else:
                    vp, dp, mp = (os.path.join(d, f"verified_{ref}.tsv"), os.path.join(d, f"decodes_{ref}.txt"),
                                  os.path.join(d, f"markers_{ref}.tsv"))
                pairs, rows = verified(vp)
                marks = {}
                if mp and os.path.exists(mp):
                    for line in open(mp).read().splitlines():
                        slot, freq, mark, text = line.split("\t")
                        marks[(slot, int(freq), text.strip())] = mark
                refs[ref] = (pairs, ref_lines(dp, rows, marks))
                for slot, snr, dt, freq, text, snd, tier, mark in refs[ref][1]:
                    for o in ours.get(slot, []):
                        if strip(o["text"]) == strip(text) and abs(o["f0"] - freq) < 3:
                            dt_off.append(dt - o["dt"])
                            snr_off[ref].append(snr - o["snr"])
                            break
            by_slot = {x: defaultdict(list) for x in a.refs}
            for x in a.refs:
                for slot, snr, dt, freq, text, snd, tier, mark in refs[x][1]:
                    by_slot[x][slot].append((snr, dt, freq, snd))
            # receiver DT offset per slot as DT re-centring sees it: median of our decoded DTs (this slot)
            med_dt = {s: float(np.median([o["dt"] for o in lst])) for s, lst in ours.items() if len(lst) >= 6}
            got = {}
            for ref in a.refs:
                pairs, lines = refs[ref]
                for slot, snr, dt, freq, text, snd, tier, mark in lines:
                    if tier != "slot" or pairs.get((slot, snd)) != "slot" or (slot, snd) in our_pairs:
                        continue
                    k = (slot, snd)
                    g = got.setdefault(k, dict(run=r, st=st, slot=slot, sender=snd, refs={}))
                    g["refs"].setdefault(ref, (snr, dt, freq, text, mark))
            for (slot, snd), g in got.items():
                # representative line: JTDX defaults, else deep JTDX, else jt9
                rep = next(g["refs"][x] for x in ("jtdx", "jtdx_deep", "jt9") if x in g["refs"])
                snr, dt, freq, text, mark = rep
                t0, stext = T(slot), strip(text)
                lst = ours.get(slot, [])
                # scorer: we have it, verify.py can't score it
                scorer = any(strip(o["text"]) == stext for o in lst) or any(
                    f"<{snd}>" in o["text"] or re.search(rf"<[^>]*{re.escape(snd)}[^>]*>", o["text"]) for o in lst)
                med = med_dt.get(slot, 0.0)
                shift = med if abs(med) >= 0.8 else 0.0
                dtwin = not (-1.5 + shift <= dt <= 2.5 + shift)
                edge = freq < 200 or freq > 3000
                near = [o for o in lst if abs(o["f0"] - freq) <= 10 and abs(o["dt"] - dt) <= 0.4
                        and v_sender(strip(o["text"])) != snd]
                nb = min(near, key=lambda o: abs(o["f0"] - freq)) if near else None
                both, nb_ref_snr = False, ""
                if nb:
                    nsnd = v_sender(strip(nb["text"]))
                    both = any(nsnd and refs[x][0].get((slot, nsnd)) for x in g["refs"])
                    # the neighbour's SNR as the representative reference reports it, if it decoded it
                    rr = next(x for x in ("jtdx", "jtdx_deep", "jt9") if x in g["refs"])
                    for (snr2, dt2, f2, sn2) in by_slot[rr].get(slot, []):
                        if (nsnd and sn2 == nsnd) or abs(f2 - nb["f0"]) < 2:
                            nb_ref_snr = snr2
                            break
                rep_ctx = [(tt, f) for tt, f in by_text.get(stext, []) if t0 - tt in (30, 60) and abs(f - freq) <= 10]
                rep_any = [(tt, f) for tt, f in by_text.get(stext, []) if 30 <= t0 - tt <= 300]
                rep_next = [(tt, f) for tt, f in by_text.get(stext, []) if tt - t0 in (30, 60) and abs(f - freq) <= 10]
                w2 = words2(text)
                ctx = [o for back in (30, 60) for o in ours.get(datetime.fromtimestamp(t0 - back).strftime("%y%m%d_%H%M%S"), [])
                       if abs(o["f0"] - freq) <= 10 and abs(o["dt"] - dt) <= 0.5 + 0.0]
                pair = bool(w2) and any(words2(o["text"]) == w2 for o in ctx)
                sender_ctx = any(v_sender(strip(o["text"])) == snd for o in ctx)
                g.update(snr=snr, dt=dt, freq=freq, text=text, mark=mark, scorer=scorer, dtwin=dtwin, edge=edge,
                         cochan=bool(nb), cochan_both=both,
                         nb_df=(nb["f0"] - freq) if nb else "", nb_ddt=(nb["dt"] - dt) if nb else "",
                         nb_snr=nb["snr"] if nb else "", nb_ref_snr=nb_ref_snr,
                         nb_text=nb["text"] if nb else "", nb_method=nb["method"] if nb else "",
                         repeat=bool(rep_ctx), rep_any=bool(rep_any), rep_next=bool(rep_next), pair=pair, sender_ctx=sender_ctx,
                         heard=snd in heard, med_dt=med)
                misses.append(g)

    print(f"conventions: reference DT - our DT on the same decode: median {np.median(dt_off):+.3f} s (n={len(dt_off)}); "
          + ", ".join(f"{k} SNR - our SNR median {np.median(v):+.1f} dB" for k, v in snr_off.items()))
    cols = ["run", "st", "slot", "sender", "refset", "snr", "dt", "freq", "mark", "text", "scorer", "dtwin", "edge",
            "cochan", "cochan_both", "nb_df", "nb_ddt", "nb_snr", "nb_ref_snr", "nb_method", "nb_text", "repeat",
            "rep_any", "pair", "sender_ctx", "heard", "med_dt", "jt9_snr", "jtdx_snr", "deep_snr", "rep_next"]
    with open(a.tsv, "w") as fh:
        fh.write("\t".join(cols) + "\n")
        for m in misses:
            m["refset"] = "+".join(x for x in a.refs if x in m["refs"])
            for x, c in (("jt9", "jt9_snr"), ("jtdx", "jtdx_snr"), ("jtdx_deep", "deep_snr")):
                m[c] = m["refs"][x][0] if x in m["refs"] else ""
            fh.write("\t".join(str(int(v) if isinstance(v, bool) else (f"{v:.2f}" if isinstance(v, float) else v))
                               for v in (m[c] for c in cols)) + "\n")
    n = len(misses)
    print(f"{n:,} (station, slot, sender) pairs slot-confirmed by at least one of {a.refs}, missed by {a.ours}")
    print("per run:", dict(Counter(m["run"] for m in misses)))
    print("which references have it:", dict(Counter(m["refset"] for m in misses).most_common()))
    for ref in a.refs:
        print(f"  {ref}-only (vs ours) slot-confirmed: {sum(ref in m['refs'] for m in misses):,}")
    for f in ("scorer", "dtwin", "edge", "cochan", "cochan_both", "repeat", "rep_any", "rep_next", "pair", "sender_ctx", "heard"):
        print(f"  flag {f:12s} {sum(m[f] for m in misses):6,}  {100 * sum(m[f] for m in misses) / n:5.1f}%")


if __name__ == "__main__":
    main()
