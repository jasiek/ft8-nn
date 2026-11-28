"""Score FT8 decodes against PSK Reporter spots captured during the same session.

  python3 tools/verify.py                                 # jt9 baseline (recordings/*/decodes.txt)
  python3 tools/verify.py --decodes ../ft8-nn/data/val --tag nn_v2   # <root>/<station>/decodes_<tag>.txt
  python3 tools/verify.py --tag nn_v2 --decodes ... --list   # also print every unconfirmed decode

Accepts jt9 lines ("<slot> <hhmmss> <snr> <dt> <freq> ~ <text>") and ft8nn tab lines
("<slot>\\t<f0>\\t<dt>\\t<method>\\t<text>[\\t...]").

For each decode the *transmitting* call is extracted (2nd call of a standard message, or the call
after CQ [modifier]). Tiers, strongest first:
  slot    - a reporter more than --near-km (default 50) km from this receiver spotted that call on this band
            in this exact 15 s slot, within FREQ_TOL Hz. Nearer reporters may be decoding our own audio (a
            Kiwi owner's skimmer, an OpenWebRX instance's uploads). Distance from the reporter's `rl` locator
            centre to meta.json lat/lon; reporters without a locator count as far. --near-km 0: any reporter
            (the tiers runs 1-4 were published with)
  xconf   - another of our receivers, >= XCONF_KM away on the same band, decoded the same message in the same
            slot: same normalised text (an unresolved <...> matches any call), within FREQ_TOL Hz after both
            receivers' offsets, DT within XCONF_DT_TOL s of that pair's median DT difference. The confirming
            decode is never a-priori (jt9 a1-a7/? marker; OpenWebRX streams strip the marker, so they never
            confirm CQ), and nothing with K1ABC counts. Confirmers: every receiver of the run (jt9
            decodes.txt) plus --xconf-from run folders recorded at the same time, and their OpenWebRX
            confirmers/ streams. --xconf off: the tiers runs 1-4 were published with
  active  - call spotted on this band within +/- ACTIVE_WIN s (real station, slot/freq not matched)
  repeat  - not spotted, but this decoder heard the same call in >= 2 slots on this receiver
            (random false decodes don't repeat a callsign; usually a station only local reporters hear)
  unconf  - call never spotted on this band during the session, decoded once
  n/a     - no checkable transmitter call (hashed <...> sender, free text, telemetry)
jt9 has a very low false-decode rate, so its 'unconf' share measures PSK Reporter coverage gaps;
a decoder whose 'unconf' share is well above jt9's is producing false decodes.
Writes <station>/verified_<tag>.tsv next to the decodes (with --xconf on, an extra `xconf_n` column: how many
other receivers decoded the same message, whatever the tier).
Legacy scoring (runs 1-4's stored verified_jt9.tsv): --near-km 0 --xconf off.

--set sets/<name>.txt: score only that set's blocks in this run. One block per line, `RUN RECEIVER FIRST_SLOT`
(RUN = the run folder relative to the repo root, e.g. run4 or runs/r05; FIRST_SLOT = the first *scored* slot,
YYMMDD_HHMMSS); `#` starts a comment (the seed line). A block is BLOCK_SLOTS scored slots; the LEAD_IN slots
before it are decoded but never scored. Confirmers (spots, other receivers) are not limited to the set. The
receiver's frequency offset comes from its full jt9 decodes.txt when there is one. Writes
verified_<tag>_<set>.tsv; --run names this run when the folder isn't where the set says (a copy).

xconf-strict (--others TAG[,TAG]): a decode that none of those decoders found on the same receiver is
*exclusive*. It counts only if another receiver >= XCONF_KM away decoded the same message in the same slot within
STRICT_F_TOL Hz and STRICT_DT_TOL s of that (decoder, confirmer) pair's median differences (DT per 6-min window,
because Kiwi timing drifts), outside zero-filled gaps on either side: tier `xconf-strict`. Otherwise
`unconfirmable`, whatever PSK Reporter says (its normal tier stays in the `base_tier` column). The TSV gets
`excl`, `strict_n` and `base_tier` columns.
"""
import argparse, json, math, os, re
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from datetime import datetime, timezone

ROOT = os.path.abspath(os.environ.get("V") or os.path.join(os.path.dirname(__file__), "..", "datasets", "ft8-validation-set"))
FREQ_TOL = 50       # Hz, after removing this receiver's median offset (reporters differ too)
ACTIVE_WIN = 600    # s
CQ_MODS = re.compile(r"^([A-Z]{1,4}|\d{3})$")  # CQ DX / CQ POTA / CQ NA / CQ 123 ...
CALL = re.compile(r"^[A-Z0-9]{1,3}[0-9][A-Z0-9]{0,4}$")


def base_call(c):
    """W1ABC/P -> W1ABC, EA8/DL1XYZ -> DL1XYZ (the part that looks most like a callsign)."""
    c = c.strip("<>").upper()
    parts = [p for p in c.split("/") if CALL.match(p)]
    return max(parts, key=len) if parts else None


def sender(text):
    w = text.split()
    if not w:
        return None
    if w[0] == "CQ":
        rest = w[1:]
        if len(rest) >= 2 and CQ_MODS.match(rest[0]) and not CALL.match(rest[0]):
            rest = rest[1:]
        return base_call(rest[0]) if rest and "<" not in rest[0] else None
    if ";" in text or len(w) < 2 or "<" in w[1]:
        return None  # DXpedition/hashed sender: not checkable
    return base_call(w[1])


def norm_text(t):
    return re.sub(r"(\s+(\?|a\d))+\s*$", "", t.strip()).strip()


def parse_decodes(path):
    out = []
    for line in open(path):
        line = line.rstrip("\n")
        if not line.strip():
            continue
        if "\t" in line:
            p = line.split("\t")
            out.append((p[0], float(p[1]), norm_text(p[4])))
        else:
            m = re.match(r"(\d{6}_\d{6})\s+\d{6}\s+(-?\d+)\s+(-?[\d.]+)\s+(\d+)\s+~\s+(.*)$", line)
            if m:
                out.append((m.group(1), float(m.group(4)), norm_text(m.group(5))))
    return out


def slot_t(slot):
    return datetime.strptime(slot, "%y%m%d_%H%M%S").replace(tzinfo=timezone.utc).timestamp()


def load_spots(path, bands):
    """index[(band, call)] -> sorted list of (t_tx, f_hz, reporter, reporter locator)"""
    idx = defaultdict(list)
    reporters = defaultdict(set)
    with open(path) as fh:
        lines = fh.readlines()
    for line in lines:
        try:
            d = json.loads(line)
        except ValueError:
            continue
        b = d.get("b")
        if b not in bands:
            continue
        c = base_call(d.get("sc", ""))
        if not c:
            continue
        ttx = d.get("t_tx") or (d["t"] - 13) // 15 * 15
        idx[(b, c)].append((int(ttx), d["f"], d.get("rc", ""), d.get("rl") or ""))
        reporters[b].add(d.get("rc", ""))
    for v in idx.values():
        v.sort()
    return idx, reporters


T_TOL = 2           # s; some reporters' clocks are a little off
NEAR_KM = 50        # reporters this close to the receiver don't confirm `slot` (PLAN.md 2a)
XCONF_KM = 50       # an xconf confirmer must be at least this far from the receiver
XCONF_DT_TOL = 0.5  # s around the pair's median DT difference ("DT is similar"; 2b-strict tightens it)
STRICT_F_TOL = 4    # Hz around the (decoder, confirmer) pair's median frequency difference (p99 of jt9 pairs)
STRICT_DT_TOL = 0.2  # s around that pair's median DT difference in a 6-min window (p99 of jt9 pairs, 2-min blocks)
BLOCK_S = 120       # the set's block: 8 slots
EPS = 1e-6          # DT comes in 0.1 s steps: 0.8 - 0.5 is 0.30000000000000004, which must still be "0.3"
BLOCK_SLOTS, LEAD_IN = 8, 2  # scored slots per block; slots before it that are decoded but not scored


def locator_latlon(loc):
    """Centre of a Maidenhead locator of 2-10 characters, or None."""
    loc = (loc or "").strip().upper()
    if len(loc) < 2 or len(loc) % 2 or not re.match(r"^[A-R]{2}([0-9]{2}([A-X]{2}([0-9]{2}([A-X]{2})?)?)?)?$", loc):
        return None
    lon, lat, w, h = -180.0, -90.0, 20.0, 10.0
    lon += (ord(loc[0]) - 65) * w
    lat += (ord(loc[1]) - 65) * h
    for i in range(2, len(loc), 2):
        n = 10 if i % 4 == 2 else 24  # digits split by 10, letters by 24
        w, h = w / n, h / n
        a, b = loc[i], loc[i + 1]
        lon += (int(a) if a.isdigit() else ord(a) - 65) * w
        lat += (int(b) if b.isdigit() else ord(b) - 65) * h
    return lat + h / 2, lon + w / 2


def _jload(path):
    with open(path) as f:
        return json.load(f)


def km(a, b):
    la1, lo1, la2, lo2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 2 * 6371.0 * math.asin(min(1.0, math.sqrt(h)))


def same_slot(idx, band, call, t):
    lst = idx.get((band, call), ())
    return lst[bisect_left(lst, (t - T_TOL,)):bisect_right(lst, (t + T_TOL, float("inf")))]


def freq_offset(idx, band, decs, dial):
    """Median (spot freq - our freq) over same-slot matches: absorbs this Kiwi's calibration error."""
    deltas = []
    for slot, f0, text in decs:
        call = sender(text)
        if call:
            near = [ff - (dial + f0) for _, ff, *_ in same_slot(idx, band, call, slot_t(slot)) if abs(ff - dial - f0) < 400]
            if near:
                deltas.append(min(near, key=abs))
    deltas.sort()
    return deltas[len(deltas) // 2] if len(deltas) >= 5 else 0.0


_LOC = {}


def is_near(rl, rx, near_km):
    """Reporter locator within near_km of the receiver (lat, lon). No locator, or no receiver location: far."""
    if not near_km or not rx:
        return False
    if rl not in _LOC:
        _LOC[rl] = locator_latlon(rl)
    p = _LOC[rl]
    return p is not None and km(p, rx) <= near_km


def classify(idx, band, call, t, f_hz, rx=None, near_km=0):
    """-> (tier, n, near_only): near_only is True when only reporters within near_km matched the slot."""
    lst = idx.get((band, call))
    if not lst:
        return "unconf", 0, False
    lo, hi = bisect_left(lst, (t - T_TOL,)), bisect_right(lst, (t + T_TOL, float("inf")))
    matched = [(r, rl) for tt, ff, r, rl in lst[lo:hi] if abs(ff - f_hz) <= FREQ_TOL]
    hits = {r for r, rl in matched if not is_near(rl, rx, near_km)}
    if hits:
        return "slot", len(hits), False
    near_only = bool(matched)
    lo, hi = bisect_left(lst, (t - ACTIVE_WIN,)), bisect_right(lst, (t + ACTIVE_WIN, float("inf")))
    return ("active", hi - lo, near_only) if hi > lo else ("unconf", 0, near_only)


AP_METHOD = re.compile(r"^rp|_ap\d|_cx")  # ft8-nn stages that use prior information (repeat, a-priori, context)
JTDX_AP_MARKS = "*^"  # JTDX: '*' a-priori, '^' its ft8sd stage (decodes known messages)


def parse_full(path):
    """Like parse_decodes, plus DT and the a-priori flag: [(slot, f0, dt, text, ap)].
    A-priori: jt9 `a1`-`a7`/`?` after the text; ft8-nn (tab) lines whose method is a repeat (`rp`), a-priori
    (`_apN`) or context (`_cx`) stage; JTDX decodes marked `*` or `^` in the markers_<tag>.tsv next to
    decodes_<tag>.txt (tools/jtdx_run.py writes it)."""
    out = []
    with open(path) as fh:
        lines = fh.readlines()
    marks = {}
    mk = re.match(r"decodes_(.+)\.txt$", os.path.basename(path))
    mp = os.path.join(os.path.dirname(path), f"markers_{mk.group(1)}.tsv") if mk else None
    if mp and os.path.exists(mp):
        with open(mp) as fh:
            for l in fh:
                q = l.rstrip("\n").split("\t")
                if len(q) >= 4:
                    marks[(q[0], int(q[1]), " ".join(q[3].split()))] = q[2]
    for line in lines:
        line = line.rstrip("\n")
        if not line.strip():
            continue
        if "\t" in line:
            p = line.split("\t")
            out.append((p[0], float(p[1]), float(p[2]), norm_text(p[4]), bool(AP_METHOD.search(p[3]))))
        else:
            m = re.match(r"(\d{6}_\d{6})\s+\d{6}\s+(-?\d+)\s+(-?[\d.]+)\s+(\d+)\s+~\s+(.*)$", line)
            if m:
                raw = m.group(5).strip()
                t = norm_text(raw)
                ap = t != raw or marks.get((m.group(1), int(m.group(4)), " ".join(raw.split())), "-") in JTDX_AP_MARKS
                out.append((m.group(1), float(m.group(4)), float(m.group(3)), t, ap))
    return out


def site_of(src):
    """A pool source is `<receiver>` (jt9) or `<receiver>#<tag>`: one receiver, possibly several decoders."""
    return src.split("#", 1)[0]


HASHED = "<...>"


def match_text(t):
    """Text for matching: collapsed spaces, resolved <CALL> -> CALL (an unresolved <...> stays)."""
    return " ".join(w if w == HASHED else w.strip("<>") for w in t.split())


def same_message(a, b):
    """Identical, or identical except where either side has an unresolved hash."""
    if a == b:
        return True
    wa, wb = a.split(), b.split()
    if len(wa) != len(wb) or (HASHED not in wa and HASHED not in wb):
        return False
    return all(x == y or HASHED in (x, y) for x, y in zip(wa, wb))


def load_pool(run_dirs, idx_by_run, tags=("jt9",), roots=(), log=print):
    """Confirming decodes of every receiver in run_dirs: {band: {slot: [(src, (lat, lon), f_abs, dt, text)]}}.
    src is the receiver, plus `#<tag>` for decoders other than jt9 (`tags`; decodes_<tag>.txt in the receiver's
    folder or in <root>/<receiver>/ for any of `roots`). f_abs has the receiver's frequency offset (vs that run's
    spots, from its jt9 decodes.txt when present) removed. A-priori and K1ABC decodes, and OpenWebRX CQs, are
    left out."""
    pool = defaultdict(lambda: defaultdict(list))
    sites, slot_sites = {}, defaultdict(lambda: defaultdict(set))  # slot_sites: who was listening, per slot
    gap_slots = defaultdict(set)  # site -> slots with zero-filled audio (kiwi_info.json "gaps")
    for run in run_dirs:
        idx = idx_by_run[run]
        rec = os.path.join(run, "recordings")
        for st in sorted(os.listdir(rec)) if os.path.isdir(rec) else []:
            mp = os.path.join(rec, st, "meta.json")
            if not os.path.exists(mp):
                continue
            srcs = []
            for tag in tags:
                cands = ([os.path.join(rec, st, "decodes.txt")] if tag == "jt9" else
                         [os.path.join(d, st, f"decodes_{tag}.txt") for d in (rec,) + tuple(roots)])
                dp = next((c for c in cands if os.path.exists(c)), None)
                if dp:
                    srcs.append((tag, parse_full(dp)))
            if not srcs:
                continue
            m = _jload(mp)
            ll = (m["lat"], m["lon"]) if m.get("lat") is not None else None
            dial = m["freq_khz"] * 1000
            decs = dict(srcs).get("jt9") or srcs[0][1]
            dial += freq_offset(idx, m["band"], [(sl, f0, t) for sl, f0, _, t, _ in decs], dial)
            site = os.path.relpath(os.path.join(rec, st), ROOT)
            sites[site] = (m["band"], ll)
            ki = os.path.join(rec, st, "kiwi_info.json")
            for arrival, fill in (_jload(ki).get("gaps") or []) if os.path.exists(ki) else []:
                for t in range(int((arrival - fill / 12000.0 - 1) // 15) * 15, int(arrival) + 1, 15):
                    gap_slots[site].add(datetime.fromtimestamp(t, timezone.utc).strftime("%y%m%d_%H%M%S"))
            for tag, decs in srcs:
                src = site if tag == "jt9" else f"{site}#{tag}"
                for sl, f0, dt, t, ap in decs:
                    slot_sites[m["band"]][sl].add(site)
                    if not ap and "K1ABC" not in t:
                        pool[m["band"]][sl].append((src, ll, dial + f0, dt, match_text(t)))
        conf = os.path.join(run, "confirmers")
        for st in sorted(os.listdir(conf)) if os.path.isdir(conf) else []:
            mp, dp = os.path.join(conf, st, "meta.json"), os.path.join(conf, st, "owrx_decodes.jsonl")
            if not (os.path.exists(mp) and os.path.exists(dp)):
                continue
            m = _jload(mp)
            g = m.get("lat_lon") or {}
            ll = (g["lat"], g["lon"]) if g.get("lat") is not None else None
            with open(dp) as fh:
                recs = [json.loads(l) for l in fh]
            recs = [r for r in recs if r.get("slot") and r.get("freq") and r.get("msg")]
            dial = m["dial_hz"]
            off = freq_offset(idx, m["band"], [(r["slot"], r["freq"] - dial, norm_text(r["msg"])) for r in recs], dial)
            site = os.path.relpath(os.path.join(conf, st), ROOT)
            sites[site] = (m["band"], ll)
            for r in recs:
                slot_sites[m["band"]][r["slot"]].add(site)
                t = norm_text(r["msg"])
                if not t.startswith("CQ ") and "K1ABC" not in t:
                    pool[m["band"]][r["slot"]].append((site, ll, r["freq"] + off, r.get("dt") or 0.0, match_text(t)))
    log(f"xconf pool: {len(sites)} receivers in {len(run_dirs)} run folder(s), "
        f"{sum(len(v) for b in pool.values() for v in b.values())} confirming decodes")
    return pool, sites, slot_sites, gap_slots


def strict_counts(pool, band, me, rx, decs, dial, excl, gap_slots):
    """Distinct strict confirmers for each decode with excl[i] True (others: 0). Pair offsets come from all of
    this decoder's matches with that confirmer (decoders differ in DT and frequency conventions)."""
    matches = []  # (i, site, df, ddt, block)
    for i, (slot, f0, dt, text, _) in enumerate(decs):
        if "K1ABC" in text:
            continue
        mt = match_text(text)
        for src, ll, f_abs, cdt, ct in pool.get(band, {}).get(slot, ()):
            if site_of(src) == me or abs(f_abs - (dial + f0)) > FREQ_TOL + EPS or not same_message(mt, ct):
                continue
            if not rx or not ll or km(rx, ll) < XCONF_KM:
                continue
            matches.append((i, src, f_abs - (dial + f0), cdt - dt, int(slot_t(slot) // BLOCK_S)))
    fmed, dblk = defaultdict(list), defaultdict(list)
    for i, src, df, ddt, b in matches:
        fmed[src].append(df)
        dblk[(src, b)].append(ddt)
    fmed = {k: sorted(v)[len(v) // 2] for k, v in fmed.items()}
    by = defaultdict(lambda: defaultdict(list))
    for (site, b), ds in dblk.items():
        by[site][b] = ds

    def dt_ref(site, b):
        return window_median(by[site], b)

    out = [set() for _ in decs]
    for i, src, df, ddt, b in matches:  # offsets per source (receiver + decoder); count receivers
        slot = decs[i][0]
        if not excl[i] or slot in gap_slots.get(me, ()) or slot in gap_slots.get(site_of(src), ()):
            continue
        if abs(df - fmed[src]) <= STRICT_F_TOL + EPS and abs(ddt - dt_ref(src, b)) <= STRICT_DT_TOL + EPS:
            out[i].add(site_of(src))
    return [len(x) for x in out]


def exclusive_flags(decs, others):
    """True for each decode that no other decoder's decode list has (same slot, same message)."""
    by_slot = defaultdict(list)
    for o in others:
        for sl, _, _, t, _ in o:
            by_slot[sl].append(match_text(t))
    return [not any(same_message(match_text(t), ot) for ot in by_slot.get(sl, ())) for sl, _, _, t, _ in decs]


def xconf_candidates(pool, band, me, rx, decs, dial):
    """For each decode index: [(site, dt_diff, block)] of other receivers >= XCONF_KM away with the same message
    in the same slot within FREQ_TOL Hz. DT isn't checked yet (see xconf_counts)."""
    out = []
    for slot, f0, dt, text, _ in decs:
        hits = []
        if "K1ABC" not in text:
            mt = match_text(text)
            for src, ll, f_abs, cdt, ct in pool.get(band, {}).get(slot, ()):
                if site_of(src) == me or abs(f_abs - (dial + f0)) > FREQ_TOL + EPS or not same_message(mt, ct):
                    continue
                if not rx or not ll or km(rx, ll) < XCONF_KM:
                    continue  # unknown or too close: could be the same antenna, or the same audio
                hits.append((src, cdt - dt, int(slot_t(slot) // BLOCK_S)))
        out.append(hits)
    return out


def window_median(by_block, b):
    """Median over blocks b-1..b+1 (6 min); the whole series if that window has < 5 values."""
    w = [d for bb in (b - 1, b, b + 1) for d in by_block.get(bb, ())]
    if len(w) < 5:
        w = [d for ds in by_block.values() for d in ds]
    return sorted(w)[len(w) // 2]


def xconf_counts(cands):
    """Distinct confirming receivers per decode, keeping a hit only if its DT difference is within
    XCONF_DT_TOL of that receiver pair's median DT difference over a 6-min window (Kiwi timing drifts)."""
    by = defaultdict(lambda: defaultdict(list))
    for hits in cands:
        for site, d, b in hits:
            by[site][b].append(d)
    return [len({site_of(src) for src, d, b in hits if abs(d - window_median(by[src], b)) <= XCONF_DT_TOL + EPS})
            for hits in cands]


def load_set(path):
    """sets/<name>.txt -> {(run, receiver): set(scored slot names)}."""
    out = defaultdict(set)
    for n, line in enumerate(open(path), 1):
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        w = line.split()
        if len(w) != 3 or not re.match(r"^\d{6}_\d{6}$", w[2]):
            raise SystemExit(f"{path}:{n}: expected `RUN RECEIVER YYMMDD_HHMMSS`, got {line!r}")
        run = os.path.normpath(w[0])
        t0 = slot_t(w[2])
        if t0 % 15:
            raise SystemExit(f"{path}:{n}: {w[2]} isn't on a 15 s slot boundary")
        for k in range(BLOCK_SLOTS):
            out[(run, w[1])].add(datetime.fromtimestamp(t0 + 15 * k, timezone.utc).strftime("%y%m%d_%H%M%S"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recordings", default=os.path.join(ROOT, "recordings"))
    ap.add_argument("--spots", default=os.path.join(ROOT, "pskr_spots.jsonl"))
    ap.add_argument("--decodes", help="root holding <station>/decodes_<tag>.txt (default: recordings, jt9)")
    ap.add_argument("--tag", default="jt9")
    ap.add_argument("--list", action="store_true", help="print unconfirmed decodes")
    ap.add_argument("--xconf", choices=("on", "off"), default="on", help="score the xconf tier (default on)")
    ap.add_argument("--pool-tags", default="jt9", metavar="TAG[,TAG]",
                    help="decoders whose decodes on the other receivers can confirm (default jt9 = decodes.txt; "
                         "others: decodes_<TAG>.txt in the receiver folder or the --decodes root)")
    ap.add_argument("--xconf-from", action="append", default=[], metavar="RUN_DIR",
                    help="another run folder recorded at the same time (recordings/, confirmers/, "
                         "pskr_spots.jsonl); repeatable. The run being scored is always included")
    ap.add_argument("--others", default="", metavar="TAG[,TAG]",
                    help="other decoders on the same receivers (jt9 = decodes.txt, else decodes_<TAG>.txt in the "
                         "--decodes root, then in --recordings): enables xconf-strict for exclusive decodes")
    ap.add_argument("--set", help="sets/<name>.txt: score only these blocks (see above)")
    ap.add_argument("--run", help="this run's name in the set file (default: the run folder relative to the repo)")
    ap.add_argument("--near-km", type=float, default=NEAR_KM,
                    help=f"reporters this close don't confirm `slot` (default {NEAR_KM}; 0 = any reporter)")
    a = ap.parse_args()

    stations = sorted(d for d in os.listdir(a.recordings) if os.path.isdir(os.path.join(a.recordings, d)))
    scored = None
    if a.set:
        run_name = os.path.normpath(a.run or os.path.relpath(os.path.dirname(os.path.abspath(a.recordings)), ROOT))
        blocks = load_set(a.set)
        scored = {rcv: sl for (run, rcv), sl in blocks.items() if run == run_name}
        missing = sorted(set(scored) - set(stations))
        if missing:
            raise SystemExit(f"--set: receivers not in {a.recordings}: {', '.join(missing)}")
        stations = [st for st in stations if st in scored]
        for st in stations:  # a block outside the recording would silently score nothing
            sd = os.path.join(a.recordings, st, "slots")
            if os.path.isdir(sd):
                have = {f[:13] for f in os.listdir(sd) if f.endswith((".wav", ".flac"))}
                lost = sorted(scored[st] - have)
                if lost:
                    print(f"warning: {st}: {len(lost)} scored slot(s) not recorded, e.g. {lost[0]}")
        set_name = os.path.splitext(os.path.basename(a.set))[0]
        print(f"set {set_name}: {sum(len(v) for v in scored.values()) // BLOCK_SLOTS} blocks in run {run_name} "
              f"({len(stations)} receivers; {sum(len(v) for v in blocks.values()) // BLOCK_SLOTS} in the whole set)")
    metas = {s: _jload(os.path.join(a.recordings, s, "meta.json")) for s in stations}
    idx, reporters = load_spots(a.spots, {m["band"] for m in metas.values()})
    print(f"spots indexed: {sum(map(len, idx.values()))} for {len(idx)} (band, call) pairs; "
          f"reporters per band: " + ", ".join(f"{b}={len(r)}" for b, r in sorted(reporters.items())))
    xc = a.xconf == "on"
    if xc:
        own = os.path.dirname(os.path.abspath(a.recordings))
        runs = [own] + [os.path.abspath(r) for r in a.xconf_from if os.path.abspath(r) != own]
        idx_by_run = {own: idx}
        for r in runs[1:]:
            rm = [_jload(os.path.join(r, "recordings", d, "meta.json"))
                  for d in os.listdir(os.path.join(r, "recordings")) if os.path.exists(os.path.join(r, "recordings", d, "meta.json"))]
            cm = [_jload(os.path.join(r, "confirmers", d, "meta.json"))
                  for d in (os.listdir(os.path.join(r, "confirmers")) if os.path.isdir(os.path.join(r, "confirmers")) else [])
                  if os.path.exists(os.path.join(r, "confirmers", d, "meta.json"))]
            idx_by_run[r] = load_spots(os.path.join(r, "pskr_spots.jsonl"), {m["band"] for m in rm + cm})[0]
        cm = [_jload(os.path.join(own, "confirmers", d, "meta.json"))
              for d in (os.listdir(os.path.join(own, "confirmers")) if os.path.isdir(os.path.join(own, "confirmers")) else [])
              if os.path.exists(os.path.join(own, "confirmers", d, "meta.json"))]
        if {m["band"] for m in cm} - {m["band"] for m in metas.values()}:
            idx_by_run[own] = load_spots(a.spots, {m["band"] for m in list(metas.values()) + cm})[0]
        pool, sites, slot_sites, gap_slots = load_pool(runs, idx_by_run, tags=tuple(t for t in a.pool_tags.split(",") if t),
                                                       roots=(a.decodes,) if a.decodes else ())

    tiers = ["slot", "xconf", "active", "repeat", "unconf", "n/a"] if xc else ["slot", "active", "repeat", "unconf", "n/a"]
    others = [t for t in a.others.split(",") if t]
    if others and not xc:
        ap.error("--others needs --xconf on")
    if others:
        tiers = ["xconf-strict", "unconfirmable"] + tiers
    estat = Counter()
    print(f"\n{'station':42s} {'band':>4s} {'dec':>5s} " + " ".join(f"{t:>7s}" for t in tiers) + "  conf%")
    grand, unconf_list, offsets, near_only = Counter(), [], {}, 0
    xstat = Counter()  # over decodes on receivers that have an xconf partner on their band
    for s in stations:
        m = metas[s]
        droot = a.decodes or a.recordings
        path = os.path.join(droot, s, "decodes.txt" if a.tag == "jt9" and not a.decodes else f"decodes_{a.tag}.txt")
        if not os.path.exists(path):
            print(f"{s:42s} (no {os.path.relpath(path, ROOT)})")
            continue
        decs_full = parse_full(path)
        decs = [(sl, f0, t) for sl, f0, _, t, _ in decs_full]
        off_decs = decs
        if scored is not None:
            base = os.path.join(a.recordings, s, "decodes.txt")  # the full jt9 baseline: enough matches for an offset
            if os.path.exists(base):
                off_decs = [(sl, f0, t) for sl, f0, _, t, _ in parse_full(base)]
        rx = (m["lat"], m["lon"]) if m.get("lat") is not None and m.get("lon") is not None else None
        if a.near_km and not rx:
            print(f"{s:42s} (no lat/lon in meta.json: the {a.near_km:g} km rule is off for it)")
        dial = m["freq_khz"] * 1000
        off = freq_offset(idx, m["band"], off_decs, dial)
        dial += off
        offsets[s] = off
        if xc:
            me = os.path.relpath(os.path.join(a.recordings, s), ROOT)  # this receiver, whichever decoder is scored
            xn = xconf_counts(xconf_candidates(pool, m["band"], me, rx, decs_full, dial))
            if others:
                odecs = []
                for t in others:
                    cands = ([os.path.join(a.recordings, s, "decodes.txt")] if t == "jt9" else
                             [os.path.join(r, s, f"decodes_{t}.txt") for r in (a.decodes, a.recordings) if r])
                    op = next((c for c in cands if os.path.exists(c)), None)
                    if op is None:
                        ap.error(f"--others {t}: no decodes for {s} (tried {', '.join(cands)})")
                    odecs.append(parse_full(op))
                excl = exclusive_flags(decs_full, odecs)
                sn = strict_counts(pool, m["band"], me, rx, decs_full, dial, excl, gap_slots)
            far = {site for site, (b, ll) in sites.items()
                   if b == m["band"] and site != me and rx and ll and km(rx, ll) >= XCONF_KM}
        cnt, rows, seen = Counter(), [], set()
        slots_by_call = defaultdict(set)
        for slot, f0, text in decs:
            c = sender(text)
            if c:
                slots_by_call[c].add(slot)
        for i, (slot, f0, text) in enumerate(decs):
            if (slot, text) in seen:  # some decoders emit the same message twice per slot
                continue
            if scored is not None and slot not in scored[s]:
                continue  # lead-in or outside the set's blocks
            seen.add((slot, text))
            call = sender(text)
            tier, n, near = ("n/a", 0, False) if not call else classify(idx, m["band"], call, slot_t(slot),
                                                                          dial + f0, rx, a.near_km)
            near_only += near
            x = xn[i] if xc else 0
            if xc and x and tier in ("active", "repeat", "unconf"):
                tier, n = "xconf", x
            if tier == "unconf" and len(slots_by_call[call]) >= 2:
                tier, n = "repeat", len(slots_by_call[call])
            base = tier
            if others and excl[i] and tier != "n/a":
                tier, n = ("xconf-strict", sn[i]) if sn[i] else ("unconfirmable", 0)
                estat[(tier, base)] += 1
                if far & slot_sites[m["band"]].get(slot, set()):
                    estat[("shared", tier)] += 1
            if xc and far & slot_sites[m["band"]].get(slot, set()):  # a partner was listening in this slot
                xstat["dec"] += 1
                xstat["x"] += x > 0
                if tier != "n/a":
                    xstat["chk"] += 1
                    xstat["conf"] += tier in ("slot", "xconf")
                    xstat["slot"] += tier == "slot"
            cnt[tier] += 1
            rows.append(f"{slot}\t{f0:.1f}\t{text}\t{call or ''}\t{tier}\t{n}" + (f"\t{x}" if xc else "")
                        + (f"\t{int(excl[i])}\t{sn[i]}\t{base}" if others else ""))
            if tier == "unconf":
                unconf_list.append((s, slot, f0, text))
        out_name = f"verified_{a.tag}_{set_name}.tsv" if scored is not None else f"verified_{a.tag}.tsv"
        with open(os.path.join(os.path.dirname(path), out_name), "w") as fh:
            fh.write("slot\tfreq\ttext\tsender\ttier\tn_reporters_or_spots" + ("\txconf_n" if xc else "")
                     + ("\texcl\tstrict_n\tbase_tier" if others else "") + "\n"
                     + "\n".join(rows) + "\n")
        tot = sum(cnt.values())
        chk = tot - cnt["n/a"]
        conf = (chk - cnt["unconf"]) / chk if chk else 0
        print(f"{s:42s} {m['band']:>4s} {tot:5d} " + " ".join(f"{cnt[t]:7d}" for t in tiers) + f"  {conf:5.1%}")
        grand.update(cnt)
    tot = sum(grand.values())
    chk = tot - grand["n/a"]
    print(f"{'TOTAL':42s} {'':4s} {tot:5d} " + " ".join(f"{grand[t]:7d}" for t in tiers)
          + f"  {(chk - grand['unconf']) / max(chk, 1):5.1%}")
    print(f"\nof checkable decodes: slot-confirmed {grand['slot'] / max(chk, 1):.1%}, "
          + (f"xconf-only {grand['xconf'] / max(chk, 1):.1%}, " if xc else "")
          + f"active-only {grand['active'] / max(chk, 1):.1%}, repeat-only {grand['repeat'] / max(chk, 1):.1%}, unconfirmed {grand['unconf'] / max(chk, 1):.1%}")
    if xc and xstat["dec"]:
        print(f"in slots shared with a receiver >= {XCONF_KM} km away: {xstat['dec']} decodes, cross-confirmed {xstat['x']} "
              f"({xstat['x'] / xstat['dec']:.1%}); of {xstat['chk']} checkable: slot {xstat['slot']} ({xstat['slot'] / xstat['chk']:.1%}), "
              f"confirmed (slot or xconf) {xstat['conf']} ({xstat['conf'] / xstat['chk']:.1%})")
    if others:
        ne = sum(v for (t, _), v in estat.items() if t != "shared")
        print(f"exclusive decodes (not found by {', '.join(others)}): {ne}; xconf-strict "
              f"{sum(v for (t, _), v in estat.items() if t == 'xconf-strict')}, unconfirmable "
              f"{sum(v for (t, _), v in estat.items() if t == 'unconfirmable')}")
        sh = estat[("shared", "xconf-strict")] + estat[("shared", "unconfirmable")]
        if sh:
            print(f"  in slots a receiver >= {XCONF_KM} km away was listening: {sh}, xconf-strict "
                  f"{estat[('shared', 'xconf-strict')]} ({estat[('shared', 'xconf-strict')] / sh:.1%})")
        for t in ("xconf-strict", "unconfirmable"):
            print(f"  {t:13s} by base tier: " + ", ".join(f"{b} {estat[(t, b)]}" for b in
                                                          ("slot", "xconf", "active", "repeat", "unconf") if estat[(t, b)]))
    if a.near_km:
        print(f"slot matches only from reporters within {a.near_km:g} km (not counted as `slot`): {near_only}")
    print("per-receiver freq offset vs spots (Hz): " + ", ".join(f"{k[:2]}={v:+.0f}" for k, v in offsets.items()))
    if a.list:
        print("\nunconfirmed:")
        for s, slot, f0, text in unconf_list:
            print(f"  {s[:30]:30s} {slot} {f0:7.1f}  {text}")


if __name__ == "__main__":
    main()
