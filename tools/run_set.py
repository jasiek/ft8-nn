#!/usr/bin/env python3
"""Decode and score a validation block set (sets/{small,third,full}.txt) across decoders.

Each block in sets/<name>.txt is 2 lead-in slots + 8 scored slots of one receiver,
decoded in chronological order as its own short sequence so stateful decoders
(ft8-nn context/Chase/hashes, JTDX message memory/hashes) start identically across
set sizes. Only the 8 scored slots' decodes are written to:
  <out>/<run>/<receiver>/decodes_<tag>.txt   (+ markers_<tag>.tsv for JTDX)

Examples:
  python3 tools/run_set.py --set small --decoder ft8nn --tag nn_v4 --verify
  python3 tools/run_set.py --set small --decoder jt9 --tag jt9 --verify
  python3 tools/run_set.py --set small --decoder jtdx --tag jtdx --verify
  python3 tools/run_set.py --set small --decoder jtdx_deep --tag jtdx_deep --verify
  python3 tools/run_set.py --set small --decoder ft8lib --tag ft8lib --verify
  python3 tools/run_set.py --set small --verify-only --tag nn_v4 \
      --others jt9,jtdx,jtdx_deep,ft8lib --pool-tags jt9,nn_v4,jtdx,jtdx_deep,ft8lib
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V = os.path.abspath(os.environ.get("V") or os.path.join(REPO, "datasets", "ft8-validation-set"))
BLOCK, LEAD = 8, 2

FT8LIB_LINE = re.compile(r"^\d{6}\s+([+-][\d.]+)\s+([+-][\d.]+)\s+(\d+)\s+~\s+(.*?)\s*$")


def resolve_set(spec):
    if os.path.exists(spec):
        return os.path.abspath(spec)
    cand = os.path.join(V, "sets", spec if spec.endswith(".txt") else f"{spec}.txt")
    if os.path.exists(cand):
        return cand
    sys.exit(f"set file not found: {spec} (tried {cand})")


def slot_t(slot):
    return int(datetime.strptime(slot, "%y%m%d_%H%M%S").replace(tzinfo=timezone.utc).timestamp())


def slot_names(first):
    t0 = slot_t(first)
    return [datetime.fromtimestamp(t0 + 15 * k, timezone.utc).strftime("%y%m%d_%H%M%S")
            for k in range(-LEAD, BLOCK)]


def read_set(path):
    out = []
    for n, line in enumerate(open(path), 1):
        w = line.split("#", 1)[0].split()
        if not w:
            continue
        if len(w) != 3 or not re.match(r"^\d{6}_\d{6}$", w[2]):
            sys.exit(f"{path}:{n}: expected `RUN RECEIVER YYMMDD_HHMMSS`")
        out.append((os.path.normpath(w[0]), w[1], w[2]))
    return out


def find_jt9(explicit=None):
    if explicit:
        return explicit
    for p in [shutil.which("jt9"),
              os.path.expanduser("~/wsjtx-prefix/build/jt9"),
              "/usr/local/bin/jt9",
              "/Applications/wsjtx.app/Contents/MacOS/jt9"]:
        if p and os.access(p, os.X_OK):
            return p
    sys.exit("jt9 executable not found; pass --exe /path/to/jt9 or add it to PATH")


def parse_jtdx(raw):
    if len(raw) < 24 or raw[21] != "~" or not raw[:6].strip().isdigit():
        return None
    try:
        return (raw[:6], int(raw[6:10]), float(raw[10:15]), int(raw[15:20]),
                raw[23:49].strip(), raw[49:50].strip())
    except ValueError:
        return None


def decode_block_cli(args):
    decoder, job, exe, model, extra = args
    run, rcv, first = job
    sd = os.path.join(V, run, "recordings", rcv, "slots")
    names = slot_names(first)
    scored_set = set(names[LEAD:])
    wavs = [os.path.join(sd, s + ".wav") for s in names]
    for s, w in zip(names, wavs):
        if not os.path.exists(w):
            sys.exit(f"{run} {rcv} {first}: slot {s} missing ({w}); unpack validation set first")

    t0 = time.time()
    scored_lines, scored_marks = [], []

    if decoder == "ft8nn":
        cmd = [exe, "--threads", "1"] + list(extra) + [model] + wavs
        p = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if p.returncode != 0:
            sys.exit(f"ft8nn_cli failed ({p.returncode}) on {run} {rcv} {first}: {p.stderr[-400:]}")
        for raw in p.stdout.splitlines():
            if not raw.strip():
                continue
            sl = raw.split("\t", 1)[0]
            if sl in scored_set:
                scored_lines.append(raw)

    elif decoder == "jt9":
        for k, (s, w) in enumerate(zip(names, wavs)):
            with tempfile.TemporaryDirectory() as td:
                p = subprocess.run([exe, "-8", "-d", "3", "-a", td, "-t", td, w],
                                   cwd=td, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=180)
            if p.returncode != 0:
                sys.exit(f"jt9 failed ({p.returncode}) on {w}: {p.stderr[-400:]}")
            if k >= LEAD:
                for l in p.stdout.splitlines():
                    l = l.rstrip()
                    if l.strip() and not l.startswith("<Decode") and "~  K1ABC " not in l:
                        scored_lines.append(f"{s} {l}")

    elif decoder in ("jtdx", "jtdx_deep"):
        jextra = (["-w", "-c", "3", "-s", "2"] if decoder == "jtdx_deep" else []) + list(extra)
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = "1"
        with tempfile.TemporaryDirectory() as td:
            p = subprocess.run([exe] + jextra + wavs, cwd=td, env=env, text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if p.returncode != 0:
            sys.exit(f"jtdx_ft8_file failed ({p.returncode}) on {run} {rcv} {first}: {p.stderr[-400:]}")
        cur_slot = None
        for raw in p.stdout.splitlines():
            if raw.startswith("#F "):
                cur_slot = os.path.basename(raw[3:].strip())[:-4]
                continue
            d = parse_jtdx(raw)
            if d is None or cur_slot not in scored_set:
                continue
            hhmmss, snr, dt, freq, msg, marker = d
            if not msg:
                continue
            scored_lines.append(f"{cur_slot} {hhmmss} {snr:3d} {dt:4.1f} {freq:4d} ~  {msg}")
            scored_marks.append(f"{cur_slot}\t{freq}\t{marker or '-'}\t{msg}")

    elif decoder == "ft8lib":
        for k, (s, w) in enumerate(zip(names, wavs)):
            hhmmss = s[7:13]
            p = subprocess.run([exe, w], text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            if p.returncode != 0:
                sys.exit(f"decode_ft8 failed ({p.returncode}) on {w}")
            if k >= LEAD:
                for raw in p.stdout.splitlines():
                    m = FT8LIB_LINE.match(raw)
                    if m and m.group(4):
                        snr, dt, freq, msg = m.groups()
                        scored_lines.append(f"{s} {hhmmss} {round(float(snr)):3d} {float(dt):4.1f} {int(freq):4d} ~  {msg}")

    dt_s = time.time() - t0
    info = dict(run=run, receiver=rcv, first=first, decodes=len(scored_lines), seconds=round(dt_s, 2))
    return info, scored_lines, scored_marks


def run_ft8nn_py(jobs, out_root, tag, model, workers, extra):
    """Run Python bench/run_real.py on each block laid out as a temporary 10-slot station folder."""
    t0 = time.time()
    with tempfile.TemporaryDirectory(prefix="ft8nn_set_") as td:
        in_dir = os.path.join(td, "in")
        out_dir = os.path.join(td, "out")
        block_map = {}
        for idx, (run, rcv, first) in enumerate(jobs):
            bname = f"{idx:04d}_{rcv}_{first}"
            sdir = os.path.join(in_dir, bname, "slots")
            os.makedirs(sdir, exist_ok=True)
            names = slot_names(first)
            for s in names:
                src = os.path.join(V, run, "recordings", rcv, "slots", s + ".wav")
                if not os.path.exists(src):
                    sys.exit(f"{run} {rcv} {first}: missing {src}")
                os.symlink(src, os.path.join(sdir, s + ".wav"))
            block_map[bname] = (run, rcv, first, set(names[LEAD:]))
        cmd = [sys.executable, os.path.join(REPO, "bench", "run_real.py"), in_dir,
               "--model", model, "--also-classical", "--tag", tag,
               "--workers", str(workers), "--out", out_dir] + list(extra)
        p = subprocess.run(cmd, cwd=REPO)
        if p.returncode != 0:
            sys.exit(p.returncode)
        wall = time.time() - t0
        by_rx = {}
        blocks = []
        for bname, (run, rcv, first, scored_set) in sorted(block_map.items()):
            dec_path = os.path.join(out_dir, bname, f"decodes_{tag}.txt")
            kept = [l for l in open(dec_path).read().splitlines()
                    if l.strip() and l.split("\t", 1)[0] in scored_set]
            by_rx.setdefault((run, rcv), []).extend(kept)
            blocks.append(dict(run=run, receiver=rcv, first=first, decodes=len(kept), seconds=0.0))
        for (run, rcv), lines in by_rx.items():
            d = os.path.join(out_root, run, rcv)
            os.makedirs(d, exist_ok=True)
            lines.sort(key=lambda l: l[:13])
            with open(os.path.join(d, f"decodes_{tag}.txt"), "w") as fh:
                fh.write("".join(l + "\n" for l in lines))
        return wall, sum(b["decodes"] for b in blocks), blocks


def machine_desc():
    try:
        cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                             capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        cpu = platform.processor() or "CPU"
    return f"{cpu} ({os.cpu_count()} cores, {platform.machine()})"


def verify_set(set_path, out_root, tag, others="", pool_tags="jt9", list_unconf=False):
    jobs = read_set(set_path)
    runs = sorted({r for r, _, _ in jobs})
    set_name = os.path.splitext(os.path.basename(set_path))[0]
    verify_py = os.path.join(REPO, "collect", "verify.py")
    log_lines = []
    grand = Counter()
    excl_stat = Counter()
    unconf_all = []

    for run in runs:
        rec = os.path.join(V, run, "recordings")
        spots = os.path.join(V, run, "pskr_spots.jsonl")
        dec_dir = os.path.join(out_root, run)
        cmd = [sys.executable, verify_py,
               "--recordings", rec,
               "--spots", spots,
               "--set", set_path,
               "--run", run,
               "--tag", tag]
        if os.path.isdir(dec_dir):
            cmd += ["--decodes", dec_dir]
        elif tag != "jt9":
            sys.exit(f"missing decode folder {dec_dir} for tag {tag}")
        if others:
            cmd += ["--others", others]
        if pool_tags:
            cmd += ["--pool-tags", pool_tags]
        if list_unconf:
            cmd += ["--list"]
        env = os.environ.copy()
        env["V"] = V
        p = subprocess.run(cmd, env=env, text=True, capture_output=True)
        if p.returncode != 0:
            sys.stderr.write(p.stdout + p.stderr)
            sys.exit(p.returncode)
        log_lines.append(f"=== {run} ===\n" + p.stdout)

        # Aggregate directly from the written verified_<tag>_<set>.tsv files for exact totals
        stations = sorted({rcv for r, rcv, _ in jobs if r == run})
        for st in stations:
            vroot = dec_dir if os.path.isdir(dec_dir) else os.path.join(rec)
            vp = os.path.join(vroot, st, f"verified_{tag}_{set_name}.tsv")
            if not os.path.exists(vp):
                vp = os.path.join(rec, st, f"verified_{tag}_{set_name}.tsv")
            with open(vp) as fh:
                hdr = fh.readline().rstrip("\n").split("\t")
                has_excl = "base_tier" in hdr
                ti = hdr.index("tier")
                bi = hdr.index("base_tier") if has_excl else ti
                ei = hdr.index("excl") if has_excl else None
                for row in fh:
                    f = row.rstrip("\n").split("\t")
                    if len(f) <= bi:
                        continue
                    base = f[bi]
                    grand[base] += 1
                    if base == "unconf":
                        unconf_all.append((run, st, f[0], f[1], f[2]))
                    if has_excl and f[ei] == "1" and base != "n/a":
                        st_tier = f[ti]
                        excl_stat[st_tier] += 1
                        excl_stat[f"{st_tier}:{base}"] += 1

    tot = sum(grand.values())
    chk = tot - grand["n/a"]
    conf = grand["slot"] + grand["xconf"]
    summary = (
        f"\n=== POOLED {set_name} ({len(jobs)} blocks, {len(jobs) * BLOCK} scored slots) — tag {tag} ===\n"
        f"decodes: {tot:,} (checkable {chk:,}, n/a {grand['n/a']:,})\n"
        f"  slot:      {grand['slot']:7,} ({100 * grand['slot'] / max(chk, 1):5.1f}%)\n"
        f"  xconf:     {grand['xconf']:7,} ({100 * grand['xconf'] / max(chk, 1):5.1f}%)\n"
        f"  confirmed: {conf:7,} ({100 * conf / max(chk, 1):5.1f}%)\n"
        f"  active:    {grand['active']:7,} ({100 * grand['active'] / max(chk, 1):5.1f}%)\n"
        f"  repeat:    {grand['repeat']:7,} ({100 * grand['repeat'] / max(chk, 1):5.1f}%)\n"
        f"  unconf:    {grand['unconf']:7,} ({100 * grand['unconf'] / max(chk, 1):5.2f}%, "
        f"{grand['unconf'] / max(len(jobs) * BLOCK, 1):.4f}/slot)\n"
    )
    if others:
        ne = excl_stat["xconf-strict"] + excl_stat["unconfirmable"]
        xs = excl_stat["xconf-strict"]
        uc = excl_stat["unconfirmable"]
        summary += (
            f"exclusives vs [{others}]: {ne:,} — "
            f"xconf-strict {xs:,} ({100 * xs / max(ne, 1):.1f}%), unconfirmable {uc:,}\n"
        )
        for st_tier in ("xconf-strict", "unconfirmable"):
            parts = [f"{b} {excl_stat[f'{st_tier}:{b}']}"
                     for b in ("slot", "xconf", "active", "repeat", "unconf")
                     if excl_stat[f"{st_tier}:{b}"]]
            if parts:
                summary += f"  {st_tier:13s} by base tier: {', '.join(parts)}\n"
    if list_unconf and unconf_all:
        summary += "unconfirmed:\n" + "".join(
            f"  {r} {st[:28]:28s} {sl} {float(fq):7.1f}  {tx}\n"
            for r, st, sl, fq, tx in unconf_all
        )
    os.makedirs(out_root, exist_ok=True)
    suffix = "_strict" if others else ""
    log_path = os.path.join(out_root, f"verify_{tag}_{set_name}{suffix}.txt")
    with open(log_path, "w") as fh:
        fh.write("\n".join(log_lines) + summary)
    print(summary, end="", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="small", help="small | third | full | path/to/set.txt")
    ap.add_argument("--decoder", choices=["ft8nn", "ft8nn_py", "jt9", "jtdx", "jtdx_deep", "ft8lib"],
                    default="ft8nn")
    ap.add_argument("--tag", default=None, help="output tag (default: decoder name)")
    ap.add_argument("--out", default=None, help="output root (default: data/val/<set>)")
    ap.add_argument("--exe", default=None, help="decoder binary override")
    ap.add_argument("--model", default=None, help="model path (ft8nn: models/v4/ft8nn.onnx; ft8nn_py: models/v4/model.pt)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--verify", action="store_true", help="score with collect/verify.py after decoding")
    ap.add_argument("--verify-only", action="store_true", help="skip decoding; only run collect/verify.py")
    ap.add_argument("--others", default="", help="verify.py --others TAG[,TAG] for xconf-strict")
    ap.add_argument("--pool-tags", default="jt9", help="verify.py --pool-tags TAG[,TAG]")
    ap.add_argument("--list", action="store_true", help="list all unconfirmed decodes")
    ap.add_argument("extra", nargs="*", help="extra flags passed to the underlying decoder")
    a = ap.parse_args()

    set_path = resolve_set(a.set)
    set_name = os.path.splitext(os.path.basename(set_path))[0]
    out_root = os.path.abspath(a.out or os.path.join(REPO, "data", "val", set_name))
    tag = a.tag or a.decoder

    if not a.verify_only:
        jobs = read_set(set_path)
        n_dec_slots = len(jobs) * (BLOCK + LEAD)
        n_sc_slots = len(jobs) * BLOCK
        print(f"{set_name}: {len(jobs)} blocks ({n_dec_slots} decoded slots, {n_sc_slots} scored), "
              f"decoder={a.decoder} (tag={tag}), {a.workers} workers", flush=True)

        if a.decoder == "ft8nn_py":
            model = a.model or os.path.join(REPO, "models", "v4", "model.pt")
            t0 = time.time()
            wall, tot_dec, blocks = run_ft8nn_py(jobs, out_root, tag, model, a.workers, a.extra)
            cpu_s = 0.0
            exe = "bench/run_real.py"
        else:
            if a.decoder == "ft8nn":
                exe = a.exe or os.path.join(REPO, "c", "build", "ft8nn_cli")
                model = a.model or os.path.join(REPO, "models", "v4", "ft8nn.onnx")
            elif a.decoder == "jt9":
                exe = find_jt9(a.exe)
                model = None
            elif a.decoder in ("jtdx", "jtdx_deep"):
                exe = a.exe or os.path.join(REPO, "third_party", "jtdx-build", "build", "jtdx_ft8_file")
                model = None
            elif a.decoder == "ft8lib":
                exe = a.exe or os.path.join(REPO, "third_party", "ft8_lib", "decode_ft8")
                model = None
            if not os.access(exe, os.X_OK):
                sys.exit(f"decoder binary not executable: {exe}")

            t0 = time.time()
            # LPT order (busiest blocks first, from jt9 baseline count) prevents tail stragglers across workers.
            def _busy_key(ij):
                _, (r, rcv, fst) = ij
                dp = os.path.join(V, r, "recordings", rcv, "decodes.txt")
                if not os.path.exists(dp):
                    return 0
                ss = set(slot_names(fst))
                return -sum(1 for l in open(dp) if l[:13] in ss)

            indexed = sorted(enumerate(jobs), key=_busy_key)
            task_args = [(a.decoder, j, exe, model, a.extra) for _, j in indexed]
            # Each worker spawns a subprocess and waits on pipes; ThreadPoolExecutor has zero pickle overhead.
            with ThreadPoolExecutor(max_workers=a.workers) as ex:
                res_ord = list(ex.map(decode_block_cli, task_args))
            res = [None] * len(jobs)
            for (orig_i, _), item in zip(indexed, res_ord):
                res[orig_i] = item
            wall = time.time() - t0

            by_rx, marks_by_rx = {}, {}
            for info, lines, marks in res:
                by_rx.setdefault((info["run"], info["receiver"]), []).extend(lines)
                if marks:
                    marks_by_rx.setdefault((info["run"], info["receiver"]), []).extend(marks)
            for (run, rcv), lines in by_rx.items():
                d = os.path.join(out_root, run, rcv)
                os.makedirs(d, exist_ok=True)
                lines.sort(key=lambda l: l[:13])
                with open(os.path.join(d, f"decodes_{tag}.txt"), "w") as fh:
                    fh.write("".join(l + "\n" for l in lines))
            for (run, rcv), marks in marks_by_rx.items():
                d = os.path.join(out_root, run, rcv)
                marks.sort(key=lambda l: l[:13])
                with open(os.path.join(d, f"markers_{tag}.tsv"), "w") as fh:
                    fh.write("".join(l + "\n" for l in marks))

            blocks = [info for info, _, _ in res]
            cpu_s = round(sum(b["seconds"] for b in blocks), 1)
            tot_dec = sum(b["decodes"] for b in blocks)

        summary = dict(
            set=set_name,
            decoder=a.decoder,
            tag=tag,
            exe=exe,
            model=model,
            workers=a.workers,
            machine=machine_desc(),
            started_utc=datetime.fromtimestamp(t0, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            wall_s=round(wall, 1),
            cpu_s=cpu_s,
            slots_decoded=n_dec_slots,
            slots_scored=n_sc_slots,
            decodes=tot_dec,
            blocks=blocks,
        )
        os.makedirs(out_root, exist_ok=True)
        with open(os.path.join(out_root, f"timing_{tag}_{set_name}.json"), "w") as fh:
            json.dump(summary, fh, indent=1)
        slow = max(blocks, key=lambda b: b["seconds"])
        print(f"done: wall {wall / 60:.2f} min ({wall:.1f} s), block-seconds {cpu_s:.1f} "
              f"({cpu_s / max(n_dec_slots, 1):.2f} s/decoded-slot/worker), {tot_dec:,} scored decodes; "
              f"slowest block {slow['seconds']:.1f} s ({slow['run']} {slow['receiver']} {slow['first']}, "
              f"{slow['decodes']} decodes)", flush=True)

    if a.verify or a.verify_only:
        verify_set(set_path, out_root, tag, others=a.others, pool_tags=a.pool_tags, list_unconf=a.list)


if __name__ == "__main__":
    main()
