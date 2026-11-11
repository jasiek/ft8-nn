#!/usr/bin/env python3
"""Decode validation-set slots with JTDX's decoder; write jt9-format
decodes_<tag>.txt files for collect/verify.py.

One jtdx_ft8_file process per station with all of its slots in chronological
order, so JTDX keeps the cross-slot state it has live (even/odd message memory,
the callsign hash table that resolves <...> calls).  Stations run in parallel.

  V=datasets/ft8-validation-set
  python3 tools/jtdx_run.py --recordings $V/run2/recordings --out data/val/run2 \
      --tag jtdx --workers 6                      # JTDX defaults
  python3 tools/jtdx_run.py --recordings $V/run2/recordings --out data/val/run2 \
      --tag jtdx_deep --workers 6 -- -w -c 3 -s 2  # JTDX at its most sensitive
"""
import argparse, os, subprocess, time
from concurrent.futures import ProcessPoolExecutor, as_completed

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_EXE = os.path.join(REPO, "third_party", "jtdx-build", "build",
                           "jtdx_ft8_file")


def parse(raw):
    """decoder.f90 ft8_decoded: i6.6,i4,f5.1,i5,1x,'~',1x,a26,a1 ->
    nutc | snr | dt | freq | ~ | 26-char message | JTDX service marker."""
    if len(raw) < 24 or raw[21] != "~" or not raw[:6].strip().isdigit():
        return None
    try:
        return (raw[:6], int(raw[6:10]), float(raw[10:15]), int(raw[15:20]),
                raw[23:49].strip(), raw[49:50].strip())
    except ValueError:
        return None


def decode_station(args):
    exe, station_dir, out_dir, tag, extra, workdir = args
    name = os.path.basename(station_dir)
    sdir = os.path.join(station_dir, "slots")
    slots = sorted(f for f in os.listdir(sdir) if f.endswith(".wav"))
    paths = [os.path.join(sdir, f) for f in slots]
    t0 = time.time()
    p = subprocess.run([exe] + extra + paths, cwd=workdir, text=True,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    lines, marks, slot = [], [], None
    for raw in p.stdout.splitlines():
        if raw.startswith("#F "):
            slot = os.path.basename(raw[3:])[:-4]
            continue
        d = parse(raw)
        if d is None or slot is None:
            continue
        hhmmss, snr, dt, freq, msg, marker = d
        if not msg:
            continue
        lines.append(f"{slot} {hhmmss} {snr:3d} {dt:4.1f} {freq:4d} ~  {msg}")
        marks.append(f"{slot}\t{freq}\t{marker or '-'}\t{msg}")
    os.makedirs(os.path.join(out_dir, name), exist_ok=True)
    with open(os.path.join(out_dir, name, f"decodes_{tag}.txt"), "w") as fh:
        fh.write("\n".join(lines) + ("\n" if lines else ""))
    # JTDX's per-decode service marker, for analysis (verify.py never reads it):
    # '-' plain, '*' a-priori, '^' its ft8sd stage, ',' '.' free text, '1' i3=1
    with open(os.path.join(out_dir, name, f"markers_{tag}.tsv"), "w") as fh:
        fh.write("\n".join(marks) + ("\n" if marks else ""))
    return name, len(slots), len(lines), time.time() - t0, p.returncode, p.stderr[-400:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recordings", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="jtdx")
    ap.add_argument("--exe", default=DEFAULT_EXE)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("extra", nargs="*", help="options for jtdx_ft8_file")
    a = ap.parse_args()

    workdir = os.path.join(REPO, "data", "jtdx-work")   # JTDX writes ALLCALL7.TXT here
    os.makedirs(workdir, exist_ok=True)
    a.recordings = os.path.abspath(a.recordings)   # the decoder runs in workdir
    stations = sorted(os.path.join(a.recordings, d)
                      for d in os.listdir(a.recordings)
                      if os.path.isdir(os.path.join(a.recordings, d, "slots")))
    print(f"{len(stations)} stations, {a.workers} workers, "
          f"opts: {' '.join(a.extra) or '(JTDX defaults)'}", flush=True)
    os.environ["OMP_NUM_THREADS"] = "1"
    t0, total = time.time(), 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(decode_station,
                          (a.exe, s, a.out, a.tag, list(a.extra), workdir))
                for s in stations]
        for f in as_completed(futs):
            name, ns, nd, dt, rc, err = f.result()
            total += nd
            print(f"  {name:48s} {ns:4d} slots {nd:6d} decodes {dt/60:6.1f} min "
                  f"rc={rc}" + (f" ERR {err}" if err.strip() else ""), flush=True)
    print(f"total {total} decodes in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
