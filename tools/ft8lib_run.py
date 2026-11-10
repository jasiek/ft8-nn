#!/usr/bin/env python3
"""Decode validation-set slots with ft8_lib's decode_ft8; write jt9-format
decodes_<tag>.txt files for collect/verify.py.

ft8_lib (kgoba/ft8_lib) stands in for the FT8CN / FT8TW decoder, which is
published only as a prebuilt Android arm64 .so.  It is one pass with no
subtraction and no OSD, so read it as a floor for that family, not as a
measurement of the app itself.  See docs/decoder-comparison.md.

  V=datasets/ft8-validation-set
  python3 tools/ft8lib_run.py --recordings $V/run2/recordings --out data/val/run2 \
      --tag ft8lib --workers 4
"""
import argparse, os, re, subprocess, time
from concurrent.futures import ProcessPoolExecutor, as_completed

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# decode_ft8 prints: "000000 +18.5 +0.96  753 ~  CQ N8UVM EM89"
LINE = re.compile(r"^\d{6}\s+([+-][\d.]+)\s+([+-][\d.]+)\s+(\d+)\s+~\s+(.*?)\s*$")


def decode_station(args):
    exe, station_dir, out_dir, tag = args
    name = os.path.basename(station_dir)
    sdir = os.path.join(station_dir, "slots")
    slots = sorted(f for f in os.listdir(sdir) if f.endswith(".wav"))
    lines, t0 = [], time.time()
    for f in slots:
        slot = f[:-4]
        hhmmss = slot.split("_")[1]
        p = subprocess.run([exe, os.path.join(sdir, f)], text=True,
                           stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        for raw in p.stdout.splitlines():
            m = LINE.match(raw)
            if not m:
                continue
            snr, dt, freq, msg = m.groups()
            if msg:
                lines.append(f"{slot} {hhmmss} {round(float(snr)):3d} "
                             f"{float(dt):4.1f} {int(freq):4d} ~  {msg}")
    os.makedirs(os.path.join(out_dir, name), exist_ok=True)
    with open(os.path.join(out_dir, name, f"decodes_{tag}.txt"), "w") as fh:
        fh.write("\n".join(lines) + ("\n" if lines else ""))
    return name, len(slots), len(lines), time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recordings", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="ft8lib")
    ap.add_argument("--exe", default=os.path.join(REPO, "third_party", "ft8_lib",
                                                  "decode_ft8"))
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    stations = sorted(os.path.join(a.recordings, d)
                      for d in os.listdir(a.recordings)
                      if os.path.isdir(os.path.join(a.recordings, d, "slots")))
    t0, total = time.time(), 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(decode_station, (a.exe, s, a.out, a.tag))
                for s in stations]
        for fu in as_completed(futs):
            name, ns, nd, dt = fu.result()
            total += nd
            print(f"  {name:48s} {ns:4d} slots {nd:6d} decodes {dt/60:5.1f} min",
                  flush=True)
    print(f"total {total} decodes in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
