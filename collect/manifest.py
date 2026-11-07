"""SHA-256 manifest of a recording run, so a copy of the data can be checked against what the results used.

  python3 collect/manifest.py make  RUN_DIR [FILE ...] > collect/runs/<run>/MANIFEST.sha256
  python3 collect/manifest.py check collect/runs/<run>/MANIFEST.sha256 RUN_DIR

make hashes every file under RUN_DIR/recordings (WAVs, meta.json, jt9 decodes.txt) plus the extra FILEs given
relative to RUN_DIR (e.g. pskr_spots.jsonl). Our own outputs (decodes_<tag>.txt, verified_<tag>.tsv) are skipped:
they change with every decoder version. Format: `shasum -a 256` (also checkable with `shasum -a 256 -c`).
"""
import hashlib, os, sys

SKIP = ("decodes_", "verified_", ".DS_Store")


def files(root, extra):
    for d, _, fs in os.walk(os.path.join(root, "recordings")):
        for f in fs:
            if not f.startswith(SKIP):
                yield os.path.relpath(os.path.join(d, f), root)
    yield from extra


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    if sys.argv[1] == "make":
        root = sys.argv[2]
        for rel in sorted(files(root, sys.argv[3:])):
            print(f"{sha(os.path.join(root, rel))}  {rel}")
    else:
        bad = n = 0
        for line in open(sys.argv[2]):
            h, rel = line.rstrip("\n").split("  ", 1)
            n += 1
            p = os.path.join(sys.argv[3], rel)
            if not os.path.exists(p) or sha(p) != h:
                bad += 1
                print("MISMATCH" if os.path.exists(p) else "MISSING ", rel)
        print(f"{n - bad}/{n} files OK")
        sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
