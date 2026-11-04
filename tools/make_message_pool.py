"""Build a pool of realistic FT8 messages with their 77-bit payloads, using WSJT-X `ft8code`.

Output: ft8nn/data/messages.json  [{"text": ..., "bits": "0101..."}]
Used so benchmark WAVs carry real messages that jt9 can unpack, and we can score by text.
"""
import json
import random
import string
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

FT8CODE = "/Applications/wsjtx.app/Contents/MacOS/ft8code"
L, D = string.ascii_uppercase, string.digits


def callsign(r):
    pre = r.choice(L) + r.choice(L + D) if r.random() < 0.7 else r.choice("KWNGFI")
    if pre[-1].isdigit():
        pre = pre[0] + r.choice(L)
    return pre + r.choice(D) + "".join(r.choice(L) for _ in range(r.randint(1, 3)))


def grid(r):
    return r.choice("ABCDEFGHIJKLMNOPQR") + r.choice("ABCDEFGHIJKLMNOPQR") + r.choice(D) + r.choice(D)


def message(r):
    a, b = callsign(r), callsign(r)
    k = r.random()
    if k < 0.25:
        return f"CQ {a} {grid(r)}"
    if k < 0.5:
        return f"{a} {b} {grid(r)}"
    if k < 0.7:
        return f"{a} {b} {r.randint(-24, 10):+03d}"
    if k < 0.85:
        return f"{a} {b} R{r.randint(-24, 10):+03d}"
    return f"{a} {b} {r.choice(['RR73', 'RRR', '73'])}"


def encode(msg):
    lines = subprocess.run([FT8CODE, msg], capture_output=True, text=True).stdout.splitlines()
    try:
        decoded_line = next(l for l in lines if l.strip().startswith("1."))
        bits = lines[next(i for i, l in enumerate(lines) if l.startswith("Source-encoded")) + 1].strip()
    except StopIteration:
        return None
    # Row looks like " 1. <msg> <decoded> ..." -> require the round trip to reproduce the text.
    if decoded_line.count(msg) < 2 or len(bits) != 77:
        return None
    return {"text": msg, "bits": bits}


if __name__ == "__main__":
    r = random.Random(1234)
    msgs = list(dict.fromkeys(message(r) for _ in range(1200)))
    with ThreadPoolExecutor(8) as ex:
        pool = [m for m in ex.map(encode, msgs) if m]
    out = Path("ft8nn/data/messages.json")
    out.write_text(json.dumps(pool, indent=0))
    print(f"{len(pool)}/{len(msgs)} messages round-tripped -> {out}")
