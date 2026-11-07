# Real-data collection (KiwiSDR recordings, jt9 labels, PSK Reporter spots)

The NN is trained only on synthetic data (`ft8nn/synth_bb.py`, generated on the fly by `train.py`). Real
off-air recordings are used for two things:

- **Evaluation:** the training recordings are the "real" benchmark, decoded by `bench/run_real.py` and compared
  with jt9's labels. Validation runs 1 and 2 add PSK Reporter spots as ground truth for false decodes.
- **Design:** the v2 training distribution (`synth_bb.py`, "built from real-recording failures") and most
  decoder stages came from analysing what failed on these recordings.

Reproducing the pipeline therefore means: (1) the same recordings, checkable against the manifests in
`collect/runs/`, and (2) the tools that made them, in `collect/`. Live radio can't be recorded twice, so a new
collection run gives a *new* data set made the same way, not the same files.

## Tools (`collect/`, Python 3 stdlib only)

| file | what |
|---|---|
| `kiwi.py` | minimal KiwiSDR WebSocket client: USB mode 200–3000 Hz, AGC on, uncompressed 16-bit PCM at 12 kHz, written as 15 s WAVs aligned to UTC :00/:15/:30/:45 and named `YYMMDD_HHMMSS.wav` (a slot is kept only if recording started within 0.2 s of its boundary) |
| `pskr.py` | minimal MQTT 3.1.1 subscriber for `mqtt.pskreporter.info:1883`, topics `pskr/filter/v2/<band>/FT8/#`; one JSON line per spot plus our arrival time `rx_t` |
| `val.py` | the run driver: `probe`, `probe-fresh`, `record`, `decode` (below) |
| `verify.py` | scores a decoder's output against the spots (tiers slot > active > repeat > unconf) |
| `manifest.py` | SHA-256 manifest of a run (`make`, `check`) |

These are copies of `../ft8-validation-set/tools/*.py` (2026-11-04). Only the default paths changed: `val.py`
writes to `$OUTDIR` (default `.`) and reads the training receiver list from `$TRAIN_SELECTION` (default
`collect/runs/train/selection.json`); `verify.py` reads the validation set from `$V` (default
`../ft8-validation-set`). `tools/ab_run2.sh` and `tools/validate_full.sh` call `collect/verify.py`.

Requirements: network access to the Kiwis (HTTP/WebSocket, usually port 8073) and to PSK Reporter MQTT;
`jt9` from WSJT-X on `PATH` for `decode` and the probes (mini: `/usr/local/bin/jt9`, WSJT-X 3.0.2 Improved; it also
ships inside `wsjtx.app/Contents/MacOS/`).

## One collection run

```sh
export OUTDIR=data/collect/2026-11-xx        # new run folder (gitignored data/)
python3 collect/val.py probe                  # re-probe the training receivers, choose a band each
# or: random receiverbook Kiwis, excluding every training site
RECEIVERS_JSON=collect/runs/val2/receivers.json python3 collect/val.py probe-fresh
python3 collect/val.py record                 # MINUTES=30 of audio from every receiver + spots (+12 min tail)
python3 collect/val.py decode                 # jt9 -8 -d 3 labels -> decodes.txt, plus a 30-min WAV
python3 collect/manifest.py make $OUTDIR pskr_spots.jsonl selection.json session.json > MANIFEST.sha256
```

- **probe:** records one slot on each band that suits the receiver's local time (day: 20/17/15/10 m; dusk and
  dawn: 20/30/40/17 m; night: 40/80/30/20 m), decodes it with jt9, then picks greedily: busiest
  (receiver, band) pairs first, one band per receiver, at most 2 receivers per band, at least 3 decodes.
- **probe-fresh:** shuffles the receiverbook list (`RECEIVERS_JSON`, a flattened receiverbook.de/map snapshot),
  keeps one Kiwi per site, drops training sites (by URL and by lat/lon rounded to 0.1°), keeps only Kiwis whose
  `/status` shows them online with at least 2 free channels and 0–30 MHz coverage, skips any with an inactivity
  or IP time limit shorter than the run, probes up to 4 per band in parallel and takes up to 10 with
  ≥ 5 decodes (first 1 per band, then a second where it pays). The shuffle is unseeded.
- **record:** all receivers in parallel from the next slot boundary, reconnecting on errors (up to 5 attempts
  per receiver), while `pskr.py` captures spots for the same bands until 12 min after the last slot: reporters
  upload in batches.
- **decode:** jt9 labels, with jt9's invented `K1ABC …` decodes dropped. Without a MyCall jt9 uses the
  placeholder K1ABC, and its MyCall a-priori pass then "decodes" messages to it.

Etiquette built in: one audio channel per receiver, at least one channel left free for people, client name
`ft8-recorder`.

## Runs archived so far

| run | `collect/runs/` | data folder | when (UTC) | receivers | slots | spots |
|---|---|---|---|---|---|---|
| training recordings | `train/` | `../ft8-training-data/recordings` | 2026-11-04 03:02–03:33 | 10 | 1,197 | – |
| validation run 1 | `val1/` | `../ft8-validation-set` | 2026-11-04 22:17:45–22:47:45 | 9 (training receivers, other bands) | 1,077 | 920k |
| validation run 2 | `val2/` | `../ft8-validation-set/run2` | 2026-11-04 22:28:30–22:58:30 | 9 (fresh sites) | 1,077 | 853k |

Each `collect/runs/<run>/` holds `selection.json` (receivers, bands, probe results), `session.json` (times,
bands, MQTT topics) and `MANIFEST.sha256`. Run 2 also holds the receiverbook snapshot it was drawn from.
Check a copy of the data with

```sh
python3 collect/manifest.py check collect/runs/train/MANIFEST.sha256 ../ft8-training-data
python3 collect/manifest.py check collect/runs/val1/MANIFEST.sha256 ../ft8-validation-set
python3 collect/manifest.py check collect/runs/val2/MANIFEST.sha256 ../ft8-validation-set/run2
```

## Known gaps (training recordings)

- The training run predates these tools; its own scripts and its original `selection.json` (with probe
  results) live in `../ft8-training-data` on the main machine, and only `recordings/` was copied to the mini.
  `collect/runs/train/selection.json` and `session.json` were rebuilt from the recordings' `meta.json` files and
  slot names. `../ft8-validation-set/probe.log` confirms it is the same 10 receivers `val.py probe` read.
  The recordings have the same layout and `meta.json` fields as the validation runs, so they were made by an
  earlier version of `val.py`. How those 10 Kiwis were picked from receiverbook, and whether spots were captured,
  is only recorded on the main machine.
- Its jt9 labels still contain 5 `K1ABC …` lines: the older script didn't filter them.
- The jt9 build used for the labels isn't recorded. The mini has WSJT-X 3.0.2 (Improved).
