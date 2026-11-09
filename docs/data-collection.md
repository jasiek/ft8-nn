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
| `val.py` | the run driver for the validation runs: `probe`, `probe-fresh`, `record`, `decode` (below) |
| `ft8rec.py` | the run driver of the training recordings (`probe`, `record`, `decode`); `val.py` is its successor |
| `fill.py` | second probe pass of the training run: fills bands `ft8rec.py probe` left empty |
| `verify.py` | scores a decoder's output against the spots (tiers slot > active > repeat > unconf) |
| `manifest.py` | SHA-256 manifest of a run (`make`, `check`) |
| `datarepo.py` | packs a run folder for its Git LFS data repo (FLAC + xz) and unpacks it byte for byte |

These are copies of the data repos' `tools/*.py` (`datasets/ft8-validation-set`, 2026-11-04; `datasets/ft8-training-data`
`ft8rec.py`, `fill.py`, 2026-11-03; their `kiwi.py` is byte-identical). Only the default paths changed: `val.py`
and `ft8rec.py` write to `$OUTDIR` (default `.`) and reads the training receiver list from `$TRAIN_SELECTION` (default
`collect/runs/train/selection.json`); `verify.py` reads the validation set from `$V` (default
`datasets/ft8-validation-set`). `tools/ab_run2.sh` and `tools/validate_full.sh` call `collect/verify.py`.

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
  keeps one Kiwi per site, drops training sites and the sites of every run in `collect/runs/` (by URL and by lat/lon rounded to 0.1°), keeps only Kiwis whose
  `/status` shows them online with at least 2 free channels and 0–30 MHz coverage, skips any with an inactivity
  or IP time limit shorter than the run, probes up to `PER_BAND` (4) per band in parallel and takes up to `N_RX` (10) with
  ≥ 5 decodes (first 1 per band, then a second and a third where it pays). The shuffle is unseeded.
- **record:** all receivers in parallel from the next slot boundary, reconnecting on errors (up to 5 attempts
  per receiver), while `pskr.py` captures spots for the same bands until 12 min after the last slot: reporters
  upload in batches.
- **decode:** jt9 labels, with jt9's invented `K1ABC …` decodes dropped. Without a MyCall jt9 uses the
  placeholder K1ABC, and its MyCall a-priori pass then "decodes" messages to it.

The training recordings were made with the older driver:

```sh
export OUTDIR=data/collect/train-2026-xx RECEIVERS_JSON=<receiverbook snapshot>
python3 collect/ft8rec.py probe                # random receiverbook Kiwis; seed printed and saved in selection.json
python3 collect/fill.py 8                     # bands still empty: up to 8 more candidates per band
python3 collect/ft8rec.py record
python3 collect/ft8rec.py decode
```

`ft8rec.py probe` shuffles the receiverbook list with a seed (saved in `selection.json`), keeps one Kiwi per
site that passes the same `/status` checks as `probe-fresh`, then probes in up to 8 rounds: each round pairs every
band still unfilled with up to 2 untried receivers whose local time suits it (day 08–17 h: 20/17/15/12/10/30 m,
otherwise 160/80/60/40/30/20 m; any band from round 4) and takes the first with ≥ 3 decodes per band, up to 10.
`fill.py` then probes up to N more site-distinct candidates for each band still missing (day bands 09–16 h local,
night bands 19–05 h). Differences from `val.py`: no PSK Reporter capture, jt9's `K1ABC …` decodes are kept.

Etiquette built in: one audio channel per receiver, at least one channel left free for people, client name
`ft8-recorder`.

## Getting the data on another computer

The recordings are in two public data repos, stored compressed in Git LFS (slot WAVs as FLAC, spots as `.xz`, no
30-minute WAVs; 446 MB + 235 MB instead of 2.6 GB). They are git submodules of this repo under `datasets/`, pinned
to the commit the results were made with. Fetch them and unpack in place; unpack restores every file byte for byte
(30-minute WAVs rebuilt from the slots, ignored by the data repos' `.gitignore`, so the submodules stay clean) and
checks the manifests. Needs `git-lfs`, `flac`, `xz`.

```sh
git submodule update --init                      # or clone with --recurse-submodules
(cd datasets/ft8-validation-set && python3 tools/datarepo.py unpack . --manifest MANIFEST.sha256 --manifest run2/MANIFEST.sha256)
(cd datasets/ft8-training-data && python3 tools/datarepo.py unpack . --manifest MANIFEST.sha256)
```

Code that needs no recordings (the C library, the unit tests, CI) works without the submodules; a plain
`git clone` does not fetch them.

`collect/datarepo.py pack SRC DST` made them (`unpack` is the same file). GitHub's free LFS bandwidth is 10 GiB a
month for the account, so about 14 full clones; `GIT_LFS_SKIP_SMUDGE=1 git clone` plus `git lfs pull --include=...`
fetches part of a set (`GIT_LFS_SKIP_SMUDGE=1 git submodule update --init`, then
`git -C datasets/<repo> lfs pull --include=...`).

## Runs archived so far

| run | `collect/runs/` | data folder | when (UTC) | receivers | slots | spots |
|---|---|---|---|---|---|---|
| training recordings | `train/` | `datasets/ft8-training-data/recordings` | 2026-11-04 03:02–03:33 | 10 (random receiverbook sites) | 1,197 | none captured |
| validation run 1 | `val1/` | `datasets/ft8-validation-set` | 2026-11-04 22:17:45–22:47:45 | 9 (training receivers, other bands) | 1,077 | 920k |
| validation run 2 | `val2/` | `datasets/ft8-validation-set/run2` | 2026-11-04 22:28:30–22:58:30 | 9 (fresh sites) | 1,077 | 853k |
| validation run 3 | `val3/` | `datasets/ft8-validation-set/run3` | 2026-11-09 05:32:00–06:02:00 (Mon) | 12 (fresh sites, none from run 2) | 1,435 | 556k |
| validation run 4 | `val4/` | `datasets/ft8-validation-set/run4` | 2026-11-09 15:00:00–15:30:00 (Mon) | 15 (fresh sites, none from runs 2–3) | 1,790 | 1,473k |

Each `collect/runs/<run>/` holds `selection.json` (receivers, bands, probe results), `session.json` (times,
bands, MQTT topics) and `MANIFEST.sha256`. Run 2 also holds the receiverbook snapshot it was drawn from (runs 3 and 4 reused it, `N_RX=16 PER_BAND=6`: 12 and 15 passed; run 4 via the data repo's `tools/scheduled_run.sh`, which waits for a set UTC time); the
training run holds its `record.log`, and its `selection.json` is the original (probe seed 372751130, plus the
receiver that was replaced).
Check a copy of the data with

```sh
python3 collect/manifest.py check collect/runs/train/MANIFEST.sha256 datasets/ft8-training-data
python3 collect/manifest.py check collect/runs/val1/MANIFEST.sha256 datasets/ft8-validation-set
python3 collect/manifest.py check collect/runs/val2/MANIFEST.sha256 datasets/ft8-validation-set/run2
```

## Known gaps (training recordings)

- The receiverbook snapshot the training probe used was not saved, so the seed alone can't replay the selection.
  (`collect/runs/val2/receivers.json` is a later snapshot, from 2026-11-04 17:27.)
- The 160 m receiver: the chosen one (EI6IZ, `http://88.81.100.130:8073/`, listed under `replaced` in
  `selection.json`) was too busy for all 5 attempts (`record.log`), and K1RA's 160 m Kiwi was recorded instead from
  03:03:30Z (`[04 160m replacement] done: 120 slots`). That replacement was done outside these scripts.
- Its jt9 labels still contain 5 `K1ABC …` lines: `ft8rec.py` doesn't filter them.
- No PSK Reporter spots: `ft8rec.py` doesn't capture them, so the training recordings have jt9 labels only.
- The jt9 build used for the labels isn't recorded (Python was 3.12.14, from the run's `.tool-versions`). The mini
  has WSJT-X 3.0.2 (Improved).
