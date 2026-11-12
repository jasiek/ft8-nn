# ft8nn — neural soft-demodulation FT8 decoder

Goal: an FT8 decoder more sensitive than WSJT-X, packaged so other software can embed it.

Approach: keep FT8's LDPC(174,91) + CRC-14 (near-optimal, gives false-decode protection) and replace
the hand-made soft metrics with a small neural net that turns baseband symbol spectra into bit LLRs.

## Results vs stock WSJT-X `jt9`

Baseline everywhere below: WSJT-X's own FT8 decoder run as `jt9 -8 -d 3`, i.e. depth 3, its deepest routine
search (`bench/run_jt9.py`; the real-recording baselines were made with WSJT-X Improved 3.0.2, see
`docs/data-collection.md`).
Ours: the shipped default, `models/v3` + the decoder defaults (4 passes with subtraction, OSD, CQ and context
a-priori decoding, DT re-centring for receivers whose clock drifts, the co-channel retry that finds a second station
hidden under one already decoded, and the repeat stage that tests a station's message from 30/60 s earlier as a
whole known codeword), tag `nn_v3_cc`. The AWGN and fading rows are unaffected by the last three (no slot times, so
no context; the re-centring gate never arms; synthetic slots hold no co-channel pairs - AWGN re-measured with
`nn_v3_cc`: identical) and were measured with `nn_v3`.

### Sensitivity: about 1.2 dB on AWGN

Synthetic benchmark sets (`bench/make_set.py`, commands under [Reproduce](#reproduce)): 10 signals per 15 s slot,
30 slots per SNR, real messages; a decode counts only on exact text match and DT ≤ 0.5 s. The number is the SNR
(in 2.5 kHz, the WSJT-X convention) at which 50% of the signals decode, so **lower is better and the gain is the
difference**:

| set | `jt9 -d 3` | ft8nn | gain |
|---|---|---|---|
| AWGN | −21.31 dB | **−22.49 dB** | **1.18 dB** |
| moderate fading (0.5 Hz spread / 1 ms delay) | −19.40 dB | **−20.12 dB** | **0.72 dB** |
| disturbed fading (1 Hz / 2 ms) | −19.19 dB | **−20.16 dB** | **0.97 dB** |
| 10,000 noise-only slots | – | 0 false decodes | – |

All three ft8nn rows are the shipped default, measured 2026-11-10 (90% thresholds: −21.24, −17.48 and −18.09 dB;
no false decodes on any of the three sets). The `jt9` column is the recorded baseline for the same sets, not re-run
for this table — the sets regenerate byte-identically from their seeds (1, 2, 3 in [Reproduce](#reproduce)), and the
AWGN 50% and 90% thresholds came out identical to the earlier run on different hardware, which is the check that
they do. 1.2 dB is roughly the difference between hearing a station and not: on FT8's exponential-ish SNR
distribution it is what produces the double-digit decode gains below.

### Extra decodes on real air: +23.5% over `jt9`, on receivers we never trained on

The validation set (`datasets/ft8-validation-set`) is KiwiSDR recordings of 120 consecutive 15 s slots per receiver,
together with every FT8 spot that PSK Reporter logged worldwide in those same slots. That outside spot list is the
ground truth: the set's own `tools/verify.py` (here `collect/verify.py`) checks each decode's sender against it
and puts the decode in a tier — **slot-confirmed** (somebody else also spotted that station in that very slot: as close to proof as real air gets)
down to **unconf** (nobody anywhere heard that station, so an unconfirmed decode is very probably a false one).
Both decoders are scored the same way on the same audio.

| run (unseen = no receiver shared with training) | slots | `jt9` decodes | ft8nn decodes | extra decodes | slot-confirmed | unconf (≈ false) |
|---|---|---|---|---|---|---|
| 1 — 9 training-site Kiwis, other bands and times | 1086 | 27,730 | 34,585 | **+6,855 (+24.7%)** | 21,492 → 26,505 (+23.3%) | 13 → 1 |
| 2 — 9 unseen Kiwis, Wed 22:30 UTC, 160–10 m | 1086 | 24,291 | 29,588 | **+5,297 (+21.8%)** | 17,993 → 21,751 (+20.9%) | 16 → 1 |
| 3 — 12 unseen Kiwis, Mon 05:32 UTC, 80–10 m | 1447 | 23,524 | 28,295 | **+4,771 (+20.3%)** | 17,888 → 21,324 (+19.2%) | 17 → 1 |
| 4 — 15 unseen Kiwis, Mon 15:00 UTC, 80–10 m | 1790 | 38,432 | 48,668 | **+10,236 (+26.6%)** | 28,053 → 35,249 (+25.7%) | 29 → 2 |
| **runs 2 + 3 + 4 — unseen receivers only** | **4323** | **86,247** | **106,551** | **+20,304 (+23.5%)** | 63,934 → 78,324 (+22.5%) | 62 → 4 |
| all four runs, 45 receivers | 5409 | 113,977 | 141,136 | +27,159 (+23.8%) | 85,426 → 104,829 (+22.7%) | 75 → 5 |

What this says:

- **About 20,300 extra decodes** out of 4,323 slots of unseen-receiver audio, i.e. **+4.7 decodes in every 15 s
  slot** on top of `jt9` — +5.7 on run 4's busy daytime high bands, +3.3 on run 3's quiet dawn ones.
- The gain is not a counting artefact: slot-confirmed decodes, the ones independently spotted by other receivers,
  rise almost as much (+22.5%); before the repeat stage, 99% of our extra decodes named a callsign `jt9` itself
  heard at that receiver.
- It is **not bought with false decodes** — the opposite. Across the four runs `jt9` leaves 75 unconfirmed
  decodes and ft8nn 5, and ft8nn finds no decode at all in 10,000 noise-only slots. FT8's CRC-14 plus the LDPC
  code does the filtering, and we kept both; the repeat stage, which skips both, is held to 0 accepted in 223,643
  tests with a wrong receiver's context, and of its decodes that `jt9` or JTDX also decoded, 1,936 of 1,939 carry
  the same message (`STATUS.md` plan 12). The co-channel retry, which decodes at places no sync search proposed,
  made 0 false decodes from 1.17 million hypotheses forced onto noise-only audio, and none of its 874 decodes on
  the four runs is unconfirmed (`STATUS.md` plan 7.1 item 1c).
- Runs 2, 3 and 4 are the honest numbers (receivers, days, times and bands never used for training, and never
  used to tune a threshold). Run 1 reuses training receivers on other bands and times, and scores in the same
  range. The gain grows with band activity: run 4's crowded afternoon high bands (up to 45 decodes per slot)
  give +26.6%, run 3's quiet dawn bands +20.3%.
- ft8nn is not a superset of `jt9`: some decodes are `jt9`-only (948 of run 4's 38,432, 850 on run 2 when that
  count was last measured, with `nn_v3`), and surprisingly many of them are strong signals rather than marginal ones — 59 of
  run 4's at ≥ 0 dB. Diagnosing the remainder is open work — see `STATUS.md`.

### Against JTDX: +14.1% over its defaults, +9.0% over its most sensitive settings

JTDX 2.2.x (built from `git://git.code.sf.net/p/jtdx/code` by `tools/jtdx_build.py`) is the WSJT-X fork tuned for
weak-signal FT8, so it is the stronger classical baseline. Same unseen runs, same audio, same scorer as above;
"JTDX deep" is JTDX at its most sensitive settings (`-w -c 3 -s 2`: SWL mode, 9 subtraction passes, extra
low-threshold pass), which costs it 5× the CPU of its defaults. `nn_v3` is our decoder before the repeat stage, DT re-centring and the co-channel retry, for reference.
Decodes / slot-confirmed / unconf:

| run | `jt9` | JTDX defaults | JTDX deep | `nn_v3` | **ft8nn (`nn_v3_cc`)** |
|---|---|---|---|---|---|
| 2 | 24,291 / 17,993 / 16 | 26,293 / 19,435 / 1 | 27,417 / 20,202 / 16 | 28,198 / 20,701 / 1 | **29,588 / 21,751 / 1** |
| 3 | 23,524 / 17,888 / 17 | 25,279 / 19,125 / 3 | 26,321 / 19,855 / 9 | 26,688 / 20,105 / 0 | **28,295 / 21,324 / 1** |
| 4 | 38,432 / 28,053 / 29 | 41,820 / 30,599 / 7 | 43,973 / 32,103 / 19 | 45,744 / 33,070 / 2 | **48,668 / 35,249 / 2** |
| **runs 2 + 3 + 4** | 86,247 / 63,934 / 62 | 93,392 / 69,159 / 11 | 97,711 / 72,160 / 44 | 100,630 / 73,876 / 3 | **106,551 / 78,324 / 4** |

ft8nn's margin over JTDX, decodes / slot-confirmed:

| run | `nn_v3` vs JTDX defaults | **ft8nn vs JTDX defaults** | `nn_v3` vs JTDX deep | **ft8nn vs JTDX deep** |
|---|---|---|---|---|
| 2 | +7.2% / +6.5% | **+12.5% / +11.9%** | +2.8% / +2.5% | **+7.9% / +7.7%** |
| 3 | +5.6% / +5.1% | **+11.9% / +11.5%** | +1.4% / +1.3% | **+7.5% / +7.4%** |
| 4 | +9.4% / +8.1% | **+16.4% / +15.2%** | +4.0% / +3.0% | **+10.7% / +9.8%** |
| **runs 2 + 3 + 4** | +7.8% / +6.8% | **+14.1% / +13.3%** | +3.0% / +2.4% | **+9.0% / +8.5%** |

- Pooled, that is 13,159 more decodes than JTDX at its defaults (9,165 more slot-confirmed) and 8,840 more than
  JTDX at its ceiling (6,164 more slot-confirmed).
- JTDX's extra sensitivity costs it false decodes: its deep settings raise unconf from 11 to 44, toward `jt9`'s 62;
  ft8nn has 4.
- Most of the jump from `nn_v3` comes from the repeat stage (a station's message from 30/60 s earlier tested whole
  against the audio — the idea behind JTDX's own `^` stage, which this comparison found was most of what JTDX
  decoded and we missed), about 1 point from DT re-centring and half a point from the co-channel retry (a second
  station sharing a decoded station's frequency and time, whose sync the subtraction of the first took away). Details, the JTDX build and the FT8TW / ft8_lib
  comparison: `docs/decoder-comparison.md`.

`STATUS.md` has the per-decoder-version history behind these tables and the commands that produce them.

## System requirements

Two machines have been tested: Apple M1 Pro (10 cores, 16 GB, macOS 26) and Mac mini M4 (4 performance +
6 efficiency cores, 16 GB, macOS 27), both arm64. The C timings below were measured on the M4; the Python ones on
the M1 Pro. Other platforms and smaller machines have not been tried.

### Decoding with the C library (`c/`, `libft8nn`)

| | Requirement |
|---|---|
| OS | macOS arm64 (tested) and Linux: CI (`.github/workflows/linux.yml`) builds and tests on Ubuntu 24.04 x64 and arm64, including a C-vs-Python decode check (arm64 also verified locally with `act`). Native Windows needs its own build files; the code has `_WIN32` branches for threads, clocks and paths that have not been compiled yet. |
| Build tools | C11 compiler (tested: Apple clang 21; the code is plain C11 without variable-length arrays or 128-bit integers) and `make` |
| Libraries | ONNX Runtime 1.31.1, the official release: `tools/get_onnxruntime.sh` downloads the build for this machine (macOS arm64/x86_64, Linux x64/aarch64, about 35 MB) into `third_party/onnxruntime`, where `make -C c` finds it. Homebrew's `onnxruntime` is used as a fallback, but it runs this model 2.3× slower. `make -C c ORT_PREFIX=<dir>` uses any other copy. No other dependencies. |
| Model | `ft8nn.onnx`, 3.5 MB (exported from a trained checkpoint by `tools/export_onnx.py`) |
| Memory | about 540 MB peak resident per decoding context on the busiest receivers (360–570 MB across stations) |
| CPU | To keep up in real time every 15 s slot must finish decoding within 15 s. On one core, slots average 2.8 s (first 20 slots of each of the 10 training stations, max 7.5 s). The busiest receiver tested (40 m, about 48 decodes per slot) takes 6.9 s per slot (max 8.0 s), so one core keeps up with room to spare. Measured with `ft8nn_cli --time` and the official ONNX Runtime build, machine awake and otherwise idle (Homebrew's build: 10.9 s per busy slot). `--threads N` splits every pass across N cores with exactly the same decodes (0.3% more instructions); the WSJT-X sidecar uses one thread per performance core and hands out decodes while a pass is still running, so the strongest signals of a busy slot appear about 0.3 s after the decode starts (STATUS.md plan 6). |

### Decoding with the Python reference (`ft8nn/`)

| | Requirement |
|---|---|
| Python | ≥ 3.12, managed with [uv](https://docs.astral.sh/uv/) |
| Packages | numpy, scipy, torch (the environment is about 0.9 GB on disk) |
| Model | `model.pt`, 3.4 MB |
| Memory | about 690 MB peak resident for one decoding process |
| CPU | about 4 s per slot per core on average (torch uses Accelerate on Apple Silicon), measured before the 4th decode pass and context a-priori decoding were added, so now higher |

### Training and benchmarks

- Training: the v2 recipe takes about 4 h on M-series with MPS (Apple GPU). It has not been tried on CPU only
  or with less than 16 GB RAM.
- jt9 baselines need WSJT-X's `jt9` at `/usr/local/bin/jt9`. The real-recording benchmarks need the recordings: git
  submodules under `datasets/` (Git LFS, ~680 MB packed), fetched and unpacked as in `docs/data-collection.md`:
  `git submodule update --init`, then `tools/datarepo.py unpack` in each.

## Layout

| Path | What |
|---|---|
| `models/v3/` | the shipped model: `model.pt` (PyTorch), `ft8nn.onnx` (for the C library), training `log.json` |
| `models/v2/` | its predecessor and starting point (synthetic noise only) |
| `ft8nn/ft8.py` | CRC-14, LDPC encode, tone map, GFSK waveform (bit-exact vs WSJT-X `ft8code`) |
| `ft8nn/channel.py` | 12 kHz slot synthesis, AWGN/Watterson fading, ft8sim-calibrated SNR |
| `ft8nn/frontend.py` | candidate search, 200 Hz downsampler, semi-coherent Costas fine sync, classical metrics |
| `ft8nn/ldpc.py` | batched BP + ordered-statistics decoding (OSD) |
| `ft8nn/synth_bb.py` | on-the-fly batched training data at baseband (torch), matched to the front end |
| `ft8nn/model.py` | `SoftDemod` (conv + transformer) and feature extraction |
| `ft8nn/decoder.py` | end-to-end slot decoder with pluggable metric |
| `train.py` | training with built-in NN-vs-classical BP validation |
| `bench/` | benchmark set generation, jt9 runner, our runner, comparison report |
| `tools/` | LDPC table extraction, message pool, sync diagnostics |
| `datasets/` | submodules: the training recordings and the validation set (packed; unpack in place) |
| `collect/` | real-data collection: KiwiSDR recorder, PSK Reporter capture, jt9 labelling, spot scoring, run records and SHA-256 manifests (`docs/data-collection.md`) |
| `wsjtx/` | `ft8nn_sidecar`: this decoder as WSJT-X's external FT8 decoder (`WSJTX_FT8_DECODER`, JSON lines + raw audio on a pipe), so WSJT-X decodes FT8 with it (`wsjtx/README.md`) |

## Tools

### Live receiver directory and comparison

`collect/receiverbook_directory.py` builds `datasets/receiverbook-directory.json` from the OpenWebRX receivers
listed by [Receiverbook](https://www.receiverbook.de/). A receiver-band pair is included only when its
`metrics.json` advertises an FT8 decoder, a public WebSocket session can select a profile covering the standard
FT8 dial frequency, and that session actually delivers audio samples. The builder takes two `metrics.json`
snapshots 60 seconds apart and records the per-band counter delta as `ft8_decodes_per_minute`; the directory is
sorted from highest to lowest measured rate. It also retains the second snapshot's cumulative counter as
`ft8_decodes_total` for context. A counter reset or unavailable first snapshot produces a zero rate. Use
`--traffic-window SECONDS` to change the sampling interval; rates are normalized to one minute.

```sh
python3 collect/receiverbook_directory.py                 # measure 60 s traffic, then verify streams
make -C c                                                 # build libft8nn if needed
python3 tools/live_compare.py                             # choose a receiver and compare live decodes
```

`live_compare.py` presents a curses receiver picker with arrow-key navigation and substring filtering. Once a
receiver-band pair is selected, it consumes the same OpenWebRX audio and remote FT8-decoder messages, divides the
audio into UTC FT8 slots, and decodes it locally with `libft8nn`. Local and remote results are shown in two
frequency-ordered columns: matching messages share a row, while a decode found by only one side has a blank in
the other column. The picker and streaming header show the measured FT8 decodes per minute; the header also shows
the receiver's `metrics.json` URL and cumulative counter snapshot. Press `q` to stop the comparison.

### Script reference

| Script | Description |
|---|---|
| `tools/ab_run2.sh` | Runs the quick decoder-change A/B gate: noise-only false-decode testing followed by validation run 2 and PSK Reporter scoring. |
| `tools/calib_snr.py` | Calibrates the decoder's estimated SNR against known SNRs in synthetic benchmark slots. |
| `tools/ci_smoke.py` | Generates a small synthetic workload and checks that the C and Python decoding pipelines produce identical results without needing recorded data. |
| `tools/compare_decoders.py` | Compares verified decoder outputs across validation runs, including totals, confirmation tiers, overlap, and exclusive signals. |
| `tools/diag_ctx_miss.py` | Diagnoses reference-only repeated messages that context a-priori decoding should have been able to recover. |
| `tools/diag_real_miss.py` | Replays a real slot and traces each `jt9`-only signal through residual candidate search, fine synchronization, and NN/LDPC decoding. |
| `tools/diag_ref_only.py` | Profiles signals decoded by a reference decoder, normally JTDX, but missed by ft8nn. |
| `tools/diag_strong_miss.py` | Classifies stronger `jt9`-only signals by the pipeline stage at which ft8nn loses them. |
| `tools/diag_sync.py` | Measures where weak synthetic signals are lost by comparing candidate search and fine synchronization with oracle synchronization. |
| `tools/diag_sync2.py` | Compares the old and semi-coherent fine synchronizers for timing/frequency accuracy and decode rate. |
| `tools/export_onnx.py` | Exports a PyTorch `SoftDemod` checkpoint to ONNX and verifies its output with ONNX Runtime. |
| `tools/extract_ldpc.py` | Extracts the FT8 LDPC generator and parity-check tables from ft8_lib into `ft8nn/data/ldpc.npz`. |
| `tools/ft8lib_build.sh` | Builds kgoba/ft8_lib's file decoder for use as an FT8CN/FT8TW-family comparison baseline. |
| `tools/ft8lib_run.py` | Runs ft8_lib over validation recordings and writes `jt9`-format output for the verification pipeline. |
| `tools/gen_c_tables.py` | Generates the C LDPC header from the same NumPy tables used by the Python decoder. |
| `tools/gen_call_blocks.py` | Generates the Python and C amateur-call-prefix tables used for message plausibility checks. |
| `tools/get_onnxruntime.sh` | Downloads the official ONNX Runtime release for the current platform and installs the repository-local link used by the C build. |
| `tools/golden_c.py` | Checks C-decoder output field by field against saved Python-decoder results on real or synthetic slots. |
| `tools/jtdx_build.py` | Clones and builds the JTDX FT8 decoder together with the repository's WAV-file driver. |
| `tools/jtdx_ft8_file.f90` | Provides a command-line WAV-file front end for JTDX's otherwise shared-memory-only FT8 decoder. |
| `tools/jtdx_run.py` | Runs JTDX over validation stations in chronological order and writes output for verification. |
| `tools/live_compare.py` | Selects a verified OpenWebRX receiver and compares local ft8nn and remote FT8 decodes live in aligned columns. |
| `tools/make_message_pool.py` | Uses WSJT-X `ft8code` to generate realistic FT8 messages and their 77-bit payloads for benchmarks. |
| `tools/make_noise_bank.py` | Builds a training bank of real residual receiver noise after subtracting known decodes from recorded slots. |
| `tools/mini.sh` | Synchronizes code, data, and results with a configured Mac mini and runs experiments there. |
| `tools/rp_eval.py` | Evaluates the repeat-decoding stage from saved base decodes without rerunning the complete decoder. |
| `tools/rp_stale.py` | Detects repeat-stage decodes that incorrectly reuse an older message after a station has moved on. |
| `tools/try_real_slots.py` | Quickly compares ft8nn and `jt9` on a selected set of real receiver slots. |
| `tools/validate_full.sh` | Runs the complete pre-commit decoder validation: real runs, noise, AWGN, C build, and C/Python golden comparisons. |

## Reproduce

```sh
uv run pytest -q tests                                     # encoder vs ft8code, LDPC, calibration
# benchmark sets (byte-identical to the ones behind all results; 10 signals per 15 s slot, 30 slots per SNR)
uv run python bench/make_set.py --out data/bench/awgn --channel awgn --snrs=-25:-16 --files 30 --seed 1
uv run python bench/make_set.py --out data/bench/moderate --channel moderate --snrs=-24:-12 --files 30 --seed 2
uv run python bench/make_set.py --out data/bench/disturbed --channel disturbed --snrs=-24:-10 --files 30 --seed 3
uv run python bench/make_set.py --out data/bench/noise --channel awgn --snrs=-80:-80 --files 100 --seed 9   # noise only
uv run python bench/make_set.py --out data/bench/noise10k --channel awgn --snrs=-80:-80 --files 10000 --seed 19   # 10,000 noise-only slots (false-rate bound)
uv run python bench/run_jt9.py data/bench/awgn             # WSJT-X baseline (needs jt9 on PATH or /usr/local/bin)
# the shipped model is models/v3; to retrain it: v1 and v2 (74 min on an M4 with MPS, ~4 h on an M1 Pro) ...
uv run python train.py --cfg v1 --steps 6000 --out runs/v1
uv run python train.py --cfg v2 --init runs/v1/model.pt --lr 5e-4 --steps 8000 --out runs/v2
# ... then v3 = v2 fine-tuned with real receiver noise from the training recordings (needs datasets/, 21 min on the M4):
uv run python bench/run_real.py datasets/ft8-training-data/recordings --model models/v2/model.pt --also-classical --tag nn_v2_pl --workers 6
uv run python tools/make_noise_bank.py datasets/ft8-training-data/recordings data/real nn_v2_pl data/noise/train_bank.npy
uv run python train.py --cfg v2 --init models/v2/model.pt --lr 2e-4 --steps 4000 --noise-bank data/noise/train_bank.npy --real-noise-p 0.5 --seed 1 --workers 8 --out runs/v3
uv run python bench/run_ours.py data/bench/awgn --tag nn --model models/v3/model.pt --also-classical
uv run python bench/report.py data/bench/awgn --dt-max 0.5
# real recordings and the validation set: see STATUS.md and docs/data-collection.md
```

C library:

```sh
tools/get_onnxruntime.sh                                    # official ONNX Runtime into third_party/ (once)
make -C c                                                   # c/build/libft8nn.{a,dylib|so}, c/build/ft8nn_cli
uv run python tools/export_onnx.py models/v3/model.pt models/v3/ft8nn.onnx
c/build/ft8nn_cli models/v3/ft8nn.onnx slot.wav               # one line per decode; --threads N, --time
```

Inside WSJT-X (`wsjtx/README.md` has the install steps and the limits):

```sh
make -C wsjtx                                                # wsjtx/build/ft8nn_sidecar
make -C wsjtx test                                           # drives the whole sidecar protocol
```

Benchmark scoring: 10 signals per 15 s slot, real messages from `ft8nn/data/messages.json`;
a decode counts only on exact text match, anything else is a false decode.
