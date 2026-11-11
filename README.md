# ft8nn — neural soft-demodulation FT8 decoder

Goal: an FT8 decoder more sensitive than WSJT-X, packaged so other software can embed it.

Approach: keep FT8's LDPC(174,91) + CRC-14 (near-optimal, gives false-decode protection) and replace
the hand-made soft metrics with a small neural net that turns baseband symbol spectra into bit LLRs.

## Results vs stock WSJT-X `jt9`

Baseline everywhere below: WSJT-X's own FT8 decoder run as `jt9 -8 -d 3`, i.e. depth 3, its deepest routine
search (`bench/run_jt9.py`; the real-recording baselines were made with WSJT-X Improved 3.0.2, see
`docs/data-collection.md`).
Ours: the shipped default, `models/v3` + the decoder defaults (4 passes with subtraction, OSD, CQ and context
a-priori decoding, and the repeat stage that tests a station's message from 30/60 s earlier as a whole known
codeword), tag `nn_v3_rpf`. The AWGN and fading rows are unaffected by the repeat stage (no slot times, so no
context) and were measured with `nn_v3`.

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

### Extra decodes on real air: +22% over `jt9`, on receivers we never trained on

The validation set (`datasets/ft8-validation-set`) is KiwiSDR recordings of 120 consecutive 15 s slots per receiver,
together with every FT8 spot that PSK Reporter logged worldwide in those same slots. That outside spot list is the
ground truth: the set's own `tools/verify.py` (here `collect/verify.py`) checks each decode's sender against it
and puts the decode in a tier — **slot-confirmed** (somebody else also spotted that station in that very slot: as close to proof as real air gets)
down to **unconf** (nobody anywhere heard that station, so an unconfirmed decode is very probably a false one).
Both decoders are scored the same way on the same audio.

| run (unseen = no receiver shared with training) | slots | `jt9` decodes | ft8nn decodes | extra decodes | slot-confirmed | unconf (≈ false) |
|---|---|---|---|---|---|---|
| 1 — 9 training-site Kiwis, other bands and times | 1086 | 27,730 | 34,295 | **+6,565 (+23.7%)** | 21,492 → 26,299 (+22.4%) | 13 → 1 |
| 2 — 9 unseen Kiwis, Wed 22:30 UTC, 160–10 m | 1086 | 24,291 | 29,435 | **+5,144 (+21.2%)** | 17,993 → 21,646 (+20.3%) | 16 → 1 |
| 3 — 12 unseen Kiwis, Mon 05:32 UTC, 80–10 m | 1447 | 23,524 | 27,711 | **+4,187 (+17.8%)** | 17,888 → 20,883 (+16.7%) | 17 → 1 |
| 4 — 15 unseen Kiwis, Mon 15:00 UTC, 80–10 m | 1790 | 38,432 | 48,008 | **+9,576 (+24.9%)** | 28,053 → 34,840 (+24.2%) | 29 → 2 |
| **runs 2 + 3 + 4 — unseen receivers only** | **4323** | **86,247** | **105,154** | **+18,907 (+21.9%)** | 63,934 → 77,369 (+21.0%) | 62 → 4 |
| all four runs, 45 receivers | 5409 | 113,977 | 139,449 | +25,472 (+22.3%) | 85,426 → 103,668 (+21.4%) | 75 → 5 |

What this says:

- **About 18,900 extra decodes** out of 4,323 slots of unseen-receiver audio, i.e. **+4.4 decodes in every 15 s
  slot** on top of `jt9` — +5.3 on run 4's busy daytime high bands, +2.9 on run 3's quiet dawn ones.
- The gain is not a counting artefact: slot-confirmed decodes, the ones independently spotted by other receivers,
  rise almost as much (+21.0%); before the repeat stage, 99% of our extra decodes named a callsign `jt9` itself
  heard at that receiver.
- It is **not bought with false decodes** — the opposite. Across the four runs `jt9` leaves 75 unconfirmed
  decodes and ft8nn 5, and ft8nn finds no decode at all in 10,000 noise-only slots. FT8's CRC-14 plus the LDPC
  code does the filtering, and we kept both; the repeat stage, which skips both, is held to 0 accepted in 223,643
  tests with a wrong receiver's context, and of its decodes that `jt9` or JTDX also decoded, 2,022 of 2,025 carry
  the same message (`STATUS.md` plan 12).
- Runs 2, 3 and 4 are the honest numbers (receivers, days, times and bands never used for training, and never
  used to tune a threshold). Run 1 reuses training receivers on other bands and times, and scores in the same
  range. The gain grows with band activity: run 4's crowded afternoon high bands (up to 45 decodes per slot)
  give +24.9%, run 3's quiet dawn bands +17.8%.
- ft8nn is not a superset of `jt9`: some decodes are `jt9`-only (948 of run 4's 38,432, 850 on run 2 when that
  count was last measured, with `nn_v3`), and surprisingly many of them are strong signals rather than marginal ones — 59 of
  run 4's at ≥ 0 dB. Diagnosing the remainder is open work — see `STATUS.md`.

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
| CPU | To keep up in real time every 15 s slot must finish decoding within 15 s. On one core, slots average 2.8 s (first 20 slots of each of the 10 training stations, max 7.5 s). The busiest receiver tested (40 m, about 48 decodes per slot) takes 6.9 s per slot (max 8.0 s), so one core keeps up with room to spare; `--threads 2` (ONNX Runtime threads) gives only 6.2 s, because the NN is no longer most of the time. Measured with `ft8nn_cli --time` and the official ONNX Runtime build, machine awake and otherwise idle (Homebrew's build: 10.9 s per busy slot). |

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
