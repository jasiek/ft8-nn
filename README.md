# ft8nn — neural soft-demodulation FT8 decoder

`ft8nn` is a high-sensitivity FT8 decoder implemented as an embeddable C11 library (`libft8nn`), a drop-in external FT8 decoder sidecar for WSJT-X (`ft8nn_sidecar`), and a bit-matched Python/PyTorch reference and training pipeline.

Instead of replacing FT8's error-correcting code, `ft8nn` keeps FT8's exact **LDPC(174,91) + CRC-14** structure — preserving algebraic protection against false decodes — and replaces hand-crafted tone-energy soft metrics with **`SoftDemod`** (`models/v4`, 0.84M parameters: 1-D dilated convolutions + 4-layer Transformer), which maps 200 Hz baseband complex symbol spectra directly into calibrated per-bit log-likelihood ratios (LLRs). Around the neural demodulator, the decoder runs:

- **4 decode-and-subtract passes** with exact `sin²`-shaped GFSK waveform reconstruction and interference cancellation;
- **Belief propagation (BP) + ordered-statistics decoding (OSD-2)**, with gated **OSD-3** fallback inside multi-constraint verified stages;
- **DT re-centring** for receivers with clock offset or packet drift, plus **SNR baseline estimation** matched to WSJT-X 3.0 (`±0.1 dB` median vs `jt9` across 80k shared decodes);
- **3-minute station context a-priori (`30..180 s`)** for standard and `i3=4` non-standard callsigns, **cross-slot soft LLR Chase combining (`30/60 s`)**, **full-codeword repeat & clear-winning successor correlation**, **co-channel collision recovery**, and **3×3 micro-shift refinement** on marginal hypotheses.

---

## Results

All comparisons below evaluate the shipped **`models/v4`** decoder (`libft8nn` / `bench/run_real.py` defaults) against four external decoders on identical 12 kHz mono WAV recordings:

- **`jt9` deep (`jt9 -8 -d 3`)** — WSJT-X 3.0.2 Improved at its deepest search setting;
- **`jtdx` (`JTDX 2.2.x` defaults)** — `ndepth=3`, `nft8cycles=1`, low sync thresholds, duplicate suppression on (`tools/jtdx_ft8_file.f90`);
- **`jtdx` deep (`JTDX -w -c 3 -s 2`)** — JTDX at its sensitivity ceiling (SWL 51-block mode, 9 subtraction passes, extra sub-pass; ~8–10× `ft8nn` CPU);
- **`ft8lib` (`kgoba/ft8_lib` `decode_ft8`)** — lightweight single-pass BP baseline (the reference implementation family used by mobile apps such as FT8CN/FT8TW).

### Sensitivity

Synthetic 15 s benchmark sets (`bench/make_set.py`: 10 signals per slot, 30 slots per 1 dB SNR step, real messages from `ft8nn/data/messages.json`; a decode counts only on exact text match with `|DT| ≤ 0.5 s`, anything else is a false decode). Thresholds are SNR in 2,500 Hz (WSJT-X convention) for 50% decode probability — **lower is better**:

| Channel (`bench/make_set.py`, `models/v4`) | `jt9 -8 -d 3` (50%) | `ft8nn` (50% / 90%) | Gain at 50% | False decodes |
|---|---:|---:|---:|---:|
| **AWGN** (`seed 1`, 300 slots) | −21.31 dB | **−22.55 dB** / −21.29 dB | **+1.24 dB** | 0 / 3,000 signals |
| **Moderate Watterson fading** (`0.5 Hz` Doppler / `1 ms` delay, `seed 2`, 390 slots) | −19.40 dB | **−20.27 dB** / −17.70 dB | **+0.87 dB** | 0 / 3,900 signals |
| **Disturbed Watterson fading** (`1.0 Hz` Doppler / `2 ms` delay, `seed 3`, 450 slots) | −19.19 dB | **−20.23 dB** / −18.19 dB | **+1.04 dB** | 0 / 4,500 signals |
| **Pure Gaussian noise** (`10,000` slots `noise10k` + `100` slots `noise`) | – | – | – | **0 false decodes** |

On HF's steep weak-signal density distribution, ~1.0–1.2 dB of threshold gain translates into **+22–28% more real off-air decodes** per 15 s period without raising the false-decode floor.

### Extra Decodes

#### Multi-receiver HF validation set (`datasets/ft8-validation-set`, `runs/r02` & `r05`–`r14`)

The off-air validation set comprises **7,172 slots** from **60 public KiwiSDR receivers** across all 10 HF FT8 bands (`160 m`–`10 m`), all 6 continents, and day/twilight/night propagation — strictly disjoint in receiver location (`0.1°` + URL) from the training recordings and from the earlier v1 validation runs (`excluded_sites.json`). Each band is recorded simultaneously on groups of 3–4 KiwiSDRs `50–1,500 km` apart alongside live OpenWebRX real-time FT8 decode streams and global PSK Reporter MQTT spots.

Each block in `sets/{small,third,full}.txt` decodes 2 unscored lead-in slots followed by 8 scored slots (2 min) as an independent sequence so stateful decoders (`ft8nn`, `JTDX`) have identical warm-up across set sizes. `collect/verify.py` classifies every checkable decode (unhashed transmitter callsign) into:

1. **`slot`** — transmitter spotted on PSK Reporter by a reporter **> 50 km** from our receiver in the exact 15 s slot (`±50 Hz` after per-receiver calibration);
2. **`xconf`** — identical message decoded without a-priori information in the same slot by another recorded KiwiSDR or live OpenWebRX stream **≥ 50 km** away (`±50 Hz`, `±0.5 s` of 6-min pair median DT);
3. **`confirmed` = `slot + xconf`** — independent same-slot external corroboration;
4. **`active`** / **`repeat`** — transmitter spotted on the band within `±10 min`, or decoded in `≥ 2` slots on the same receiver;
5. **`unconf`** — never spotted on the band during the session and heard only once (false-decode proxy);
6. **`xconf-strict` (`--others`)** — for decodes *exclusive* to a single decoder on that receiver, requires another SDR `≥ 50 km` away to have decoded the identical message within **4 Hz** and **0.2 s**.

**5-decoder head-to-head on `sets/third.txt` (`265` blocks, `2,650` decoded / `2,120` scored slots across all 11 runs `r02`, `r05`–`r14`; 8-core Apple M1, `6` workers):**

| Decoder | Decodes | vs `jt9` | Checkable | `slot` | `xconf` | **Confirmed (`slot+xconf`)** | vs `jt9` | Conf % | `active` | `unconf` | Core-s / slot |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `ft8lib` (`decode_ft8`) | 29,780 | −44.2% | 29,175 | 22,029 | 6,527 | 28,556 | −43.6% | 97.9% | 618 | **0** | 0.02 s |
| `jt9` deep (`jt9 -8 -d 3`) | 53,322 | – | 52,301 | 39,445 | 11,228 | 50,673 | – | 96.9% | 1,577 | **44** | 3.66 s |
| `jtdx` (2.2.x defaults) | 56,642 | +6.2% | 56,120 | 41,862 | 12,168 | 54,030 | +6.6% | 96.3% | 2,081 | **5** | 4.39 s |
| `jtdx` deep (`-w -c 3 -s 2`) | 59,340 | +11.3% | 58,800 | 43,704 | 12,665 | 56,369 | +11.2% | 95.9% | 2,403 | **20** | 33.05 s |
| **`ft8nn` (`models/v4`, C `3A`)** | **65,919** | **+23.6%** | **64,783** | **48,169** | **13,649** | **61,818** | **+22.0%** | **95.4%** | **2,951** | **6** | **4.04 s** |

- **Gain over WSJT-X `jt9 -8 -d 3`:** **`+12,597` extra decodes (`+23.6%`, `+5.94` extra decodes in every 15 s slot)** and **`+11,145` extra confirmed decodes (`+22.0%`)**, winning on **all 11 of 11 runs** (`+9.5%` on quiet dawn 60 m up to `+41.8%` on crowded evening 40/30 m).
- **Gain over JTDX:** **`+16.4%` decodes (`+9,277`) / `+14.4%` confirmed (`+7,788`)** over `jtdx` defaults while running **faster per core** (`4.04 s` vs `4.39 s`), and **`+11.1%` decodes (`+6,579`) / `+9.7%` confirmed (`+5,449`)** over `jtdx` deep (`-w -c 3 -s 2`) at **`8.2×` lower CPU** (`29.8 min` vs `244.0 min` wall).
- **False-decode audit (`unconf`):** Of `ft8nn`'s **6** `unconf` rows in `65,919` decodes, **4 are genuine unspotted or free-text on-air signals** (`CQ D01GMW JO31` and `CQ DX IN80CE`, both decoded via plain BP by all four deep decoders; `<UA1AAP> R8QBH/1`, an active QSO partner also decoded by `jt9` and `jtdx_deep`; and `DL1NB JO30`, a free-text grid message from active station `DL1NB` where `verify.py` parsed `JO30` as the sender), leaving **only 2 actual false decodes in 65,919 (`0.003%`)** — vs **17** random false decodes in `jtdx` deep (including 4 `DE <call> TU` + contest-exchange double hallucinations) and **41** in `jt9`.

**Strict exclusive confirmation on `sets/third.txt` (`verify.py --others`, 4 Hz / 0.2 s on another SDR ≥ 50 km away):**

| Decoder | Exclusive decodes | `xconf-strict` | Strict % | `unconfirmable` | `unconfirmable` base tiers |
|---|---:|---:|---:|---:|---|
| `ft8lib` | 5 | 3 | 60.0% | 2 | `slot`: 1, `active`: 1 |
| `jtdx` | 96 | 45 | 46.9% | 51 | `slot`: 26, `active`: 24, `unconf`: 1 |
| `jt9` deep | 167 | 88 | 52.7% | 79 | `slot`: 27, `active`: 11, `unconf`: 41 |
| `jtdx` deep | 516 | 186 | 36.0% | 330 | `slot`: 173, `xconf`: 1, `active`: 140, `unconf`: 16 |
| **`ft8nn` (`models/v4`)** | **7,102** | **4,590** | **64.6%** | **2,512** | **`slot`: 1,658, `xconf`: 8, `active`: 843, `unconf`: 3** |

`ft8nn` recovers **7,102 decodes that none of `{jt9, ft8lib, jtdx, jtdx_deep}` found on the same receiver**; **64.6% (`4,590`)** are independently verified within `4 Hz` and `0.2 s` on another SDR `≥ 50 km` away, and `99.9%` of the remainder (`2,509 / 2,512`, mostly on single-Kiwi continents or weaker than the partner SDRs' `jt9` confirmers) are PSK Reporter `slot` or `active`.

**Pairwise callsign-slot overlap on `sets/third.txt` (`(run, station, slot, sender)` pairs):**

| Comparison (`A` vs `B`) | Decoded by both | `A`-only (`confirmed` / `active` / `unconf`) | `B`-only (`confirmed` / `active` / `unconf`) |
|---|---:|---|---|
| `jt9` vs `ft8nn` (`nn_v4`) | 51,416 | 769 (`694` / `33` / `41`) | **13,211 (`11,808` / `1,398` / `3`)** |
| `jtdx` vs `ft8nn` (`nn_v4`) | 54,265 | 1,675 (`1,473` / `199` / `2`) | **10,362 (`9,241` / `1,113` / `4`)** |
| `jtdx` deep vs `ft8nn` (`nn_v4`) | 56,414 | 2,150 (`1,829` / `307` / `13`) | **8,213 (`7,287` / `922` / `3`)** |

**Quick-check `sets/small.txt` (`16` blocks, `128` scored slots across all 10 bands & 6 continents):** ranks all 5 decoders in the exact same order — `ft8lib` `1,545` (`−44.6%`, `0` `unconf`), `jt9` `2,789` (`2` `unconf`), `jtdx` `2,970` (`+6.5%`, `0` `unconf`), `jtdx` deep `3,128` (`+12.2%`, `2` `unconf`), and **`ft8nn` `3,396` (`+21.8%` vs `jt9`, `+14.3%` vs `jtdx`, `+8.6%` vs `jtdx` deep, `0` `unconf`, `3.22 s/slot/core`)**.

#### Archived v1 unseen validation runs (`val2`–`val4`, full 30-minute continuous streams)

On the earlier v1 full-receiver validation runs (`120` consecutive slots per receiver, PSK Reporter `slot` confirmation without `xconf`; see `docs/decoder-comparison.md` and `STATUS.md`):

- **Unseen runs 2 + 3 (`21` fresh KiwiSDRs, `2,524` slots, `models/v4`):** **`61,169` decodes / `45,313` slot-confirmed / `2` unconf** (1 real unspotted DX `K1BMW UA0LAQ -17` spotted 13 min later, 1 actual false) vs `jt9 -d 3`'s **`47,815 / 35,881 / 33`** (**`+27.9%` decodes / `+26.3%` slot-confirmed**), `JTDX` defaults' **`51,572 / 38,560 / 4`** (**`+18.6%` / `+17.5%`**), and `JTDX` deep's **`53,738 / 40,057 / 25`** (**`+13.8%` / `+13.1%`**).
- **Repeat & Chase stage integrity:** across wrong-band shuffle nulls (`223,643` repeat tests and `106,000` cross-band Chase pairs), false acceptances are **0**, and on unseen Run 2 every one of the `205` `rp` decodes whose sender `jt9` also decoded matches `jt9` **205 / 205 (`100%`)** in exact message text (`0` stale repeats).

---

## Performance & Resource Footprint

`libft8nn` (`c/`) is plain C11 with zero external runtime dependencies on macOS (links Apple `-framework Accelerate`) and a single shared library (`onnxruntime`) on Linux and Windows:

| Platform / Backend | Busy 40 m (`~54` dec/slot), 1 thread | Busy 40 m, 4 P-cores (`--threads 0`) | Live 11.8 s early pass (`--early 141696`, 4T) | Peak RSS (4T) |
|---|---:|---:|---:|---:|
| **macOS arm64 default** (`c/src/nn_accel.c`: `Accelerate` AMX + NEON, parses `ft8nn.onnx` directly) | **3.94 s/slot** (max 4.18 s) | **1.61 s/slot** (max 1.72 s) | **0.80 s** (max 0.84 s; budget 2.60 s) | **155 MiB** |
| **Linux / Windows default** (`FT8NN_BACKEND=ort`: ONNX Runtime 1.31.1 + `230 KB` `screener.onnx`) | **5.69 s/slot** | **1.83 s/slot** | **0.62 s** (max 0.69 s; budget 2.60 s) | **287 MiB** (187 MiB 1T) |

*(Measured on an idle 8-core Apple M1, `caffeinate -dims`, 0 sleep events; Mac mini M4 is ~1.8× faster. Decoding is deterministic across thread counts; in streaming mode, the strongest pass-0 signals are emitted ~0.2–0.3 s after a request begins.)*

- **Option 3A (macOS):** `c/src/nn_accel.c` loads float32 weights straight from `models/v4/ft8nn.onnx` (`3.38 MB`) and executes `SoftDemod` in thread-local scratch via `cblas_sgemm` + ARM NEON (`max |accel − ort| = 6.7e-6`, **zero decode changes** vs ONNX Runtime), falling back to ONNX Runtime automatically for custom graphs or `FT8NN_BACKEND=ort`.
- **Option 3B (Linux / Windows):** pairs `ft8nn.onnx` with `models/v4/screener.onnx` (`53k` params, `230 KB` 1-layer `Screener` trained to predict `v4` expected bit errors), screening main-pass candidate hypotheses at `exp_err < 69.0` while running targeted `cochannel`, `refine`, and `repeat` stages unscreened (`99.99%` unseen real-air parity with unscreened `v4`, `0` noise false decodes).
- **ONNX Runtime privacy:** `ft8nn_open()` sets `ORT_DISABLE_TELEMETRY=1` and calls `DisableTelemetryEvents` before session creation.

---

## Quick Start

### 1. Build the C library & CLI (`c/`)

Supported in CI on **macOS arm64**, **Ubuntu 24.04 x64 & arm64** (`.github/workflows/linux.yml`), and **Windows x64** via MSYS2 MinGW-w64 UCRT64 (`.github/workflows/windows.yml`).

```sh
./tools/get_onnxruntime.sh      # downloads pinned official ONNX Runtime 1.31.1 into third_party/ (once)
make -C c                       # builds c/build/libft8nn.{a,dylib|so|dll} and c/build/ft8nn_cli

# Decode 15 s 12 kHz mono WAV slots (chronological order enables cross-slot context, Chase & hash resolution):
c/build/ft8nn_cli --threads 0 --time models/v4/ft8nn.onnx slot1.wav slot2.wav
```

Key C API entry points (`c/include/ft8nn.h`):
- `ft8nn_open(onnx_path, n_threads)` / `ft8nn_close(ctx)` — `n_threads = 0` uses one worker per performance core;
- `ft8nn_decode_at(ctx, audio12k, n, slot_time, out, max_out)` — full-slot decode with cross-slot context, Chase, and callsign hash learning;
- `ft8nn_decode_stream(...)` — streams finalized decode batches via callback during each pass;
- `ft8nn_decode_early(...)` + `ft8nn_decode_final(...)` — two-stage live decoding at `11.8 s` and `14.4 s`;
- `ft8nn_set_range(ctx, nfa, nfb)` — clamps search range to `100..4910 Hz` matching `jt9`.

### 2. Use inside WSJT-X (`wsjtx/ft8nn_sidecar`)

`ft8nn_sidecar` implements the external FT8 decoder pipe protocol (`WSJTX_FT8_DECODER`, documented in [`wsjtx/README.md`](wsjtx/README.md)): WSJT-X streams period metadata as JSON + 12 kHz PCM over stdin and receives streaming FT8 decodes over stdout, falling back cleanly to `jt9` if the sidecar exits or no model is found.

```sh
make -C c && make -C wsjtx
make -C wsjtx test              # runs 8 protocol, range, early/final sequence, and fallback checks

WSJTX_FT8_DECODER="$PWD/wsjtx/build/ft8nn_sidecar --model $PWD/models/v4/ft8nn.onnx" \
  /Applications/wsjtx.app/Contents/MacOS/wsjtx
```

### 3. Live off-air side-by-side comparison (`tools/live_compare.py`)

Streams live 12 kHz audio (`pcm-zstd` lossless when supported, Opus fallback) and remote FT8 decodes from public UberSDR instances with the skimmer add-on, private `--instance HOST:PORT` receivers, or live OpenWebRX servers, displaying local `libft8nn` decodes, remote decodes, and live TLS PSK Reporter MQTT spots (`mqtt.pskreporter.info:1884`) in aligned curses columns:

```sh
make -C c
python3 tools/live_compare.py                             # interactive picker (fair skimmer/OpenWebRX sources)
python3 tools/live_compare.py --instance 192.168.1.20:8080  # compare against a specific UberSDR instance
```

---

## Repository Layout

| Path | Contents |
|---|---|
| `models/v4/` | Shipped model: `model.pt` (PyTorch), `ft8nn.onnx` (`3.38 MB`), `screener.{pt,onnx}` (`230 KB` ORT pre-screener), `log.json` (`models/v3/` kept as its starting point) |
| `c/` | `libft8nn` portable C11 decoder (`src/nn_accel.c`, `decode.c`, `frontend.c`, `ldpc.c`, `subtract.c`, `fft.c`, `par.c`) and `ft8nn_cli` |
| `wsjtx/` | `ft8nn_sidecar` external FT8 decoder for WSJT-X + `fake_wsjtx` protocol test harness (`wsjtx/README.md`) |
| `ft8nn/` | Python reference decoder (`ft8.py`, `frontend.py`, `model.py`, `nn_metric.py`, `ldpc.py`, `decoder.py`, `subtract.py`, `message.py`) and synthetic baseband generator (`synth_bb.py`, `channel.py`) |
| `train.py` | Synthetic + real-noise-bank PyTorch training loop (`--cfg v4`) |
| `bench/` | Synthetic benchmark generator (`make_set.py`), `jt9` & `ft8nn` runners (`run_jt9.py`, `run_ours.py`, `run_real.py`), SNR threshold reporter (`report.py`) |
| `collect/` | KiwiSDR recorder, PSK Reporter MQTT capture, `verify.py` ground-truth scorer, SHA-256 manifest tools, and archived run metadata (`docs/data-collection.md`) |
| `tools/` | Validation set block runner (`run_set.py`), multi-decoder comparison (`compare_decoders.py`), ONNX exporter (`export_onnx.py`), screener trainer (`train_screener.py`), C==Python golden verifier (`golden_c.py`), JTDX & `ft8_lib` build/run drivers, and live comparison UI (`live_compare.py`) |
| `datasets/` | Read-only Git LFS submodules (`ft8-training-data`, `ft8-validation-set`; packed FLAC + `.jsonl.xz`) |

---

## Reproducing Benchmarks & Training

### Python environment & unit tests

Requires Python ≥ 3.12 with [`uv`](https://docs.astral.sh/uv/):

```sh
uv sync
uv run pytest -q
uv run python tools/ci_smoke.py                           # C == Python check on synthetic slots (no LFS data needed)
```

### Synthetic sensitivity & noise benchmarks

```sh
uv run python bench/make_set.py --out data/bench/awgn      --channel awgn      --snrs=-25:-16 --files 30 --seed 1
uv run python bench/make_set.py --out data/bench/moderate  --channel moderate  --snrs=-24:-12 --files 30 --seed 2
uv run python bench/make_set.py --out data/bench/disturbed --channel disturbed --snrs=-24:-10 --files 30 --seed 3
uv run python bench/make_set.py --out data/bench/noise     --channel awgn      --snrs=-80:-80 --files 100 --seed 9
uv run python bench/make_set.py --out data/bench/noise10k  --channel awgn      --snrs=-80:-80 --files 10000 --seed 19

uv run python bench/run_ours.py data/bench/awgn --tag nn_v4 --model models/v4/model.pt --also-classical
uv run python bench/report.py data/bench/awgn --dt-max 0.5
```

### Off-air validation set & 5-decoder comparison

Fetch and unpack the validation recordings (`docs/data-collection.md`), then run and score any block set (`small`, `third`, or `full`):

```sh
git submodule update --init
(cd datasets/ft8-validation-set && python3 tools/datarepo.py unpack runs \
  --manifest runs/r02/MANIFEST.sha256 --manifest runs/r05/MANIFEST.sha256 --manifest runs/r06/MANIFEST.sha256 \
  --manifest runs/r07/MANIFEST.sha256 --manifest runs/r08/MANIFEST.sha256 --manifest runs/r09/MANIFEST.sha256 \
  --manifest runs/r10/MANIFEST.sha256 --manifest runs/r11/MANIFEST.sha256 --manifest runs/r12/MANIFEST.sha256 \
  --manifest runs/r13/MANIFEST.sha256 --manifest runs/r14/MANIFEST.sha256)

# Optional external baselines (requires gfortran, cmake, pkg-config, fftw, boost for JTDX):
python3 tools/jtdx_build.py
sh tools/ft8lib_build.sh

# Decode & score a validation block set (small = 16 blocks, third = 265 blocks, full = 807 blocks):
for dec in ft8lib jt9 jtdx jtdx_deep; do
  python3 tools/run_set.py --set third --decoder "$dec" --tag "$dec" --workers 6
done
python3 tools/run_set.py --set third --decoder ft8nn --tag nn_v4 --workers 6

POOL="jt9,nn_v4,jtdx,jtdx_deep,ft8lib"
python3 tools/run_set.py --set third --verify-only --tag nn_v4 --others jt9,ft8lib,jtdx,jtdx_deep --pool-tags "$POOL" --list
python3 tools/compare_decoders.py --set third --tags jt9 ft8lib jtdx jtdx_deep nn_v4
```

### Retraining `models/v4` & exporting ONNX

```sh
# 1. Build residual real-noise bank from training recordings only:
uv run python tools/make_noise_bank.py datasets/ft8-training-data/recordings data/real nn_v3_opt8 data/noise/train_bank_v4.npy

# 2. Fine-tune calibrated multipath SoftDemod (v4) from models/v3 and export ONNX + screener:
uv run python train.py --cfg v4 --init models/v3/model.pt --lr 2e-4 --steps 6000 \
  --noise-bank data/noise/train_bank_v4.npy --real-noise-p 0.5 --seed 1 --workers 8 --out runs/v4_cal
uv run python tools/train_screener.py --teacher runs/v4_cal/model.pt --noise-bank data/noise/train_bank_v4.npy --out runs/v4_cal/screener.pt
uv run python tools/export_onnx.py runs/v4_cal/model.pt runs/v4_cal/ft8nn.onnx
```

Detailed development history, ablation numbers, and per-stage validations are recorded in [`STATUS.md`](STATUS.md) and [`docs/decoder-comparison.md`](docs/decoder-comparison.md).
