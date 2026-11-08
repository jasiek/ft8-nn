# ft8nn — neural soft-demodulation FT8 decoder

Goal: an FT8 decoder more sensitive than WSJT-X, packaged so other software can embed it.

Approach: keep FT8's LDPC(174,91) + CRC-14 (near-optimal, gives false-decode protection) and replace
the hand-made soft metrics with a small neural net that turns baseband symbol spectra into bit LLRs.

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
- jt9 baselines need WSJT-X's `jt9` at `/usr/local/bin/jt9`. The real-recording benchmarks need the recording
  folders described in `STATUS.md`.

## Layout

| Path | What |
|---|---|
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
| `collect/` | real-data collection: KiwiSDR recorder, PSK Reporter capture, jt9 labelling, spot scoring, run records and SHA-256 manifests (`docs/data-collection.md`) |

## Reproduce

```sh
uv run pytest -q tests                                     # encoder vs ft8code, LDPC, calibration
uv run python bench/make_set.py --out data/bench/awgn --channel awgn --snrs=-25:-16 --files 30 --seed 1
uv run python bench/run_jt9.py data/bench/awgn             # WSJT-X baseline (needs /usr/local/bin/jt9)
uv run python train.py --steps 6000 --out runs/v1
uv run python bench/run_ours.py data/bench/awgn --tag nn --model runs/v1/model.pt
uv run python bench/report.py data/bench/awgn --dt-max 0.5
```

C library:

```sh
tools/get_onnxruntime.sh                                    # official ONNX Runtime into third_party/ (once)
make -C c                                                   # c/build/libft8nn.{a,dylib|so}, c/build/ft8nn_cli
uv run python tools/export_onnx.py runs/v2/model.pt runs/v2/ft8nn.onnx
c/build/ft8nn_cli runs/v2/ft8nn.onnx slot.wav               # one line per decode; --threads N, --time
```

Benchmark scoring: 10 signals per 15 s slot, real messages from `ft8nn/data/messages.json`;
a decode counts only on exact text match, anything else is a false decode.
