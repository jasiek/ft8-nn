# ft8nn — neural soft-demodulation FT8 decoder

Goal: an FT8 decoder more sensitive than WSJT-X, packaged so other software can embed it.

Approach: keep FT8's LDPC(174,91) + CRC-14 (near-optimal, gives false-decode protection) and replace
the hand-made soft metrics with a small neural net that turns baseband symbol spectra into bit LLRs.

## System requirements

Only one machine has been tested: Apple M1 Pro (10 cores, 16 GB), macOS 26, arm64. The numbers below were
measured there. Other platforms and smaller machines have not been tried.

### Decoding with the C library (`c/`, `libft8nn`)

| | Requirement |
|---|---|
| OS | macOS arm64 (tested). The Makefile is macOS-specific: it finds ONNX Runtime via `brew` and builds a `.dylib`. Linux would need Makefile changes. |
| Build tools | C11 compiler (tested: Apple clang 21) and `make` |
| Libraries | ONNX Runtime ≥ 1.31 (`brew install onnxruntime`, about 21 MB). No other dependencies. |
| Model | `ft8nn.onnx`, 3.5 MB (exported from a trained checkpoint by `tools/export_onnx.py`) |
| Memory | about 370 MB peak resident per decoding context, on a busy 40 m slot |
| CPU | One core averages 6.4 s per 15 s slot (training recordings, idle machine). Crowded bands take longer: one busy 40 m slot took 18 s on one core while other jobs were running. To keep up in real time, every slot must finish within 15 s, so plan on at least 2 cores (`--threads N` sets ONNX Runtime's thread count). |

### Decoding with the Python reference (`ft8nn/`)

| | Requirement |
|---|---|
| Python | ≥ 3.12, managed with [uv](https://docs.astral.sh/uv/) |
| Packages | numpy, scipy, torch (the environment is about 0.9 GB on disk) |
| Model | `model.pt`, 3.4 MB |
| Memory | about 690 MB peak resident for one decoding process |
| CPU | about 4 s per slot per core on average (torch uses Accelerate on Apple Silicon) |

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

## Reproduce

```sh
uv run pytest -q tests                                     # encoder vs ft8code, LDPC, calibration
uv run python bench/make_set.py --out data/bench/awgn --channel awgn --snrs=-25:-16 --files 30 --seed 1
uv run python bench/run_jt9.py data/bench/awgn             # WSJT-X baseline (needs /usr/local/bin/jt9)
uv run python train.py --steps 6000 --out runs/v1
uv run python bench/run_ours.py data/bench/awgn --tag nn --model runs/v1/model.pt
uv run python bench/report.py data/bench/awgn --dt-max 0.5
```

Benchmark scoring: 10 signals per 15 s slot, real messages from `ft8nn/data/messages.json`;
a decode counts only on exact text match, anything else is a false decode.
